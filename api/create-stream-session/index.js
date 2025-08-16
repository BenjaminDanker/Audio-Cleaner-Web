const { v4: uuidv4 } = require('uuid')
const crypto = require('crypto')
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware')
const AzureSDKConfig = require('../shared/azureSDKConfig')

module.exports = async function (context, req) {
  // Handle OPTIONS for CORS preflight
  if (req.method === 'OPTIONS') {
    context.res = {
      status: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal, x-api-key'
      },
      body: ''
    }
    return
  }

  if (req.method !== 'POST') {
    context.res = {
      status: 405,
      headers: { 'Access-Control-Allow-Origin': '*' },
      jsonBody: { message: 'Method not allowed' }
    }
    return
  }

  const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING)
  // Allow API key clients (e.g., OBS companion) or SWA-authenticated browser clients
  const secResult = await security.checkSecurity(context, req, { requireAuth: true, allowApiKey: true })
  if (!secResult.allowed) {
    context.res = {
      status: secResult.status,
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      jsonBody: secResult.body
    }
    return
  }

  try {
    const body = req.body || {}
    const languagesRequested = Array.isArray(body.languagesRequested) ? body.languagesRequested : ['en']

    // Get user info for billing (works with both SWA auth and API key auth)
    const userId = secResult.userInfo?.userId
    if (!userId) {
      context.res = {
        status: 400,
        headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
        jsonBody: { message: 'User identification required for streaming' }
      }
      return
    }

    // Check account balance for streaming (similar to enqueue-job)
    if (process.env.COSMOS_CONNECTION_STRING) {
      try {
        const cosmosClient = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING)
        const database = cosmosClient.database(process.env.COSMOS_DB_NAME || 'app')
        // Use consistent container name for accounts
        const accountContainer = database.container('accounts')
        
        const { resource: account } = await accountContainer.item(userId, userId).read()
        if (!account) {
          context.res = {
            status: 402,
            headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
            jsonBody: { message: 'Account not found. Please set up billing.' }
          }
          return
        }

        const minStreamingBalance = parseFloat(process.env.MIN_STREAMING_BALANCE || '1.0')
        if (account.balance < minStreamingBalance) {
          context.res = {
            status: 402,
            headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
            jsonBody: { 
              message: `Insufficient balance for streaming. Minimum $${minStreamingBalance} required.`,
              balance: account.balance 
            }
          }
          return
        }
      } catch (accountErr) {
        context.log.warn('Could not check account balance:', accountErr.message)
        // Continue anyway in case billing is not fully set up
      }
    }

    // Trigger container scaling if configured - MUST succeed for streaming to work
    const scalingResult = await triggerStreamingScale(context)
    if (!scalingResult.success) {
      context.res = {
        status: 503,
        headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
        jsonBody: { 
          message: 'Streaming service unavailable. Unable to scale processing capacity.',
          error: scalingResult.error
        }
      }
      return
    }

    // Persist/allocate to come later
    const sessionId = uuidv4()
    // Short-lived, signed session token to gate WS connections
    // Token format: base64url(payload).base64url(HMAC_SHA256(payload, key))
    // Load signing key; support both raw string and base64/base64url encoded secrets
    const signingKeyRaw = process.env.STREAM_SESSION_SIGNING_KEY
    const signingKey = getSigningKeyBytes(signingKeyRaw)
    const ttlMinutes = parseInt(process.env.STREAM_SESSION_TTL_MINUTES || '30', 10)
    const payload = {
      sid: sessionId,
      userId: userId, // Include userId in token for billing
      mode: 'stream',
      exp: Math.floor(Date.now() / 1000) + ttlMinutes * 60,
    }
    const payloadStr = Buffer.from(JSON.stringify(payload)).toString('base64url')
    if (!signingKey) {
      throw new Error('STREAM_SESSION_SIGNING_KEY not configured')
    }
    const mac = crypto
      .createHmac('sha256', signingKey)
      .update(payloadStr)
      .digest('base64url')
    const sessionToken = `${payloadStr}.${mac}`

    // Determine WebSocket URL based on environment
    let wsUrl
    const streamingEndpoint = process.env.STREAMING_ENDPOINT
    if (streamingEndpoint) {
      // Production: use configured streaming container endpoint
      const wsProtocol = streamingEndpoint.startsWith('https://') ? 'wss://' : 'ws://'
      const endpoint = streamingEndpoint.replace(/^https?:\/\//, '')
      // SECURITY FIX: Pass token in response, not URL to avoid logging
      wsUrl = `${wsProtocol}${endpoint}/stream/${sessionId}`
    } else {
      // Development: return relative path for local proxy
      wsUrl = `/stream/${sessionId}`
    }

    context.res = {
      status: 200,
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      jsonBody: {
        sessionId,
        wsUrl,
        languagesRequested,
        token: sessionToken, // Client should pass this as header, not URL param
        expiresInMinutes: ttlMinutes,
      },
    }
  } catch (err) {
    context.log.error('create-stream-session error', err)
    context.res = {
      status: 500,
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      jsonBody: { message: 'Failed to create stream session' },
    }
  }
}

/**
 * Trigger scaling of streaming container instances using Azure Container Apps API
 * @returns {Promise<{success: boolean, error?: string}>}
 */
async function triggerStreamingScale(context) {
  // Dev bypass: allow local runs without Azure credentials or network
  if (process.env.DISABLE_STREAMING_SCALE === '1') {
    context.log.warn('DISABLE_STREAMING_SCALE=1 set; skipping Azure Container Apps scaling.');
    return { success: true };
  }
  try {
    const subscriptionId = process.env.AZURE_SUBSCRIPTION_ID
    const resourceGroupName = process.env.AZURE_RESOURCE_GROUP_NAME
    const containerAppName = process.env.STREAMING_CONTAINER_APP_NAME
    
    if (!subscriptionId || !resourceGroupName || !containerAppName) {
      const error = 'Container scaling not configured (missing env vars)'
      context.log.error(error)
      return { success: false, error }
    }

    // Use managed identity to authenticate with Azure
    const { DefaultAzureCredential } = require('@azure/identity')
    const { ContainerAppsAPIClient } = require('@azure/arm-appcontainers')
    
    const credential = new DefaultAzureCredential()
    const client = new ContainerAppsAPIClient(credential, subscriptionId)

    // Get current container app configuration
    const containerApp = await client.containerApps.get(resourceGroupName, containerAppName)
    
    if (!containerApp || !containerApp.configuration?.template) {
      const error = 'Could not retrieve container app configuration for scaling'
      context.log.error(error)
      return { success: false, error }
    }

    const currentMinReplicas = containerApp.configuration.template.scale?.minReplicas || 0
    const currentMaxReplicas = containerApp.configuration.template.scale?.maxReplicas || 10
    const targetMinReplicas = Math.max(1, currentMinReplicas) // Ensure at least 1 replica for streaming

    // Only scale if we need to increase min replicas
    if (currentMinReplicas >= targetMinReplicas) {
      context.log.info(`Container already scaled appropriately (min: ${currentMinReplicas}, target: ${targetMinReplicas})`)
      return { success: true }
    }

    // Update container app with new scaling configuration
    const updatePayload = {
      ...containerApp,
      configuration: {
        ...containerApp.configuration,
        template: {
          ...containerApp.configuration.template,
          scale: {
            ...containerApp.configuration.template.scale,
            minReplicas: targetMinReplicas,
            maxReplicas: Math.max(targetMinReplicas, currentMaxReplicas)
          }
        }
      }
    }

    context.log.info(`Scaling streaming container from ${currentMinReplicas} to ${targetMinReplicas} min replicas`)
    
    // Apply the update and WAIT for it to complete
    const updateOperation = await client.containerApps.beginUpdate(resourceGroupName, containerAppName, updatePayload)
    await updateOperation.pollUntilDone()
    
    context.log.info('Container scaling completed successfully')
    return { success: true }

  } catch (scaleErr) {
    const error = `Failed to scale streaming container: ${scaleErr.message}`
    context.log.error(error)
    return { success: false, error }
  }
}

// Helpers
function getSigningKeyBytes(raw) {
  if (!raw) throw new Error('STREAM_SESSION_SIGNING_KEY not configured')
  // Try base64 first
  try {
    // Standard base64
    const buf = Buffer.from(raw, 'base64')
    if (buf.length > 0) return buf
  } catch {}
  try {
    // Base64url (no padding)
    const pad = raw.length % 4 === 0 ? '' : '='.repeat(4 - (raw.length % 4))
    const norm = raw.replace(/-/g, '+').replace(/_/g, '/') + pad
    const buf = Buffer.from(norm, 'base64')
    if (buf.length > 0) return buf
  } catch {}
  // Fallback to raw string bytes
  return Buffer.from(raw, 'utf8')
}

const crypto = require('crypto')
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware')
const AzureSDKConfig = require('../shared/azureSDKConfig')

const DB_NAME = process.env.COSMOS_DB_NAME || 'app'
const ACCOUNTS_CONTAINER = 'accounts'  // Store API keys in user accounts

module.exports = async function (context, req) {
  const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING)
  // Require SWA user for key admin; do not allow API key auth here
  const sec = await security.checkSecurity(context, req, { requireAuth: true, allowApiKey: false })
  if (!sec.allowed) {
    context.res = { status: sec.status, headers: security.getSecurityHeaders(), jsonBody: sec.body }
    return
  }

  const userId = sec.userInfo?.userId
  if (!userId) {
    context.res = { 
      status: 400, 
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      jsonBody: { message: 'User identification required' } 
    }
    return
  }

  // Only basic CORS headers
  const headers = { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() }
  const method = req.method

  const cosmos = security.cosmosClient || AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING)
  const db = cosmos.database(DB_NAME)
  const container = db.container(ACCOUNTS_CONTAINER)

  try {
    if (method === 'OPTIONS') {
      context.res = { status: 200, headers }
      return
    }

    // Get user account
    const { resource: account } = await container.item(userId, userId).read()
    if (!account) {
      context.res = { status: 404, headers, jsonBody: { message: 'Account not found' } }
      return
    }

    if (method === 'GET') {
      // Return current API key info (but not the key itself)
      const keyInfo = account.apiKeyHash ? {
        hasKey: true,
        name: account.apiKeyName || 'API Key',
        createdAt: account.apiKeyCreatedAt
      } : {
        hasKey: false
      }
      
      context.res = { status: 200, headers, jsonBody: keyInfo }
      return
    }

    if (method === 'POST') {
      const body = req.body || {}
      const name = body.name || 'OBS Streaming Key'
      
      // Don't allow creating a key if one already exists
      if (account.apiKeyHash) {
        context.res = { status: 409, headers, jsonBody: { 
          message: 'API key already exists. Use PUT to rotate it.' 
        }}
        return
      }
      
      // Generate API key with user identification: userId_randomKey
      const keyPart = crypto.randomBytes(32).toString('base64url')
      const key = `${userId}_${keyPart}`
      const hash = crypto.createHash('sha256').update(key).digest('hex')
      
      // Update account with new API key
      const updatedAccount = {
        ...account,
        apiKeyHash: hash,
        apiKeyName: name,
        apiKeyCreatedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      await container.item(userId, userId).replace(updatedAccount)
      context.res = { status: 201, headers, jsonBody: { 
        name: name, 
        apiKey: key,
        message: 'API key created successfully'
      }}
      return
    }

    if (method === 'DELETE') {
      // Remove API key from account
      if (!account.apiKeyHash) {
        context.res = { status: 404, headers, jsonBody: { message: 'No API key found' } }
        return
      }
      
      const updatedAccount = {
        ...account,
        apiKeyHash: null,
        apiKeyName: null,
        apiKeyCreatedAt: null,
        updatedAt: new Date().toISOString()
      }
      
      await container.item(userId, userId).replace(updatedAccount)
      context.res = { status: 200, headers, jsonBody: { message: 'API key deleted' } }
      return
    }

    if (method === 'PUT') {
      // API Key rotation
      if (!account.apiKeyHash) {
        context.res = { status: 404, headers, jsonBody: { message: 'No API key to rotate' } }
        return
      }
      
      // Generate new key with user identification: userId_randomKey  
      const newKeyPart = crypto.randomBytes(32).toString('base64url')
      const newKey = `${userId}_${newKeyPart}`
      const newHash = crypto.createHash('sha256').update(newKey).digest('hex')
      
      // Update account with rotated key
      const updatedAccount = {
        ...account,
        apiKeyHash: newHash,
        apiKeyCreatedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      await container.item(userId, userId).replace(updatedAccount)
      context.res = { status: 200, headers, jsonBody: { 
        name: account.apiKeyName,
        apiKey: newKey,
        message: 'API key rotated successfully'
      }}
      return
    }

    context.res = { status: 405, headers, jsonBody: { message: 'Method not allowed' } }
  } catch (err) {
    context.log.error('api-keys error', err)
    context.res = { status: 500, headers, jsonBody: { message: 'Server error' } }
  }
}

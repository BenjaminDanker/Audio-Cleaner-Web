const { v4: uuidv4 } = require('uuid')
const crypto = require('crypto')
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware')

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

    // Persist/allocate to come later
    const sessionId = uuidv4()
  // Short-lived session token (HMAC or random) – keep simple random for now
  const sessionToken = crypto.randomBytes(18).toString('base64url')

  // Return relative WS path so local dev proxy can reroute to streaming service
  const wsUrl = `/stream/${sessionId}?t=${sessionToken}`

    context.res = {
      status: 200,
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      jsonBody: {
        sessionId,
        wsUrl,
    languagesRequested,
  token: sessionToken,
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

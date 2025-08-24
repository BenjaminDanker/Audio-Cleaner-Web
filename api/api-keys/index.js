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
    context.res = { status: sec.status, headers: security.getSecurityHeaders(), body: sec.body }
    return
  }

  const userId = sec.userInfo?.userId
  if (!userId) {
    context.res = { 
      status: 400, 
      headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() },
      body: { message: 'User identification required' } 
    }
    return
  }

  // Only basic CORS headers
  const headers = { 
    'Access-Control-Allow-Origin': '*', 
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
    'Cache-Control': 'no-store',
    ...security.getSecurityHeaders() 
  }
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
      context.res = { status: 404, headers, body: { message: 'Account not found' } }
      return
    }

  // GET removed in simplified model

    if (method === 'POST') {
      // Always generate a new key (create or rotate)
      const keyPart = crypto.randomBytes(32).toString('base64url')
      const key = `${userId}_${keyPart}`
      const hash = crypto.createHash('sha256').update(key).digest('hex')
      
      // Update account with new API key (no name field in simplified model)
      const updatedAccount = {
        ...account,
        apiKeyHash: hash,
        apiKeyCreatedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      await container.item(userId, userId).replace(updatedAccount)
      const message = account.apiKeyHash ? 'API key rotated successfully' : 'API key created successfully'
      context.res = { status: 201, headers, body: { apiKey: key, message } }
      return
    }

    if (method === 'DELETE') {
      // Revocation/deletion not supported in simplified single-key model
      context.res = { status: 405, headers, body: { message: 'Method not allowed' } }
      return
    }

    if (method === 'PUT') {
      // API Key rotation (kept for compatibility)
      if (!account.apiKeyHash) {
        context.res = { status: 404, headers, body: { message: 'No API key to rotate' } }
        return
      }
      
      const newKeyPart = crypto.randomBytes(32).toString('base64url')
      const newKey = `${userId}_${newKeyPart}`
      const newHash = crypto.createHash('sha256').update(newKey).digest('hex')
      
      const updatedAccount = {
        ...account,
        apiKeyHash: newHash,
        apiKeyCreatedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
      
      await container.item(userId, userId).replace(updatedAccount)
      context.res = { status: 200, headers, body: { apiKey: newKey, message: 'API key rotated successfully' } }
      return
    }

    context.res = { status: 405, headers, body: { message: 'Method not allowed' } }
  } catch (err) {
    context.log.error('api-keys error', err)
    context.res = { status: 500, headers, body: { message: 'Server error' } }
  }
}

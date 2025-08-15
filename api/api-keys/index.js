const { CosmosClient } = require('@azure/cosmos')
const crypto = require('crypto')
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware')

const DB_NAME = process.env.COSMOS_DB_NAME || 'app'
const KEYS_CONTAINER = process.env.COSMOS_API_KEYS_CONTAINER || 'ApiKeys'

module.exports = async function (context, req) {
  const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING)
  // Require SWA user for key admin; do not allow API key auth here
  const sec = await security.checkSecurity(context, req, { requireAuth: true, allowApiKey: false })
  if (!sec.allowed) {
    context.res = { status: sec.status, headers: security.getSecurityHeaders(), jsonBody: sec.body }
    return
  }

  // Only basic CORS headers
  const headers = { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() }
  const method = req.method

  const cosmos = security.cosmosClient || new CosmosClient(process.env.COSMOS_CONNECTION_STRING)
  const db = cosmos.database(DB_NAME)
  const container = db.container(KEYS_CONTAINER)

  try {
    if (method === 'OPTIONS') {
      context.res = { status: 200, headers }
      return
    }

    if (method === 'GET') {
      const { resources } = await container.items.query({ query: 'SELECT c.id, c.name, c.isActive, c.createdAt FROM c ORDER BY c.createdAt DESC' }).fetchAll()
      context.res = { status: 200, headers, jsonBody: { keys: resources || [] } }
      return
    }

    if (method === 'POST') {
      const body = req.body || {}
      const name = body.name || 'obs'
      const key = crypto.randomBytes(24).toString('base64url')
      const hash = crypto.createHash('sha256').update(key).digest('hex')
      const doc = {
        id: crypto.randomUUID(),
        name,
        apiKeyHash: hash,
        isActive: true,
        createdAt: new Date().toISOString(),
        createdBy: sec.userInfo?.userId || 'system',
      }
      await container.items.create(doc)
      context.res = { status: 201, headers, jsonBody: { id: doc.id, name: doc.name, apiKey: key } }
      return
    }

    if (method === 'DELETE') {
      const id = context.bindingData.id
      if (!id) {
        context.res = { status: 400, headers, jsonBody: { message: 'Missing id' } }
        return
      }
      // Soft delete: set isActive=false
      const { resource } = await container.item(id, id).read()
      if (!resource) {
        context.res = { status: 404, headers, jsonBody: { message: 'Not found' } }
        return
      }
      resource.isActive = false
      await container.item(id, id).replace(resource)
      context.res = { status: 200, headers, jsonBody: { id, isActive: false } }
      return
    }

    context.res = { status: 405, headers, jsonBody: { message: 'Method not allowed' } }
  } catch (err) {
    context.log.error('api-keys error', err)
    context.res = { status: 500, headers, jsonBody: { message: 'Server error' } }
  }
}

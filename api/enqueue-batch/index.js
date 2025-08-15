const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware')

module.exports = async function (context, req) {
  if (req.method === 'OPTIONS') {
    context.res = { status: 200, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal' }, body: '' }
    return
  }
  if (req.method !== 'POST') {
    context.res = { status: 405, headers: { 'Access-Control-Allow-Origin': '*' }, jsonBody: { error: 'Method not allowed' } }
    return
  }
  const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING)
  const sec = await security.checkSecurity(context, req, { requireAuth: true })
  if (!sec.allowed) {
    context.res = { status: sec.status, headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() }, jsonBody: sec.body }
    return
  }
  try {
    const body = req.body || {}
    const items = Array.isArray(body.items) ? body.items : []
    if (!items.length) {
      context.res = { status: 400, headers: { 'Access-Control-Allow-Origin': '*' }, jsonBody: { error: 'items[] required' } }
      return
    }
    // Minimal placeholder: return accepted and let frontend enqueue individually for now
    context.res = { status: 202, headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() }, jsonBody: { accepted: items.length } }
  } catch (e) {
    context.log.error('enqueue-batch error', e)
    context.res = { status: 500, headers: { 'Access-Control-Allow-Origin': '*', ...security.getSecurityHeaders() }, jsonBody: { error: 'Failed to enqueue batch' } }
  }
}

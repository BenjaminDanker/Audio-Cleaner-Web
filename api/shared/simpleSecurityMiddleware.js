/**
 * Simple Security Middleware for Azure Functions
 * Simplified version focusing on basic rate limiting and authentication
 */

const { CosmosClient } = require('@azure/cosmos');
const crypto = require('crypto');

class SimpleSecurityMiddleware {
    constructor(cosmosConnectionString) {
        this.cosmosClient = (cosmosConnectionString && 
                           cosmosConnectionString !== 'file-based-for-local-dev' && 
                           cosmosConnectionString.includes('AccountEndpoint')) 
                           ? new CosmosClient(cosmosConnectionString) : null;
        this.initialized = false;
        // Load API keys from env for lightweight validation without Cosmos
        // STREAMING_API_KEY supports a single key, STREAMING_API_KEYS supports comma-separated keys
        const single = process.env.STREAMING_API_KEY ? [process.env.STREAMING_API_KEY] : [];
        const many = (process.env.STREAMING_API_KEYS || '')
            .split(',')
            .map(k => k.trim())
            .filter(k => !!k);
        this.localApiKeys = new Set([...single, ...many]);
    }

    /**
     * Simple security check - basic auth validation only
     */
    async checkSecurity(context, req, options = {}) {
        try {
            // Simple auth check
            if (options.requireAuth !== false) {
                // Allow either SWA auth or API key auth (for non-browser clients like OBS companion)
                const allowApiKey = options.allowApiKey !== false; // default true
                const apiKey = allowApiKey ? this.getApiKey(req) : null;
                if (apiKey) {
                    const valid = await this.validateApiKey(apiKey);
                    if (!valid) {
                        return { allowed: false, status: 401, body: { error: 'Invalid API key' } };
                    }
                } else {
                    const authCheck = this.validateAuthentication(req);
                    if (!authCheck.valid) {
                        return {
                            allowed: false,
                            status: 401,
                            body: { error: 'Authentication required' }
                        };
                    }
                }
            }

            return {
                allowed: true,
                userInfo: options.requireAuth !== false ? this.getUserInfo(req) : null
            };

        } catch (error) {
            if (context) {
                context.log.error('Security check error:', error.message);
            }
            
            // Fail securely - deny request if security check fails
            return {
                allowed: false,
                status: 500,
                body: { error: 'Security validation failed' }
            };
        }
    }

    /**
     * Extract API key from headers
     */
    getApiKey(req) {
        const headerKey = req.headers['x-api-key'];
        if (headerKey && typeof headerKey === 'string' && headerKey.trim().length > 0) {
            return headerKey.trim();
        }
        const auth = req.headers['authorization'];
        if (auth && typeof auth === 'string' && auth.toLowerCase().startsWith('bearer ')) {
            return auth.substring(7).trim();
        }
        return null;
    }

    /**
     * Validate API key via env allowlist or Cosmos (optional if configured)
     */
    async validateApiKey(apiKey) {
        if (!apiKey) return false;
        if (this.localApiKeys && this.localApiKeys.has(apiKey)) return true;
        // Optional Cosmos lookup: container ApiKeys with id/apiKey and isActive=true
        if (!this.cosmosClient) return false; // No DB configured, only env keys allowed
        try {
            const db = this.cosmosClient.database(process.env.COSMOS_DB_NAME || 'app');
            const container = db.container(process.env.COSMOS_API_KEYS_CONTAINER || 'ApiKeys');
            const hash = crypto.createHash('sha256').update(apiKey).digest('hex');
            const query = {
                query: 'SELECT TOP 1 c.id FROM c WHERE (c.apiKey = @k OR c.apiKeyHash = @h) AND c.isActive = true',
                parameters: [
                    { name: '@k', value: apiKey },
                    { name: '@h', value: hash },
                ],
            };
            const { resources } = await container.items.query(query).fetchAll();
            return Array.isArray(resources) && resources.length > 0;
        } catch {
            return false;
        }
    }

    /**
     * Simple authentication check
     */
    validateAuthentication(req) {
        const clientPrincipal = req.headers['x-ms-client-principal'];
        
        if (!clientPrincipal) {
            return { valid: false, error: 'No authentication' };
        }

        try {
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            return { valid: true, principal };
        } catch (error) {
            return { valid: false, error: 'Invalid authentication' };
        }
    }

    /**
     * Get user info from request
     */
    getUserInfo(req) {
        try {
            const clientPrincipal = req.headers['x-ms-client-principal'];
            if (!clientPrincipal) {
                return null;
            }
            
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            return {
                userId: principal.userId || principal.userDetails,
                email: principal.userDetails || principal.userId
            };
        } catch {
            return null;
        }
    }

    /**
     * Get basic security headers
     */
    getSecurityHeaders() {
        return {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal, x-api-key',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Cache-Control': 'no-store'
        };
    }
}

module.exports = SimpleSecurityMiddleware;

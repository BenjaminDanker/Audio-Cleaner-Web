/**
 * Simple Security Middleware for Azure Functions
 * Simplified version focusing on basic rate limiting and authentication
 */

const { CosmosClient } = require('@azure/cosmos');
const crypto = require('crypto');
const AzureSDKConfig = require('./azureSDKConfig');

class SimpleSecurityMiddleware {
    constructor(cosmosConnectionString) {
        this.cosmosClient = (cosmosConnectionString && 
                           cosmosConnectionString !== 'file-based-for-local-dev' && 
                           cosmosConnectionString.includes('AccountEndpoint')) 
                           ? AzureSDKConfig.createCosmosClient(cosmosConnectionString) : null;
        this.initialized = false;
        // Load API keys from env for lightweight validation without Cosmos (non-production only)
        // In production, disable env-based API keys to avoid accidental bypass
        if ((process.env.NODE_ENV || '').toLowerCase() !== 'production') {
            const single = process.env.STREAMING_API_KEY ? [process.env.STREAMING_API_KEY] : [];
            const many = (process.env.STREAMING_API_KEYS || '')
                .split(',')
                .map(k => k.trim())
                .filter(k => !!k);
            this.localApiKeys = new Set([...single, ...many]);
        } else {
            this.localApiKeys = new Set();
        }
    }

    /**
     * Simple security check - basic auth validation only
     */
    async checkSecurity(context, req, options = {}) {
        try {
            let userInfo = null;
            let authMethod = null;
            
            // Simple auth check
            if (options.requireAuth !== false) {
                // Allow either SWA auth or API key auth (for non-browser clients like OBS companion)
                const allowApiKey = options.allowApiKey !== false; // default true
                const apiKey = allowApiKey ? this.getApiKey(req) : null;
                
                if (apiKey) {
                    const keyValidation = await this.validateApiKey(apiKey);
                    if (!keyValidation) {
                        return { allowed: false, status: 401, body: { error: 'Invalid API key' } };
                    }
                    
                    // Handle both boolean (env keys) and object (Cosmos keys) returns
                    if (typeof keyValidation === 'object' && keyValidation.userId) {
                        userInfo = { userId: keyValidation.userId, keyId: keyValidation.keyId };
                        authMethod = 'apikey';
                    } else {
                        // Environment API key - no user info available
                        userInfo = null;
                        authMethod = 'apikey';
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
                    userInfo = this.getUserInfo(req);
                    authMethod = 'swa';
                }
            }

            return {
                allowed: true,
                userInfo: userInfo,
                authMethod: authMethod
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
     * @param {string} apiKey - The API key to validate
     * @returns {boolean|object} - false if invalid, true if valid env key, or {userId, keyId} if Cosmos key
     */
    async validateApiKey(apiKey) {
        if (!apiKey) return false;
    if (this.localApiKeys && this.localApiKeys.size > 0 && this.localApiKeys.has(apiKey)) return true;
        
        // Check if API key contains user identification (format: userId_randomKey)
        if (!this.cosmosClient) return false; // No DB configured, only env keys allowed
        
        try {
            // Parse API key format: userId_randomKey
            const parts = apiKey.split('_');
            if (parts.length !== 2) {
                return false; // Invalid format
            }
            
            const [userId, keyPart] = parts;
            if (!userId || !keyPart || keyPart.length < 32) {
                return false; // Invalid user ID or key too short
            }
            
            const db = this.cosmosClient.database(process.env.COSMOS_DB_NAME || 'app');
            const container = db.container('accounts');
            const hash = crypto.createHash('sha256').update(apiKey).digest('hex');
            
            // Only check the specific user's account - prevents cross-user access
            const { resource: account } = await container.item(userId, userId).read();
            if (!account || !account.apiKeyHash) {
                return false; // User not found or no API key set
            }
            
            // Compare hash against this specific user's stored hash only
            // Use constant-time comparison to prevent timing attacks
            if (this.constantTimeEquals(account.apiKeyHash, hash)) {
                return {
                    userId: userId
                };
            }
            
            return false;
        } catch {
            return false;
        }
    }

    /**
     * Constant-time string comparison to prevent timing attacks
     */
    constantTimeEquals(a, b) {
        if (!a || !b || a.length !== b.length) {
            return false;
        }
        
        let result = 0;
        for (let i = 0; i < a.length; i++) {
            result |= a.charCodeAt(i) ^ b.charCodeAt(i);
        }
        return result === 0;
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

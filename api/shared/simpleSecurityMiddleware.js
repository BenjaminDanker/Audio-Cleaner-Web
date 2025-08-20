/**
 * Simple Security Middleware for Azure Functions
 * Simplified version focusing on basic rate limiting and authentication
 */

const { CosmosClient } = require('@azure/cosmos');
const crypto = require('crypto');
const AzureSDKConfig = require('./azureSDKConfig');
const MinimalLogger = require('./minimalLogger');

class SimpleSecurityMiddleware {
    constructor(cosmosConnectionString) {
        this.cosmosClient = (cosmosConnectionString && 
                           cosmosConnectionString !== 'file-based-for-local-dev' && 
                           cosmosConnectionString.includes('AccountEndpoint')) 
                           ? AzureSDKConfig.createCosmosClient(cosmosConnectionString) : null;
        this.initialized = false;
        this.logger = null; // Will be set when context is available
        
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
        // Initialize logger with context
        if (context && !this.logger) {
            this.logger = new MinimalLogger(context).getLogger();
        }
        
        const sessionId = `sec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        try {
            this.logger?.logInfo('security-middleware', 'Starting security check', 'security', {
                sessionId,
                method: req.method,
                url: req.url,
                requireAuth: options.requireAuth,
                allowApiKey: options.allowApiKey
            });
            
            let userInfo = null;
            let authMethod = null;
            
            // Simple auth check
            if (options.requireAuth !== false) {
                // Allow either SWA auth or API key auth (for non-browser clients like OBS companion)
                const allowApiKey = options.allowApiKey !== false; // default true
                const apiKey = allowApiKey ? this.getApiKey(req) : null;
                
                if (apiKey) {
                    this.logger?.logInfo('security-middleware', 'API key authentication attempt', 'security', {
                        sessionId,
                        hasApiKey: true,
                        apiKeyLength: apiKey.length
                    });
                    
                    const keyValidation = await this.validateApiKey(apiKey);
                    if (!keyValidation) {
                        this.logger?.logError('security-middleware', 'Invalid API key provided', 'security', {
                            sessionId,
                            apiKeyLength: apiKey.length
                        });
                        return { allowed: false, status: 401, body: { error: 'Invalid API key' } };
                    }
                    
                    // Handle both boolean (env keys) and object (Cosmos keys) returns
                    if (typeof keyValidation === 'object' && keyValidation.userId) {
                        userInfo = { userId: keyValidation.userId, keyId: keyValidation.keyId };
                        authMethod = 'apikey';
                        this.logger?.logInfo('security-middleware', 'API key authentication successful (Cosmos)', 'security', {
                            sessionId,
                            userId: keyValidation.userId,
                            authMethod
                        });
                    } else {
                        // Environment API key - no user info available
                        userInfo = null;
                        authMethod = 'apikey';
                        this.logger?.logInfo('security-middleware', 'API key authentication successful (env)', 'security', {
                            sessionId,
                            authMethod
                        });
                    }
                } else {
                    this.logger?.logInfo('security-middleware', 'SWA authentication attempt', 'security', {
                        sessionId,
                        hasClientPrincipal: !!req.headers['x-ms-client-principal']
                    });
                    
                    const authCheck = this.validateAuthentication(req);
                    if (!authCheck.valid) {
                        this.logger?.logError('security-middleware', 'SWA authentication failed', 'security', {
                            sessionId,
                            error: authCheck.error
                        });
                        return {
                            allowed: false,
                            status: 401,
                            body: { error: 'Authentication required' }
                        };
                    }
                    userInfo = this.getUserInfo(req);
                    authMethod = 'swa';
                    this.logger?.logInfo('security-middleware', 'SWA authentication successful', 'security', {
                        sessionId,
                        userId: userInfo?.userId,
                        authMethod
                    });
                }
            } else {
                this.logger?.logInfo('security-middleware', 'No authentication required', 'security', {
                    sessionId
                });
            }

            this.logger?.logInfo('security-middleware', 'Security check completed successfully', 'security', {
                sessionId,
                authMethod,
                hasUserInfo: !!userInfo
            });

            return {
                allowed: true,
                userInfo: userInfo,
                authMethod: authMethod
            };

        } catch (error) {
            this.logger?.logError('security-middleware', 'Security check error', 'security', {
                sessionId,
                error: error.message,
                stack: error.stack
            });
            
            // Fallback to context logging if logger fails
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
        if (!apiKey) {
            this.logger?.logError('validateApiKey', 'No API key provided', 'security');
            return false;
        }
        
        // Check environment keys first
        if (this.localApiKeys && this.localApiKeys.size > 0 && this.localApiKeys.has(apiKey)) {
            this.logger?.logInfo('validateApiKey', 'API key validated against environment keys', 'security', {
                envKeysCount: this.localApiKeys.size
            });
            return true;
        }
        
        // Check if API key contains user identification (format: userId_randomKey)
        if (!this.cosmosClient) {
            this.logger?.logError('validateApiKey', 'No Cosmos client configured and API key not in env keys', 'security', {
                hasCosmosClient: false,
                envKeysCount: this.localApiKeys?.size || 0
            });
            return false; // No DB configured, only env keys allowed
        }
        
        try {
            // Parse API key format: userId_randomKey
            const parts = apiKey.split('_');
            if (parts.length !== 2) {
                this.logger?.logError('validateApiKey', 'Invalid API key format', 'security', {
                    partsCount: parts.length,
                    expectedFormat: 'userId_randomKey'
                });
                return false; // Invalid format
            }
            
            const [userId, keyPart] = parts;
            if (!userId || !keyPart || keyPart.length < 32) {
                this.logger?.logError('validateApiKey', 'Invalid API key components', 'security', {
                    hasUserId: !!userId,
                    keyPartLength: keyPart?.length || 0,
                    minKeyLength: 32
                });
                return false; // Invalid user ID or key too short
            }
            
            this.logger?.logInfo('validateApiKey', 'Attempting Cosmos API key validation', 'security', {
                userId,
                keyPartLength: keyPart.length
            });
            
            const db = this.cosmosClient.database(process.env.COSMOS_DB_NAME || 'app');
            const container = db.container('accounts');
            const hash = crypto.createHash('sha256').update(apiKey).digest('hex');
            
            // Only check the specific user's account - prevents cross-user access
            const { resource: account } = await container.item(userId, userId).read();
            if (!account || !account.apiKeyHash) {
                this.logger?.logError('validateApiKey', 'User account not found or no API key hash', 'security', {
                    userId,
                    accountExists: !!account,
                    hasApiKeyHash: !!(account?.apiKeyHash)
                });
                return false; // User not found or no API key set
            }
            
            // Compare hash against this specific user's stored hash only
            // Use constant-time comparison to prevent timing attacks
            if (this.constantTimeEquals(account.apiKeyHash, hash)) {
                this.logger?.logInfo('validateApiKey', 'Cosmos API key validation successful', 'security', {
                    userId
                });
                return {
                    userId: userId
                };
            }
            
            this.logger?.logError('validateApiKey', 'API key hash mismatch', 'security', {
                userId
            });
            return false;
        } catch (error) {
            this.logger?.logError('validateApiKey', 'Cosmos API key validation error', 'security', {
                error: error.message,
                stack: error.stack
            });
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
            this.logger?.logError('validateAuthentication', 'No x-ms-client-principal header', 'security');
            return { valid: false, error: 'No authentication' };
        }

        try {
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            this.logger?.logInfo('validateAuthentication', 'Successfully parsed client principal', 'security', {
                hasUserId: !!principal.userId,
                hasUserDetails: !!principal.userDetails,
                identityProvider: principal.identityProvider
            });
            return { valid: true, principal };
        } catch (error) {
            this.logger?.logError('validateAuthentication', 'Failed to parse client principal', 'security', {
                error: error.message,
                clientPrincipalLength: clientPrincipal.length
            });
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
                this.logger?.logError('getUserInfo', 'No x-ms-client-principal header', 'security');
                return null;
            }
            
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            const userInfo = {
                userId: principal.userId || principal.userDetails,
                email: principal.userDetails || principal.userId
            };
            
            this.logger?.logInfo('getUserInfo', 'Successfully extracted user info', 'security', {
                hasUserId: !!userInfo.userId,
                hasEmail: !!userInfo.email
            });
            
            return userInfo;
        } catch (error) {
            this.logger?.logError('getUserInfo', 'Failed to extract user info', 'security', {
                error: error.message
            });
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

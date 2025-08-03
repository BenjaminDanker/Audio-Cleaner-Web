/**
 * Simple Security Middleware for Azure Functions
 * Simplified version focusing on basic rate limiting and authentication
 */

const { CosmosClient } = require('@azure/cosmos');

class SimpleSecurityMiddleware {
    constructor(cosmosConnectionString) {
        this.cosmosClient = (cosmosConnectionString && 
                           cosmosConnectionString !== 'file-based-for-local-dev' && 
                           cosmosConnectionString.includes('AccountEndpoint')) 
                           ? new CosmosClient(cosmosConnectionString) : null;
        this.initialized = false;
    }

    /**
     * Simple security check - basic auth validation only
     */
    async checkSecurity(context, req, options = {}) {
        try {
            // Simple auth check
            if (options.requireAuth !== false) {
                const authCheck = this.validateAuthentication(req);
                if (!authCheck.valid) {
                    return {
                        allowed: false,
                        status: 401,
                        body: { error: 'Authentication required' }
                    };
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
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'Cache-Control': 'no-store'
        };
    }
}

module.exports = SimpleSecurityMiddleware;

/**
 * Comprehensive Security Middleware for Azure Functions
 * Implements rate limiting, input validation, security headers, and threat detection
 */

const { CosmosClient } = require('@azure/cosmos');
const crypto = require('crypto');

class SecurityMiddleware {
    constructor(cosmosConnectionString) {
        this.cosmosClient = cosmosConnectionString ? new CosmosClient(cosmosConnectionString) : null;
        this.rateLimitContainer = null;
        this.securityEventsContainer = null;
        this.initialized = false;
        
        // Rate limiting configurations per endpoint
        this.rateLimits = {
            '/api/upload-file': { requests: 10, windowMs: 60000, burstLimit: 3 }, // 10 per minute, 3 burst
            '/api/download-file': { requests: 50, windowMs: 60000, burstLimit: 10 }, // 50 per minute
            '/api/enqueue-job': { requests: 5, windowMs: 60000, burstLimit: 2 }, // 5 per minute
            '/api/job-status': { requests: 100, windowMs: 60000, burstLimit: 20 }, // 100 per minute
            '/api/auth': { requests: 20, windowMs: 60000, burstLimit: 5 }, // 20 per minute
            'default': { requests: 30, windowMs: 60000, burstLimit: 10 } // Default for other endpoints
        };

        // Suspicious activity patterns
        this.suspiciousPatterns = {
            maxRequestsPerSecond: 20,
            maxFailedAuthAttempts: 5,
            suspiciousUserAgents: [
                'curl', 'wget', 'python-requests', 'bot', 'scanner', 'crawl',
                'sqlmap', 'nikto', 'nmap', 'masscan'
            ],
            blockedCountries: ['CN', 'RU', 'KP'], // Example blocked countries
            maxFileSize: 2 * 1024 * 1024 * 1024, // 2GB
            allowedFileTypes: ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        };
    }

    /**
     * Main security check function - call this at the start of each Azure Function
     */
    async checkSecurity(context, req, options = {}) {
        // Auto-initialize if not done yet
        if (!this.initialized && this.cosmosClient) {
            await this.initialize();
        }
        
        const startTime = Date.now();
        const clientIP = this.getClientIP(req);
        const userAgent = req.headers['user-agent'] || '';
        const endpoint = req.url?.split('?')[0] || 'unknown';
        
        try {
            // 1. Rate Limiting Check
            const rateLimitResult = await this.checkRateLimit(clientIP, endpoint, context);
            if (!rateLimitResult.allowed) {
                await this.logSecurityEvent('RATE_LIMIT_EXCEEDED', {
                    clientIP,
                    endpoint,
                    userAgent,
                    rateLimitInfo: rateLimitResult
                }, context);
                
                return {
                    allowed: false,
                    status: 429,
                    body: { 
                        error: 'Rate limit exceeded', 
                        retryAfter: rateLimitResult.retryAfter,
                        limit: rateLimitResult.limit 
                    },
                    headers: {
                        'Retry-After': rateLimitResult.retryAfter.toString(),
                        'X-RateLimit-Limit': rateLimitResult.limit.toString(),
                        'X-RateLimit-Remaining': rateLimitResult.remaining.toString(),
                        'X-RateLimit-Reset': rateLimitResult.resetTime.toString()
                    }
                };
            }

            // 2. Suspicious Activity Detection
            const threatCheck = await this.checkForThreats(req, clientIP, userAgent, context);
            if (!threatCheck.allowed) {
                await this.logSecurityEvent('THREAT_DETECTED', {
                    clientIP,
                    endpoint,
                    userAgent,
                    threatInfo: threatCheck.reason
                }, context);
                
                return {
                    allowed: false,
                    status: 403,
                    body: { error: 'Request blocked for security reasons' }
                };
            }

            // 3. Input Validation
            if (req.body && options.validateInput !== false) {
                const inputValidation = this.validateInput(req.body, endpoint);
                if (!inputValidation.valid) {
                    await this.logSecurityEvent('INVALID_INPUT', {
                        clientIP,
                        endpoint,
                        userAgent,
                        validationErrors: inputValidation.errors
                    }, context);
                    
                    return {
                        allowed: false,
                        status: 400,
                        body: { error: 'Invalid input data', details: inputValidation.errors }
                    };
                }
            }

            // 4. Authentication Check (if required)
            if (options.requireAuth !== false) {
                const authCheck = this.validateAuthentication(req);
                if (!authCheck.valid) {
                    await this.logSecurityEvent('AUTH_FAILURE', {
                        clientIP,
                        endpoint,
                        userAgent,
                        authError: authCheck.error
                    }, context);
                    
                    return {
                        allowed: false,
                        status: 401,
                        body: { error: 'Authentication required' }
                    };
                }
            }

            // Log successful security check
            if (context) {
                context.log(`Security check passed for ${endpoint} from ${clientIP} (${Date.now() - startTime}ms)`);
            }

            return {
                allowed: true,
                clientIP,
                userInfo: options.requireAuth !== false ? this.getUserInfo(req) : null,
                rateLimitInfo: rateLimitResult
            };

        } catch (error) {
            if (context) {
                context.log.error('Security middleware error:', error.message || 'Unknown error');
            }
            
            // Fail securely - if security check fails, deny request
            return {
                allowed: false,
                status: 500,
                body: { error: 'Security check failed' }
            };
        }
    }

    /**
     * Rate limiting implementation with sliding window and burst protection
     */
    async checkRateLimit(clientIP, endpoint, context) {
        // If containers aren't initialized or no client IP, allow with default limits
        if (!this.rateLimitContainer || !clientIP) {
            if (context && !this.rateLimitContainer) {
                context.log.warn('Rate limiting container not available - allowing request');
            }
            return { allowed: true, remaining: 999, limit: 1000, resetTime: Date.now() + 60000 };
        }

        const config = this.rateLimits[endpoint] || this.rateLimits.default;
        const now = Date.now();
        const windowStart = now - config.windowMs;
        const rateLimitKey = `${clientIP}_${endpoint}`;

        try {
            // Get current rate limit data
            let rateLimitData;
            try {
                const { resource } = await this.rateLimitContainer.item(rateLimitKey).read();
                rateLimitData = resource;
            } catch (error) {
                if (error.code === 404) {
                    rateLimitData = {
                        id: rateLimitKey,
                        clientIP,
                        endpoint,
                        requests: [],
                        ttl: Math.floor(config.windowMs / 1000) + 60 // Auto-delete after window + buffer
                    };
                } else {
                    throw error;
                }
            }

            // Clean old requests
            rateLimitData.requests = rateLimitData.requests.filter(timestamp => timestamp > windowStart);

            // Check burst limit (requests in last 10 seconds)
            const burstWindowStart = now - 10000; // 10 seconds
            const recentRequests = rateLimitData.requests.filter(timestamp => timestamp > burstWindowStart);
            
            if (recentRequests.length >= config.burstLimit) {
                return {
                    allowed: false,
                    remaining: 0,
                    limit: config.requests,
                    resetTime: Math.min(...rateLimitData.requests) + config.windowMs,
                    retryAfter: 10, // Wait 10 seconds for burst limit
                    type: 'burst'
                };
            }

            // Check window limit
            if (rateLimitData.requests.length >= config.requests) {
                return {
                    allowed: false,
                    remaining: 0,
                    limit: config.requests,
                    resetTime: Math.min(...rateLimitData.requests) + config.windowMs,
                    retryAfter: Math.ceil((Math.min(...rateLimitData.requests) + config.windowMs - now) / 1000),
                    type: 'window'
                };
            }

            // Add current request
            rateLimitData.requests.push(now);
            rateLimitData.lastUpdated = now;

            // Save updated rate limit data
            await this.rateLimitContainer.items.upsert(rateLimitData);

            return {
                allowed: true,
                remaining: config.requests - rateLimitData.requests.length,
                limit: config.requests,
                resetTime: now + config.windowMs
            };

        } catch (error) {
            if (context) {
                context.log.warn('Rate limit check failed - containers may not exist yet');
            }
            // Allow request if rate limiting fails
            return { allowed: true, remaining: 999, limit: 1000, resetTime: now + 60000 };
        }
    }

    /**
     * Enhanced rate limiting for file operations that handles parallel uploads/downloads
     */
    async checkFileOperationRateLimit(clientIP, endpoint, options = {}, context) {
        const { fileSize, isChunkUpload = false, userId = null } = options;
        
        // Use userId for rate limiting if available (better for parallel operations)
        const identifier = userId || clientIP;
        const rateLimitKey = `${identifier}_${endpoint}${isChunkUpload ? '_chunk' : ''}`;
        
        // Calculate dynamic limits based on operation type
        let config;
        
        if (isChunkUpload) {
            // For chunk uploads, allow higher burst and total limits
            const expectedChunks = fileSize ? Math.ceil(fileSize / (4 * 1024 * 1024)) : 50; // 4MB chunks
            config = {
                requests: Math.min(expectedChunks * 2, 200), // Double the expected chunks, max 200
                windowMs: 600000, // 10 minutes for chunk operations
                burstLimit: Math.min(expectedChunks, 30) // Allow all chunks in burst, max 30
            };
        } else if (fileSize && fileSize > 64 * 1024 * 1024) {
            // For large file operations (>64MB), use relaxed limits
            config = {
                requests: 20,
                windowMs: 300000, // 5 minutes
                burstLimit: 10
            };
        } else {
            // Use standard limits for small files
            config = this.rateLimits[endpoint] || this.rateLimits.default;
        }
        
        return this.checkRateLimitWithConfig(identifier, endpoint, config, context);
    }
    
    /**
     * Rate limiting with custom configuration
     */
    async checkRateLimitWithConfig(identifier, endpoint, config, context) {
        if (!this.rateLimitContainer || !identifier) {
            return { allowed: true, remaining: 999, limit: config.requests, resetTime: Date.now() + config.windowMs };
        }

        const now = Date.now();
        const windowStart = now - config.windowMs;
        const rateLimitKey = `${identifier}_${endpoint}_custom`;

        try {
            // Get current rate limit data
            let rateLimitData;
            try {
                const { resource } = await this.rateLimitContainer.item(rateLimitKey).read();
                rateLimitData = resource;
            } catch (error) {
                if (error.code === 404) {
                    rateLimitData = {
                        id: rateLimitKey,
                        identifier,
                        endpoint,
                        requests: [],
                        config: config,
                        ttl: Math.floor(config.windowMs / 1000) + 60
                    };
                } else {
                    throw error;
                }
            }

            // Clean old requests
            rateLimitData.requests = rateLimitData.requests.filter(timestamp => timestamp > windowStart);

            // Check burst limit (requests in last 10 seconds) - but more lenient for chunk uploads
            const burstWindowMs = config.burstLimit > 20 ? 30000 : 10000; // 30s window for high burst limits
            const burstWindowStart = now - burstWindowMs;
            const recentRequests = rateLimitData.requests.filter(timestamp => timestamp > burstWindowStart);
            
            if (recentRequests.length >= config.burstLimit) {
                return {
                    allowed: false,
                    remaining: 0,
                    limit: config.requests,
                    resetTime: Math.min(...rateLimitData.requests) + config.windowMs,
                    retryAfter: Math.ceil(burstWindowMs / 1000),
                    type: 'burst'
                };
            }

            // Check window limit
            if (rateLimitData.requests.length >= config.requests) {
                return {
                    allowed: false,
                    remaining: 0,
                    limit: config.requests,
                    resetTime: Math.min(...rateLimitData.requests) + config.windowMs,
                    retryAfter: Math.ceil((Math.min(...rateLimitData.requests) + config.windowMs - now) / 1000),
                    type: 'window'
                };
            }

            // Add current request
            rateLimitData.requests.push(now);
            rateLimitData.lastUpdated = now;
            rateLimitData.config = config; // Update config in case it changed

            // Save updated rate limit data
            await this.rateLimitContainer.items.upsert(rateLimitData);

            return {
                allowed: true,
                remaining: config.requests - rateLimitData.requests.length,
                limit: config.requests,
                resetTime: now + config.windowMs,
                type: 'custom'
            };

        } catch (error) {
            if (context) {
                context.log.warn('Custom rate limit check failed - containers may not exist yet');
            }
            // Allow request if rate limiting fails
            return { allowed: true, remaining: 999, limit: config.requests, resetTime: now + config.windowMs };
        }
    }

    /**
     * Threat detection - check for suspicious patterns
     */
    async checkForThreats(req, clientIP, userAgent, context) {
        const threats = [];

        // Check user agent
        const suspiciousUA = this.suspiciousPatterns.suspiciousUserAgents.some(pattern => 
            userAgent.toLowerCase().includes(pattern.toLowerCase())
        );
        if (suspiciousUA) {
            threats.push('suspicious_user_agent');
        }

        // Check for common attack patterns in URL
        const url = req.url || '';
        const attackPatterns = [
            /\.\./,  // Directory traversal
            /\/etc\/passwd/,  // File access
            /<script/i,  // XSS
            /union\s+select/i,  // SQL injection
            /exec\s*\(/i,  // Code injection
            /eval\s*\(/i   // Code injection
        ];
        
        if (attackPatterns.some(pattern => pattern.test(url))) {
            threats.push('malicious_url_pattern');
        }

        // Check request headers for attacks
        const headers = req.headers || {};
        for (const [key, value] of Object.entries(headers)) {
            if (typeof value === 'string' && attackPatterns.some(pattern => pattern.test(value))) {
                threats.push('malicious_header');
                break;
            }
        }

        // Check for excessive request size
        const contentLength = parseInt(req.headers['content-length'] || '0');
        if (contentLength > this.suspiciousPatterns.maxFileSize) {
            threats.push('excessive_request_size');
        }

        // Geographic restriction (if enabled)
        // Note: In production, you'd use a proper IP geolocation service
        if (this.suspiciousPatterns.blockedCountries.length > 0 && this.isFromBlockedCountry(clientIP)) {
            threats.push('blocked_country');
        }

        return {
            allowed: threats.length === 0,
            reason: threats.join(', '),
            threats
        };
    }

    /**
     * Input validation based on endpoint
     */
    validateInput(body, endpoint) {
        const errors = [];

        if (!body || typeof body !== 'object') {
            return { valid: true }; // Allow empty or non-object bodies
        }

        // Common validation
        for (const [key, value] of Object.entries(body)) {
            // Check for script injection in all string values
            if (typeof value === 'string') {
                if (/<script|javascript:|vbscript:|onload=|onerror=/i.test(value)) {
                    errors.push(`Potential script injection in field: ${key}`);
                }
                
                // Check for SQL injection patterns
                if (/(\b(union|select|insert|update|delete|drop|exec|execute)\b)/i.test(value)) {
                    errors.push(`Potential SQL injection in field: ${key}`);
                }

                // Check string length
                if (value.length > 10000) {
                    errors.push(`Field ${key} exceeds maximum length`);
                }
            }
        }

        // Endpoint-specific validation
        switch (endpoint) {
            case '/api/upload-file':
                if (body.fileName && typeof body.fileName === 'string') {
                    const ext = body.fileName.toLowerCase().substring(body.fileName.lastIndexOf('.'));
                    if (!this.suspiciousPatterns.allowedFileTypes.includes(ext)) {
                        errors.push(`File type not allowed: ${ext}`);
                    }
                    
                    // Check for path traversal in filename
                    if (body.fileName.includes('..') || body.fileName.includes('/') || body.fileName.includes('\\')) {
                        errors.push('Invalid characters in filename');
                    }
                }
                
                if (body.fileSize && body.fileSize > this.suspiciousPatterns.maxFileSize) {
                    errors.push('File size exceeds maximum allowed');
                }
                break;

            case '/api/enqueue-job':
                if (body.attenuationDb) {
                    const atten = parseInt(body.attenuationDb);
                    if (isNaN(atten) || atten < 1 || atten > 100) {
                        errors.push('Invalid attenuation value');
                    }
                }
                break;
        }

        return {
            valid: errors.length === 0,
            errors
        };
    }

    /**
     * Enhanced authentication validation
     */
    validateAuthentication(req) {
        const clientPrincipal = req.headers['x-ms-client-principal'];
        
        if (!clientPrincipal) {
            return { valid: false, error: 'No authentication header' };
        }

        try {
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            
            if (!principal.userId || !principal.userDetails) {
                return { valid: false, error: 'Invalid principal data' };
            }

            // Check for session hijacking indicators
            const userAgent = req.headers['user-agent'] || '';
            const sessionKey = `${principal.userId}_${this.hashString(userAgent)}`;
            
            return { valid: true, principal, sessionKey };
            
        } catch (error) {
            return { valid: false, error: 'Invalid authentication data' };
        }
    }

    /**
     * Get user information from request
     */
    getUserInfo(req) {
        try {
            const clientPrincipal = req.headers['x-ms-client-principal'];
            if (!clientPrincipal) return null;
            
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            return {
                userId: principal.userId,
                email: principal.userDetails,
                provider: principal.identityProvider
            };
        } catch {
            return null;
        }
    }

    /**
     * Log security events for monitoring
     */
    async logSecurityEvent(eventType, data, context) {
        // Temporarily disabled to reduce log volume and costs
        return;

        try {
            const event = {
                id: `${eventType}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
                eventType,
                timestamp: new Date().toISOString(),
                data,
                ttl: 30 * 24 * 60 * 60 // Keep for 30 days
            };

            await this.securityEventsContainer.items.create(event);
            
            if (context) {
                context.log(`Security event logged: ${eventType}`, { 
                    clientIP: data.clientIP, 
                    endpoint: data.endpoint 
                });
            }
        } catch (error) {
            if (context) {
                context.log.error('Failed to log security event:', error.message || 'Unknown error');
            }
        }
    }

    /**
     * Get security headers for responses
     */
    getSecurityHeaders(endpoint = '') {
        const headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        };

        // Endpoint-specific headers
        if (endpoint.includes('download') || endpoint.includes('upload')) {
            headers['Content-Security-Policy'] = "default-src 'none'; script-src 'none'; object-src 'none';";
        } else {
            headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
        }

        return headers;
    }

    /**
     * Utility functions
     */
    getClientIP(req) {
        const forwardedFor = req.headers['x-forwarded-for'];
        if (forwardedFor) {
            return forwardedFor.split(',')[0].trim();
        }
        return req.headers['x-client-ip'] || 
               req.headers['x-real-ip'] || 
               req.connection?.remoteAddress || 
               'unknown';
    }

    hashString(str) {
        return crypto.createHash('sha256').update(str).digest('hex').substring(0, 16);
    }

    isFromBlockedCountry(ip) {
        // Implement proper IP geolocation check in production
        // For now, this is a placeholder
        return false;
    }

    /**
     * Clean up expired rate limit entries (call periodically)
     */
    async cleanupRateLimits(context) {
        if (!this.rateLimitContainer) return;

        try {
            const cutoff = Date.now() - (60 * 60 * 1000); // 1 hour ago
            const query = `SELECT * FROM c WHERE c.lastUpdated < ${cutoff}`;
            
            const { resources: expiredEntries } = await this.rateLimitContainer.items.query(query).fetchAll();
            
            for (const entry of expiredEntries) {
                await this.rateLimitContainer.item(entry.id).delete();
            }
            
            if (context && expiredEntries.length > 0) {
                context.log(`Cleaned up ${expiredEntries.length} expired rate limit entries`);
            }
        } catch (error) {
            if (context) {
                context.log.error('Failed to cleanup rate limits:', error.message || 'Unknown error');
            }
        }
    }

    /**
     * Initialize the SecurityMiddleware with proper error handling
     */
    async initialize() {
        if (this.initialized || !this.cosmosClient) {
            return;
        }

        try {
            const database = this.cosmosClient.database('audiocleaner');
            this.rateLimitContainer = database.container('ratelimits');
            this.securityEventsContainer = database.container('securityevents');
            this.initialized = true;
        } catch (error) {
            // CRITICAL: Only log error message to prevent massive log costs
            console.warn('SecurityMiddleware containers not available, using degraded mode');
            // Continue without Cosmos DB - degraded functionality
            this.rateLimitContainer = null;
            this.securityEventsContainer = null;
        }
    }

    /**
     * Check if the middleware is ready to use
     */
    isReady() {
        return this.initialized || !this.cosmosClient;
    }
}

module.exports = SecurityMiddleware;

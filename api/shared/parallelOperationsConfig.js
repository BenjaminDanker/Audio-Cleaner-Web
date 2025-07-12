/**
 * Parallel Operations Configuration for Audio Cleaner Web
 * Optimizes rate limiting for concurrent uploads and downloads
 */

class ParallelOperationsConfig {
    /**
     * Get recommended upload configuration based on file size and connection
     */
    static getUploadConfig(fileSize, connectionSpeed = 'medium') {
        const configs = {
            small: { // < 64MB
                useParallel: false,
                maxConcurrency: 1,
                chunkSize: 4 * 1024 * 1024, // 4MB
                rateLimitStrategy: 'standard'
            },
            medium: { // 64MB - 500MB
                useParallel: true,
                maxConcurrency: 4,
                chunkSize: 4 * 1024 * 1024, // 4MB
                rateLimitStrategy: 'enhanced'
            },        large: { // 500MB - 2GB
            useParallel: true,
            maxConcurrency: 6,
            chunkSize: 8 * 1024 * 1024, // 8MB
            rateLimitStrategy: 'bulk'
        },
        xlarge: { // > 2GB
            useParallel: true,
            maxConcurrency: 8,
            chunkSize: 16 * 1024 * 1024, // 16MB
            rateLimitStrategy: 'enterprise'
        }
        };

        // Determine size category
        if (fileSize < 64 * 1024 * 1024) return configs.small;
        if (fileSize < 500 * 1024 * 1024) return configs.medium;
        if (fileSize < 2 * 1024 * 1024 * 1024) return configs.large;
        return configs.xlarge;
    }

    /**
     * Get rate limiting configuration for different strategies
     */
    static getRateLimitConfig(strategy, isChunkUpload = false) {
        const strategies = {
            standard: {
                requests: 10,
                windowMs: 60000,
                burstLimit: 3,
                chunkMultiplier: 1
            },
            enhanced: {
                requests: 25,
                windowMs: 120000, // 2 minutes
                burstLimit: 8,
                chunkMultiplier: 3
            },
            bulk: {
                requests: 50,
                windowMs: 300000, // 5 minutes  
                burstLimit: 15,
                chunkMultiplier: 5
            },
            enterprise: {
                requests: 100,
                windowMs: 600000, // 10 minutes
                burstLimit: 25,
                chunkMultiplier: 8
            }
        };

        const config = strategies[strategy] || strategies.standard;
        
        if (isChunkUpload) {
            return {
                requests: config.requests * config.chunkMultiplier,
                windowMs: config.windowMs,
                burstLimit: config.burstLimit * config.chunkMultiplier
            };
        }
        
        return config;
    }

    /**
     * Calculate optimal concurrency based on rate limits
     */
    static calculateOptimalConcurrency(fileSize, rateLimitConfig) {
        const expectedChunks = Math.ceil(fileSize / (4 * 1024 * 1024)); // 4MB chunks
        const windowSeconds = rateLimitConfig.windowMs / 1000;
        const requestsPerSecond = rateLimitConfig.requests / windowSeconds;
        
        // Calculate how many chunks we can upload per second within burst limits
        const burstCapacity = rateLimitConfig.burstLimit;
        const sustainedCapacity = Math.floor(requestsPerSecond * 30); // 30 second sustained rate
        
        // Use the more conservative of burst or sustained capacity
        const effectiveCapacity = Math.min(burstCapacity, sustainedCapacity);
        
        // Optimal concurrency should not exceed rate limits
        const optimalConcurrency = Math.min(
            effectiveCapacity,
            expectedChunks,
            8 // Hard limit to prevent overwhelming the server
        );
        
        return Math.max(1, optimalConcurrency);
    }

    /**
     * Frontend configuration for parallel uploads
     */
    static getFrontendConfig(fileSize) {
        const uploadConfig = this.getUploadConfig(fileSize);
        const rateLimitConfig = this.getRateLimitConfig(uploadConfig.rateLimitStrategy, true);
        const optimalConcurrency = this.calculateOptimalConcurrency(fileSize, rateLimitConfig);
        
        return {
            // Upload settings
            useParallelUpload: uploadConfig.useParallel,
            maxConcurrency: Math.min(uploadConfig.maxConcurrency, optimalConcurrency),
            chunkSize: uploadConfig.chunkSize,
            
            // Rate limiting awareness
            retryConfig: {
                maxRetries: 5,
                baseDelay: 1000, // 1 second
                maxDelay: 30000, // 30 seconds
                backoffFactor: 2
            },
            
            // Progress and error handling
            progressUpdateInterval: 500, // 500ms
            timeoutPerChunk: 120000, // 2 minutes per chunk
            
            // Headers to help server identify parallel operations
            headers: {
                'X-Upload-Strategy': uploadConfig.rateLimitStrategy,
                'X-Expected-Chunks': Math.ceil(fileSize / uploadConfig.chunkSize).toString(),
                'X-Chunk-Upload': 'true'
            }
        };
    }

    /**
     * Server-side detection of parallel operations
     */
    static detectParallelOperation(req) {
        return {
            isChunkUpload: req.headers['x-chunk-upload'] === 'true' ||
                          req.query?.comp === 'block' ||
                          !!req.query?.blockid,
            expectedChunks: parseInt(req.headers['x-expected-chunks'] || '0'),
            uploadStrategy: req.headers['x-upload-strategy'] || 'standard',
            isRangeRequest: !!req.headers['range']
        };
    }
}

module.exports = ParallelOperationsConfig;

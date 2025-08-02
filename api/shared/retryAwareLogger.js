/**
 * Retry-Aware Error Logger
 * Prevents log spam from Azure SDK retries and function-level retries
 * Deduplicates errors based on content and timing
 */
class RetryAwareLogger {
    constructor(baseLogger, maxDuplicatesPerMinute = 3) {
        this.baseLogger = baseLogger;
        this.maxDuplicatesPerMinute = maxDuplicatesPerMinute;
        this.errorCache = new Map(); // { errorHash: { count, firstSeen, lastSeen } }
        this.cleanupInterval = setInterval(() => this.cleanup(), 60000); // Cleanup every minute
    }

    /**
     * Generate a hash for error deduplication
     */
    generateErrorHash(functionName, errorMessage, operation = '') {
        // Create a hash based on function, error message, and operation
        // Ignore timestamps, request IDs, and other unique identifiers
        const normalizedMessage = errorMessage
            .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/g, 'TIMESTAMP') // Remove timestamps
            .replace(/[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/gi, 'UUID') // Remove UUIDs
            .replace(/job_\d+_[a-z0-9]+/g, 'JOB_ID') // Remove job IDs
            .replace(/\d+ms/g, 'Xms') // Remove timing
            .replace(/attempt \d+/gi, 'attempt X'); // Remove attempt numbers

        return `${functionName}:${operation}:${normalizedMessage}`;
    }

    /**
     * Check if error should be logged or suppressed
     */
    shouldLogError(errorHash) {
        const now = Date.now();
        const entry = this.errorCache.get(errorHash);

        if (!entry) {
            // First occurrence - always log
            this.errorCache.set(errorHash, {
                count: 1,
                firstSeen: now,
                lastSeen: now
            });
            return { shouldLog: true, isRetry: false };
        }

        // Update last seen
        entry.lastSeen = now;
        entry.count++;

        // Check if we're within the same minute and under the limit
        const withinMinute = (now - entry.firstSeen) < 60000;
        if (withinMinute && entry.count <= this.maxDuplicatesPerMinute) {
            return { shouldLog: true, isRetry: true, count: entry.count };
        }

        // If more than a minute has passed, reset the counter
        if (!withinMinute) {
            entry.firstSeen = now;
            entry.count = 1;
            return { shouldLog: true, isRetry: false };
        }

        // Suppress this log - too many duplicates
        return { shouldLog: false, isRetry: true, count: entry.count, suppressed: true };
    }

    /**
     * Log error with retry awareness
     */
    logError(functionName, error, userId = 'system', metadata = {}) {
        const errorMessage = typeof error === 'string' ? error : (error?.message || 'Unknown error');
        const operation = metadata.operation || '';
        const errorHash = this.generateErrorHash(functionName, errorMessage, operation);
        
        const logDecision = this.shouldLogError(errorHash);
        
        if (logDecision.shouldLog) {
            const enhancedMetadata = {
                ...metadata,
                retryInfo: logDecision.isRetry ? {
                    isRetry: true,
                    attemptCount: logDecision.count,
                    ...(logDecision.count === this.maxDuplicatesPerMinute && { 
                        note: 'Further identical errors will be suppressed for 1 minute' 
                    })
                } : undefined
            };

            this.baseLogger.logError(functionName, errorMessage, userId, enhancedMetadata);
        } else if (logDecision.suppressed) {
            // Log a suppression notice every 10th occurrence
            if (logDecision.count % 10 === 0) {
                this.baseLogger.logError(functionName, `Error suppressed - occurred ${logDecision.count} times`, userId, {
                    ...metadata,
                    suppressedError: errorMessage.substring(0, 100) + '...',
                    suppressionInfo: true
                });
            }
        }
    }

    /**
     * Log info with basic deduplication
     */
    logInfo(functionName, message, userId = 'system', metadata = {}) {
        // For info messages, we're less aggressive with deduplication
        // Only deduplicate if it's clearly a retry operation
        if (metadata.isRetry || message.toLowerCase().includes('retry') || message.toLowerCase().includes('attempt')) {
            const hash = this.generateErrorHash(functionName, message, metadata.operation || '');
            const logDecision = this.shouldLogError(hash);
            
            if (logDecision.shouldLog) {
                const enhancedMetadata = {
                    ...metadata,
                    retryInfo: logDecision.isRetry ? { 
                        isRetry: true, 
                        attemptCount: logDecision.count 
                    } : undefined
                };
                this.baseLogger.logInfo(functionName, message, userId, enhancedMetadata);
            }
        } else {
            // Regular info messages - log normally
            this.baseLogger.logInfo(functionName, message, userId, metadata);
        }
    }

    /**
     * Pass through other logging methods
     */
    logPerformance(functionName, operation, duration, userId = 'system', metadata = {}) {
        this.baseLogger.logPerformance(functionName, operation, duration, userId, metadata);
    }

    logDebug(functionName, message, userId = 'system', metadata = {}) {
        this.baseLogger.logDebug(functionName, message, userId, metadata);
    }

    /**
     * Initialize method for compatibility
     */
    async initialize() {
        return this.baseLogger.initialize ? await this.baseLogger.initialize() : true;
    }

    /**
     * Clean up old cache entries
     */
    cleanup() {
        const now = Date.now();
        const fiveMinutesAgo = now - (5 * 60 * 1000);

        for (const [hash, entry] of this.errorCache.entries()) {
            if (entry.lastSeen < fiveMinutesAgo) {
                this.errorCache.delete(hash);
            }
        }
    }

    /**
     * Cleanup on destruction
     */
    destroy() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        this.errorCache.clear();
    }
}

module.exports = RetryAwareLogger;

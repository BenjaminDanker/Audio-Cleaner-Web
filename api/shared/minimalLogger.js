/**
 * Minimal Application Insights Logger
 * Replaces BlobLogger with cost-effective Application Insights logging
 * Only logs essential information for debugging the video upload/denoise/download pipeline
 */
const RetryAwareLogger = require('./retryAwareLogger');

class MinimalLogger {
    constructor(context) {
        this.context = context;
        this.initialized = true; // Always ready
        this.retryLogger = new RetryAwareLogger(this, 2); // Max 2 duplicates per minute
    }

    /**
     * Get the retry-aware logger
     */
    getLogger() {
        return this.retryLogger;
    }

    /**
     * Log only essential info - just message and minimal metadata
     */
    logInfo(functionName, message, userId = 'system', metadata = {}) {
        if (!this.context) return;
        
        // Only log essential fields to minimize cost
        const logData = {
            level: 'INFO',
            function: functionName,
            message: message,
            userId: userId,
            timestamp: new Date().toISOString()
        };

        // Only include critical metadata fields
        if (metadata.sessionId) logData.sessionId = metadata.sessionId;
        if (metadata.fileSize) logData.fileSize = metadata.fileSize;
        if (metadata.fileName) logData.fileName = metadata.fileName?.substring(0, 20) + '...'; // Truncate for cost
        if (metadata.jobId) logData.jobId = metadata.jobId;
        if (metadata.status) logData.status = metadata.status;
        if (metadata.retryInfo) logData.retryInfo = metadata.retryInfo;

        this.context.log(`[${functionName}]`, JSON.stringify(logData));
    }

    /**
     * Log errors - only message, no full error objects
     */
    logError(functionName, error, userId = 'system', metadata = {}) {
        if (!this.context) return;
        
        const errorMessage = typeof error === 'string' ? error : (error?.message || 'Unknown error');
        
        const logData = {
            level: 'ERROR',
            function: functionName,
            error: errorMessage, // Only the message, not the full object
            userId: userId,
            timestamp: new Date().toISOString()
        };

        // Only include critical metadata
        if (metadata.sessionId) logData.sessionId = metadata.sessionId;
        if (metadata.operation) logData.operation = metadata.operation;
        if (metadata.retryInfo) logData.retryInfo = metadata.retryInfo;
        if (metadata.suppressionInfo) logData.suppressionInfo = metadata.suppressionInfo;

        this.context.log.error(`[${functionName}]`, JSON.stringify(logData));
    }

    /**
     * Log performance metrics - essential for monitoring pipeline
     */
    logPerformance(functionName, operation, duration, userId = 'system', metadata = {}) {
        if (!this.context) return;
        
        const logData = {
            level: 'PERF',
            function: functionName,
            operation: operation,
            duration: duration,
            userId: userId,
            timestamp: new Date().toISOString()
        };

        // Only include essential metadata
        if (metadata.sessionId) logData.sessionId = metadata.sessionId;
        if (metadata.fileSize) logData.fileSize = metadata.fileSize;

        this.context.log(`[${functionName}]`, JSON.stringify(logData));
    }

    /**
     * Log debug info - minimal, only for critical debugging
     */
    logDebug(functionName, message, userId = 'system', metadata = {}) {
        if (!this.context) return;
        
        // Only log debug in development or when explicitly needed
        if (process.env.NODE_ENV === 'production') return;
        
        const logData = {
            level: 'DEBUG',
            function: functionName,
            message: message,
            userId: userId,
            timestamp: new Date().toISOString()
        };

        // Minimal metadata for debug
        if (metadata.sessionId) logData.sessionId = metadata.sessionId;

        this.context.log(`[${functionName}]`, JSON.stringify(logData));
    }

    /**
     * Warning log (backwards compatibility for previous logger API)
     * Some existing function code calls logger.logWarn / logger.logWarning which did not exist
     * Implement it as a thin wrapper around info with a distinct level.
     */
    logWarning(functionName, message, userId = 'system', metadata = {}) {
        if (!this.context) return;

        const logData = {
            level: 'WARN',
            function: functionName,
            message: message,
            userId: userId,
            timestamp: new Date().toISOString()
        };

        if (metadata.sessionId) logData.sessionId = metadata.sessionId;
        if (metadata.operation) logData.operation = metadata.operation;

        // Use warn channel if available to aid filtering
        if (this.context.log.warn) {
            this.context.log.warn(`[${functionName}]`, JSON.stringify(logData));
        } else {
            this.context.log(`[${functionName}]`, JSON.stringify(logData));
        }
    }

    // Alias for code that uses logWarn
    logWarn(functionName, message, userId = 'system', metadata = {}) {
        return this.logWarning(functionName, message, userId, metadata);
    }

    /**
     * Initialize method for compatibility with BlobLogger
     */
    async initialize() {
        return true; // Always succeeds
    }

    /**
     * Cleanup method
     */
    destroy() {
        this.retryLogger.destroy();
    }
}

module.exports = MinimalLogger;

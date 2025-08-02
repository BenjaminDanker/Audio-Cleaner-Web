/**
 * Azure SDK Configuration Helper
 * Configures Azure SDKs with optimized retry policies and reduced logging
 */
const { CosmosClient } = require('@azure/cosmos');
const { BlobServiceClient } = require('@azure/storage-blob');

class AzureSDKConfig {
    /**
     * Create a Cosmos DB client with optimized retry configuration
     */
    static createCosmosClient(connectionString, options = {}) {
        if (!connectionString) {
            throw new Error('Cosmos DB connection string is required');
        }

        const defaultOptions = {
            // Retry configuration to reduce log spam
            retryOptions: {
                maxRetryAttemptCount: 3, // Reduced from default 9
                fixedRetryIntervalInMilliseconds: 1000, // 1 second
                maxRetryWaitTimeInSeconds: 10 // Reduced from default 30
            },
            // Disable verbose SDK logging
            userAgentSuffix: 'AudioCleaner/1.0 (reduced-logging)',
            // Connection options
            connectionPolicy: {
                requestTimeout: 30000, // 30 seconds
                enableEndpointDiscovery: true
            }
        };

        const mergedOptions = { ...defaultOptions, ...options };
        return new CosmosClient(connectionString, mergedOptions);
    }

    /**
     * Create a Blob Storage client with optimized retry configuration
     */
    static createBlobServiceClient(connectionString, options = {}) {
        if (!connectionString) {
            throw new Error('Storage connection string is required');
        }

        const defaultOptions = {
            retryOptions: {
                maxTries: 3, // Reduced from default 4
                tryTimeoutInMs: 30000, // 30 seconds per try
                retryDelayInMs: 1000, // 1 second base delay
                maxRetryDelayInMs: 10000 // 10 seconds max delay
            },
            // Keep alive for connection reuse
            keepAliveOptions: {
                enable: true
            }
        };

        const mergedOptions = { ...defaultOptions, ...options };
        return BlobServiceClient.fromConnectionString(connectionString, mergedOptions);
    }

    /**
     * Create operation-specific retry configuration
     */
    static getOperationRetryConfig(operationType) {
        const configs = {
            // Critical operations - more retries
            'upload': {
                maxRetries: 3,
                baseDelay: 1000,
                maxDelay: 15000,
                backoffFactor: 2
            },
            'download': {
                maxRetries: 3,
                baseDelay: 500,
                maxDelay: 10000,
                backoffFactor: 2
            },
            'database': {
                maxRetries: 3,
                baseDelay: 1000,
                maxDelay: 8000,
                backoffFactor: 1.5
            },
            // Non-critical operations - fewer retries
            'cleanup': {
                maxRetries: 2,
                baseDelay: 2000,
                maxDelay: 10000,
                backoffFactor: 2
            },
            'metadata': {
                maxRetries: 2,
                baseDelay: 1000,
                maxDelay: 5000,
                backoffFactor: 1.5
            }
        };

        return configs[operationType] || configs['metadata'];
    }

    /**
     * Execute operation with custom retry logic and reduced logging
     */
    static async executeWithRetry(operation, operationType, logger, context) {
        const config = this.getOperationRetryConfig(operationType);
        let lastError;
        let attempt = 0;

        for (attempt = 1; attempt <= config.maxRetries; attempt++) {
            try {
                const result = await operation();
                
                // Log success only if there were previous failures
                if (attempt > 1) {
                    logger.logInfo('azure-sdk', `Operation succeeded on attempt ${attempt}`, 'system', {
                        operationType,
                        attempt,
                        retrySuccess: true
                    });
                }
                
                return result;
            } catch (error) {
                lastError = error;
                
                // Determine if we should retry
                const shouldRetry = this.shouldRetryError(error, attempt, config.maxRetries);
                
                if (shouldRetry && attempt < config.maxRetries) {
                    const delay = Math.min(
                        config.baseDelay * Math.pow(config.backoffFactor, attempt - 1),
                        config.maxDelay
                    );
                    
                    // Only log retries at a reduced frequency
                    if (attempt === 1 || attempt === config.maxRetries - 1) {
                        logger.logError('azure-sdk', `Operation failed, retrying in ${delay}ms`, 'system', {
                            operationType,
                            attempt,
                            maxRetries: config.maxRetries,
                            error: error.message || 'Unknown error',
                            willRetry: true
                        });
                    }
                    
                    await this.delay(delay);
                } else {
                    // Final failure - log with full context
                    logger.logError('azure-sdk', `Operation failed after ${attempt} attempts`, 'system', {
                        operationType,
                        attempt,
                        maxRetries: config.maxRetries,
                        error: error.message || 'Unknown error',
                        finalFailure: true
                    });
                    break;
                }
            }
        }

        throw lastError;
    }

    /**
     * Determine if an error should trigger a retry
     */
    static shouldRetryError(error, attempt, maxRetries) {
        if (attempt >= maxRetries) return false;

        // Don't retry on authentication errors
        if (error.status === 401 || error.status === 403) return false;
        
        // Don't retry on client errors (400-499) except specific ones
        if (error.status >= 400 && error.status < 500) {
            // Retry on rate limiting and request timeout
            return error.status === 408 || error.status === 429;
        }

        // Retry on server errors (500-599) and network errors
        if (error.status >= 500 || !error.status) return true;

        // Retry on specific Azure error codes
        if (error.code) {
            const retryableErrors = [
                'RequestTimeout',
                'InternalServerError',
                'ServerBusy',
                'ServiceUnavailable',
                'OperationTimedOut',
                'ThrottledError'
            ];
            return retryableErrors.includes(error.code);
        }

        return false;
    }

    /**
     * Utility delay function
     */
    static delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Wrap Azure SDK clients with retry-aware logging
     */
    static wrapClientWithLogging(client, clientType, logger) {
        // This would intercept SDK calls and add retry-aware logging
        // For now, we'll rely on the executeWithRetry method for operations
        return client;
    }
}

module.exports = AzureSDKConfig;

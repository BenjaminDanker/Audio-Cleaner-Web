const { CosmosClient } = require('@azure/cosmos');
const { BlobServiceClient } = require('@azure/storage-blob');
const { ServiceBusClient } = require('@azure/service-bus');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const { calculateProcessingCost } = require('../shared/pricingUtils');

module.exports = async function (context, req) {
    const startTime = Date.now();
    
    // Generate session ID for this request
    const sessionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize retry-aware minimal logger
    const logger = new MinimalLogger(context).getLogger();
    
    // Minimal logging for Application Insights
    logger.logInfo('enqueue-job', 'Enqueue job endpoint called', 'system', {
        sessionId,
        requestMethod: req.method
    });
    
    // Handle OPTIONS request for CORS
    if (req.method === 'OPTIONS') {
        context.res = {
            status: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal'
            },
            body: ''
        };
        return;
    }

    // Only allow POST requests
    if (req.method !== 'POST') {
        logger.logError('enqueue-job', `Method ${req.method} not allowed`, 'system', { sessionId });
        context.res = {
            status: 405,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: { error: 'Method not allowed' }
        };
        return;
    }
    
    try {
        // Security check with simple middleware
        const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        const securityResult = await security.checkSecurity(context, req, { requireAuth: true });
        
        if (!securityResult.allowed) {
            await logger.logError('enqueue-job', 'Security check failed', 'system', { 
                sessionId, 
                status: securityResult.status,
                reason: securityResult.body?.error 
            });
            context.res = {
                status: securityResult.status,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    ...securityResult.headers
                },
                body: securityResult.body
            };
            return;
        }
        
        let userId = securityResult.userInfo?.userId;

        // Verify authentication
        const clientPrincipal = req.headers['x-ms-client-principal'];
        if (!clientPrincipal) {
            logger.logError('enqueue-job', 'Unauthorized - No client principal found', 'system', { sessionId });
            context.res = {
                status: 401,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Unauthorized - No client principal found' }
            };
            return;
        }
        // Ensure Cosmos DB connection string is configured
        const connectionString = process.env.COSMOS_CONNECTION_STRING;
        if (!connectionString) {
            logger.logError('enqueue-job', 'COSMOS_CONNECTION_STRING is not set', 'system', { sessionId });
            context.res = {
                status: 500,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Server configuration error: missing COSMOS_CONNECTION_STRING' }
            };
            return;
        }
         const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
         userId = principal.userId; // Update userId with principal data

         // Basic input validation
        const { fileName, processingType, attenuationDb } = req.body || {};

        if (!fileName) {
            logger.logError('enqueue-job', 'Missing required field: fileName (blobName)', userId, { sessionId });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'fileName (blobName) is required' }
            };
            return;
        }

        // Treat fileName as blobName (path relative to uploads container). Must start with userId/ for ownership.
        const blobName = fileName.trim();
        if (!blobName.startsWith(userId + '/')) {
            logger.logError('enqueue-job', `Blob name does not belong to user userId=${userId} blobName=${blobName}`, userId, { sessionId });
            context.res = {
                status: 403,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Blob does not belong to authenticated user' }
            };
            return;
        }
        // Basic pattern check (alphanumeric, separators . _ - / )
        if (!/^[-A-Za-z0-9_./]+$/.test(blobName)) {
            logger.logError('enqueue-job', 'Invalid blobName pattern', userId, { sessionId });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Invalid blob name' }
            };
            return;
        }

        // Get actual file size from blob storage (container fixed 'uploads')
        let actualFileSizeBytes;
        const derivedContainerName = 'uploads';
        let attemptsTried = 0;
        try {
            const storageConnectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
            if (!storageConnectionString) {
                throw new Error('AZURE_STORAGE_CONNECTION_STRING not set');
            }
            logger.logInfo('enqueue-job', `Using provided blobName container=${derivedContainerName} blob=${blobName}`, userId, { sessionId });

            // Use helper to correctly create client from connection string (previous direct constructor caused Invalid URL)
            const blobServiceClient = AzureSDKConfig.createBlobServiceClient(storageConnectionString);
            const containerClient = blobServiceClient.getContainerClient(derivedContainerName);
            const blobClient = containerClient.getBlobClient(blobName);

            const maxAttempts = 5;
            const baseDelayMs = 300;
            let lastErr;
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                attemptsTried = attempt;
                try {
                    const properties = await blobClient.getProperties();
                    actualFileSizeBytes = properties.contentLength;
                    if (!actualFileSizeBytes || actualFileSizeBytes <= 0) {
                        throw new Error(`Invalid contentLength (${actualFileSizeBytes})`);
                    }
                    logger.logInfo(
                        'enqueue-job',
                        `Retrieved blob size attempt=${attempt}/${maxAttempts} container=${derivedContainerName} blob=${blobName} sizeBytes=${actualFileSizeBytes}`,
                        userId,
                        { sessionId }
                    );
                    break; // success
                } catch (innerErr) {
                    lastErr = innerErr;
                    // If 404 or similar, wait then retry (upload may not be committed yet)
                    if (attempt < maxAttempts) {
                        await new Promise(r => setTimeout(r, baseDelayMs * attempt));
                        logger.logInfo(
                            'enqueue-job',
                            `Retrying blob properties after error attempt=${attempt} container=${derivedContainerName} blob=${blobName} err=${innerErr?.name || 'Error'}:${innerErr?.message}`,
                            userId,
                            { sessionId }
                        );
                        continue;
                    }
                }
            }

            if (!actualFileSizeBytes) {
                throw lastErr || new Error('Unknown error retrieving blob properties');
            }
        } catch (blobError) {
            const stackFirst = blobError?.stack?.split('\n')[0] || '';
            logger.logError(
                'enqueue-job',
                `Failed to get blob size container=${derivedContainerName} blob=${blobName} attempts=${attemptsTried} err=${blobError?.name || 'Error'}:${blobError?.message} hasConn=${!!process.env.AZURE_STORAGE_CONNECTION_STRING} stackFirst=${stackFirst}`,
                userId,
                { sessionId }
            );
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Could not retrieve file information from storage' }
            };
            return;
        }

    // Pricing util returns cost in USD (float, 2 decimals). Convert to integer cents for storage consistency.
    const actualCostUsd = calculateProcessingCost(actualFileSizeBytes);
    const actualCost = Math.round(actualCostUsd * 100); // store in cents

        // Initialize optimized Cosmos client with retry-aware configuration
        const cosmosClient = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const jobsContainer = database.container('Jobs');
        const accountsContainer = database.container('accounts');
        const transactionsContainer = database.container('transactions');

    // Track the transaction we create so we can update it with jobId later without a query
    let pendingTransactionId = null;

    // Check account balance and deduct cost
        try {
            const { resource: account } = await accountsContainer.item(userId, userId).read();
            
            if (!account || account.balance < actualCost) {
                logger.logError('enqueue-job', 'Insufficient account balance', userId, { 
                    sessionId, 
                balance: account?.balance || 0, 
                requiredCostCents: actualCost, 
                requiredCostUsd: actualCost / 100
                });
                context.res = {
                    status: 402, // Payment Required
                    headers: {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    body: { 
                        error: 'Insufficient account balance',
                        currentBalance: account?.balance || 0,
                        requiredAmountCents: actualCost,
                        requiredAmountUsd: actualCost / 100
                    }
                };
                return;
            }

            // Deduct cost from account balance
            account.balance -= actualCost; // both in cents
            account.updatedAt = new Date().toISOString();
            await accountsContainer.item(account.id, account.userId).replace(account);

            // Create transaction record
            const transactionId = `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const transaction = {
                id: transactionId,
                userId: userId,
                type: 'processing',
                amount: actualCost, // cents
                description: `Media processing: ${fileName}`,
                jobId: null, // Will be updated after job creation
                createdAt: new Date().toISOString()
            };

            await transactionsContainer.items.create(transaction);
            pendingTransactionId = transactionId; // remember for later update

            logger.logInfo('enqueue-job', 'Account balance updated and transaction created', userId, {
                sessionId,
                previousBalance: account.balance + actualCost, // log pre-deduction (in cents)
                newBalance: account.balance,
                transactionId,
                actualCostCents: actualCost,
                actualCostUsd: actualCost / 100
            });

        } catch (error) {
            if (error.code === 404) {
                // Account doesn't exist
                logger.logError('enqueue-job', 'Account not found', userId, { sessionId });
                context.res = {
                    status: 404,
                    headers: {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    body: { error: 'Account not found. Please ensure your account is set up.' }
                };
                return;
            }
            throw error; // Re-throw other errors
        }

        // Generate job ID and output file name
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        // Preserve original extension for output artifact; just append _cleaned before extension
        let outputFileName;
        if (fileName) {
            const lastDot = fileName.lastIndexOf('.')
            if (lastDot !== -1) {
                const base = fileName.substring(0, lastDot)
                const ext = fileName.substring(lastDot)
                outputFileName = `${base}_cleaned${ext}`
            } else {
                outputFileName = `${fileName}_cleaned`
            }
        } else {
            outputFileName = 'cleaned_media'
        }
        
        const jobRecord = {
            id: jobId,
            userId: userId,
            fileName: fileName,
            input_blob_url: `https://${process.env.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${derivedContainerName}/${blobName}`,
            processingType: processingType,
            attenuationDb: attenuationDb,
            actualCost: actualCost, // cents
            fileSizeBytes: actualFileSizeBytes,
            status: 'queued',
            progress: 0,
            message: 'Job queued successfully',
            createdAt: new Date().toISOString()
        };
        
        await jobsContainer.items.create(jobRecord);

        // Update the previously created transaction with the jobId (avoid problematic SQL query)
        if (pendingTransactionId) {
            try {
                const { resource: txn } = await transactionsContainer.item(pendingTransactionId, userId).read();
                if (txn) {
                    txn.jobId = jobId;
                    await transactionsContainer.item(txn.id, txn.userId).replace(txn);
                    await logger.logInfo('enqueue-job', 'Transaction updated with jobId', userId, { sessionId, jobId, transactionId: pendingTransactionId });
                } else {
                    await logger.logError('enqueue-job', 'Pending transaction not found for update', userId, { sessionId, jobId, transactionId: pendingTransactionId });
                }
            } catch (txnErr) {
                await logger.logError('enqueue-job', `Failed to update transaction with jobId err=${txnErr.message}`, userId, { sessionId, jobId, transactionId: pendingTransactionId });
            }
        } else {
            await logger.logError('enqueue-job', 'No pendingTransactionId recorded to update', userId, { sessionId, jobId });
        }

        await logger.logInfo('enqueue-job', 'Job successfully created in database', userId, {
            sessionId,
            jobId,
            jobRecord
        });

        // Send message to Service Bus to trigger processor
        try {
            const serviceBusConnectionString = process.env.AZURE_SERVICE_BUS_CONNECTION_STRING;
            
            await logger.logInfo('enqueue-job', 'Attempting Service Bus connection', userId, {
                sessionId,
                jobId,
                hasConnectionString: !!serviceBusConnectionString,
                connectionStringStart: serviceBusConnectionString ? serviceBusConnectionString.substring(0, 30) + '...' : 'undefined'
            });
            
            if (serviceBusConnectionString) {
                const sbClient = new ServiceBusClient(serviceBusConnectionString);
                const sender = sbClient.createSender('video-processing-jobs');
                
                const message = {
                    body: {
                        jobId: jobId,
                        userId: userId,
                        sessionId: sessionId,
                        timestamp: new Date().toISOString()
                    },
                    messageId: jobId,
                    sessionId: sessionId
                };

                await sender.sendMessages(message);
                await sender.close();
                await sbClient.close();

                await logger.logInfo('enqueue-job', 'Job message sent to Service Bus', userId, {
                    sessionId,
                    jobId,
                    queueName: 'video-processing-jobs'
                });
            } else {
                await logger.logError('enqueue-job', 'Service Bus connection string not configured', userId, {
                    sessionId,
                    jobId
                });
            }
        } catch (serviceBusError) {
            // Don't fail the entire request if Service Bus fails
            await logger.logError('enqueue-job', `Failed to send message to Service Bus: ${serviceBusError.message}`, userId, {
                sessionId,
                jobId,
                error: serviceBusError.message,
                errorCode: serviceBusError.code,
                errorName: serviceBusError.name,
                errorStack: serviceBusError.stack?.substring(0, 300),
                connectionStringExists: !!process.env.AZURE_SERVICE_BUS_CONNECTION_STRING,
                connectionStringLength: process.env.AZURE_SERVICE_BUS_CONNECTION_STRING?.length || 0
            });
        }

        // Return success response
        const responseBody = {
            id: jobId,
            status: 'queued',
            message: 'Job queued successfully',
            fileName: fileName,
            processingType: processingType
        };

        const duration = Date.now() - startTime;
        await logger.logPerformance('enqueue-job', 'Job enqueue complete', duration, userId, {
            sessionId,
            jobId,
            responseBody
        });

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: responseBody
        };

    } catch (error) {
        const duration = Date.now() - startTime;
        logger.logError('enqueue-job', error, 'system', {
            sessionId: sessionId || 'unknown',
            duration: `${duration}ms`
        });
        
        context.res = {
            status: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: { error: 'Internal server error', details: error.message }
        };
    }
};

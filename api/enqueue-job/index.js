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
        const { fileName, fileUrl, processingType, attenuationDb } = req.body || {};

        if (!fileName || !fileUrl) {
            logger.logError('enqueue-job', 'Missing required fields: fileName and fileUrl', userId, { 
                sessionId
            });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'fileName and fileUrl are required' }
            };
            return;
        }

        // Get actual file size from blob storage
        let actualFileSizeBytes;
        try {
            const blobServiceClient = new BlobServiceClient(process.env.AZURE_STORAGE_CONNECTION_STRING);
            const containerClient = blobServiceClient.getContainerClient('uploads');
            
            // Extract blob name from URL
            const url = new URL(fileUrl);
            const pathParts = url.pathname.split('/');
            const blobName = pathParts[pathParts.length - 1];
            
            const blobClient = containerClient.getBlobClient(blobName);
            const properties = await blobClient.getProperties();
            actualFileSizeBytes = properties.contentLength;
            
            if (!actualFileSizeBytes || actualFileSizeBytes <= 0) {
                throw new Error('Invalid file size from blob properties');
            }
            
            logger.logInfo('enqueue-job', 'Retrieved actual file size from blob', userId, {
                sessionId,
                blobName,
                fileSizeBytes: actualFileSizeBytes
            });
            
        } catch (blobError) {
            logger.logError('enqueue-job', 'Failed to get file size from blob storage', userId, {
                sessionId,
                fileUrl,
                error: blobError.message
            });
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

        const actualCost = calculateProcessingCost(actualFileSizeBytes);

        // Initialize optimized Cosmos client with retry-aware configuration
        const cosmosClient = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const jobsContainer = database.container('Jobs');
        const accountsContainer = database.container('accounts');
        const transactionsContainer = database.container('transactions');

        // Check account balance and deduct cost
        try {
            const { resource: account } = await accountsContainer.item(userId, userId).read();
            
            if (!account || account.balance < actualCost) {
                logger.logError('enqueue-job', 'Insufficient account balance', userId, { 
                    sessionId, 
                    balance: account?.balance || 0, 
                    requiredCost: actualCost 
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
                        requiredAmount: actualCost
                    }
                };
                return;
            }

            // Deduct cost from account balance
            account.balance -= actualCost;
            account.updatedAt = new Date().toISOString();
            await accountsContainer.item(account.id, account.userId).replace(account);

            // Create transaction record
            const transactionId = `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            const transaction = {
                id: transactionId,
                userId: userId,
                type: 'processing',
                amount: actualCost,
                description: `Video processing: ${fileName}`,
                jobId: null, // Will be updated after job creation
                createdAt: new Date().toISOString()
            };

            await transactionsContainer.items.create(transaction);

            logger.logInfo('enqueue-job', 'Account balance updated and transaction created', userId, {
                sessionId,
                previousBalance: account.balance + actualCost,
                newBalance: account.balance,
                transactionId,
                actualCost
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
        const outputFileName = fileName ? fileName.replace(/\.[^/.]+$/, '_cleaned.mp3') : 'cleaned_audio.mp3';
        
        const jobRecord = {
            id: jobId,
            userId: userId,
            fileName: fileName,
            input_blob_url: fileUrl,
            processingType: processingType,
            attenuationDb: attenuationDb,
            actualCost: actualCost, // Store actual cost for potential refund
            fileSizeBytes: actualFileSizeBytes,
            status: 'queued',
            progress: 0,
            message: 'Job queued successfully',
            createdAt: new Date().toISOString()
        };
        
        await jobsContainer.items.create(jobRecord);

        // Update transaction with jobId
        const transactionQuery = {
            query: 'SELECT * FROM c WHERE c.userId = @userId AND c.jobId IS NULL ORDER BY c.createdAt DESC OFFSET 0 LIMIT 1',
            parameters: [{ name: '@userId', value: userId }]
        };
        const { resources: transactions } = await transactionsContainer.items.query(transactionQuery).fetchAll();
        if (transactions.length > 0) {
            const transaction = transactions[0];
            transaction.jobId = jobId;
            await transactionsContainer.item(transaction.id, transaction.userId).replace(transaction);
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
            fileUrl: fileUrl,
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

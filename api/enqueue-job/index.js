const { CosmosClient } = require('@azure/cosmos');
const { BlobServiceClient } = require('@azure/storage-blob');
const { ServiceBusClient } = require('@azure/service-bus');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');

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

        // Generate job ID and output file name
        const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const outputFileName = fileName ? fileName.replace(/\.[^/.]+$/, '_cleaned.mp3') : 'cleaned_audio.mp3';

        // Initialize optimized Cosmos client with retry-aware configuration
        const cosmosClient = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const container = database.container('Jobs');
        
        const jobRecord = {
            id: jobId,
            userId: userId,
            fileName: fileName,
            input_blob_url: fileUrl,
            processingType: processingType,
            attenuationDb: attenuationDb,
            status: 'queued',
            progress: 0,
            message: 'Job queued successfully',
            createdAt: new Date().toISOString()
        };
        
        await container.items.create(jobRecord);

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

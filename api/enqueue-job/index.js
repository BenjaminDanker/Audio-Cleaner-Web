const { ServiceBusClient } = require('@azure/service-bus');
const { CosmosClient } = require('@azure/cosmos');

module.exports = async function (context, req) {
    context.log('Enqueue job endpoint called');
    
    try {
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Local development mode - using local processor');
            
            // Validate request body
            const { fileName, fileUrl, processingType = 'denoise', attenuationDb = 30 } = req.body;
            if (!fileName || !fileUrl) {
                context.res = {
                    status: 400,
                    body: { error: 'fileName and fileUrl are required' }
                };
                return;
            }

            // Validate attenuation parameter
            const atten = parseInt(attenuationDb);
            if (isNaN(atten) || atten < 10 || atten > 50) {
                context.res = {
                    status: 400,
                    body: { error: 'attenuationDb must be a number between 10 and 50' }
                };
                return;
            }

            // Generate job ID
            const jobId = 'dev-job-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            
            // In local development, we'll add the job to the local processor
            // This simulates adding to Service Bus + Cosmos DB
            const jobData = {
                jobId: jobId,
                fileName: fileName,
                fileUrl: fileUrl,
                processingType: processingType,
                attenuation: atten,
                userId: 'dev-user',
                userEmail: 'dev@example.com'
            };

            // For local development, write individual job files that the processor monitors
            const fs = require('fs');
            const path = require('path');
            
            try {
                // Create jobs directory if it doesn't exist
                const jobsDir = path.join(process.cwd(), 'temp', 'jobs');
                if (!fs.existsSync(jobsDir)) {
                    fs.mkdirSync(jobsDir, { recursive: true });
                }
                
                // Create job data with file path for processing
                // Convert fileUrl to actual file path if localPath not provided
                let actualFilePath = req.body.localPath;
                if (!actualFilePath && fileUrl.startsWith('local://')) {
                    // Extract filename from local:// URL and construct full path
                    const filename = fileUrl.replace('local://', '');
                    actualFilePath = path.join(process.cwd(), 'temp', 'uploads', filename);
                }
                
                const jobFileData = {
                    id: jobId,
                    fileName: fileName,
                    fileUrl: fileUrl,
                    filePath: actualFilePath || fileUrl, // Use actual file path for processing
                    processingType: processingType,
                    attenuation: atten,
                    userId: 'dev-user',
                    userEmail: 'dev@example.com',
                    status: 'queued',
                    progress: 0,
                    createdAt: new Date().toISOString()
                };
                
                // Write individual job file
                const jobFilePath = path.join(jobsDir, `${jobId}.json`);
                fs.writeFileSync(jobFilePath, JSON.stringify(jobFileData, null, 2));
                
                context.res = {
                    status: 200,
                    body: {
                        success: true,
                        id: jobId,
                        jobId: jobId,
                        status: 'queued',
                        message: 'Job queued successfully (local development)',
                        estimatedProcessingTime: '2-5 minutes'
                    }
                };
                return;
                
            } catch (error) {
                context.log.error('Local job creation error:', error);
                context.res = {
                    status: 500,
                    body: { error: 'Failed to create local job: ' + error.message }
                };
                return;
            }
        }

        // Production code - verify authentication
        if (!clientPrincipal) {
            context.res = {
                status: 401,
                body: { error: 'Unauthorized - No client principal found' }
            };
            return;
        }

        // Decode the client principal
        const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = principal.userId;
        const userEmail = principal.userDetails;

        // Validate request body
        const { fileName, fileUrl, processingType = 'denoise', attenuationDb = 30 } = req.body;
        if (!fileName || !fileUrl) {
            context.res = {
                status: 400,
                body: { error: 'fileName and fileUrl are required' }
            };
            return;
        }

        // Validate attenuation parameter
        const atten = parseInt(attenuationDb);
        if (isNaN(atten) || atten < 10 || atten > 50) {
            context.res = {
                status: 400,
                body: { error: 'attenuationDb must be a number between 10 and 50' }
            };
            return;
        }

        // Generate job ID
        const jobId = 'job-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);

        // Initialize clients
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('audiocleaner');
        const jobsContainer = database.container('jobs');

        const serviceBusClient = new ServiceBusClient(process.env.AZURE_SERVICE_BUS_CONNECTION_STRING);
        const sender = serviceBusClient.createSender('audio-processing-queue');

        // Create job record in Cosmos DB
        const jobRecord = {
            id: jobId,
            userId: userId,
            userEmail: userEmail,
            fileName: fileName,
            fileUrl: fileUrl,
            processingType: processingType,
            attenuation: atten,
            status: 'queued',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        };

        await jobsContainer.items.create(jobRecord);

        // Send message to Service Bus queue
        const message = {
            body: JSON.stringify({
                jobId: jobId,
                userId: userId,
                fileName: fileName,
                fileUrl: fileUrl,
                processingType: processingType,
                attenuation: atten
            }),
            messageId: jobId,
            contentType: 'application/json'
        };

        await sender.sendMessages(message);
        await sender.close();
        await serviceBusClient.close();

        context.res = {
            status: 200,
            body: {
                id: jobId,
                status: 'queued',
                message: 'Job queued successfully',
                estimatedTime: '2-5 minutes',
                fileName: fileName,
                processingType: processingType
            }
        };

    } catch (error) {
        context.log.error('Error enqueueing job:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error' }
        };
    }
};

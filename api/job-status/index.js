const { CosmosClient } = require('@azure/cosmos');
const fs = require('fs').promises;
const path = require('path');

module.exports = async function (context, req) {
    context.log('Job status endpoint called');
    
    try {
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.AZURE_COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Local development mode - checking local job status');
            
            // Get job ID from query parameters
            const jobId = req.query.jobId;
            if (!jobId) {
                context.res = {
                    status: 400,
                    body: { error: 'jobId query parameter is required' }
                };
                return;
            }

            try {
                // Check for local job status file
                const jobStatusPath = path.join(process.cwd(), 'temp', 'jobs', `${jobId}.json`);
                const jobStatusExists = await fs.access(jobStatusPath).then(() => true).catch(() => false);
                
                if (jobStatusExists) {
                    const jobData = JSON.parse(await fs.readFile(jobStatusPath, 'utf8'));
                    context.res = {
                        status: 200,
                        body: jobData
                    };
                } else {
                    // Return mock job status for development if no file exists
                    const mockStatuses = ['queued', 'processing', 'completed', 'failed'];
                    const randomStatus = mockStatuses[Math.floor(Math.random() * mockStatuses.length)];
                    
                    context.res = {
                        status: 200,
                        body: {
                            id: jobId,
                            status: randomStatus,
                            fileName: 'test-video.mp4',
                            processingType: 'denoise',
                            progress: randomStatus === 'processing' ? Math.floor(Math.random() * 100) : 
                                     randomStatus === 'completed' ? 100 : 0,
                            createdAt: new Date(Date.now() - 300000).toISOString(),
                            updatedAt: new Date().toISOString(),
                            downloadUrl: randomStatus === 'completed' ? 
                                `http://localhost:7071/api/download/${jobId}` : null,
                            message: randomStatus === 'failed' ? 'Processing failed' : 
                                    randomStatus === 'completed' ? 'Processing completed successfully' :
                                    'Processing in progress'
                        }
                    };
                }
            } catch (error) {
                context.log.error('Error reading local job status:', error);
                context.res = {
                    status: 500,
                    body: { error: 'Error reading job status' }
                };
            }
            return;
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

        // Get job ID from query parameters
        const jobId = req.query.jobId;
        if (!jobId) {
            context.res = {
                status: 400,
                body: { error: 'jobId query parameter is required' }
            };
            return;
        }

        // Initialize Cosmos client
        const client = new CosmosClient(process.env.AZURE_COSMOS_CONNECTION_STRING);
        const database = client.database('audiocleaner');
        const container = database.container('jobs');

        try {
            // Get job record from Cosmos DB
            const { resource: job } = await container.item(jobId, jobId).read();
            
            if (!job) {
                context.res = {
                    status: 404,
                    body: { error: 'Job not found' }
                };
                return;
            }

            // Verify job belongs to user
            if (job.userId !== userId) {
                context.res = {
                    status: 403,
                    body: { error: 'Access denied - Job belongs to different user' }
                };
                return;
            }

            context.res = {
                status: 200,
                body: {
                    id: job.id,
                    status: job.status,
                    progress: job.progress || 0,
                    fileName: job.fileName,
                    processingType: job.processingType,
                    downloadUrl: job.downloadUrl || null,
                    message: job.message || 'Processing in progress',
                    createdAt: job.createdAt,
                    updatedAt: job.updatedAt,
                    completedAt: job.completedAt || null
                }
            };

        } catch (error) {
            if (error.code === 404) {
                context.res = {
                    status: 404,
                    body: { error: 'Job not found' }
                };
            } else {
                context.log.error('Error querying Cosmos DB:', error);
                context.res = {
                    status: 500,
                    body: { error: 'Database error' }
                };
            }
        }

    } catch (error) {
        context.log.error('Error getting job status:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error' }
        };
    }
};

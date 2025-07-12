const { CosmosClient } = require('@azure/cosmos');

module.exports = async function (context, req) {
    context.log('Job status endpoint called');
    
    try {
        // Verify authentication
        const clientPrincipal = req.headers['x-ms-client-principal'];
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
        const client = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
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
                    downloadUrl: job.downloadUrl || job.output_blob_url || null,
                    output_blob_url: job.output_blob_url || null,
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
                context.log.error('Error querying Cosmos DB:', error.message || 'Unknown error');
                context.res = {
                    status: 500,
                    body: { error: 'Database error' }
                };
            }
        }

    } catch (error) {
        context.log.error('Error getting job status:', error.message || 'Unknown error');
        context.res = {
            status: 500,
            body: { error: 'Internal server error' }
        };
    }
};

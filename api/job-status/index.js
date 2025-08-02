const { CosmosClient } = require('@azure/cosmos');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');

module.exports = async function (context, req) {
    // Initialize retry-aware minimal logger
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        // Security check with simple middleware
        const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        const securityResult = await security.checkSecurity(context, req, { requireAuth: true });
        
        if (!securityResult.allowed) {
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
        
        const userId = securityResult.userInfo?.userId;
        context.log('Security result userInfo:', securityResult.userInfo);
        context.log('Extracted userId:', userId);

        if (!userId) {
            context.res = {
                status: 401,
                body: { error: 'Unauthorized - No user ID found' }
            };
            return;
        }

        // Get job ID from query parameters
        const jobId = req.query.jobId;
        
        if (!jobId) {
            context.res = {
                status: 400,
                body: { error: 'jobId query parameter is required' }
            };
            return;
        }

        // Initialize optimized Cosmos client
        const client = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = client.database('AudioCleanerDB');
        const container = database.container('Jobs');

        try {
            const { resource: job } = await container.item(jobId, userId).read();
            
            
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

            const responseData = {
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
            };

            context.res = {
                status: 200,
                body: responseData
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

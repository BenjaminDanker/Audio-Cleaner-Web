const { CosmosClient } = require('@azure/cosmos');
const { BlobServiceClient } = require('@azure/storage-blob');

module.exports = async function (context, req) {
    context.log('Clear jobs function processed a request.');
    
    try {
        // Check if user is authenticated
        const clientPrincipal = req.headers['x-ms-client-principal'];
        context.log('Auth header present:', !!clientPrincipal);
        
        if (!clientPrincipal) {
            context.log('No authentication header found');
            context.res = {
                status: 401,
                body: { success: false, error: 'Unauthorized - no authentication header' }
            };
            return;
        }

        const user = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = user.userId;
        context.log('User ID:', userId);

        if (!userId) {
            context.log('No user ID in authentication token');
            context.res = {
                status: 401,
                body: { success: false, error: 'Unauthorized - no user ID' }
            };
            return;
        }

        // Connect to Cosmos DB
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const container = database.container('Jobs');

        // Connect to Azure Storage
        const storageConnectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
        if (!storageConnectionString) {
            throw new Error('Storage connection string not found');
        }
        const blobServiceClient = BlobServiceClient.fromConnectionString(storageConnectionString);
        const uploadContainer = blobServiceClient.getContainerClient('uploads');
        const processedContainer = blobServiceClient.getContainerClient('processed-videos');

        // Check if this is a single job deletion or all jobs deletion
        const singleJobId = req.query.jobId;
        context.log('Single job ID:', singleJobId);
        
        let querySpec;
        if (singleJobId) {
            // Delete single job
            querySpec = {
                query: 'SELECT * FROM c WHERE c.userId = @userId AND c.id = @jobId',
                parameters: [
                    {
                        name: '@userId',
                        value: userId
                    },
                    {
                        name: '@jobId',
                        value: singleJobId
                    }
                ]
            };
        } else {
            // Delete all jobs for user
            querySpec = {
                query: 'SELECT * FROM c WHERE c.userId = @userId',
                parameters: [
                    {
                        name: '@userId',
                        value: userId
                    }
                ]
            };
        }

        const { resources: jobs } = await container.items.query(querySpec).fetchAll();
        context.log(`Found ${jobs.length} jobs for user ${userId}`);
        
        if (singleJobId && jobs.length === 0) {
            context.log(`Job ${singleJobId} not found for user ${userId}`);
            context.res = {
                status: 404,
                body: { success: false, error: 'Job not found or does not belong to user' }
            };
            return;
        }
        
        let deletedCount = 0;
        let errorCount = 0;
        let blobsDeleted = 0;
        const errors = [];

        // Delete each job and its associated output blobs (input blobs are auto-deleted after processing)
        for (const job of jobs) {
            try {
                // Only delete output blob since input blob is automatically deleted after processing
                if (job.output_blob_url) {
                    try {
                        // Extract blob name from URL
                        const url = new URL(job.output_blob_url);
                        const pathParts = url.pathname.split('/');
                        if (pathParts.length >= 3) {
                            const outputBlobName = pathParts.slice(2).join('/'); // Skip empty string and container name
                            const outputBlobClient = processedContainer.getBlobClient(outputBlobName);
                            const deleteResult = await outputBlobClient.deleteIfExists();
                            if (deleteResult.succeeded) {
                                blobsDeleted++;
                                context.log(`Deleted output blob: ${outputBlobName}`);
                            } else {
                                context.log(`Output blob did not exist or was already deleted: ${outputBlobName}`);
                            }
                        }
                    } catch (blobError) {
                        context.log.warn(`Could not delete output blob from ${job.output_blob_url}:`, blobError.message);
                        // Don't fail the entire operation for blob deletion issues
                    }
                }

                // Also try to delete any potential input blob that might still exist
                if (job.input_blob_name) {
                    try {
                        const inputBlobClient = uploadContainer.getBlobClient(job.input_blob_name);
                        const deleteResult = await inputBlobClient.deleteIfExists();
                        if (deleteResult.succeeded) {
                            context.log(`Deleted orphaned input blob: ${job.input_blob_name}`);
                        }
                    } catch (inputBlobError) {
                        context.log.warn(`Could not delete input blob ${job.input_blob_name}:`, inputBlobError.message);
                        // Don't fail the entire operation for blob deletion issues
                    }
                }

                // Delete job record
                await container.item(job.id, job.userId).delete();
                deletedCount++;
                context.log(`Deleted job: ${job.id}`);
            } catch (error) {
                errorCount++;
                errors.push(`Failed to delete job ${job.id}: ${error.message}`);
                context.log.error(`Error deleting job ${job.id}:`, error.message || 'Unknown error');
            }
        }

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json'
            },
            body: {
                success: true,
                message: singleJobId 
                    ? `Deleted job and ${blobsDeleted} output files (input files are auto-deleted after processing)`
                    : `Deleted ${deletedCount} jobs and ${blobsDeleted} output files (input files are auto-deleted after processing)`,
                deletedCount,
                blobsDeleted,
                errorCount,
                totalJobs: jobs.length,
                errors: errors.length > 0 ? errors : undefined
            }
        };

    } catch (error) {
        context.log.error('Clear jobs error:', error.message || 'Unknown error');
        context.res = {
            status: 500,
            body: { 
                success: false, 
                error: error.message || 'Internal server error' 
            }
        };
    }
};

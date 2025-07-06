const { BlobServiceClient } = require('@azure/storage-blob');
const { CosmosClient } = require('@azure/cosmos');

module.exports = async function (context, myTimer) {
    context.log('Cleanup orphaned blobs function started via timer trigger');
    
    try {
        // Initialize clients
        const blobServiceClient = BlobServiceClient.fromConnectionString(
            process.env.AzureWebJobsStorage
        );
        
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('audiocleaner');
        const jobsContainer = database.container('jobs');
        
        // Get all active job file URLs from Cosmos DB (only queued/processing jobs that are recent)
        const { resources: jobs } = await jobsContainer.items
            .query('SELECT c.fileName, c.fileUrl, c.output_blob_url, c.status, c.createdAt FROM c WHERE c.status IN ("queued", "processing")')
            .fetchAll();
        
        const activeFiles = new Set();
        const now = Date.now();
        
        jobs.forEach(job => {
            // Only protect files from jobs that are recent (less than 24 hours old)
            const jobAge = now - (job.createdAt || 0);
            const isRecentJob = jobAge < 24 * 60 * 60 * 1000; // Less than 24 hours
            
            if (isRecentJob) {
                context.log(`Protecting files for active job (${job.status}, age: ${Math.round(jobAge / 1000 / 60)} min): ${job.fileName}`);
                if (job.fileName) activeFiles.add(job.fileName);
                if (job.fileUrl) {
                    // Extract filename from URL
                    const filename = job.fileUrl.split('/').pop();
                    activeFiles.add(filename);
                    // Also add the full path for user-scoped files
                    activeFiles.add(job.fileUrl.split('/').slice(-2).join('/')); // userId/filename
                }
                if (job.output_blob_url) {
                    const filename = job.output_blob_url.split('/').pop();
                    activeFiles.add(filename);
                    // Also add the full path for user-scoped files
                    activeFiles.add(job.output_blob_url.split('/').slice(-2).join('/')); // userId/filename
                }
            } else {
                context.log(`Not protecting old stuck job (${job.status}, age: ${Math.round(jobAge / 1000 / 60 / 60)} hours): ${job.fileName}`);
            }
        });
        
        // Check uploads container for orphaned files
        const uploadsContainer = blobServiceClient.getContainerClient('uploads');
        const processedContainer = blobServiceClient.getContainerClient('processed');
        
        let orphanedCount = 0;
        let cleanedSize = 0;
        
        // Clean uploads container
        for await (const blob of uploadsContainer.listBlobsFlat()) {
            const fileName = blob.name.split('/').pop(); // Handle user ID prefix
            const blobAge = Date.now() - new Date(blob.properties.lastModified).getTime();
            const isOldBlob = blobAge > 24 * 60 * 60 * 1000; // Older than 24 hours
            
            if (!activeFiles.has(fileName) && !activeFiles.has(blob.name) && isOldBlob) {
                context.log(`Deleting orphaned upload blob: ${blob.name}`);
                await uploadsContainer.deleteBlob(blob.name);
                orphanedCount++;
                cleanedSize += blob.properties.contentLength || 0;
            }
        }
        
        // Clean processed container (keep files for 7 days after job completion)
        for await (const blob of processedContainer.listBlobsFlat()) {
            const fileName = blob.name.split('/').pop();
            const blobAge = Date.now() - new Date(blob.properties.lastModified).getTime();
            const isOldBlob = blobAge > 7 * 24 * 60 * 60 * 1000; // Older than 7 days
            
            if (!activeFiles.has(fileName) && !activeFiles.has(blob.name) && isOldBlob) {
                context.log(`Deleting old processed blob: ${blob.name}`);
                await processedContainer.deleteBlob(blob.name);
                orphanedCount++;
                cleanedSize += blob.properties.contentLength || 0;
            }
        }
        
        context.log(`Cleanup completed: ${orphanedCount} files deleted, ${cleanedSize} bytes freed`);
        
        const responseMessage = `Cleaned up ${orphanedCount} orphaned files, freed ${Math.round(cleanedSize / 1024 / 1024)} MB`;
        context.log(`Timer cleanup result: ${responseMessage}`);
        
    } catch (error) {
        context.log.error('Error during cleanup:', error);
    }
};

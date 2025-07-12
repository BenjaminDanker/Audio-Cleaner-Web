const { BlobServiceClient } = require('@azure/storage-blob');

module.exports = async function (context, req) {
    context.log('Manual blob cleanup function called');
    
    // Handle CORS preflight requests
    if (req.method === 'OPTIONS') {
        context.res = {
            status: 200,
            headers: {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Access-Control-Max-Age': '86400'
            },
            body: ''
        };
        return;
    }
    
    try {
        // Check if user is authenticated
        const clientPrincipal = req.headers['x-ms-client-principal'];
        if (!clientPrincipal) {
            context.res = {
                status: 401,
                body: { success: false, error: 'Unauthorized' }
            };
            return;
        }

        const user = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = user.userId;

        // Get blob name from request body
        const { blobName } = req.body || {};
        
        if (!blobName) {
            context.res = {
                status: 400,
                body: { success: false, error: 'blobName is required' }
            };
            return;
        }

        // Validate that the blob belongs to the authenticated user
        if (!blobName.startsWith(`${userId}/`)) {
            context.res = {
                status: 403,
                body: { 
                    success: false, 
                    error: 'Access denied: You can only delete your own blobs' 
                }
            };
            return;
        }

        // Initialize blob service client
        const connectionString = process.env.AzureWebJobsStorage;
        if (!connectionString) {
            throw new Error('Storage connection string not configured');
        }

        const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
        const containerClient = blobServiceClient.getContainerClient('uploads');
        const blobClient = containerClient.getBlobClient(blobName);

        // Check if blob exists before attempting deletion
        const exists = await blobClient.exists();
        
        if (!exists) {
            context.log(`Blob does not exist: ${blobName}`);
            context.res = {
                status: 200,
                body: { 
                    success: true, 
                    message: 'Blob does not exist (already cleaned up)',
                    deleted: false
                }
            };
            return;
        }

        // Delete the blob
        await blobClient.delete();
        
        context.log(`Successfully deleted blob: ${blobName}`);
        
        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            body: {
                success: true,
                message: 'Blob deleted successfully',
                deleted: true,
                blobName: blobName
            }
        };

    } catch (error) {
        context.log.error('Blob cleanup error:', error);
        context.res = {
            status: 500,
            body: { 
                success: false, 
                error: error.message || 'Internal server error' 
            }
        };
    }
};

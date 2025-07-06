const { BlobServiceClient } = require('@azure/storage-blob');
const SASTokenManager = require('../shared/sasTokenManager');
const fs = require('fs').promises;
const path = require('path');

module.exports = async function (context, req) {
    context.log('Upload file function processed a request.');
    try {
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Development mode: simulating file upload');
            
            const { fileName, fileSize } = req.body || {};
            if (!fileName) {
                context.res = {
                    status: 400,
                    body: { success: false, error: 'fileName is required in development mode' }
                };
                return;
            }

            // Simulate SAS URL generation for development
            const mockFileUrl = `local://uploads/${Date.now()}_${Math.random().toString(36).substr(2, 5)}.${fileName.split('.').pop()}`;
            
            context.res = {
                status: 200,
                body: {
                    success: true,
                    uploadUrl: mockFileUrl, // This would be the SAS URL in production
                    blobName: fileName,
                    fileName: fileName,
                    fileUrl: mockFileUrl // Final URL after upload
                }
            };
            return;
        }

        // Production: Check if user is authenticated
        if (!clientPrincipal) {
            context.res = {
                status: 401,
                body: { success: false, error: 'Unauthorized' }
            };
            return;
        }

        const user = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = user.userId;

        // Handle SAS token request for direct blob upload
        const { fileName, fileSize } = req.body || {};
        
        if (!fileName) {
            context.res = {
                status: 400,
                body: { success: false, error: 'fileName is required' }
            };
            return;
        }

        // Validate file size (optional - Azure Blob Storage can handle very large files)
        const maxFileSize = 2 * 1024 * 1024 * 1024; // 2GB
        if (fileSize && fileSize > maxFileSize) {
            context.res = {
                status: 413,
                body: { 
                    success: false, 
                    error: `File size ${Math.round(fileSize / 1024 / 1024)}MB exceeds maximum allowed size of ${maxFileSize / 1024 / 1024}MB` 
                }
            };
            return;
        }

        // Create blob service client
        const connectionString = process.env.AzureWebJobsStorage;
        if (!connectionString) {
            throw new Error('Storage connection string not configured');
        }

        const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
        const containerClient = blobServiceClient.getContainerClient('uploads');

        // Ensure container exists (private container by default)
        await containerClient.createIfNotExists();

        // Generate unique blob name
        const timestamp = Date.now();
        const randomId = Math.random().toString(36).substr(2, 5);
        const fileExtension = fileName.split('.').pop();
        const blobName = `${userId}/${timestamp}_${randomId}.${fileExtension}`;

        const blobClient = containerClient.getBlobClient(blobName);
        const blockBlobClient = blobClient.getBlockBlobClient();

        // Initialize SAS Token Manager
        const sasManager = new SASTokenManager(
            connectionString, 
            process.env.COSMOS_CONNECTION_STRING
        );
        
        // Get client IP for SAS restriction (Rule #3) - temporarily disabled
        const clientIP = null; // Temporarily disable IP restrictions
        // const clientIP = SASTokenManager.getClientIP(req);
        
        context.log(`Generating SAS token for blob: ${blobName}, user: ${userId}`);
        
        // Generate secure SAS token following all 5 rules
        const sasResult = await sasManager.generateSASToken({
            containerName: 'uploads',
            blobName: blobName,
            permissions: 'cw', // create and write only (Rule #2)
            expiryMinutes: 30, // Increased to 30 minutes for larger uploads
            clientIP: clientIP, // IP restriction if available (Rule #3)
            userId: userId, // For tracking and revocation (Rule #5)
            context: context
        });
        
        const uploadUrl = `${blockBlobClient.url}?${sasResult.sasToken}`;

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-blob-type'
            },
            body: {
                success: true,
                uploadUrl: uploadUrl, // SAS URL is only in response body (Rule #4)
                blobName: blobName,
                fileName: fileName,
                fileUrl: blockBlobClient.url, // This will be the final URL after upload
                debug: {
                    containerName: 'uploads',
                    storageAccount: sasManager.accountName,
                    sasExpiry: sasResult.expiresAt.toISOString(),
                    sasType: sasResult.sasType, // Indicate SAS type
                    hasIPRestriction: !!clientIP
                }
            }
        };

    } catch (error) {
        context.log.error('Upload file error:', error);
        context.res = {
            status: 500,
            body: { 
                success: false, 
                error: error.message || 'Internal server error' 
            }
        };
    }
};

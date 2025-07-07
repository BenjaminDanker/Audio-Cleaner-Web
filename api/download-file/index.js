const fs = require('fs');
const path = require('path');
const { BlobServiceClient } = require('@azure/storage-blob');
const SASTokenManager = require('../shared/sasTokenManager');

module.exports = async function (context, req) {
    context.log('Download endpoint called');
    
    try {
        // Get the filename from the URL parameter
        const filename = req.params.filename || req.query.filename;
        
        if (!filename) {
            context.res = {
                status: 400,
                body: { error: 'Filename parameter is required' }
            };
            return;
        }
        
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Local development mode - serving file locally');
            
            // Construct the file path
            const downloadsDir = path.join(process.cwd(), 'temp', 'downloads');
            const filePath = path.join(downloadsDir, filename);
            
            // Check if file exists
            if (!fs.existsSync(filePath)) {
                context.res = {
                    status: 404,
                    body: { error: 'File not found' }
                };
                return;
            }
            
            // Get file stats
            const stats = fs.statSync(filePath);
            const fileSize = stats.size;
            
            // Handle Range requests for partial content
            const rangeHeader = req.headers['range'];
            if (rangeHeader) {
                const [startStr, endStr] = rangeHeader.replace(/bytes=/, '').split('-');
                const start = parseInt(startStr, 10);
                const end = endStr ? parseInt(endStr, 10) : fileSize - 1;
                const chunkSize = end - start + 1;
                const buffer = Buffer.alloc(chunkSize);
                const fd = fs.openSync(filePath, 'r');
                fs.readSync(fd, buffer, 0, chunkSize, start);
                fs.closeSync(fd);
                context.res = {
                    status: 206,
                    headers: {
                        'Content-Type': 'video/mp4',
                        'Content-Length': chunkSize.toString(),
                        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': `attachment; filename="${filename}"`
                    },
                    body: buffer,
                    isRaw: true
                };
                return;
            }
            // For HEAD requests, return only headers
            if (req.method === 'HEAD') {
                context.res = {
                    status: 200,
                    headers: {
                        'Content-Type': 'video/mp4',
                        'Content-Disposition': `attachment; filename="${filename}"`,
                        'Content-Length': fileSize.toString(),
                        'Accept-Ranges': 'bytes'
                    }
                };
                return;
            }
            // Read the file
            const fileBuffer = fs.readFileSync(filePath);
            
            context.res = {
                status: 200,
                headers: {
                    'Content-Type': 'video/mp4',
                    'Content-Disposition': `attachment; filename="${filename}"`,
                    'Content-Length': fileSize.toString(),
                    'Accept-Ranges': 'bytes'
                },
                body: fileBuffer,
                isRaw: true
            };
            return;
        }
        
        // Production code - generate SAS URL for direct download from Azure Blob Storage
        try {
            context.log('Environment check:');
            context.log('- AzureWebJobsStorage exists:', !!process.env.AzureWebJobsStorage);
            context.log('- Requested filename:', filename);
            
            if (!process.env.AzureWebJobsStorage) {
                throw new Error('AzureWebJobsStorage environment variable is not set');
            }
            
            const { BlobSASPermissions, generateBlobSASQueryParameters, StorageSharedKeyCredential } = require('@azure/storage-blob');
            
            context.log('Creating BlobServiceClient...');
            const blobServiceClient = BlobServiceClient.fromConnectionString(process.env.AzureWebJobsStorage);
            
            // Extract account name and key from connection string
            const connectionString = process.env.AzureWebJobsStorage;
            const accountNameMatch = connectionString.match(/AccountName=([^;]+)/);
            const accountKeyMatch = connectionString.match(/AccountKey=([^;]+)/);
            
            if (!accountNameMatch || !accountKeyMatch) {
                throw new Error('Could not extract account name or key from connection string');
            }
            
            const accountName = accountNameMatch[1];
            const accountKey = accountKeyMatch[1];
            const sharedKeyCredential = new StorageSharedKeyCredential(accountName, accountKey);
            
            context.log('Getting container client...');
            const containerClient = blobServiceClient.getContainerClient('processed');
            
            // Try direct filename first
            let blobClient = containerClient.getBlobClient(filename);
            let blobName = filename;
            let exists = await blobClient.exists();
            
            // If not found and filename doesn't contain a path, try searching for it
            if (!exists && !filename.includes('/')) {
                context.log(`Direct filename not found, searching for blob containing: ${filename}`);
                
                // List blobs that end with this filename
                for await (const blob of containerClient.listBlobsFlat()) {
                    if (blob.name.endsWith(filename) || blob.name.includes(filename)) {
                        context.log(`Found matching blob: ${blob.name}`);
                        blobClient = containerClient.getBlobClient(blob.name);
                        blobName = blob.name;
                        exists = true;
                        break;
                    }
                }
            }
            
            if (!exists) {
                context.log(`Blob not found: ${filename}`);
                context.res = {
                    status: 404,
                    body: { error: 'File not found' }
                };
                return;
            }
            
            // Get current user for tracking
            const clientPrincipal = req.headers['x-ms-client-principal'];
            let userId = null;
            if (clientPrincipal) {
                const user = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
                userId = user.userId;
            }
            
            // Initialize SAS Token Manager
            const sasManager = new SASTokenManager(
                process.env.AzureWebJobsStorage,
                process.env.COSMOS_CONNECTION_STRING
            );
            
            // Get client IP for SAS restriction (Rule #3)  
            const clientIP = SASTokenManager.getClientIP(req);
            
            // Generate secure SAS token for download
            const sasResult = await sasManager.generateSASToken({
                containerName: 'processed',
                blobName: blobName,
                permissions: 'r', // read-only (Rule #2)
                expiryMinutes: 10, // Very short-lived for downloads (Rule #2)
                clientIP: clientIP, // IP restriction if available (Rule #3)
                userId: userId, // For tracking and revocation (Rule #5)
                context: context
            });
            
            const sasUrl = `${blobClient.url}?${sasResult.sasToken}`;
            
            // For HEAD requests, get blob properties and return headers
            if (req.method === 'HEAD') {
                try {
                    const properties = await blobClient.getProperties();
                    context.res = {
                    status: 200,
                    headers: {
                        'Content-Type': properties.contentType || 'video/mp4',
                        'Content-Length': properties.contentLength?.toString() || '0',
                        'Accept-Ranges': 'bytes',
                        'Location': sasUrl
                    }
                    };
                } catch (e) {
                    context.log.error('HEAD request failed:', e.message);
                    context.res = {
                    status: 404,
                    body: { error: 'Blob not found or inaccessible' }
                    };
                }
                return;
            }

            // For GET requests, redirect to the SAS URL
            context.res = {
                status: 302,
                headers: {
                    'Location': sasUrl
                }
            };
            
        } catch (error) {
            context.log.error('Error downloading from blob storage:', error);
            context.res = {
                status: 500,
                body: { error: 'Error downloading file from storage' }
            };
        }
        
    } catch (error) {
        context.log.error('Error downloading file:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error: ' + error.message }
        };
    }
};

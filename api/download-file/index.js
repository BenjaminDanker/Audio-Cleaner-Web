const fs = require('fs');
const path = require('path');
const { BlobServiceClient } = require('@azure/storage-blob');

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
            
            // Read the file
            const fileBuffer = fs.readFileSync(filePath);
            
            context.res = {
                status: 200,
                headers: {
                    'Content-Type': 'video/mp4',
                    'Content-Disposition': `attachment; filename="${filename}"`,
                    'Content-Length': fileSize.toString()
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
            
            // Generate SAS token for direct download (valid for 1 hour)
            const sasOptions = {
                containerName: 'processed',
                blobName: blobName,
                permissions: BlobSASPermissions.parse('r'), // read permission
                startsOn: new Date(),
                expiresOn: new Date(new Date().valueOf() + 3600 * 1000), // 1 hour from now
            };
            
            const sasToken = generateBlobSASQueryParameters(sasOptions, sharedKeyCredential).toString();
            const sasUrl = `${blobClient.url}?${sasToken}`;
            
            context.log(`Generated SAS URL for blob: ${blobName}`);
            
            // Redirect to the SAS URL for direct download
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

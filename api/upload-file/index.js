const { BlobServiceClient } = require('@azure/storage-blob');
const busboy = require('busboy');
const fs = require('fs').promises;
const path = require('path');

module.exports = async function (context, req) {
    context.log('Upload file endpoint called');
    
    try {
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.AZURE_COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Local development mode - saving file locally');
            
            // Handle Express multer upload
            if (req.file) {
                const fileName = req.file.originalname;
                const fileSize = req.file.size;
                const filePath = req.file.path;
                
                context.log(`File uploaded: ${fileName}, size: ${fileSize}, path: ${filePath}`);
                
                context.res = {
                    status: 200,
                    body: {
                        success: true,
                        fileName: fileName,
                        fileUrl: `local://${req.file.filename}`,
                        localPath: filePath,
                        fileSize: fileSize,
                        message: 'File uploaded successfully (development mode)'
                    }
                };
                return;
            }
            
            // Fallback for other upload methods
            const fileName = req.headers['x-file-name'] || 'uploaded-file.mp4';
            const fileSize = req.headers['content-length'] || 0;
            
            context.log(`Mock upload: ${fileName}, size: ${fileSize}`);
            
            // Create uploads directory if it doesn't exist
            const uploadsDir = path.join(process.cwd(), 'temp', 'uploads');
            try {
                await fs.mkdir(uploadsDir, { recursive: true });
            } catch (error) {
                // Directory might already exist
            }
            
            // Generate a unique filename for the mock
            const uniqueFileName = `dev-${Date.now()}-${fileName}`;
            const localPath = path.join(uploadsDir, uniqueFileName);
            
            // For now, we'll just create a placeholder file and use the test video
            // In a real implementation, you'd save the actual uploaded file
            const testVideoPath = path.resolve('video/C1395.MP4');
            
            context.res = {
                status: 200,
                body: {
                    success: true,
                    fileName: fileName,
                    fileUrl: `local://${uniqueFileName}`,
                    localPath: testVideoPath, // Use test video for processing
                    fileSize: fileSize,
                    message: 'File uploaded successfully (development mode - using test video)'
                }
            };
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

        // Parse multipart form data
        const chunks = [];
        let fileName = '';
        let fileSize = 0;

        const bb = busboy({
            headers: req.headers,
            limits: {
                fileSize: 500 * 1024 * 1024 // 500MB limit
            }
        });

        bb.on('file', (fieldname, file, filename, encoding, mimetype) => {
            fileName = `${Date.now()}-${filename.filename}`;
            
            file.on('data', (data) => {
                chunks.push(data);
                fileSize += data.length;
            });
        });

        bb.on('close', async () => {
            try {
                if (chunks.length === 0) {
                    context.res = {
                        status: 400,
                        body: { error: 'No file data received' }
                    };
                    return;
                }

                // Combine all chunks
                const fileBuffer = Buffer.concat(chunks);

                // Upload to Azure Blob Storage
                const blobServiceClient = BlobServiceClient.fromConnectionString(
                    process.env.AzureWebJobsStorage
                );
                
                const containerClient = blobServiceClient.getContainerClient('uploads');
                
                // Ensure container exists
                await containerClient.createIfNotExists({
                    access: 'blob'
                });
                
                const blockBlobClient = containerClient.getBlockBlobClient(fileName);
                
                // Upload the file
                await blockBlobClient.upload(fileBuffer, fileBuffer.length, {
                    blobHTTPHeaders: {
                        blobContentType: 'video/mp4'
                    }
                });

                const fileUrl = blockBlobClient.url;

                context.res = {
                    status: 200,
                    body: {
                        success: true,
                        fileName: fileName,
                        fileUrl: fileUrl,
                        fileSize: fileSize,
                        message: 'File uploaded successfully'
                    }
                };

            } catch (uploadError) {
                context.log.error('Upload error:', uploadError);
                context.res = {
                    status: 500,
                    body: { error: 'File upload failed: ' + uploadError.message }
                };
            }
        });

        bb.on('error', (error) => {
            context.log.error('Busboy error:', error);
            context.res = {
                status: 400,
                body: { error: 'Invalid file format: ' + error.message }
            };
        });

        // Write the request body to busboy
        if (req.body) {
            bb.write(req.body);
        }
        bb.end();

    } catch (error) {
        context.log.error('Error uploading file:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error: ' + error.message }
        };
    }
};

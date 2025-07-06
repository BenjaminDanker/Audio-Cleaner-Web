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
            context.log('Local development mode - direct blob simulation');
            
            // Handle JSON request (simplified local upload)
            if (req.body && req.body.mockUpload) {
                const fileName = req.body.fileName || 'test-video.mp4';
                const fileSize = req.body.fileSize || 0;
                const testVideoPath = path.resolve('../video/C1395.MP4');
                const uniqueFileName = `dev-${Date.now()}-${fileName}`;
                
                // Copy test video to uploads directory to simulate blob storage
                const uploadsDir = path.join(process.cwd(), 'temp', 'uploads');
                try {
                    await fs.mkdir(uploadsDir, { recursive: true });
                    const targetPath = path.join(uploadsDir, uniqueFileName);
                    await fs.copyFile(testVideoPath, targetPath);
                    
                    context.res = {
                        status: 200,
                        body: {
                            success: true,
                            fileName: fileName,
                            fileUrl: `local://${uniqueFileName}`,
                            localPath: targetPath,
                            fileSize: (await fs.stat(targetPath)).size,
                            message: 'File uploaded successfully (direct blob simulation)'
                        }
                    };
                } catch (error) {
                    context.log.error('Error in blob simulation:', error);
                    context.res = {
                        status: 500,
                        body: { error: 'Failed to simulate blob upload: ' + error.message }
                    };
                }
                return;
            }
            
            // Handle Express multer upload (when coming through SWA proxy)
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
            
            // Handle direct Azure Functions request (fallback)
            context.log('Direct Azure Functions upload request - using mock response');
            const fileName = req.headers['x-file-name'] || 'test-video.mp4';
            const testVideoPath = path.resolve('../video/C1395.MP4');  // Go up one level from api folder
            const uniqueFileName = `dev-${Date.now()}-${fileName}`;
            
            // Copy test video to uploads directory
            const uploadsDir = path.join(process.cwd(), 'temp', 'uploads');
            try {
                await fs.mkdir(uploadsDir, { recursive: true });
                const targetPath = path.join(uploadsDir, uniqueFileName);
                await fs.copyFile(testVideoPath, targetPath);
                
                context.res = {
                    status: 200,
                    body: {
                        success: true,
                        fileName: fileName,
                        fileUrl: `local://${uniqueFileName}`,
                        localPath: targetPath,
                        fileSize: (await fs.stat(targetPath)).size,
                        message: 'File uploaded successfully (development mode - using test video)'
                    }
                };
            } catch (error) {
                context.log.error('Error copying test video:', error);
                context.res = {
                    status: 500,
                    body: { error: 'Failed to upload file: ' + error.message }
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

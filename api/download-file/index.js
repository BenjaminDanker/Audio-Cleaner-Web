const fs = require('fs');
const path = require('path');
const { BlobServiceClient } = require('@azure/storage-blob');
const SASTokenManager = require('../shared/sasTokenManager');
const SecurityMiddleware = require('../shared/securityMiddleware');
const InputValidator = require('../shared/inputValidator');

module.exports = async function (context, req) {
    const startTime = Date.now();
    context.log('Download endpoint called');
    
    // Initialize security middleware with proper error handling
    const security = new SecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
    await security.initialize();
    
    // Enhanced security check that handles parallel downloads and range requests
    const securityResult = await security.checkSecurity(context, req, {
        requireAuth: true,
        validateInput: true
    });
    
    if (!securityResult.allowed) {
        context.res = {
            status: securityResult.status,
            headers: {
                ...security.getSecurityHeaders('/api/download-file'),
                ...securityResult.headers
            },
            body: securityResult.body
        };
        return;
    }
    
    try {
        // Input validation with proper schema
        const validator = new InputValidator();
        const schema = InputValidator.getSchemaForEndpoint('/api/download-file');
        
        const inputData = {
            filename: req.params.filename || req.query.filename
        };
        
        const validationResult = validator.validateInput(inputData, schema);
        if (!validationResult.valid) {
            context.log.warn('Input validation failed:', validationResult.errors);
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/download-file'),
                body: { 
                    error: 'Invalid input', 
                    details: validationResult.errors 
                }
            };
            return;
        }
        
        const filename = validationResult.data.filename;
        
        // Additional filename security checks
        if (!filename) {
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/download-file'),
                body: { error: 'Filename parameter is required' }
            };
            return;
        }
        
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.COSMOS_CONNECTION_STRING?.includes('localhost');
        
        // Get authenticated user info
        const userInfo = securityResult.userInfo;
        if (!userInfo && !isLocalDev) {
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders('/api/download-file'),
                body: { error: 'Authentication required' }
            };
            return;
        }
        
        if (isLocalDev) {
            context.log('Local development mode - serving file locally');
            
            // Construct the file path with additional security checks
            const downloadsDir = path.join(process.cwd(), 'temp', 'downloads');
            
            // Prevent directory traversal attacks
            const safePath = path.normalize(path.join(downloadsDir, path.basename(filename)));
            if (!safePath.startsWith(downloadsDir)) {
                context.log.warn(`Directory traversal attempt blocked: ${filename}`);
                context.res = {
                    status: 403,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'Access denied' }
                };
                return;
            }
            
            // Check if file exists
            if (!fs.existsSync(safePath)) {
                context.res = {
                    status: 404,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'File not found' }
                };
                return;
            }
            
            // Get file stats and validate
            const stats = fs.statSync(safePath);
            const fileSize = stats.size;
            
            // Security check: Prevent serving excessively large files
            const maxFileSize = 5 * 1024 * 1024 * 1024; // 5GB limit
            if (fileSize > maxFileSize) {
                context.log.warn(`File too large for download: ${fileSize} bytes`);
                context.res = {
                    status: 413,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'File too large' }
                };
                return;
            }
            
            // Handle Range requests for partial content (with security headers)
            const rangeHeader = req.headers['range'];
            if (rangeHeader) {
                // Validate range header format
                const rangeMatch = rangeHeader.match(/bytes=(\d+)-(\d*)/);
                if (!rangeMatch) {
                    context.res = {
                        status: 416,
                        headers: security.getSecurityHeaders('/api/download-file'),
                        body: { error: 'Invalid range request' }
                    };
                    return;
                }
                
                const start = parseInt(rangeMatch[1], 10);
                const end = rangeMatch[2] ? parseInt(rangeMatch[2], 10) : fileSize - 1;
                
                // Validate range bounds
                if (start >= fileSize || end >= fileSize || start > end) {
                    context.res = {
                        status: 416,
                        headers: {
                            ...security.getSecurityHeaders('/api/download-file'),
                            'Content-Range': `bytes */${fileSize}`
                        },
                        body: { error: 'Range not satisfiable' }
                    };
                    return;
                }
                
                const chunkSize = end - start + 1;
                
                // Security: Prevent excessive memory allocation
                const maxChunkSize = 100 * 1024 * 1024; // 100MB
                if (chunkSize > maxChunkSize) {
                    context.res = {
                        status: 413,
                        headers: security.getSecurityHeaders('/api/download-file'),
                        body: { error: 'Range too large' }
                    };
                    return;
                }
                
                const buffer = Buffer.alloc(chunkSize);
                const fd = fs.openSync(safePath, 'r');
                fs.readSync(fd, buffer, 0, chunkSize, start);
                fs.closeSync(fd);
                
                context.res = {
                    status: 206,
                    headers: {
                        ...security.getSecurityHeaders('/api/download-file'),
                        'Content-Type': 'video/mp4',
                        'Content-Length': chunkSize.toString(),
                        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
                        'Accept-Ranges': 'bytes',
                        'Content-Disposition': `attachment; filename="${path.basename(filename)}"`,
                        'Cache-Control': 'private, max-age=3600',
                        'ETag': `"${stats.mtime.getTime()}"`
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
                        ...security.getSecurityHeaders('/api/download-file'),
                        'Content-Type': 'video/mp4',
                        'Content-Disposition': `attachment; filename="${path.basename(filename)}"`,
                        'Content-Length': fileSize.toString(),
                        'Accept-Ranges': 'bytes',
                        'Cache-Control': 'private, max-age=3600',
                        'ETag': `"${stats.mtime.getTime()}"`
                    }
                };
                return;
            }
            
            // Read the file with memory management for large files
            let fileBuffer;
            try {
                if (fileSize > 50 * 1024 * 1024) { // 50MB threshold
                    // For large files, consider streaming instead
                    context.log('Large file download, consider implementing streaming');
                }
                fileBuffer = fs.readFileSync(safePath);
            } catch (error) {
                context.log.error('Error reading file:', error.message || 'Unknown error');
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'Error reading file' }
                };
                return;
            }

            context.res = {
                status: 200,
                headers: {
                    ...security.getSecurityHeaders('/api/download-file'),
                    'Content-Type': 'video/mp4',
                    'Content-Disposition': `attachment; filename="${path.basename(filename)}"`,
                    'Content-Length': fileSize.toString(),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'private, max-age=3600',
                    'ETag': `"${stats.mtime.getTime()}"`
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
            context.log('- Requested filename:', filename.substring(0, 20) + '...'); // Don't log full filename
            context.log('- User ID:', userInfo?.userId?.substring(0, 8) + '...');
            
            if (!process.env.AzureWebJobsStorage) {
                throw new Error('AzureWebJobsStorage environment variable is not set');
            }

            const { BlobSASPermissions, generateBlobSASQueryParameters, StorageSharedKeyCredential } = require('@azure/storage-blob');
            
            context.log('Creating BlobServiceClient...');
            const blobServiceClient = BlobServiceClient.fromConnectionString(process.env.AzureWebJobsStorage);
            
            // Extract account name and key from connection string with validation
            const connectionString = process.env.AzureWebJobsStorage;
            const accountNameMatch = connectionString.match(/AccountName=([^;]+)/);
            const accountKeyMatch = connectionString.match(/AccountKey=([^;]+)/);
            
            if (!accountNameMatch || !accountKeyMatch) {
                throw new Error('Could not extract account name or key from connection string');
            }

            const accountName = accountNameMatch[1];
            const accountKey = accountKeyMatch[1];
            
            // Validate extracted credentials
            if (!accountName || !accountKey || accountName.length < 3 || accountKey.length < 10) {
                throw new Error('Invalid storage account credentials');
            }
            
            const sharedKeyCredential = new StorageSharedKeyCredential(accountName, accountKey);
            
            context.log('Getting container client...');
            const containerClient = blobServiceClient.getContainerClient('processed');
            
            // Enhanced blob discovery with user-specific access control
            let blobClient = containerClient.getBlobClient(filename);
            let blobName = filename;
            let exists = false;
            
            try {
                exists = await blobClient.exists();
            } catch (error) {
                context.log.warn('Error checking blob existence:', error.message);
                exists = false;
            }
            
            // If not found and filename doesn't contain a path, try searching for it
            if (!exists && !filename.includes('/')) {
                context.log(`Direct filename not found, searching for blob containing: ${filename.substring(0, 20)}...`);
                
                let foundBlob = null;
                try {
                    // Security: Limit blob listing and search only user's files if possible
                    const searchPrefix = userInfo?.userId ? `${userInfo.userId}/` : '';
                    const listOptions = {
                        prefix: searchPrefix,
                        maxPageSize: 100 // Limit to prevent excessive API calls
                    };
                    
                    let blobCount = 0;
                    for await (const blob of containerClient.listBlobsFlat(listOptions)) {
                        blobCount++;
                        if (blobCount > 500) { // Safety limit
                            context.log.warn('Blob search limit reached');
                            break;
                        }
                        
                        if (blob.name.endsWith(filename) || blob.name.includes(filename)) {
                            // Additional security: Check if user has access to this blob
                            if (userInfo?.userId && !blob.name.startsWith(userInfo.userId + '/') && 
                                !blob.name.includes('shared/') && !blob.name.includes('public/')) {
                                context.log.warn(`Access denied to blob: ${blob.name.substring(0, 20)}... for user: ${userInfo.userId.substring(0, 8)}...`);
                                continue;
                            }
                            
                            context.log(`Found matching blob: ${blob.name.substring(0, 30)}...`);
                            foundBlob = blob;
                            break;
                        }
                    }
                } catch (listError) {
                    context.log.error('Error listing blobs:', listError.message);
                }
                
                if (foundBlob) {
                    blobClient = containerClient.getBlobClient(foundBlob.name);
                    blobName = foundBlob.name;
                    exists = true;
                }
            }
            
            if (!exists) {
                context.log(`Blob not found: ${filename.substring(0, 20)}...`);
                context.res = {
                    status: 404,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'File not found' }
                };
                return;
            }
              // Get current user for tracking and access control
            const userId = userInfo?.userId;
            if (!userId) {
                context.log.error('No user ID available for SAS token generation');
                context.res = {
                    status: 401,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'User identification required' }
                };
                return;
            }

            // Initialize SAS Token Manager
            const sasManager = new SASTokenManager(
                process.env.AzureWebJobsStorage,
                process.env.COSMOS_CONNECTION_STRING
            );

            // Get client IP for SAS restriction (Rule #3)  
            const clientIP = securityResult.clientIP;
            
            // Get blob properties for content validation and metadata
            let properties;
            try {
                properties = await blobClient.getProperties();
                
                // Security: Validate blob properties
                if (!properties.contentLength || properties.contentLength === 0) {
                    context.log.warn('Blob has no content or invalid content length');
                    context.res = {
                        status: 404,
                        headers: security.getSecurityHeaders('/api/download-file'),
                        body: { error: 'Invalid file' }
                    };
                    return;
                }
                
                // Security: Check for reasonable file size limits
                const maxDownloadSize = 10 * 1024 * 1024 * 1024; // 10GB
                if (properties.contentLength > maxDownloadSize) {
                    context.log.warn(`File too large for download: ${properties.contentLength} bytes`);
                    context.res = {
                        status: 413,
                        headers: security.getSecurityHeaders('/api/download-file'),
                        body: { error: 'File too large for download' }
                    };
                    return;
                }
                
            } catch (propertiesError) {
                context.log.error('Error getting blob properties:', propertiesError.message);
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'Error accessing file' }
                };
                return;
            }

            // Generate secure SAS token for download with enhanced security
            const sasResult = await sasManager.generateSASToken({
                containerName: 'processed',
                blobName: blobName,
                permissions: 'r', // read-only (Rule #2)
                expiryMinutes: 5, // Very short-lived for downloads (Rule #2) - reduced from 10 to 5 minutes
                clientIP: clientIP, // IP restriction if available (Rule #3)
                userId: userId, // For tracking and revocation (Rule #5)
                context: context
            });
            
            if (!sasResult || !sasResult.sasToken) {
                context.log.error('Failed to generate SAS token');
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders('/api/download-file'),
                    body: { error: 'Failed to generate secure download link' }
                };
                return;
            }

            const sasUrl = `${blobClient.url}?${sasResult.sasToken}`;
            
            // Generate ETag for caching
            const etag = properties.etag || `"${Date.now()}"`;
            
            // Log successful download request (without sensitive data)
            context.log(`Download access granted: ${blobName.substring(0, 30)}... for user: ${userId.substring(0, 8)}... (${Date.now() - startTime}ms)`);

            // Always return SAS URL and metadata as JSON with enhanced security headers
            context.res = {
                status: 200,
                headers: {
                    ...security.getSecurityHeaders('/api/download-file'),
                    'Content-Type': 'application/json',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                },
                body: {
                    sasUrl: sasUrl,
                    contentLength: properties.contentLength || 0,
                    contentType: properties.contentType || 'application/octet-stream',
                    fileName: path.basename(blobName),
                    etag: etag,
                    expiresAt: sasResult.expiresAt.toISOString(),
                    downloadSecurityInfo: {
                        sasType: sasResult.sasType,
                        hasIPRestriction: !!clientIP,
                        expiryMinutes: 5,
                        permissions: 'read-only'
                    }
                }
            };
            
        } catch (error) {
            context.log.error('Error downloading from blob storage:', error.message || 'Unknown error');
            
            // Security: Don't expose internal error details
            const safeErrorMessage = error.message?.includes('not found') ? 'File not found' : 
                                   error.message?.includes('access') ? 'Access denied' : 
                                   'Error downloading file from storage';
            
            context.res = {
                status: error.message?.includes('not found') ? 404 : 
                       error.message?.includes('access') ? 403 : 500,
                headers: security.getSecurityHeaders('/api/download-file'),
                body: { error: safeErrorMessage }
            };
        }
        
    } catch (error) {
        context.log.error('Error downloading file:', error.message || 'Unknown error');
        
        // Security: Generic error message to prevent information disclosure
        context.res = {
            status: 500,
            headers: security.getSecurityHeaders('/api/download-file'),
            body: { error: 'Internal server error' }
        };
    }
};

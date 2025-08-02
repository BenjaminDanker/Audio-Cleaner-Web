const fs = require('fs');
const path = require('path');
const { BlobServiceClient } = require('@azure/storage-blob');
const SASTokenManager = require('../shared/sasTokenManager');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const InputValidator = require('../shared/inputValidator');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');

module.exports = async function (context, req) {
    const startTime = Date.now();
    
    // Initialize retry-aware minimal logger
    const logger = new MinimalLogger(context).getLogger();
    
    // Initialize simple security middleware
    const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
    
    // Enhanced security check that handles parallel downloads and range requests
    const securityResult = await security.checkSecurity(context, req, {
        requireAuth: true,
        validateInput: true
    });
    
    if (!securityResult.allowed) {
        context.res = {
            status: securityResult.status,
            headers: {
                ...security.getSecurityHeaders(),
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
            logger.logError('download-file', 'Input validation failed', 'system', { 
                sessionId: req.headers['x-request-id'] || 'unknown' 
            });
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders(),
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
                headers: security.getSecurityHeaders(),
                body: { error: 'Filename parameter is required' }
            };
            return;
        }
        
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.COSMOS_CONNECTION_STRING?.includes('localhost');
        
        // Get authenticated user info with better error handling
        const userInfo = securityResult.userInfo;
        context.log('Authentication check:', {
            hasClientPrincipal: !!clientPrincipal,
            hasUserInfo: !!userInfo,
            isLocalDev: isLocalDev,
            userId: userInfo?.userId?.substring(0, 8) + '...' || 'none'
        });
        
        if (!userInfo && !isLocalDev) {
            context.log.error('Authentication required but no user info available');
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders(),
                body: { error: 'Authentication required' }
            };
            return;
        }

        // Production code - generate SAS URL for direct download from Azure Blob Storage
        try {
            context.log('Starting blob storage operations...');
            context.log('Environment check:');
            context.log('- AZURE_STORAGE_CONNECTION_STRING exists:', !!process.env.AZURE_STORAGE_CONNECTION_STRING);
            context.log('- Requested filename:', filename.substring(0, 20) + '...'); // Don't log full filename
            context.log('- User ID:', userInfo?.userId?.substring(0, 8) + '...');
            
            // Use the same storage connection string as upload-file for consistency
            const storageConnectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
            
            if (!storageConnectionString) {
                context.log.error('CRITICAL: AZURE_STORAGE_CONNECTION_STRING environment variable is not set');
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Storage configuration error - please contact administrator' }
                };
                return;
            }

            const { BlobSASPermissions, generateBlobSASQueryParameters, StorageSharedKeyCredential } = require('@azure/storage-blob');
            
            context.log('Creating optimized BlobServiceClient...');
            const blobServiceClient = AzureSDKConfig.createBlobServiceClient(storageConnectionString);
            
            // Extract account name and key from connection string with validation
            const connectionString = storageConnectionString;
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
            const containerClient = blobServiceClient.getContainerClient('processed-videos');
            
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
                    headers: security.getSecurityHeaders(),
                    body: { error: 'File not found' }
                };
                return;
            }
              // Get current user for tracking and access control
            const userId = userInfo?.userId;
            context.log('Using userId for SAS generation:', userId.substring(0, 8) + '...');
            
            if (!userId) {
                context.log.warn('Using fallback user ID for SAS token generation');
            }

            // Initialize SAS Token Manager
            const sasManager = new SASTokenManager(
                storageConnectionString,
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
                        headers: security.getSecurityHeaders(),
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
                        headers: security.getSecurityHeaders(),
                        body: { error: 'File too large for download' }
                    };
                    return;
                }
                
            } catch (propertiesError) {
                context.log.error('Error getting blob properties:', propertiesError.message);
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Error accessing file' }
                };
                return;
            }

            // Generate secure SAS token for download with enhanced security
            context.log('Generating SAS token...');
            const sasResult = await sasManager.generateSASToken({
                containerName: 'processed-videos',
                blobName: blobName,
                permissions: 'r', // read-only (Rule #2)
                expiryMinutes: 5, // Very short-lived for downloads (Rule #2) - reduced from 10 to 5 minutes
                clientIP: clientIP, // IP restriction if available (Rule #3)
                userId: userId, // For tracking and revocation (Rule #5)
                context: context
            });
            
            context.log('SAS generation result:', {
                success: !!sasResult,
                hasToken: !!sasResult?.sasToken,
                sasType: sasResult?.sasType || 'unknown'
            });
            
            if (!sasResult || !sasResult.sasToken) {
                context.log.error('CRITICAL: Failed to generate SAS token - sasResult:', !!sasResult, 'sasToken:', !!sasResult?.sasToken);
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Failed to generate secure download link - please try again' }
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
                    ...security.getSecurityHeaders(),
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
            context.log.error('CRITICAL ERROR in blob storage operations:', {
                message: error.message,
                stack: error.stack?.substring(0, 500) + '...',
                errorType: error.constructor.name
            });
            
            // Security: Don't expose internal error details
            const safeErrorMessage = error.message?.includes('not found') ? 'File not found' : 
                                   error.message?.includes('access') ? 'Access denied' : 
                                   error.message?.includes('Storage') ? 'Storage service error' :
                                   'Error downloading file from storage';
            
            const statusCode = error.message?.includes('not found') ? 404 : 
                             error.message?.includes('access') ? 403 : 
                             500;
            
            context.res = {
                status: statusCode,
                headers: security.getSecurityHeaders(),
                body: { error: safeErrorMessage }
            };
        }
        
    } catch (error) {
        context.log.error('Error downloading file:', error.message || 'Unknown error');
        
        // Security: Generic error message to prevent information disclosure
        context.res = {
            status: 500,
            headers: security.getSecurityHeaders(),
            body: { error: 'Internal server error' }
        };
    }
};

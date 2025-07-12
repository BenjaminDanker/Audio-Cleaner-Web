const { BlobServiceClient } = require('@azure/storage-blob');
const SASTokenManager = require('../shared/sasTokenManager');
const SecurityMiddleware = require('../shared/securityMiddleware');
const InputValidator = require('../shared/inputValidator');
const fs = require('fs').promises;
const path = require('path');

module.exports = async function (context, req) {
    const startTime = Date.now();
    context.log('Upload file function processed a request.');
    
    try {
        // Initialize security middleware with proper error handling
        const security = new SecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        await security.initialize();
        
        // Enhanced security check that handles parallel uploads
        const securityResult = await security.checkSecurity(context, req, {
            requireAuth: true,
            validateInput: true,
            fileSize: parseInt(req.headers['content-length'] || '0')
        });
        
        if (!securityResult.allowed) {
            context.res = {
                status: securityResult.status,
                headers: {
                    ...security.getSecurityHeaders('/api/upload-file'),
                    ...securityResult.headers
                },
                body: securityResult.body
            };
            return;
        }
        
        // Enhanced authentication check
        const userInfo = securityResult.userInfo;
        if (!userInfo || !userInfo.userId) {
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { success: false, error: 'Authentication required' }
            };
            return;
        }

        const userId = userInfo.userId;
        
        // Input validation with comprehensive schema
        const validator = new InputValidator();
        const schema = InputValidator.getSchemaForEndpoint('/api/upload-file');
        
        const validationResult = validator.validateInput(req.body || {}, schema);
        if (!validationResult.valid) {
            context.log.warn('Upload input validation failed:', validationResult.errors);
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { 
                    success: false, 
                    error: 'Invalid input data',
                    details: validationResult.errors
                }
            };
            return;
        }
        
        const { fileName, fileSize } = validationResult.data;
        // Enhanced validation
        if (!fileName) {
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { success: false, error: 'fileName is required' }
            };
            return;
        }

        // Additional file type validation
        const allowedExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a'];
        const fileExtension = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
        
        if (!allowedExtensions.includes(fileExtension)) {
            context.log.warn(`Blocked upload of unsupported file type: ${fileExtension}`);
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { 
                    success: false, 
                    error: `File type ${fileExtension} is not supported. Allowed types: ${allowedExtensions.join(', ')}` 
                }
            };
            return;
        }

        // Enhanced file size validation with multiple limits
        const maxFileSize = 5 * 1024 * 1024 * 1024; // 5GB absolute limit
        const warnFileSize = 2 * 1024 * 1024 * 1024; // 2GB warning threshold
        
        if (fileSize && fileSize > maxFileSize) {
            context.log.warn(`Blocked upload exceeding size limit: ${Math.round(fileSize / 1024 / 1024)}MB`);
            context.res = {
                status: 413,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { 
                    success: false, 
                    error: `File size ${Math.round(fileSize / 1024 / 1024)}MB exceeds maximum allowed size of ${maxFileSize / 1024 / 1024}MB` 
                }
            };
            return;
        }
        
        if (fileSize && fileSize > warnFileSize) {
            context.log(`Large file upload detected: ${Math.round(fileSize / 1024 / 1024)}MB for user: ${userId.substring(0, 8)}...`);
        }

        // Rate limiting check for file size (additional protection)
        const hourlyUploadLimit = 50 * 1024 * 1024 * 1024; // 50GB per hour per user
        // Note: This would require additional implementation in SecurityMiddleware for upload quotas

        // Create blob service client
        const connectionString = process.env.AzureWebJobsStorage;
        if (!connectionString) {
            throw new Error('Storage connection string not configured');
        }

        const blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
        const containerClient = blobServiceClient.getContainerClient('uploads');

        // Ensure container exists (private container by default)
        await containerClient.createIfNotExists();

        // Generate unique blob name with enhanced security
        const timestamp = Date.now();
        const randomId = Math.random().toString(36).substr(2, 9); // Increased randomness
        const safeFileName = path.basename(fileName).replace(/[^a-zA-Z0-9.-]/g, '_'); // Sanitize filename
        const blobName = `${userId}/${timestamp}_${randomId}_${safeFileName}`;

        const blobClient = containerClient.getBlobClient(blobName);
        const blockBlobClient = blobClient.getBlockBlobClient();

        // Initialize SAS Token Manager
        const sasManager = new SASTokenManager(
            connectionString, 
            process.env.COSMOS_CONNECTION_STRING
        );
        
        // Get client IP for SAS restriction (Rule #3)
        const clientIP = securityResult.clientIP;
        
        context.log(`Generating SAS token for blob: ${blobName.substring(0, 50)}... user: ${userId.substring(0, 8)}...`);
        
        // Generate secure SAS token following all 5 rules with enhanced security
        const sasResult = await sasManager.generateSASToken({
            containerName: 'uploads',
            blobName: blobName,
            permissions: 'cw', // create and write only (Rule #2)
            expiryMinutes: 60, // 1 hour for larger uploads (Rule #2)
            clientIP: clientIP, // IP restriction if available (Rule #3)
            userId: userId, // For tracking and revocation (Rule #5)
            context: context
        });
        
        if (!sasResult || !sasResult.sasToken) {
            context.log.error('Failed to generate SAS token');
            context.res = {
                status: 500,
                headers: security.getSecurityHeaders('/api/upload-file'),
                body: { 
                    success: false, 
                    error: 'Failed to generate secure upload link' 
                }
            };
            return;
        }
        
        const uploadUrl = `${blockBlobClient.url}?${sasResult.sasToken}`;
        
        // Log successful upload token generation (without sensitive data)
        context.log(`Upload token generated for user: ${userId.substring(0, 8)}... file: ${fileName.substring(0, 20)}... (${Date.now() - startTime}ms)`);

        context.res = {
            status: 200,
            headers: {
                ...security.getSecurityHeaders('/api/upload-file'),
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*', // Consider restricting this in production
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-blob-type',
                'Cache-Control': 'no-cache, no-store, must-revalidate'
            },
            body: {
                success: true,
                uploadUrl: uploadUrl, // SAS URL is only in response body (Rule #4)
                blobName: blobName,
                fileName: fileName,
                fileUrl: blockBlobClient.url, // This will be the final URL after upload
                uploadSecurityInfo: {
                    containerName: 'uploads',
                    storageAccount: sasManager.accountName,
                    sasExpiry: sasResult.expiresAt.toISOString(),
                    sasType: sasResult.sasType, // Indicate SAS type
                    hasIPRestriction: !!clientIP,
                    expiryMinutes: 60,
                    permissions: 'create,write',
                    maxFileSize: maxFileSize,
                    allowedFileTypes: allowedExtensions
                }
            }
        };

    } catch (error) {
        // CRITICAL: Only log error message to prevent massive log costs
        context.log.error('Upload file error:', error.message || 'Unknown error');
        
        // Initialize a basic security object for headers if middleware failed
        let security = null;
        try {
            security = new SecurityMiddleware();
        } catch (secError) {
            context.log.error('Failed to create security middleware for error response:', secError.message || 'Unknown error');
        }
        
        // Security: Don't expose internal error details
        const safeErrorMessage = error.message?.includes('authentication') ? 'Authentication failed' :
                                error.message?.includes('storage') ? 'Storage service unavailable' :
                                error.message?.includes('cosmos') ? 'Security service temporarily unavailable' :
                                'Upload service temporarily unavailable';
        
        context.res = {
            status: error.message?.includes('authentication') ? 401 : 500,
            headers: security ? security.getSecurityHeaders('/api/upload-file') : {},
            body: { 
                success: false, 
                error: safeErrorMessage
            }
        };
    }
};

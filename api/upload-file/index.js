const { BlobServiceClient } = require('@azure/storage-blob');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');
const InputValidator = require('../shared/inputValidator');
const SASTokenManager = require('../shared/sasTokenManager');

module.exports = async function (context, req) {
    const startTime = Date.now();
    
    // Generate session ID for this request
    const sessionId = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize retry-aware minimal logger
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        // Initialize simple security middleware
        const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        
        // Consistent security check - always require auth to match enqueue-job behavior
        const securityResult = await security.checkSecurity(context, req, {
            requireAuth: true,
            validateInput: true,
            fileSize: parseInt(req.headers['content-length'] || '0')
        });
        
        if (!securityResult.allowed) {
            context.res = {
                status: securityResult.status,
                headers: {
                    'Content-Type': 'application/json',
                    ...security.getSecurityHeaders(),
                    ...securityResult.headers
                },
                body: securityResult.body
            };
            return;
        }
        
        // Get user info - should always be available since auth is required
        const userInfo = securityResult.userInfo;
        if (!userInfo || !userInfo.userId) {
            logger.logError('upload-file', 'Authentication succeeded but user info missing', 'system', { sessionId });
            context.res = {
                status: 500,
                headers: {
                    'Content-Type': 'application/json',
                    ...security.getSecurityHeaders()
                },
                body: { error: 'Server error: User information not available' }
            };
            return;
        }
        const userId = userInfo.userId || userInfo.email;
        
        // Basic input validation
        const { fileName, fileSize } = req.body || {};
        
        if (!fileName) {
            logger.logError('upload-file', 'fileName is required', userId, { sessionId });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    ...security.getSecurityHeaders()
                },
                body: { success: false, error: 'fileName is required' }
            };
            return;
        }

        // Initialize input validator for centralized validation
        const validator = new InputValidator();
        
        // File type validation using validator
        const allowedExtensions = validator.allowedFileTypes;
        const fileExtension = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
        
        if (!allowedExtensions.includes(fileExtension)) {
            logger.logError('upload-file', `Unsupported file type: ${fileExtension}`, userId, { sessionId });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    ...security.getSecurityHeaders()
                },
                body: { 
                    success: false, 
                    error: `File type ${fileExtension} is not supported. Allowed types: ${allowedExtensions.join(', ')}` 
                }
            };
            return;
        }

        // File size validation using centralized limit
        const maxFileSize = validator.getFileUploadLimit();
        if (fileSize && fileSize > maxFileSize) {
            logger.logError('upload-file', `File size exceeds limit`, userId, { sessionId });
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    ...security.getSecurityHeaders()
                },
                body: { 
                    success: false, 
                    error: `File size ${Math.round(fileSize / 1024 / 1024)}MB exceeds maximum allowed size of ${Math.round(maxFileSize / 1024 / 1024 / 1024)}GB` 
                }
            };
            return;
        }

        // Create blob service client
        const connectionString = process.env.AZURE_STORAGE_CONNECTION_STRING;
        if (!connectionString) {
            logger.logError('upload-file', 'Storage connection string not configured', userId, { sessionId });
            context.res = {
                status: 500,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { success: false, error: 'Storage connection string not configured' }
            };
            return;
        }

        const blobServiceClient = AzureSDKConfig.createBlobServiceClient(connectionString);
        const containerClient = blobServiceClient.getContainerClient('uploads');

        // Ensure container exists
        await containerClient.createIfNotExists();

        // Generate unique blob name
        const timestamp = Date.now();
        const randomId = Math.random().toString(36).substr(2, 9);
        const safeFileName = fileName.replace(/[^a-zA-Z0-9.-]/g, '_');
        const blobName = `${userId}/${timestamp}_${randomId}_${safeFileName}`;

        const blobClient = containerClient.getBlobClient(blobName);

        // Initialize SAS Token Manager for secure token generation
        const sasManager = new SASTokenManager(connectionString, context);

        // Get client IP for SAS restriction
        const clientIP = SASTokenManager.getClientIP(req);
        
        // Generate secure upload SAS token
        context.log('Generating secure upload SAS token...');
        const sasResult = await sasManager.generateSASToken({
            containerName: 'uploads',
            blobName: blobName,
            permissions: 'cw', // create and write
            expiryMinutes: 60, // 1 hour
            clientIP: clientIP, // IP restriction for security
            context: context
        });
        
        if (!sasResult || !sasResult.sasToken) {
            logger.logError('upload-file', 'Failed to generate secure upload SAS token', userId, { sessionId });
            context.res = {
                status: 500,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    ...security.getSecurityHeaders()
                },
                body: { 
                    success: false, 
                    error: 'Failed to generate secure upload URL - please try again'
                }
            };
            return;
        }

        // Construct full upload URL
        const uploadUrl = `${blobClient.url}?${sasResult.sasToken}`;

        // Log successful operation
        const duration = Date.now() - startTime;
        
        logger.logInfo('upload-file', 'Upload URL generated successfully', userId, {
            sessionId,
            duration: `${duration}ms`,
            fileExtension
        });

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                ...security.getSecurityHeaders()
            },
            body: {
                success: true,
                uploadUrl: uploadUrl,
                blobName: blobName,
                containerId: 'uploads',
                expiresAt: sasResult.expiresAt.toISOString(),
                uploadSecurityInfo: {
                    sasType: sasResult.sasType,
                    hasIPRestriction: !!clientIP,
                    expiryMinutes: 60,
                    permissions: 'create-write-only'
                }
            }
        };

    } catch (error) {
        const duration = Date.now() - startTime;
        logger.logError('upload-file', error, 'system', {
            sessionId: sessionId || 'unknown',
            duration: `${duration}ms`
        });
        
        context.res = {
            status: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: { 
                success: false, 
                error: 'Internal server error',
                details: error.message
            }
        };
    }
};

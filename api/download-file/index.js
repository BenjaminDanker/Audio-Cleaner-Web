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
        const schema = validator.getSchemaForEndpoint('/api/download-file');
        
        const inputData = {
            jobId: req.query.jobId || req.params.jobId  // Query parameter first, then path parameter
        };
        
        const validationResult = validator.validateInput(inputData, schema);
        if (!validationResult.valid) {
            logger.logError('download-file', 'Input validation failed', 'system', { 
                sessionId: req.headers['x-request-id'] || 'unknown',
                inputData: inputData,
                validationErrors: validationResult.errors,
                schema: schema
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
        
        const jobId = validationResult.data.jobId;
        
        // Additional jobId security checks
        if (!jobId) {
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders(),
                body: { error: 'Job ID parameter is required' }
            };
            return;
        }
        
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        
        // Get authenticated user info with better error handling
        const userInfo = securityResult.userInfo;
        context.log('Authentication check:', {
            hasClientPrincipal: !!clientPrincipal,
            hasUserInfo: !!userInfo,
            userId: userInfo?.userId?.substring(0, 8) + '...' || 'none'
        });
        
        if (!userInfo) {
            context.log.error('Authentication required but no user info available');
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders(),
                body: { error: 'Authentication required' }
            };
            return;
        }

        // Get current user ID for exact path construction
        const userId = userInfo?.userId;
        if (!userId) {
            context.log.error('User ID is required for secure file access');
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders(),
                body: { error: 'User ID required for file access' }
            };
            return;
        }

        // Production code - generate SAS URL for direct download from Azure Blob Storage
        try {
            context.log('Starting job lookup and blob storage operations...');
            context.log('Environment check:');
            context.log('- AZURE_STORAGE_CONNECTION_STRING exists:', !!process.env.AZURE_STORAGE_CONNECTION_STRING);
            context.log('- Requested jobId:', jobId.substring(0, 20) + '...');
            context.log('- User ID:', userInfo?.userId?.substring(0, 8) + '...');
            
            // First, get the job record from Cosmos DB to get the exact blob name
            const { CosmosClient } = require('@azure/cosmos');
            const cosmosClient = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
            const database = cosmosClient.database('AudioCleanerDB');
            const container = database.container('Jobs');
            
            let job;
            try {
                const { resource: jobRecord } = await container.item(jobId, userId).read();
                job = jobRecord;
                context.log('Job record retrieved successfully:', {
                    jobId: jobId,
                    status: job.status,
                    hasOutputBlobName: !!job.outputBlobName,
                    hasDownloadUrl: !!job.downloadUrl,
                    userId: job.userId?.substring(0, 8) + '...'
                });
            } catch (cosmosError) {
                context.log.error('Cosmos DB error:', {
                    code: cosmosError.code,
                    message: cosmosError.message,
                    statusCode: cosmosError.statusCode
                });
                if (cosmosError.code === 404) {
                    context.log(`Job not found: ${jobId} for user: ${userId.substring(0, 8)}...`);
                    context.res = {
                        status: 404,
                        headers: security.getSecurityHeaders(),
                        body: { error: 'Job not found' }
                    };
                    return;
                } else {
                    throw cosmosError;
                }
            }
            
            // Verify job belongs to user and is completed
            if (job.userId !== userId) {
                context.log(`Access denied: Job ${jobId} belongs to different user`);
                context.res = {
                    status: 403,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Access denied' }
                };
                return;
            }
            
            if (job.status !== 'completed') {
                context.log(`Job not ready for download: ${jobId} status: ${job.status}`);
                context.res = {
                    status: 400,
                    headers: security.getSecurityHeaders(),
                    body: { error: `Job not ready for download. Status: ${job.status}` }
                };
                return;
            }
            
            // Get the exact blob name from the job record
            const blobName = job.outputBlobName;
            if (!blobName) {
                context.log(`No output blob name found for job: ${jobId}`);
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Output file information not available' }
                };
                return;
            }
            
            context.log('Retrieved blob name from job record:', blobName.substring(0, 40) + '...');
            
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
            const containerClient = blobServiceClient.getContainerClient('processed-videos');
            
            // Use exact blob name from job record - no construction needed
            const blobClient = containerClient.getBlobClient(blobName);
            
            let exists = false;
            try {
                exists = await blobClient.exists();
            } catch (error) {
                context.log.warn('Error checking blob existence:', error.message);
                exists = false;
            }
            
            if (!exists) {
                context.log(`Blob not found with exact path: ${blobName.substring(0, 30)}...`);
                context.res = {
                    status: 404,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'File not found' }
                };
                return;
            }
            // Get current user for tracking and access control
            context.log('Using userId for SAS generation:', userId.substring(0, 8) + '...');

            // Initialize SAS Token Manager
            const sasManager = new SASTokenManager(storageConnectionString, context);

            // Get client IP for SAS restriction (Rule #3)  
            const clientIP = SASTokenManager.getClientIP(req);
            
            // Security: Require client IP for SAS token generation
            if (!clientIP) {
                context.log.error('Client IP is required for secure SAS token generation');
                context.res = {
                    status: 403,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Client IP validation required for secure download' }
                };
                return;
            }
            
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
            } catch (propertiesError) {
                context.log.error('Error getting blob properties:', propertiesError.message);
                context.res = {
                    status: 500,
                    headers: security.getSecurityHeaders(),
                    body: { error: 'Error accessing file' }
                };
                return;
            }

            // Generate secure SAS token for download with mandatory IP restriction
            context.log('Generating SAS token with IP restriction...');
            const sasResult = await sasManager.generateSASToken({
                containerName: 'processed-videos',
                blobName: blobName,
                permissions: 'r', // read-only
                expiryMinutes: 60, // 1 hour
                clientIP: clientIP, // IP restriction
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

const { ServiceBusClient } = require('@azure/service-bus');
const { CosmosClient } = require('@azure/cosmos');
const SecurityMiddleware = require('../shared/securityMiddleware');
const InputValidator = require('../shared/inputValidator');

module.exports = async function (context, req) {
    const startTime = Date.now();
    context.log('Enqueue job endpoint called');
    
    // Initialize security middleware with proper error handling
    const security = new SecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
    await security.initialize();
    
    // Security check with specific options for job enqueue endpoint
    const securityResult = await security.checkSecurity(context, req, {
        requireAuth: true,
        validateInput: true
    });
    
    if (!securityResult.allowed) {
        context.res = {
            status: securityResult.status,
            headers: {
                ...security.getSecurityHeaders('/api/enqueue-job'),
                ...securityResult.headers
            },
            body: securityResult.body
        };
        return;
    }
    
    try {
        // Enhanced authentication validation
        const userInfo = securityResult.userInfo;
        if (!userInfo || !userInfo.userId) {
            context.res = {
                status: 401,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'Authentication required' }
            };
            return;
        }

        const userId = userInfo.userId;
        const userEmail = userInfo.email;
        
        // Input validation with comprehensive schema
        const validator = new InputValidator();
        const schema = InputValidator.getSchemaForEndpoint('/api/enqueue-job');
        
        const validationResult = validator.validateInput(req.body || {}, schema);
        if (!validationResult.valid) {
            context.log.warn('Job enqueue input validation failed:', validationResult.errors);
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { 
                    error: 'Invalid input data',
                    details: validationResult.errors
                }
            };
            return;
        }
        
        const { fileName, fileUrl, processingType, attenuationDb } = validationResult.data;

        // Additional validation for required fields
        if (!fileName || !fileUrl) {
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'fileName and fileUrl are required' }
            };
            return;
        }

        // Enhanced URL validation for security
        try {
            const parsedUrl = new URL(fileUrl);
            const allowedHosts = [
                'blob.core.windows.net', // Azure Storage
                'localhost', // Local development
                '127.0.0.1' // Local development
            ];
            
            const isAllowedHost = allowedHosts.some(host => 
                parsedUrl.hostname.endsWith(host) || parsedUrl.hostname === host
            );
            
            if (!isAllowedHost) {
                context.log.warn(`Blocked job with suspicious URL: ${parsedUrl.hostname}`);
                context.res = {
                    status: 400,
                    headers: security.getSecurityHeaders('/api/enqueue-job'),
                    body: { error: 'Invalid file URL source' }
                };
                return;
            }
        } catch (urlError) {
            context.log.warn('Invalid URL provided:', fileUrl);
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'Invalid file URL format' }
            };
            return;
        }

        // Set safe defaults and validate processing parameters
        const safeProcessingType = processingType || 'denoise';
        const safeAttenuationDb = attenuationDb || 30;
        
        // Validate processing type
        const allowedProcessingTypes = ['denoise', 'enhance', 'normalize'];
        if (!allowedProcessingTypes.includes(safeProcessingType)) {
            context.res = {
                status: 400,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: `Invalid processing type. Allowed: ${allowedProcessingTypes.join(', ')}` }
            };
            return;
        }

        // Check for job limits per user (additional security)
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('audiocleaner');
        const jobsContainer = database.container('jobs');
        
        // Rate limiting: Check recent jobs from this user
        const recentJobsQuery = `
            SELECT COUNT(1) as jobCount 
            FROM c 
            WHERE c.userId = "${userId}" 
            AND c.createdAt >= "${new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()}"
        `;
        
        try {
            const { resources: recentJobsResult } = await jobsContainer.items.query(recentJobsQuery).fetchAll();
            const recentJobCount = recentJobsResult[0]?.jobCount || 0;
            const maxJobsPerDay = 50; // Configurable limit
            
            if (recentJobCount >= maxJobsPerDay) {
                context.log.warn(`User ${userId.substring(0, 8)}... exceeded daily job limit: ${recentJobCount}`);
                context.res = {
                    status: 429,
                    headers: {
                        ...security.getSecurityHeaders('/api/enqueue-job'),
                        'Retry-After': '86400' // 24 hours
                    },
                    body: { 
                        error: `Daily job limit exceeded (${maxJobsPerDay} jobs per day)`,
                        retryAfter: 86400
                    }
                };
                return;
            }
        } catch (quotaError) {
            context.log.error('Error checking job quota:', quotaError.message || 'Unknown error');
            // Continue with request if quota check fails
        }

        // Generate secure job ID with enhanced entropy
        const timestamp = Date.now();
        const randomId = Math.random().toString(36).substr(2, 12); // Increased randomness
        const jobId = `job-${timestamp}-${randomId}`;

        // Initialize Service Bus client with error handling
        let serviceBusClient;
        let sender;
        
        try {
            if (!process.env.AZURE_SERVICE_BUS_CONNECTION_STRING) {
                throw new Error('Service Bus connection string not configured');
            }
            
            serviceBusClient = new ServiceBusClient(process.env.AZURE_SERVICE_BUS_CONNECTION_STRING);
            sender = serviceBusClient.createSender('audio-processing-queue');
        } catch (sbError) {
            context.log.error('Service Bus initialization failed:', sbError.message || 'Unknown error');
            context.res = {
                status: 503,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'Job processing service temporarily unavailable' }
            };
            return;
        }

        // Create job record in Cosmos DB with enhanced security fields
        const jobRecord = {
            id: jobId,
            userId: userId,
            userEmail: userEmail,
            fileName: fileName,
            fileUrl: fileUrl,
            processingType: safeProcessingType,
            attenuation: safeAttenuationDb,
            status: 'queued',
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            clientIP: securityResult.clientIP,
            userAgent: req.headers['user-agent']?.substring(0, 200) || 'unknown', // Truncate for safety
            securityContext: {
                rateLimitInfo: securityResult.rateLimitInfo,
                authenticationType: 'azure-ad',
                requestOrigin: req.headers['origin'] || 'unknown'
            }
        };

        try {
            await jobsContainer.items.create(jobRecord);
        } catch (cosmosError) {
            context.log.error('Failed to create job record:', cosmosError.message || 'Unknown Cosmos DB error');
            await sender.close();
            await serviceBusClient.close();
            
            context.res = {
                status: 500,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'Failed to create job record' }
            };
            return;
        }

        // Send message to Service Bus queue with enhanced security
        const message = {
            body: JSON.stringify({
                jobId: jobId,
                userId: userId,
                fileName: fileName,
                fileUrl: fileUrl,
                processingType: safeProcessingType,
                attenuation: safeAttenuationDb,
                createdAt: jobRecord.createdAt,
                securityHash: require('crypto').createHash('sha256')
                    .update(`${jobId}${userId}${fileName}${process.env.JOB_SECURITY_SALT || 'default-salt'}`)
                    .digest('hex')
            }),
            messageId: jobId,
            contentType: 'application/json',
            timeToLive: 7 * 24 * 60 * 60 * 1000, // 7 days TTL
            applicationProperties: {
                userId: userId,
                processingType: safeProcessingType,
                priority: fileUrl.includes('priority=high') ? 'high' : 'normal'
            }
        };

        try {
            await sender.sendMessages(message);
            context.log(`Job queued successfully: ${jobId} for user: ${userId.substring(0, 8)}... (${Date.now() - startTime}ms)`);
        } catch (sendError) {
            context.log.error('Failed to send job to queue:', sendError.message || 'Unknown error');
            
            // Try to mark job as failed in Cosmos DB
            try {
                await jobsContainer.item(jobId).patch([
                    { op: 'replace', path: '/status', value: 'failed' },
                    { op: 'replace', path: '/error', value: 'Failed to queue job' },
                    { op: 'replace', path: '/updatedAt', value: new Date().toISOString() }
                ]);
            } catch (patchError) {
                context.log.error('Failed to update job status:', patchError.message || 'Unknown error');
            }
            
            await sender.close();
            await serviceBusClient.close();
            
            context.res = {
                status: 503,
                headers: security.getSecurityHeaders('/api/enqueue-job'),
                body: { error: 'Failed to queue job for processing' }
            };
            return;
        }

        // Clean up connections
        await sender.close();
        await serviceBusClient.close();

        context.res = {
            status: 200,
            headers: security.getSecurityHeaders('/api/enqueue-job'),
            body: {
                id: jobId,
                status: 'queued',
                message: 'Job queued successfully',
                estimatedTime: '2-5 minutes',
                fileName: fileName,
                processingType: safeProcessingType,
                queuePosition: 'Processing queue', // Could be enhanced with actual position
                securityInfo: {
                    jobCreatedAt: jobRecord.createdAt,
                    hasSecurityValidation: true,
                    rateLimitRemaining: securityResult.rateLimitInfo?.remaining || 0
                }
            }
        };

    } catch (error) {
        context.log.error('Enqueue job error:', error.message || 'Unknown error');
        
        // Security: Don't expose internal error details
        const safeErrorMessage = error.message?.includes('authentication') ? 'Authentication failed' :
                                error.message?.includes('storage') ? 'Storage service unavailable' :
                                error.message?.includes('quota') ? 'Service quota exceeded' :
                                'Job processing service temporarily unavailable';
        
        context.res = {
            status: error.message?.includes('authentication') ? 401 :
                   error.message?.includes('quota') ? 429 : 500,
            headers: security.getSecurityHeaders('/api/enqueue-job'),
            body: { error: safeErrorMessage }
        };
    }
};

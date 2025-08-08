const { CosmosClient } = require('@azure/cosmos');
const MinimalLogger = require('../shared/minimalLogger');

module.exports = async function (context, req) {
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        logger.logInfo('refund-failed-job', 'Processing HTTP refund request', 'system', {
            method: req.method,
            url: req.url
        });

        // Parse the request body for job information
        let jobId, userId;

        if (req.body) {
            jobId = req.body.jobId;
            userId = req.body.userId;
        }

        if (!jobId || !userId) {
            logger.logError('refund-failed-job', 'Missing jobId or userId in request', 'system', {
                body: req.body,
                hasJobId: !!jobId,
                hasUserId: !!userId
            });
            context.res = {
                status: 400,
                body: { error: 'Missing jobId or userId in request body' }
            };
            return;
        }

        // Initialize Cosmos client
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const jobsContainer = database.container('Jobs');

        // Get the job to check if it needs refund
        const { resource: job } = await jobsContainer.item(jobId, userId).read();
        
        if (!job) {
            logger.logWarning('refund-failed-job', 'Job not found for refund request', 'system', {
                jobId,
                userId
            });
            context.res = {
                status: 404,
                body: { error: 'Job not found' }
            };
            return;
        }

        // Update job status to failed if not already
        if (job.status !== 'failed' && job.status !== 'completed') {
            job.status = 'failed';
            job.progress = 0;
            job.message = 'Job failed';
            job.updatedAt = new Date().toISOString();
            await jobsContainer.item(job.id, job.userId).replace(job);
            
            logger.logInfo('refund-failed-job', 'Updated job status to failed', userId, {
                jobId,
                reason: 'HTTP refund request'
            });
        }

        // Process refund if job had a cost and hasn't been refunded
        if (job.actualCost && job.actualCost > 0 && !job.refunded) {
            try {
                // Initialize containers for refund processing
                const accountsContainer = database.container('accounts');
                const transactionsContainer = database.container('transactions');
                
                // Get user's account
                const { resource: account } = await accountsContainer.item(userId, userId).read();
                if (!account) {
                    logger.logError('refund-failed-job', 'User account not found for refund', userId, { jobId });
                    context.res = {
                        status: 404,
                        body: { error: 'User account not found' }
                    };
                    return;
                }
                
                // Add refund amount back to account balance
                account.balance += job.actualCost;
                account.updatedAt = new Date().toISOString();
                await accountsContainer.item(account.id, account.userId).replace(account);
                
                // Create refund transaction record
                const refundTransactionId = `refund_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                const refundTransaction = {
                    id: refundTransactionId,
                    userId: userId,
                    type: 'refund',
                    amount: job.actualCost,
                    description: `Refund for failed job: ${job.fileName}`,
                    jobId: jobId,
                    createdAt: new Date().toISOString()
                };
                await transactionsContainer.items.create(refundTransaction);
                
                // Mark job as refunded
                job.refunded = true;
                job.refundTransactionId = refundTransactionId;
                job.updatedAt = new Date().toISOString();
                await jobsContainer.item(job.id, job.userId).replace(job);

                logger.logInfo('refund-failed-job', 'Refund processed successfully', userId, {
                    jobId,
                    refundAmount: job.actualCost,
                    newBalance: account.balance,
                    transactionId: refundTransactionId
                });

                context.res = {
                    status: 200,
                    body: {
                        success: true,
                        jobId,
                        refundAmount: job.actualCost,
                        transactionId: refundTransactionId,
                        message: 'Refund processed successfully'
                    }
                };

            } catch (refundError) {
                logger.logError('refund-failed-job', 'Failed to process refund', userId, {
                    jobId,
                    refundAmount: job.actualCost,
                    error: refundError.message
                });
                context.res = {
                    status: 500,
                    body: { error: 'Failed to process refund', details: refundError.message }
                };
            }
        } else {
            // Job doesn't need refund (no cost or already refunded)
            logger.logInfo('refund-failed-job', 'No refund needed for job', userId, {
                jobId,
                actualCost: job.actualCost,
                alreadyRefunded: job.refunded
            });
            
            context.res = {
                status: 200,
                body: {
                    success: true,
                    jobId,
                    message: job.refunded ? 'Job already refunded' : 'No refund needed (no cost)'
                }
            };
        }

    } catch (error) {
        logger.logError('refund-failed-job', 'Error processing failed job refund', 'system', {
            error: error.message,
            stack: error.stack
        });
        context.res = {
            status: 500,
            body: { error: 'Internal server error', details: error.message }
        };
    }
};

const { CosmosClient } = require('@azure/cosmos');
const MinimalLogger = require('../shared/minimalLogger');

module.exports = async function (context, deadLetterMsg) {
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        logger.logInfo('refund-failed-job', 'Processing failed job for refund', 'system', {
            messageId: deadLetterMsg.messageId,
            reason: deadLetterMsg.deadLetterReason,
            description: deadLetterMsg.deadLetterErrorDescription
        });

        // Parse the message body
        const messageBody = deadLetterMsg.body || deadLetterMsg;
        let jobId, userId;

        if (typeof messageBody === 'string') {
            const parsed = JSON.parse(messageBody);
            jobId = parsed.jobId;
            userId = parsed.userId;
        } else {
            jobId = messageBody.jobId;
            userId = messageBody.userId;
        }

        if (!jobId || !userId) {
            logger.logError('refund-failed-job', 'Missing jobId or userId in dead letter message', 'system', {
                messageBody,
                hasJobId: !!jobId,
                hasUserId: !!userId
            });
            return;
        }

        // Initialize Cosmos client
        const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = cosmosClient.database('AudioCleanerDB');
        const jobsContainer = database.container('Jobs');

        // Get the job to check if it needs refund
        const { resource: job } = await jobsContainer.item(jobId, userId).read();
        
        if (!job) {
            logger.logWarning('refund-failed-job', 'Job not found for dead letter message', 'system', {
                jobId,
                userId
            });
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
                reason: deadLetterMsg.deadLetterReason
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

            } catch (refundError) {
                logger.logError('refund-failed-job', 'Failed to process refund', userId, {
                    jobId,
                    refundAmount: job.actualCost,
                    error: refundError.message
                });
            }
        }

    } catch (error) {
        logger.logError('refund-failed-job', 'Error processing failed job refund', 'system', {
            error: error.message,
            stack: error.stack
        });
        throw error;
    }
};

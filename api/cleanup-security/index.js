const SecurityMiddleware = require('../shared/securityMiddleware');

module.exports = async function (context, timerTrigger) {
    const startTime = Date.now();
    context.log('Security cleanup function triggered');

    try {
        // Check if Cosmos DB connection string is available
        if (!process.env.COSMOS_CONNECTION_STRING) {
            context.log('COSMOS_CONNECTION_STRING not available, skipping cleanup');
            return;
        }

        // Initialize security middleware with error handling
        let security;
        try {
            security = new SecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
            await security.initialize(); // Ensure containers are initialized
        } catch (initError) {
            context.log.warn('Failed to initialize SecurityMiddleware:', initError.message);
            return;
        }
        
        if (!security.rateLimitContainer) {
            context.log('Rate limit container not available, skipping cleanup');
            return;
        }

        // Cleanup expired rate limit entries
        try {
            await security.cleanupRateLimits(context);
        } catch (cleanupError) {
            context.log.error('Failed to cleanup rate limits:', cleanupError.message);
        }
        
        // Cleanup old security events (keep only last 30 days)
        if (security.securityEventsContainer) {
            try {
                const cutoffDate = new Date(Date.now() - (30 * 24 * 60 * 60 * 1000)); // 30 days ago
                const query = `SELECT * FROM c WHERE c.timestamp < "${cutoffDate.toISOString()}"`;
                
                const { resources: expiredEvents } = await security.securityEventsContainer.items.query(query).fetchAll();
                
                let deletedCount = 0;
                for (const event of expiredEvents) {
                    try {
                        await security.securityEventsContainer.item(event.id).delete();
                        deletedCount++;
                    } catch (deleteError) {
                        context.log.warn(`Failed to delete security event ${event.id}:`, deleteError.message);
                    }
                    
                    // Batch processing to avoid timeout
                    if (deletedCount % 100 === 0 && deletedCount > 0) {
                        context.log(`Deleted ${deletedCount} security events...`);
                    }
                }
                
                if (deletedCount > 0) {
                    context.log(`Cleaned up ${deletedCount} expired security events`);
                } else {
                    context.log('No expired security events to cleanup');
                }
            } catch (error) {
                context.log.error('Failed to cleanup security events:', error.message);
            }
        }
        
        // Log cleanup stats
        const duration = Date.now() - startTime;
        context.log(`Security cleanup completed in ${duration}ms`);
        
    } catch (error) {
        context.log.error('Security cleanup function error:', error.message || error);
    }
};

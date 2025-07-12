const { CosmosClient } = require('@azure/cosmos');

module.exports = async function (context, req) {
    context.log('Get subscription endpoint called');
    
    try {
        // Verify authentication
        const clientPrincipal = req.headers['x-ms-client-principal'];
        if (!clientPrincipal) {
            context.res = {
                status: 401,
                body: { error: 'Unauthorized - No client principal found' }
            };
            return;
        }

        // Decode the client principal
        const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = principal.userId;

        // Initialize Cosmos client
        const client = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = client.database('audiocleaner');
        const container = database.container('subscriptions');

        try {
            // Query for user's subscription
            const { resource: subscription } = await container.item(userId, userId).read();
            
            if (subscription) {
                context.res = {
                    status: 200,
                    body: {
                        id: subscription.id,
                        status: subscription.status,
                        planName: subscription.planName,
                        plan: subscription.plan,
                        tier: subscription.tier,
                        usageLimit: subscription.usageLimit,
                        currentUsage: subscription.currentUsage,
                        nextBillingDate: subscription.nextBillingDate,
                        price: subscription.price
                    }
                };
            } else {
                // No subscription found, return free tier
                context.res = {
                    status: 200,
                    body: {
                        id: 'free-tier',
                        status: 'active',
                        planName: 'Free',
                        plan: 'free',
                        tier: 'free',
                        usageLimit: 3,
                        currentUsage: 0,
                        nextBillingDate: null,
                        price: 'Free'
                    }
                };
            }
        } catch (error) {
            if (error.code === 404) {
                // No subscription found, return free tier
                context.res = {
                    status: 200,
                    body: {
                        id: 'free-tier',
                        status: 'active',
                        planName: 'Free',
                        plan: 'free',
                        tier: 'free',
                        usageLimit: 3,
                        currentUsage: 0,
                        nextBillingDate: null,
                        price: 'Free'
                    }
                };
            } else {
                throw error;
            }
        }
    } catch (error) {
        context.log.error('Error getting subscription:', error.message || 'Unknown error');
        context.res = {
            status: 500,
            body: { error: 'Internal server error' }
        };
    }
};

const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { CosmosClient } = require('@azure/cosmos');

// Initialize Cosmos DB client
const cosmosClient = new CosmosClient(process.env.AZURE_COSMOS_CONNECTION_STRING);
const database = cosmosClient.database('audiocleaner');
const container = database.container('subscriptions');

module.exports = async function (context, req) {
    context.log('Stripe webhook endpoint called');
    
    try {
        const sig = req.headers['stripe-signature'];
        const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;
        
        if (!sig || !endpointSecret) {
            context.res = {
                status: 400,
                body: { error: 'Missing stripe signature or webhook secret' }
            };
            return;
        }

        let event;

        try {
            // Construct the event from the webhook payload
            event = stripe.webhooks.constructEvent(req.rawBody, sig, endpointSecret);
        } catch (err) {
            context.log.error('Webhook signature verification failed:', err.message);
            context.res = {
                status: 400,
                body: { error: 'Webhook signature verification failed' }
            };
            return;
        }

        // Handle the event
        switch (event.type) {
            case 'checkout.session.completed':
                await handleCheckoutSessionCompleted(context, event.data.object);
                break;
            
            case 'customer.subscription.created':
                await handleSubscriptionCreated(context, event.data.object);
                break;
            
            case 'customer.subscription.updated':
                await handleSubscriptionUpdated(context, event.data.object);
                break;
            
            case 'customer.subscription.deleted':
                await handleSubscriptionDeleted(context, event.data.object);
                break;
            
            case 'invoice.payment_succeeded':
                await handlePaymentSucceeded(context, event.data.object);
                break;
            
            case 'invoice.payment_failed':
                await handlePaymentFailed(context, event.data.object);
                break;
            
            default:
                context.log(`Unhandled event type: ${event.type}`);
        }

        context.res = {
            status: 200,
            body: { received: true }
        };

    } catch (error) {
        context.log.error('Error processing webhook:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error' }
        };
    }
};

async function handleCheckoutSessionCompleted(context, session) {
    context.log('Processing checkout session completed:', session.id);
    
    try {
        const userId = session.metadata.userId;
        const userEmail = session.metadata.userEmail;
        
        if (session.mode === 'subscription') {
            // Get the subscription details from Stripe
            const subscription = await stripe.subscriptions.retrieve(session.subscription);
            
            // Save subscription to Cosmos DB
            await container.items.create({
                id: userId,
                userId: userId,
                userEmail: userEmail,
                stripeCustomerId: session.customer,
                stripeSubscriptionId: session.subscription,
                status: subscription.status,
                currentPeriodStart: new Date(subscription.current_period_start * 1000),
                currentPeriodEnd: new Date(subscription.current_period_end * 1000),
                planId: subscription.items.data[0].price.id,
                createdAt: new Date(),
                updatedAt: new Date()
            });
            
            context.log(`Subscription created for user ${userId}`);
        }
    } catch (error) {
        context.log.error('Error handling checkout session:', error);
        throw error;
    }
}

async function handleSubscriptionCreated(context, subscription) {
    context.log('Processing subscription created:', subscription.id);
    // Additional logic if needed
}

async function handleSubscriptionUpdated(context, subscription) {
    context.log('Processing subscription updated:', subscription.id);
    
    try {
        // Find the subscription in Cosmos DB
        const querySpec = {
            query: 'SELECT * FROM c WHERE c.stripeSubscriptionId = @subscriptionId',
            parameters: [
                { name: '@subscriptionId', value: subscription.id }
            ]
        };
        
        const { resources } = await container.items.query(querySpec).fetchAll();
        
        if (resources.length > 0) {
            const existingSubscription = resources[0];
            
            // Update the subscription
            existingSubscription.status = subscription.status;
            existingSubscription.currentPeriodStart = new Date(subscription.current_period_start * 1000);
            existingSubscription.currentPeriodEnd = new Date(subscription.current_period_end * 1000);
            existingSubscription.updatedAt = new Date();
            
            await container.item(existingSubscription.id, existingSubscription.userId).replace(existingSubscription);
            
            context.log(`Subscription updated for ${existingSubscription.userId}`);
        }
    } catch (error) {
        context.log.error('Error handling subscription update:', error);
        throw error;
    }
}

async function handleSubscriptionDeleted(context, subscription) {
    context.log('Processing subscription deleted:', subscription.id);
    
    try {
        // Find and update the subscription status
        const querySpec = {
            query: 'SELECT * FROM c WHERE c.stripeSubscriptionId = @subscriptionId',
            parameters: [
                { name: '@subscriptionId', value: subscription.id }
            ]
        };
        
        const { resources } = await container.items.query(querySpec).fetchAll();
        
        if (resources.length > 0) {
            const existingSubscription = resources[0];
            existingSubscription.status = 'canceled';
            existingSubscription.updatedAt = new Date();
            
            await container.item(existingSubscription.id, existingSubscription.userId).replace(existingSubscription);
            
            context.log(`Subscription canceled for ${existingSubscription.userId}`);
        }
    } catch (error) {
        context.log.error('Error handling subscription deletion:', error);
        throw error;
    }
}

async function handlePaymentSucceeded(context, invoice) {
    context.log('Processing payment succeeded:', invoice.id);
    // Additional logic for successful payments
}

async function handlePaymentFailed(context, invoice) {
    context.log('Processing payment failed:', invoice.id);
    // Additional logic for failed payments
}

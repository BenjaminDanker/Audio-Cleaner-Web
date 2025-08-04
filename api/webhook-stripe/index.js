const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { CosmosClient } = require('@azure/cosmos');

// Initialize Cosmos DB client
const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
const database = cosmosClient.database('AudioCleanerDB');
const accountsContainer = database.container('accounts');
const transactionsContainer = database.container('transactions');

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
            
            case 'payment_intent.succeeded':
                await handlePaymentSucceeded(context, event.data.object);
                break;
            
            case 'payment_intent.payment_failed':
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
        context.log.error('Error processing webhook:', error.message || 'Unknown error');
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
        const amount = parseInt(session.metadata.amount);
        
        if (session.metadata.type === 'account_topup' && userId && amount > 0) {
            // Update account balance
            await updateAccountBalance(context, userId, amount);
            
            // Create transaction record
            await createTransaction(context, {
                userId: userId,
                type: 'payment',
                amount: amount,
                description: 'Account top-up via Stripe',
                stripePaymentIntentId: session.payment_intent
            });
            
            context.log(`Account balance updated for user ${userId}: +$${amount/100}`);
        }
    } catch (error) {
        context.log.error('Error handling checkout session:', error.message || 'Unknown error');
        throw error;
    }
}

async function handlePaymentSucceeded(context, paymentIntent) {
    context.log('Processing payment succeeded:', paymentIntent.id);
    // Additional logic for successful payments if needed
}

async function handlePaymentFailed(context, paymentIntent) {
    context.log('Processing payment failed:', paymentIntent.id);
    // Additional logic for failed payments if needed
}

async function updateAccountBalance(context, userId, amount) {
    try {
        // Get existing account or create new one
        let account;
        try {
            const { resource } = await accountsContainer.item(userId, userId).read();
            account = resource;
        } catch (error) {
            if (error.code === 404) {
                // Create new account
                account = {
                    id: userId,
                    userId: userId,
                    balance: 0,
                    currency: 'usd',
                    createdAt: new Date().toISOString(),
                    updatedAt: new Date().toISOString()
                };
            } else {
                throw error;
            }
        }
        
        // Update balance
        account.balance = (account.balance || 0) + amount;
        account.updatedAt = new Date().toISOString();
        
        // Save updated account
        if (account.id) {
            await accountsContainer.item(account.id, account.userId).replace(account);
        } else {
            await accountsContainer.items.create(account);
        }
        
        context.log(`Account balance updated: ${userId} -> $${account.balance/100}`);
    } catch (error) {
        context.log.error('Error updating account balance:', error.message);
        throw error;
    }
}

async function createTransaction(context, transactionData) {
    try {
        const transaction = {
            id: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            userId: transactionData.userId,
            type: transactionData.type,
            amount: transactionData.amount,
            description: transactionData.description,
            jobId: transactionData.jobId || null,
            stripePaymentIntentId: transactionData.stripePaymentIntentId || null,
            createdAt: new Date().toISOString()
        };
        
        await transactionsContainer.items.create(transaction);
        context.log(`Transaction created: ${transaction.id}`);
    } catch (error) {
        context.log.error('Error creating transaction:', error.message);
        throw error;
    }
}

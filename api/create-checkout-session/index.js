const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);

module.exports = async function (context, req) {
    context.log('Create checkout session endpoint called');
    
    try {
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || !process.env.STRIPE_SECRET_KEY?.startsWith('sk_live_');
        
        if (isLocalDev) {
            context.log('Local development mode - returning mock checkout URL');
            
            // Validate request body
            const { priceId, mode = 'subscription' } = req.body;
            if (!priceId) {
                context.res = {
                    status: 400,
                    body: { error: 'Price ID is required' }
                };
                return;
            }

            // Return mock checkout session for development
            context.res = {
                status: 200,
                body: {
                    sessionId: 'cs_test_development_session_id',
                    url: 'https://checkout.stripe.com/pay/cs_test_development_session_id',
                    message: 'Development mode - no actual payment will be processed'
                }
            };
            return;
        }

        // Production code - verify authentication
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
        const userEmail = principal.userDetails;

        // Validate request body
        const { priceId, mode = 'subscription' } = req.body;
        if (!priceId) {
            context.res = {
                status: 400,
                body: { error: 'Price ID is required' }
            };
            return;
        }

        // Create Stripe checkout session
        const session = await stripe.checkout.sessions.create({
            customer_email: userEmail,
            payment_method_types: ['card'],
            line_items: [
                {
                    price: priceId,
                    quantity: 1,
                },
            ],
            mode: mode,
            success_url: `${process.env.FRONTEND_URL}/dashboard?session_id={CHECKOUT_SESSION_ID}`,
            cancel_url: `${process.env.FRONTEND_URL}/dashboard`,
            metadata: {
                userId: userId,
                userEmail: userEmail
            },
            subscription_data: mode === 'subscription' ? {
                metadata: {
                    userId: userId
                }
            } : undefined,
        });

        context.res = {
            status: 200,
            body: {
                sessionId: session.id,
                url: session.url
            }
        };

    } catch (error) {
        context.log.error('Error creating checkout session:', error);
        context.res = {
            status: 500,
            body: { 
                error: 'Failed to create checkout session',
                details: process.env.NODE_ENV === 'development' ? error.message : undefined
            }
        };
    }
};

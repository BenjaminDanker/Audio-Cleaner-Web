const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const MinimalLogger = require('../shared/minimalLogger');

module.exports = async function (context, req) {
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        // Handle OPTIONS request for CORS
        if (req.method === 'OPTIONS') {
            context.res = {
                status: 200,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal'
                }
            };
            return;
        }

        // Security check with simple middleware
        const security = new SimpleSecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        const securityResult = await security.checkSecurity(context, req, { requireAuth: true });
        
        if (!securityResult.allowed) {
            context.res = {
                status: securityResult.status,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    ...securityResult.headers
                },
                body: securityResult.body
            };
            return;
        }
        
        const userId = securityResult.userInfo?.userId;
        const userEmail = securityResult.userInfo?.userDetails;

        if (!userId) {
            context.res = {
                status: 401,
                body: { error: 'Unauthorized - No user ID found' }
            };
            return;
        }

        // Validate request body
        const { amount, currency = 'usd' } = req.body;
        if (!amount || amount <= 0) {
            context.res = {
                status: 400,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                body: { error: 'Valid amount is required' }
            };
            return;
        }

        // Create Stripe checkout session
        const session = await stripe.checkout.sessions.create({
            customer_email: userEmail,
            payment_method_types: ['card'],
            line_items: [
                {
                    price_data: {
                        currency: currency,
                        product_data: {
                            name: 'Account Balance Top-up',
                            description: `Add funds to your Audio Cleaner account`
                        },
                        unit_amount: amount
                    },
                    quantity: 1,
                },
            ],
            mode: 'payment',
            success_url: `${process.env.FRONTEND_URL}/dashboard?payment_success=true`,
            cancel_url: `${process.env.FRONTEND_URL}/dashboard?payment_cancelled=true`,
            metadata: {
                userId: userId,
                userEmail: userEmail,
                type: 'account_topup',
                amount: amount.toString()
            }
        });

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                sessionId: session.id,
                url: session.url
            }
        };

    } catch (error) {
        logger.logError('create-payment-session', 'Error creating payment session', 'system', {
            error: error.message || 'Unknown error'
        });
        context.res = {
            status: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: { 
                error: 'Failed to create payment session'
            }
        };
    }
};

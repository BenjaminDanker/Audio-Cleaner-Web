const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const MinimalLogger = require('../shared/minimalLogger');

module.exports = async function (context, req) {
    const logger = new MinimalLogger(context).getLogger();
    
    try {
        // Handle OPTIONS request for CORS preflight
        if (req.method === 'OPTIONS') {
            context.res = {
                status: 200,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal',
                    'Access-Control-Max-Age': '86400'
                },
                body: ''
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
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal',
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
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal'
                },
                body: { error: 'Unauthorized - No user ID found' }
            };
            return;
        }

        // Validate request body - no amount needed, Stripe will handle it
        const { currency = 'usd' } = req.body;
        // Determine frontend origin: use actual request origin first, fallback to configured env var
        const origin = req.headers.origin || process.env.FRONTEND_URL;
        
        logger.logInfo('create-payment-session', 'Creating Stripe session', userId, {
            currency,
            userEmail,
            frontendUrl: origin
        });

        if (!process.env.STRIPE_SECRET_KEY) {
            logger.logError('create-payment-session', 'Missing STRIPE_SECRET_KEY env var');
            context.res = { status: 500, headers: { 'Access-Control-Allow-Origin': '*'}, body: { error: 'Server config error (stripe key missing)' } };
            return;
        }

        // Ensure we have / reuse a Stripe customer so payment methods can be saved
        let customerId;
        try {
            if (userEmail) {
                const existing = await stripe.customers.list({ email: userEmail, limit: 1 });
                if (existing.data.length > 0) {
                    customerId = existing.data[0].id;
                }
            }
            if (!customerId) {
                const created = await stripe.customers.create({
                    email: userEmail || undefined,
                    metadata: { userId }
                });
                customerId = created.id;
            }
        } catch (custErr) {
            logger.logWarn('create-payment-session', 'Customer lookup/create failed – proceeding without persistent customer', userId, { error: custErr.message });
        }

    // Simpler 1:1 setup: use a single price (can be a custom_unit_amount price) with quantity fixed at 1.
    const basePriceId = 'price_1RkGUKRxGAWCymT9ryuzOleo';
    const lineItem = { price: basePriceId, quantity: 1 };

        const session = await stripe.checkout.sessions.create({
            customer: customerId,
            payment_method_types: ['card'],
            mode: 'payment',
            payment_intent_data: { setup_future_usage: 'off_session' },
            line_items: [lineItem],
            success_url: `${origin}/dashboard?payment_success=true`,
            cancel_url: `${origin}/dashboard?payment_cancelled=true`,
            metadata: {
                userId,
                userEmail: userEmail || '',
                type: 'account_topup'
            }
        });

        logger.logInfo('create-payment-session', 'Stripe session created successfully', userId, {
            sessionId: session.id,
            sessionUrl: session.url
        });

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal'
            },
            body: {
                sessionId: session.id,
                url: session.url
            }
        };

    } catch (error) {
        const errorDetails = {
            message: error.message || 'Unknown error',
            code: error.code,
            type: error.type,
            param: error.param,
            statusCode: error.statusCode,
            decline_code: error.decline_code,
            charge: error.charge,
            payment_intent: error.payment_intent,
            payment_method: error.payment_method,
            setup_intent: error.setup_intent,
            source: error.source,
            raw: error.raw,
            stack: error.stack?.substring(0, 500)
        };

        logger.logError('create-payment-session', `Stripe API Error: ${error.message || 'Unknown error'}`, 'system', errorDetails);
        
        context.res = {
            status: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-ms-client-principal'
            },
            body: { 
                error: 'Failed to create payment session',
                details: error.message || 'Unknown error',
                code: error.code || 'unknown_error'
            }
        };
    }
};

const express = require("express");
const crypto = require('crypto');
// Import Azure Functions Express adapter explicitly from dist since package lacks a main field
// and cannot be required by bare specifier in some environments.
const createAzureFunctionHandler = require("@pagopa/express-azure-functions/dist/src/createAzureFunctionsHandler").default;
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { CosmosClient } = require('@azure/cosmos');

// Initialize Cosmos DB client
const cosmosClient = new CosmosClient(process.env.COSMOS_CONNECTION_STRING);
const database = cosmosClient.database('AudioCleanerDB');
const accountsContainer = database.container('accounts');
const transactionsContainer = database.container('transactions');

const app = express();

// Optional diagnostic flag
const DEBUG = process.env.WEBHOOK_DEBUG === 'true';

// Middleware to ensure we always capture an exact raw body buffer (without mutating ordering)
app.use('/api/webhook-stripe', (req, res, next) => {
  if (req._bodyCaptured) return next();
  req._bodyCaptured = true;
  // If adapter already provided rawBody (string) keep it.
  if (req.rawBody && Buffer.isBuffer(req.rawBody)) {
    req._rawBuffer = req.rawBody;
    return next();
  }
  if (req.rawBody && typeof req.rawBody === 'string') {
    req._rawBuffer = Buffer.from(req.rawBody, 'utf8');
    return next();
  }
  // Collect stream data ourselves (only if not already consumed)
  const chunks = [];
  req.on('data', (c) => chunks.push(c));
  req.on('end', () => {
    try {
      req._rawBuffer = Buffer.concat(chunks);
      // Do NOT JSON.parse here to avoid altering original string used for signature
    } catch (e) {
      req._rawBuffer = Buffer.from('');
    }
    next();
  });
});

// Stripe webhook endpoint
app.post(
  "/api/webhook-stripe",
  async (req, res) => {
    console.log('Stripe webhook endpoint called');
    console.log('Method:', req.method);
    console.log('Content-Type:', req.headers['content-type']);
    if (DEBUG) {
      console.log('Body present (req.body keys):', req.body && typeof req.body === 'object' ? Object.keys(req.body) : typeof req.body);
    }
    console.log('Raw buffer length:', req._rawBuffer ? req._rawBuffer.length : 'none');
    
    try {
  const sig = req.headers["stripe-signature"]; 
  const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;

      if (!endpointSecret) {
        // List any STRIPE* env keys to help diagnose missing secret (without values)
        const stripeEnvKeys = Object.keys(process.env).filter(k => k.startsWith('STRIPE'));
        console.warn('STRIPE_WEBHOOK_SECRET env var missing. Available STRIPE-related keys:', stripeEnvKeys);
      }

      console.log('Resolved webhook secret (single)');
      console.log(`Secret present: ${!!endpointSecret}`);
      console.log(`Stripe signature exists: ${!!sig}`);
      
      if (!sig || !endpointSecret) {
        console.error(`Missing signature or webhook secret. Sig: ${!!sig}, Secret: ${!!endpointSecret}`);
        return res.status(400).json({ error: 'Missing stripe signature or webhook secret' });
      }

      // Raw body obtained from middleware or adapter
      const rawBody = req._rawBuffer || (Buffer.isBuffer(req.body) ? req.body : Buffer.from(typeof req.body === 'string' ? req.body : ''));
      if (DEBUG) {
        const preview = rawBody.slice(0, 80).toString();
        const sha256 = crypto.createHash('sha256').update(rawBody).digest('hex');
        console.log('Raw body preview (first 80 chars):', preview);
        console.log('Raw body sha256:', sha256);
      }

      let event;
      try {
  event = stripe.webhooks.constructEvent(
          rawBody,
          sig,
          endpointSecret
        );
        console.log("✅ Signature verified successfully");
        if (DEBUG) {
          try {
            // Recompute signature (primary v1) for diagnostic only
            const timestamp = (sig.split(',').find(p => p.startsWith('t=')) || '').split('=')[1];
            if (timestamp) {
              const signedPayload = `${timestamp}.${rawBody.toString()}`;
              const expected = crypto.createHmac('sha256', endpointSecret).update(signedPayload).digest('hex');
              console.log('Computed expected v1 signature (first 16 hex):', expected.slice(0,16));
            }
          } catch (sigErr) {
            console.warn('Signature recompute diagnostic failed:', sigErr.message);
          }
        }
      } catch (err) {
        console.error("❌ Webhook signature verification failed:", err.message);
        return res.status(400).json({ error: `Webhook Error: ${err.message}` });
      }

      // Handle the event
      console.log(`Event type: ${event.type}`);
      console.log(`Event ID: ${event.id}`);
      
      switch (event.type) {
        case 'checkout.session.completed':
          console.log('Processing checkout.session.completed event');
          await handleCheckoutSessionCompleted(event.data.object, event.id);
          break;
        case 'payment_intent.succeeded':
          console.log('Processing payment_intent.succeeded event');
          await handlePaymentSucceeded(event.data.object);
          break;
        case 'payment_intent.payment_failed':
          await handlePaymentFailed(event.data.object);
          break;
        default:
          console.log(`Unhandled event type: ${event.type}`);
      }

      res.json({ received: true });

    } catch (error) {
      console.error('Error processing webhook:', error.message || 'Unknown error');
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

module.exports = createAzureFunctionHandler(app);

async function handleCheckoutSessionCompleted(session, eventId) {
  console.log('Processing checkout session completed:', session.id);
  console.log('Session metadata:', JSON.stringify(session.metadata, null, 2));
  console.log('Session amount_total:', session.amount_total);
  console.log('Session payment_status:', session.payment_status);

  try {
    if (!session || !session.metadata) {
      console.warn('Missing session metadata – skipping');
      return;
    }
    if (session.payment_status && session.payment_status !== 'paid') {
      console.log(`Session payment_status is ${session.payment_status}, not crediting yet.`);
      return;
    }
    const userId = session.metadata.userId;
    const userEmail = session.metadata.userEmail;
    const actualAmount = Number(session.amount_total) || 0; // integer (cents)
    const paymentIntentId = session.payment_intent;

    console.log(`User ID: ${userId}, Email: ${userEmail}, Amount (cents): ${actualAmount}`);

    if (!(session.metadata.type === 'account_topup' && userId && actualAmount > 0 && paymentIntentId)) {
      console.log('Conditions not met for crediting account:', {
        type: session.metadata.type,
        hasUserId: !!userId,
        actualAmount,
        paymentIntentIdPresent: !!paymentIntentId
      });
      return;
    }

    // Idempotency check: look for existing transaction with same stripePaymentIntentId
    const querySpec = {
      query: 'SELECT TOP 1 * FROM c WHERE c.userId = @userId AND c.stripePaymentIntentId = @pid',
      parameters: [
        { name: '@userId', value: userId },
        { name: '@pid', value: paymentIntentId }
      ]
    };
    const { resources: existing } = await transactionsContainer.items.query(querySpec, { partitionKey: userId }).fetchAll();
    if (existing && existing.length > 0) {
      console.log(`Duplicate event/payment intent detected (paymentIntentId=${paymentIntentId}) – skipping credit.`);
      return;
    }

    console.log('Idempotency OK, updating account balance...');
    await updateAccountBalance(userId, actualAmount);

    await createTransaction({
      userId,
      type: 'payment',
      amount: actualAmount,
      description: 'Account top-up via Stripe',
      stripePaymentIntentId: paymentIntentId,
      eventId
    });
    console.log(`Account balance updated for user ${userId}: +$${actualAmount/100}`);
  } catch (error) {
    console.error('Error handling checkout session:', error.message || 'Unknown error');
    throw error;
  }
}

async function handlePaymentSucceeded(paymentIntent) {
    console.log('Processing payment succeeded:', paymentIntent.id);
    // Additional logic for successful payments if needed
}

async function handlePaymentFailed(paymentIntent) {
    console.log('Processing payment failed:', paymentIntent.id);
    // Additional logic for failed payments if needed
}

async function updateAccountBalance(userId, amount) {
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
        
        console.log(`Account balance updated: ${userId} -> $${account.balance/100}`);
    } catch (error) {
        console.error('Error updating account balance:', error.message);
        throw error;
    }
}

async function createTransaction(transactionData) {
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
        console.log(`Transaction created: ${transaction.id}`);
    } catch (error) {
        console.error('Error creating transaction:', error.message);
        throw error;
    }
}

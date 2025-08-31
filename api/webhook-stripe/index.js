const express = require("express");
const crypto = require('crypto');
// Import Azure Functions Express adapter explicitly from dist since package lacks a main field
// and cannot be required by bare specifier in some environments.
const createAzureFunctionHandler = require("@pagopa/express-azure-functions/dist/src/createAzureFunctionsHandler").default;
const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const { CosmosClient } = require('@azure/cosmos');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');

// Lazy Cosmos initialization to avoid crashing local dev when not configured
function getCosmosContainers(logger) {
  const cs = process.env.COSMOS_CONNECTION_STRING;
  if (!cs) {
    // Use console as a fallback if logger not ready
    try { logger?.logWarning('webhook-stripe', 'COSMOS_CONNECTION_STRING not set; skipping persistence for webhook', 'system', {}); } catch {}
    return null;
  }
  try {
    const client = AzureSDKConfig.createCosmosClient(cs);
    const database = client.database(process.env.COSMOS_DB_NAME || 'app');
    return {
      accounts: database.container('accounts'),
      transactions: database.container('transactions'),
    };
  } catch (e) {
    try { logger?.logError('webhook-stripe', 'Failed to init Cosmos client', 'system', { error: e.message }); } catch {}
    return null;
  }
}

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
    // Initialize retry-aware minimal logger
    const logger = new MinimalLogger({ log: console }).getLogger();
    
    // Generate session ID for request tracking
    const sessionId = `webhook_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.logInfo('webhook-stripe', 'Stripe webhook endpoint called', 'system', {
      sessionId,
      method: req.method,
      contentType: req.headers['content-type'],
      rawBufferLength: req._rawBuffer ? req._rawBuffer.length : 0,
      hasBody: !!req.body
    });
    
    try {
      const sig = req.headers["stripe-signature"]; 
      const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;

      if (!endpointSecret) {
        // List any STRIPE* env keys to help diagnose missing secret (without values)
        const stripeEnvKeys = Object.keys(process.env).filter(k => k.startsWith('STRIPE'));
        logger.logError('webhook-stripe', 'STRIPE_WEBHOOK_SECRET env var missing', 'system', {
          sessionId,
          availableStripeKeys: stripeEnvKeys
        });
      }

      logger.logDebug('webhook-stripe', 'Webhook authentication check', 'system', {
        sessionId,
        secretPresent: !!endpointSecret,
        signaturePresent: !!sig
      });
      
      if (!sig || !endpointSecret) {
        logger.logError('webhook-stripe', 'Missing signature or webhook secret', 'system', {
          sessionId,
          hasSignature: !!sig,
          hasSecret: !!endpointSecret
        });
        return res.status(400).json({ error: 'Missing stripe signature or webhook secret' });
      }

      // Raw body obtained from middleware or adapter
      const rawBody = req._rawBuffer || (Buffer.isBuffer(req.body) ? req.body : Buffer.from(typeof req.body === 'string' ? req.body : ''));
      
      logger.logDebug('webhook-stripe', 'Raw body processing', 'system', {
        sessionId,
        rawBodyLength: rawBody.length,
        bodyType: typeof req.body,
        hasRawBuffer: !!req._rawBuffer
      });

      let event;
      try {
        event = stripe.webhooks.constructEvent(rawBody, sig, endpointSecret);
        
        logger.logInfo('webhook-stripe', 'Signature verified successfully', 'system', {
          sessionId,
          eventType: event.type,
          eventId: event.id
        });
        
      } catch (err) {
        logger.logError('webhook-stripe', 'Webhook signature verification failed', 'system', {
          sessionId,
          error: err.message,
          hasRawBody: !!rawBody,
          rawBodyLength: rawBody.length
        });
        return res.status(400).json({ error: `Webhook Error: ${err.message}` });
      }

      // Handle the event
      logger.logInfo('webhook-stripe', 'Processing webhook event', 'system', {
        sessionId,
        eventType: event.type,
        eventId: event.id
      });
      
      switch (event.type) {
        case 'checkout.session.completed':
          logger.logInfo('webhook-stripe', 'Processing checkout.session.completed event', 'system', {
            sessionId,
            checkoutSessionId: event.data.object.id
          });
          await handleCheckoutSessionCompleted(event.data.object, event.id, logger, sessionId);
          break;
        case 'payment_intent.succeeded':
          logger.logInfo('webhook-stripe', 'Processing payment_intent.succeeded event', 'system', {
            sessionId,
            paymentIntentId: event.data.object.id
          });
          await handlePaymentSucceeded(event.data.object, logger, sessionId);
          break;
        case 'payment_intent.payment_failed':
          logger.logInfo('webhook-stripe', 'Processing payment_intent.payment_failed event', 'system', {
            sessionId,
            paymentIntentId: event.data.object.id
          });
          await handlePaymentFailed(event.data.object, logger, sessionId);
          break;
        default:
          logger.logWarning('webhook-stripe', 'Unhandled event type', 'system', {
            sessionId,
            eventType: event.type,
            eventId: event.id
          });
      }

      res.json({ received: true });

    } catch (error) {
      logger.logError('webhook-stripe', 'Error processing webhook', 'system', {
        sessionId,
        error: error.message || 'Unknown error',
        stack: error.stack
      });
      res.status(500).json({ error: 'Internal server error' });
    }
  }
);

module.exports = createAzureFunctionHandler(app);

async function handleCheckoutSessionCompleted(session, eventId, logger, sessionId) {
  logger.logInfo('webhook-stripe', 'Processing checkout session completed', 'system', {
    sessionId,
    checkoutSessionId: session.id,
    amountTotal: session.amount_total,
    paymentStatus: session.payment_status,
    hasMetadata: !!session.metadata
  });

  try {
    const containers = getCosmosContainers(logger);
    if (!containers) {
      logger.logWarning('webhook-stripe', 'Cosmos not configured; skipping account credit persistence', 'system', { sessionId });
      return;
    }
    if (!session || !session.metadata) {
      logger.logWarning('webhook-stripe', 'Missing session metadata - skipping', 'system', {
        sessionId,
        checkoutSessionId: session.id
      });
      return;
    }
    
    if (session.payment_status && session.payment_status !== 'paid') {
      logger.logInfo('webhook-stripe', 'Session payment not completed yet', 'system', {
        sessionId,
        checkoutSessionId: session.id,
        paymentStatus: session.payment_status
      });
      return;
    }
    
    const userId = session.metadata.userId;
    const userEmail = session.metadata.userEmail;
    const actualAmount = Number(session.amount_total) || 0; // integer (cents)
    const paymentIntentId = session.payment_intent;

    logger.logInfo('webhook-stripe', 'Processing payment for user account topup', 'system', {
      sessionId,
      userId: userId?.substring(0, 8) + '...',
      userEmail,
      amountCents: actualAmount,
      hasPaymentIntentId: !!paymentIntentId,
      metadataType: session.metadata.type
    });

    if (!(session.metadata.type === 'account_topup' && userId && actualAmount > 0 && paymentIntentId)) {
      logger.logWarning('webhook-stripe', 'Conditions not met for crediting account', 'system', {
        sessionId,
        type: session.metadata.type,
        hasUserId: !!userId,
        actualAmount,
        hasPaymentIntentId: !!paymentIntentId
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
    
  const { resources: existing } = await containers.transactions.items.query(querySpec, { partitionKey: userId }).fetchAll();
    if (existing && existing.length > 0) {
      logger.logWarning('webhook-stripe', 'Duplicate payment intent detected - skipping credit', 'system', {
        sessionId,
        userId: userId.substring(0, 8) + '...',
        paymentIntentId,
        existingTransactionId: existing[0].id
      });
      return;
    }

    logger.logInfo('webhook-stripe', 'Idempotency check passed - updating account balance', 'system', {
      sessionId,
      userId: userId.substring(0, 8) + '...',
      amountCents: actualAmount
    });
    
  await updateAccountBalance(userId, actualAmount, logger, sessionId);

  await createTransaction({
      userId,
      type: 'payment',
      amount: actualAmount,
      description: 'Account top-up via Stripe',
      stripePaymentIntentId: paymentIntentId,
      eventId
    }, logger, sessionId);
    
    logger.logInfo('webhook-stripe', 'Account balance updated successfully', 'system', {
      sessionId,
      userId: userId.substring(0, 8) + '...',
      amountCents: actualAmount,
      amountDollars: (actualAmount/100).toFixed(2)
    });
    
  } catch (error) {
    logger.logError('webhook-stripe', 'Error handling checkout session', 'system', {
      sessionId,
      userId: userId?.substring(0, 8) + '...',
      error: error.message || 'Unknown error',
      stack: error.stack
    });
    throw error;
  }
}

async function handlePaymentSucceeded(paymentIntent, logger, sessionId) {
    logger.logInfo('webhook-stripe', 'Processing payment succeeded', 'system', {
      sessionId,
      paymentIntentId: paymentIntent.id,
      amount: paymentIntent.amount
    });
    // Additional logic for successful payments if needed
}

async function handlePaymentFailed(paymentIntent, logger, sessionId) {
    logger.logError('webhook-stripe', 'Processing payment failed', 'system', {
      sessionId,
      paymentIntentId: paymentIntent.id,
      amount: paymentIntent.amount,
      lastPaymentError: paymentIntent.last_payment_error
    });
    // Additional logic for failed payments if needed
}

async function updateAccountBalance(userId, amount, logger, sessionId) {
    try {
        const containers = getCosmosContainers(logger);
        if (!containers) {
          logger.logWarning('webhook-stripe', 'Cosmos not configured; skip updateAccountBalance', 'system', { sessionId });
          return;
        }
        // Get existing account or create new one
        let account;
        try {
            const { resource } = await containers.accounts.item(userId, userId).read();
            account = resource;
            
            logger.logDebug('webhook-stripe', 'Found existing account', 'system', {
              sessionId,
              userId: userId.substring(0, 8) + '...',
              currentBalance: account.balance
            });
            
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
            await containers.accounts.item(account.id, account.userId).replace(account);
        } else {
            await containers.accounts.items.create(account);
        }
        
        logger.logInfo('webhook-stripe', 'Account balance updated', 'system', {
          sessionId,
          userId: userId.substring(0, 8) + '...',
          amountDollars: (account.balance/100).toFixed(2)
        });
    } catch (error) {
        logger.logError('webhook-stripe', 'Error updating account balance', 'system', {
          sessionId,
          error: error.message
        });
        throw error;
    }
}

async function createTransaction(transactionData, logger, sessionId) {
    try {
        const containers = getCosmosContainers(logger);
        if (!containers) {
          logger.logWarning('webhook-stripe', 'Cosmos not configured; skip createTransaction', 'system', { sessionId });
          return;
        }
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
        
        await containers.transactions.items.create(transaction);
        logger.logInfo('webhook-stripe', 'Transaction created', 'system', {
          sessionId,
          transactionId: transaction.id,
          userId: (transactionData.userId || '').substring(0, 8) + '...'
        });
    } catch (error) {
        logger.logError('webhook-stripe', 'Error creating transaction', 'system', {
          sessionId,
          error: error.message
        });
        throw error;
    }
}

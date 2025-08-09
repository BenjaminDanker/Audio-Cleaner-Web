const { CosmosClient } = require('@azure/cosmos');
const SimpleSecurityMiddleware = require('../shared/simpleSecurityMiddleware');
const MinimalLogger = require('../shared/minimalLogger');
const AzureSDKConfig = require('../shared/azureSDKConfig');

module.exports = async function (context, req) {
    const logger = new MinimalLogger(context).getLogger();
    
    try {
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

        if (!userId) {
            context.res = {
                status: 401,
                body: { error: 'Unauthorized - No user ID found' }
            };
            return;
        }

        // Initialize optimized Cosmos client
        const client = AzureSDKConfig.createCosmosClient(process.env.COSMOS_CONNECTION_STRING);
        const database = client.database('AudioCleanerDB');
        const accountsContainer = database.container('accounts');
        const transactionsContainer = database.container('transactions');

        // Get account and transactions in parallel
        const [accountResult, transactionsResult] = await Promise.allSettled([
            accountsContainer.item(userId, userId).read(),
            transactionsContainer.items.query({
                query: 'SELECT * FROM c WHERE c.userId = @userId ORDER BY c.createdAt DESC',
                parameters: [{ name: '@userId', value: userId }]
            }).fetchAll()
        ]);

        // Handle account
        let account;
        if (accountResult.status === 'fulfilled' && accountResult.value.resource) {
            account = accountResult.value.resource;
        } else {
            // Create new account if doesn't exist
            account = {
                id: userId,
                userId: userId,
                balance: 0,
                currency: 'usd',
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            };
            await accountsContainer.items.create(account);
        }

        // Handle transactions
        const transactions = transactionsResult.status === 'fulfilled' ? 
            transactionsResult.value.resources.map(t => {
                let amt = t.amount;
                // Normalize legacy processing/refund amounts stored in USD (<1) to cents
                if ((t.type === 'processing' || t.type === 'refund') && amt > 0 && amt < 1) {
                    amt = Math.round(amt * 100);
                }
                return {
                    id: t.id,
                    type: t.type,
                    amount: amt,
                    description: t.description,
                    jobId: t.jobId,
                    createdAt: t.createdAt
                };
            }) : [];

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: {
                account: {
                    id: account.id,
                    userId: account.userId,
                    balance: account.balance,
                    currency: account.currency,
                    createdAt: account.createdAt,
                    updatedAt: account.updatedAt
                },
                transactions: transactions
            }
        };

    } catch (error) {
        logger.logError('get-account-data', 'Error getting account data', 'system', {
            error: error.message || 'Unknown error'
        });
        context.res = {
            status: 500,
            headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            body: { error: 'Internal server error' }
        };
    }
};

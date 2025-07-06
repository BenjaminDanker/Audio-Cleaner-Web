const SASTokenManager = require('../shared/sasTokenManager');

module.exports = async function (context, req) {
    context.log('Revoke SAS tokens function called');
    
    try {
        // Check if user is authenticated
        const clientPrincipal = req.headers['x-ms-client-principal'];
        if (!clientPrincipal) {
            context.res = {
                status: 401,
                body: { success: false, error: 'Unauthorized' }
            };
            return;
        }

        const user = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
        const userId = user.userId;

        // Initialize SAS Token Manager
        const sasManager = new SASTokenManager(
            process.env.AzureWebJobsStorage,
            process.env.COSMOS_CONNECTION_STRING
        );

        // Rule #5: Revoke all SAS tokens for this user
        const success = await sasManager.revokeSASTokensForUser(userId, context);

        context.res = {
            status: 200,
            headers: {
                'Content-Type': 'application/json'
            },
            body: {
                success: success,
                message: success 
                    ? 'All SAS tokens have been revoked successfully'
                    : 'Failed to revoke some SAS tokens'
            }
        };

    } catch (error) {
        context.log.error('Error revoking SAS tokens:', error);
        context.res = {
            status: 500,
            body: { 
                success: false, 
                error: 'Internal server error: ' + error.message 
            }
        };
    }
};

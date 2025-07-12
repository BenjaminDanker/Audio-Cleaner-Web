module.exports = async function (context, req) {
    context.log('Auth endpoint called');
    
    try {
        // In Azure Static Web Apps, authentication is handled by the platform
        // This endpoint returns the current user's authentication status
        const clientPrincipal = req.headers['x-ms-client-principal'];
        
        if (clientPrincipal) {
            // User is authenticated - decode the principal
            const principal = JSON.parse(Buffer.from(clientPrincipal, 'base64').toString());
            
            context.res = {
                status: 200,
                body: {
                    authenticated: true,
                    user: {
                        id: principal.userId,
                        email: principal.userDetails,
                        name: principal.userDetails,
                        provider: principal.identityProvider
                    }
                }
            };
        } else {
            // User is not authenticated
            context.res = {
                status: 200,
                body: {
                    authenticated: false,
                    loginUrl: '/.auth/login/aad' // Azure AD login
                }
            };
        }
    } catch (error) {
        context.log.error('Auth endpoint error:', error.message || 'Unknown error');
        context.res = {
            status: 500,
            body: {
                authenticated: false,
                error: 'Internal server error'
            }
        };
    }
};

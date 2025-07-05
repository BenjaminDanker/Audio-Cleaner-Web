module.exports = async function (context, req) {
    context.log('Auth endpoint called');
    
    if (req.method === 'GET') {
        // Handle authentication check
        const authHeader = req.headers.authorization;
        
        if (authHeader && authHeader.startsWith('Bearer ')) {
            // For development, we'll just return a success response
            // In production, you'd validate the token here
            context.res = {
                status: 200,
                body: {
                    authenticated: true,
                    user: {
                        id: 'local-dev-user-123',
                        email: 'developer@localhost.local',
                        name: 'Developer User'
                    }
                }
            };
        } else {
            context.res = {
                status: 401,
                body: {
                    authenticated: false,
                    error: 'No valid token provided'
                }
            };
        }
    } else if (req.method === 'POST') {
        // Handle login
        const { email, password } = req.body;
        
        // For development, accept any email/password combination
        // In production, you'd validate credentials here
        if (email && password) {
            context.res = {
                status: 200,
                body: {
                    success: true,
                    token: 'dev-token-' + Date.now(),
                    user: {
                        id: 'local-dev-user-123',
                        email: email,
                        name: 'Developer User'
                    }
                }
            };
        } else {
            context.res = {
                status: 400,
                body: {
                    success: false,
                    error: 'Email and password are required'
                }
            };
        }
    } else {
        context.res = {
            status: 405,
            body: {
                error: 'Method not allowed'
            }
        };
    }
};

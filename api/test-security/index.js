const SecurityMiddleware = require('../shared/securityMiddleware');

module.exports = async function (context, req) {
    context.log('Simple test function started');
    
    try {
        // Test SecurityMiddleware initialization
        const security = new SecurityMiddleware(process.env.COSMOS_CONNECTION_STRING);
        
        // Try to initialize
        await security.initialize();
        context.log('SecurityMiddleware initialized successfully');
        
        // Simple security check without complex validation
        const securityResult = await security.checkSecurity(context, req, {
            requireAuth: false,
            validateInput: false
        });
        
        context.log('Security check completed:', securityResult.allowed);
        
        context.res = {
            status: 200,
            headers: security.getSecurityHeaders('/api/test-security'),
            body: { 
                success: true, 
                securityCheck: securityResult.allowed,
                message: 'Security middleware test successful'
            }
        };
        
    } catch (error) {
        context.log.error('Test security error:', error.message);
        context.res = {
            status: 500,
            body: { 
                success: false, 
                error: error.message,
                stack: error.stack
            }
        };
    }
};

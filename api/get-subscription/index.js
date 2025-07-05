module.exports = async function (context, req) {
    context.log('Simple test function called');
    
    context.res = {
        status: 200,
        body: {
            message: 'Hello from Azure Functions!',
            timestamp: new Date().toISOString(),
            method: req.method,
            url: req.url
        }
    };
};

module.exports = async function (context, req) {
    context.log('API root endpoint accessed');

    const apiInfo = {
        name: "Audio Cleaner API",
        version: "1.0.0",
        status: "running",
        description: "API for audio cleaning and processing services",
        endpoints: {
            auth: "/api/auth - User authentication",
            subscription: "/api/get-subscription - Get user subscription status",
            checkout: "/api/create-checkout-session - Create payment session",
            upload: "/api/enqueue-job - Upload and process audio/video",
            status: "/api/job-status - Check processing status",
            webhook: "/api/webhook-stripe - Stripe webhook handler"
        },
        message: "Welcome to the Audio Cleaner API. Please authenticate to access services."
    };

    context.res = {
        status: 200,
        headers: {
            "Content-Type": "application/json",
            "X-API-Info": "Audio Cleaner API v1.0"
        },
        body: apiInfo
    };
};

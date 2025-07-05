const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 7071;

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// Middleware to log requests
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
    next();
});

// Mock subscription endpoint
app.get('/api/get-subscription', (req, res) => {
    console.log('📋 Get subscription called');
    res.json({
        id: 'dev-subscription',
        status: 'active',
        planName: 'Free Tier',
        plan: 'free',
        tier: 'free',
        usageLimit: 5,
        currentUsage: 0,
        nextBillingDate: null,
        price: '$0.00'
    });
});

// Mock job enqueue endpoint
app.post('/api/enqueue-job', (req, res) => {
    console.log('🚀 Enqueue job called with:', req.body);
    const { fileName, fileUrl, processingType = 'denoise' } = req.body;
    
    if (!fileName || !fileUrl) {
        return res.status(400).json({ error: 'fileName and fileUrl are required' });
    }

    const jobId = 'dev-job-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    
    res.json({
        success: true,
        jobId: jobId,
        message: 'Job queued successfully (development mode)',
        estimatedProcessingTime: '2-5 minutes'
    });
});

// Mock job status endpoint
app.get('/api/job-status', (req, res) => {
    console.log('📊 Job status called for:', req.query.jobId);
    const { jobId } = req.query;
    
    if (!jobId) {
        return res.status(400).json({ error: 'jobId query parameter is required' });
    }

    // Simulate different job statuses
    const statuses = ['queued', 'processing', 'completed', 'failed'];
    const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];
    
    res.json({
        id: jobId,
        status: randomStatus,
        fileName: 'test-video.mp4',
        processingType: 'denoise',
        progress: randomStatus === 'processing' ? Math.floor(Math.random() * 100) : 100,
        createdAt: new Date(Date.now() - 300000).toISOString(), // 5 minutes ago
        updatedAt: new Date().toISOString(),
        downloadUrl: randomStatus === 'completed' ? `http://localhost:${PORT}/api/download/${jobId}` : null
    });
});

// Mock create checkout session endpoint
app.post('/api/create-checkout-session', (req, res) => {
    console.log('💳 Create checkout session called with:', req.body);
    const { priceId, mode = 'subscription' } = req.body;
    
    if (!priceId) {
        return res.status(400).json({ error: 'Price ID is required' });
    }

    res.json({
        sessionId: 'cs_test_development_session_id',
        url: 'https://checkout.stripe.com/pay/cs_test_development_session_id',
        message: 'Development mode - no actual payment will be processed'
    });
});

// Mock auth endpoint
app.get('/api/auth', (req, res) => {
    console.log('🔐 Auth check called');
    res.json({
        authenticated: true,
        user: {
            id: 'local-dev-user-123',
            email: 'developer@localhost.local',
            name: 'Developer User'
        }
    });
});

app.post('/api/auth', (req, res) => {
    console.log('🔐 Auth login called');
    res.json({
        success: true,
        message: 'Login successful (development mode)'
    });
});

// Mock webhook endpoint
app.post('/api/webhook-stripe', (req, res) => {
    console.log('🪝 Stripe webhook called');
    res.json({ received: true });
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        message: 'Local development API server is running',
        timestamp: new Date().toISOString()
    });
});

// Default route
app.get('/', (req, res) => {
    res.json({ 
        message: 'Audio Cleaner Pro - Local Development API Server',
        version: '1.0.0',
        endpoints: [
            'GET /api/get-subscription',
            'POST /api/enqueue-job',
            'GET /api/job-status',
            'POST /api/create-checkout-session',
            'GET /api/auth',
            'POST /api/auth',
            'POST /api/webhook-stripe',
            'GET /api/health'
        ]
    });
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error('❌ Error:', err);
    res.status(500).json({ error: 'Internal server error', message: err.message });
});

app.listen(PORT, () => {
    console.log(`🚀 Local development API server running on http://localhost:${PORT}`);
    console.log(`📊 API endpoints available at http://localhost:${PORT}/api/*`);
    console.log(`💡 This is a mock server for local development only`);
});

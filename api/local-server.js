const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');

const app = express();
const PORT = 7071;

// Enable CORS for all origins in development
app.use(cors());

// Parse JSON bodies
app.use(express.json());

// Parse URL-encoded bodies
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, path.join(__dirname, 'temp', 'uploads'));
    },
    filename: function (req, file, cb) {
        // Preserve file extension
        const fileExtension = path.extname(file.originalname);
        const fileName = Date.now() + '_' + Math.random().toString(36).substring(7) + fileExtension;
        cb(null, fileName);
    }
});

const upload = multer({
    storage: storage,
    limits: {
        fileSize: 1000 * 1024 * 1024 // 1000MB limit
    }
});

// Mock context object for Azure Functions
function createContext(req, res) {
    return {
        log: console.log,
        res: {},
        req: req,
        done: (err, response) => {
            if (err) {
                console.error('Function error:', err);
                res.status(500).json({ error: err.message });
            } else if (context.res.status || context.res.body) {
                res.status(context.res.status || 200).json(context.res.body);
            } else {
                res.status(200).json(response);
            }
        }
    };
}

// Load and wrap Azure Functions
function loadFunction(functionPath) {
    try {
        const functionHandler = require(functionPath);
        return async (req, res) => {
            const context = createContext(req, res);
            
            try {
                await functionHandler(context, req);
                
                // Handle response
                if (context.res.status || context.res.body) {
                    res.status(context.res.status || 200);
                    if (context.res.body) {
                        res.json(context.res.body);
                    } else {
                        res.end();
                    }
                } else {
                    res.status(200).json({ message: 'Function executed successfully' });
                }
            } catch (error) {
                console.error(`Error in function ${functionPath}:`, error);
                res.status(500).json({ error: error.message });
            }
        };
    } catch (error) {
        console.error(`Failed to load function ${functionPath}:`, error);
        return (req, res) => {
            res.status(500).json({ error: `Function not found: ${functionPath}` });
        };
    }
}

// Register API routes
app.get('/api/auth', loadFunction('./auth/index.js'));
app.post('/api/create-checkout-session', loadFunction('./create-checkout-session/index.js'));
app.get('/api/download-file/:filename', async (req, res) => {
    // Handle file download directly in Express for better streaming
    try {
        const filename = req.params.filename;
        
        if (!filename) {
            return res.status(400).json({ error: 'Filename parameter is required' });
        }
        
        // Construct the file path
        const downloadsDir = path.join(__dirname, 'temp', 'downloads');
        const filePath = path.join(downloadsDir, filename);
        
        // Check if file exists
        const fs = require('fs');
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ error: 'File not found' });
        }
        
        // Get file stats
        const stats = fs.statSync(filePath);
        const fileSize = stats.size;
        
        // Set appropriate headers
        res.setHeader('Content-Type', 'video/mp4');
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.setHeader('Content-Length', fileSize.toString());
        
        // Stream the file
        const fileStream = fs.createReadStream(filePath);
        fileStream.pipe(res);
        
    } catch (error) {
        console.error('Error downloading file:', error);
        res.status(500).json({ error: 'Internal server error: ' + error.message });
    }
});
app.post('/api/enqueue-job', loadFunction('./enqueue-job/index.js'));
app.get('/api/get-subscription', loadFunction('./get-subscription/index.js'));
app.get('/api', loadFunction('./index/index.js'));
app.get('/api/job-status', loadFunction('./job-status/index.js'));
app.post('/api/upload-file', upload.single('file'), async (req, res) => {
    // Handle file upload specially for Express
    if (!req.file) {
        return res.status(400).json({ error: 'No file uploaded' });
    }
    
    // Mock Azure Functions context and request for upload handler
    const context = {
        log: console.log,
        res: {},
        req: {
            headers: {
                'x-file-name': req.file.originalname,
                'content-length': req.file.size
            },
            file: req.file
        }
    };
    
    // Call the upload function
    try {
        const uploadFunction = require('./upload-file/index.js');
        await uploadFunction(context, context.req);
        
        // Send the response
        if (context.res.status && context.res.body) {
            res.status(context.res.status).json(context.res.body);
        } else {
            res.status(200).json({ message: 'Upload completed' });
        }
    } catch (error) {
        console.error('Upload function error:', error);
        res.status(500).json({ error: error.message });
    }
});
app.post('/api/webhook-stripe', loadFunction('./webhook-stripe/index.js'));

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 API Server running on http://localhost:${PORT}`);
    console.log(` Available endpoints:`);
    console.log(`   GET  /api/auth`);
    console.log(`   POST /api/create-checkout-session`);
    console.log(`   GET  /api/download-file/:filename`);
    console.log(`   POST /api/enqueue-job`);
    console.log(`   GET  /api/get-subscription`);
    console.log(`   GET  /api`);
    console.log(`   GET  /api/job-status`);
    console.log(`   POST /api/upload-file`);
    console.log(`   POST /api/webhook-stripe`);
});

module.exports = app;

// Test the get-subscription function directly
const fs = require('fs');
const path = require('path');

// Mock context object
const mockContext = {
    log: console.log,
    res: {}
};

// Mock request object
const mockReq = {
    method: 'GET',
    url: '/api/get-subscription'
};

// Load and test the function
try {
    const functionPath = path.join(__dirname, 'api', 'get-subscription', 'index.js');
    console.log('Testing function at:', functionPath);
    
    if (fs.existsSync(functionPath)) {
        const functionModule = require(functionPath);
        console.log('Function loaded successfully');
        
        // Call the function
        functionModule(mockContext, mockReq).then(() => {
            console.log('Function executed successfully');
            console.log('Response:', mockContext.res);
        }).catch(err => {
            console.error('Function execution error:', err);
        });
    } else {
        console.error('Function file not found');
    }
} catch (error) {
    console.error('Error loading function:', error);
}

const fs = require('fs');
const path = require('path');

module.exports = async function (context, req) {
    context.log('Download endpoint called');
    
    try {
        // Get the filename from the URL parameter
        const filename = req.params.filename || req.query.filename;
        
        if (!filename) {
            context.res = {
                status: 400,
                body: { error: 'Filename parameter is required' }
            };
            return;
        }
        
        // Check if we're in local development
        const clientPrincipal = req.headers['x-ms-client-principal'];
        const isLocalDev = !clientPrincipal || process.env.AZURE_COSMOS_CONNECTION_STRING?.includes('localhost');
        
        if (isLocalDev) {
            context.log('Local development mode - serving file locally');
            
            // Construct the file path
            const downloadsDir = path.join(process.cwd(), 'temp', 'downloads');
            const filePath = path.join(downloadsDir, filename);
            
            // Check if file exists
            if (!fs.existsSync(filePath)) {
                context.res = {
                    status: 404,
                    body: { error: 'File not found' }
                };
                return;
            }
            
            // Get file stats
            const stats = fs.statSync(filePath);
            const fileSize = stats.size;
            
            // Read the file
            const fileBuffer = fs.readFileSync(filePath);
            
            context.res = {
                status: 200,
                headers: {
                    'Content-Type': 'video/mp4',
                    'Content-Disposition': `attachment; filename="${filename}"`,
                    'Content-Length': fileSize.toString()
                },
                body: fileBuffer,
                isRaw: true
            };
            return;
        }
        
        // Production code would download from Azure Blob Storage
        context.res = {
            status: 501,
            body: { error: 'Production download not implemented yet' }
        };
        
    } catch (error) {
        context.log.error('Error downloading file:', error);
        context.res = {
            status: 500,
            body: { error: 'Internal server error: ' + error.message }
        };
    }
};

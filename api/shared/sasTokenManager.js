const { BlobServiceClient, generateBlobSASQueryParameters, BlobSASPermissions, StorageSharedKeyCredential } = require('@azure/storage-blob');

class SASTokenManager {
    constructor(storageConnectionString, context = null) {
        // Extract account name and key from connection string
        const accountNameMatch = storageConnectionString.match(/AccountName=([^;]+)/);
        const accountKeyMatch = storageConnectionString.match(/AccountKey=([^;]+)/);
        
        if (!accountNameMatch || !accountKeyMatch) {
            throw new Error('Invalid storage connection string format - missing AccountName or AccountKey');
        }
        
        this.accountName = accountNameMatch[1];
        this.accountKey = accountKeyMatch[1];
        
        if (!this.accountName || !this.accountKey) {
            throw new Error('Empty AccountName or AccountKey in connection string');
        }

        // Initialize with account key credential
        this.credential = new StorageSharedKeyCredential(this.accountName, this.accountKey);
        this.blobServiceClient = new BlobServiceClient(
            `https://${this.accountName}.blob.core.windows.net`,
            this.credential
        );
    }

    async generateSASToken(options) {
        const {
            containerName,
            blobName,
            permissions = 'r',
            expiryMinutes = 5,
            clientIP = null,
            context = null
        } = options;

        if (context) {
            context.log('Generating account key-based SAS token...');
            context.log('Storage account:', this.accountName);
            context.log('Container:', containerName);
            context.log('Blob:', blobName);
        }
        
        try {
            // Generate account key-based SAS
            const sasOptions = {
                containerName,
                blobName,
                permissions: BlobSASPermissions.parse(permissions),
                startsOn: new Date(),
                expiresOn: new Date(Date.now() + expiryMinutes * 60 * 1000),
                ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
            };

            const sasToken = generateBlobSASQueryParameters(sasOptions, this.credential).toString();
            
            if (!sasToken) {
                throw new Error('Generated SAS token is empty');
            }
            
            if (context) {
                context.log('Successfully generated account key-based SAS token');
            }

            return {
                sasToken,
                sasType: 'AccountKey',
                expiresAt: new Date(Date.now() + expiryMinutes * 60 * 1000)
            };
            
        } catch (error) {
            if (context) {
                context.log.error('SAS token generation failed:', error.message);
            }
            throw new Error(`Failed to generate SAS token: ${error.message}`);
        }
    }

    // Get client IP from Azure Functions request headers
    static getClientIP(req) {
        const forwardedFor = req.headers['x-forwarded-for'];
        if (forwardedFor) {
            const firstIP = forwardedFor.split(',')[0].trim();
            const cleanIP = firstIP.split(':')[0];
            if (this.isValidIPv4(cleanIP)) {
                return cleanIP;
            }
        }
        
        const clientIP = req.headers['x-client-ip'];
        if (clientIP) {
            const cleanIP = clientIP.split(':')[0];
            if (this.isValidIPv4(cleanIP)) {
                return cleanIP;
            }
        }
        
        const realIP = req.headers['x-real-ip'];
        if (realIP) {
            const cleanIP = realIP.split(':')[0];
            if (this.isValidIPv4(cleanIP)) {
                return cleanIP;
            }
        }
        
        return null;
    }

    // Validate if an IP address is a valid IPv4 format
    static isValidIPv4(ip) {
        if (!ip || typeof ip !== 'string') return false;
        
        const parts = ip.split('.');
        if (parts.length !== 4) return false;
        
        return parts.every(part => {
            const num = parseInt(part, 10);
            return num >= 0 && num <= 255 && part === num.toString();
        });
    }
}

module.exports = SASTokenManager;
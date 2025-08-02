const { BlobServiceClient, StorageSharedKeyCredential, generateBlobSASQueryParameters, BlobSASPermissions } = require('@azure/storage-blob');

/**
 * SAS Token Manager for Azure Blob Storage
 * 
 * This class manages SAS token generation for Azure Blob Storage.
 * 
 * Usage:
 * 1. Connection String (Account Key auth): new SASTokenManager(connectionString)
 * 2. Azure AD (for User Delegation SAS): SASTokenManager.fromAzureAD(accountName, credential)
 * 
 * Authentication Note:
 * - When initialized with a connection string, only Account Key SAS tokens can be generated
 * - User Delegation SAS requires Azure AD authentication (DefaultAzureCredential) which is not available with connection string auth
 * - The class will attempt User Delegation SAS but fall back to Account Key SAS when using connection string
 * 
 * Security Best Practices:
 * - User Delegation SAS is more secure as it uses Azure AD and can be revoked
 * - Account Key SAS works but cannot be revoked without rotating the storage account key
 * - For production, consider using Managed Identity + DefaultAzureCredential instead of connection strings
 */
class SASTokenManager {
    constructor(storageConnectionString) {
        // Initialize blob service client
        this.blobServiceClient = BlobServiceClient.fromConnectionString(storageConnectionString);
        
        // Extract account details for SAS generation
        const accountNameMatch = storageConnectionString.match(/AccountName=([^;]+)/);
        const accountKeyMatch = storageConnectionString.match(/AccountKey=([^;]+)/);
        
        if (!accountNameMatch || !accountKeyMatch) {
            throw new Error('Invalid storage connection string format');
        }
        
        this.accountName = accountNameMatch[1];
        const accountKey = accountKeyMatch[1];
        
        if (!this.accountName || !accountKey) {
            throw new Error('Empty AccountName or AccountKey in connection string');
        }

        try {
            this.sharedKeyCredential = new StorageSharedKeyCredential(this.accountName, accountKey);
        } catch (error) {
            throw new Error(`Failed to create shared key credential: ${error.message}`);
        }
        
        this.authType = 'ConnectionString';
    }

    /**
     * Alternative constructor for Azure AD authentication (enables true user delegation SAS)
     * @param {string} accountName - Storage account name
     * @param {object} credential - Azure AD credential (e.g., DefaultAzureCredential)
     * @returns {SASTokenManager} Instance configured for Azure AD authentication
     */
    static fromAzureAD(accountName, credential) {
        const instance = Object.create(SASTokenManager.prototype);
        instance.accountName = accountName;
        instance.blobServiceClient = new BlobServiceClient(
            `https://${accountName}.blob.core.windows.net`,
            credential
        );
        instance.sharedKeyCredential = null; // No account key available
        instance.authType = 'AzureAD';
        return instance;
    }    /**
     * Generate a secure SAS token (simplified - no tracking needed for short-lived tokens)
     * Note: User delegation SAS requires Azure AD authentication, not connection string auth
     */
    async generateSASToken(options) {
        const {
            containerName,
            blobName,
            permissions = 'r',
            expiryMinutes = 5,
            clientIP = null,
            context = null
        } = options;

        let sasToken;
        let sasType = 'AccountKey';
        
        // Note: User delegation SAS will likely fail when BlobServiceClient is created from connection string
        // Connection string uses account key auth, but user delegation requires Azure AD auth
        try {
            // Try user delegation SAS first (more secure)
            if (context) {
                context.log(`Attempting user delegation SAS (auth type: ${this.authType})...`);
            }
            
            const delegationKeyStart = new Date();
            // Delegation key should live longer than the SAS token, minimum 1 hour, max 7 days
            const delegationKeyMinutes = Math.max(expiryMinutes * 2, 60); // At least double the SAS lifetime or 1 hour
            const delegationKeyExpiry = new Date(delegationKeyStart.getTime() + delegationKeyMinutes * 60 * 1000);
            
            const userDelegationKey = await this.blobServiceClient.getUserDelegationKey(
                delegationKeyStart,
                delegationKeyExpiry
            );
            
            if (userDelegationKey) {
                // Generate user delegation SAS
                const sasOptions = {
                    containerName,
                    blobName,
                    permissions: BlobSASPermissions.parse(permissions),
                    startsOn: new Date(),
                    expiresOn: new Date(Date.now() + expiryMinutes * 60 * 1000),
                    ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
                };

                // Note: generateBlobSASQueryParameters for user delegation requires: (options, userDelegationKey, accountName)
                sasToken = generateBlobSASQueryParameters(sasOptions, userDelegationKey, this.accountName).toString();
                sasType = 'UserDelegation';
                
                if (context) {
                    context.log('Generated user delegation SAS token');
                }
            }
        } catch (error) {
            if (context) {
                const expectedFailure = this.authType === 'ConnectionString' ? ' (expected with connection string auth)' : '';
                context.log(`User delegation SAS failed${expectedFailure}, falling back to account key:`, error.message);
            }
        }

        // Fallback to account key SAS (this will always be used when initialized with connection string)
        if (!sasToken) {
            if (!this.sharedKeyCredential) {
                throw new Error('Cannot generate account key SAS: no shared key credential available (Azure AD auth mode)');
            }
            
            try {
                const sasOptions = {
                    containerName,
                    blobName,
                    permissions: BlobSASPermissions.parse(permissions),
                    startsOn: new Date(),
                    expiresOn: new Date(Date.now() + expiryMinutes * 60 * 1000),
                    ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
                };

                sasToken = generateBlobSASQueryParameters(sasOptions, this.sharedKeyCredential).toString();
                sasType = 'AccountKey';
                
                if (context) {
                    context.log('Generated account key SAS token');
                }
            } catch (fallbackError) {
                if (context) {
                    context.log.error('Account key SAS also failed:', fallbackError.message);
                }
                throw new Error(`Failed to generate SAS token: ${fallbackError.message}`);
            }
        }

        if (!sasToken) {
            throw new Error('SAS token generation completed but token is empty');
        }

        return {
            sasToken,
            sasType,
            expiresAt: new Date(Date.now() + expiryMinutes * 60 * 1000)
        };
    }

    /**
     * Get client IP from Azure Functions request headers
     */
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

    /**
     * Validate if an IP address is a valid IPv4 format
     */
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
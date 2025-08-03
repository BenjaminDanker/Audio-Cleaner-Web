const { BlobServiceClient, generateBlobSASQueryParameters, BlobSASPermissions } = require('@azure/storage-blob');

class SASTokenManager {
    constructor(storageConnectionString) {
        // Initialize blob service client for User Delegation SAS
        this.blobServiceClient = BlobServiceClient.fromConnectionString(storageConnectionString);
        
        // Extract account name for SAS generation
        const accountNameMatch = storageConnectionString.match(/AccountName=([^;]+)/);
        
        if (!accountNameMatch) {
            throw new Error('Invalid storage connection string format - missing AccountName');
        }
        
        this.accountName = accountNameMatch[1];
        
        if (!this.accountName) {
            throw new Error('Empty AccountName in connection string');
        }
    }

    /**
     * Alternative constructor for Azure AD authentication
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
        return instance;
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
            context.log('Generating User Delegation SAS token...');
        }
        
        try {
            const delegationKeyStart = new Date();
            // Delegation key should live longer than the SAS token, minimum 1 hour, max 7 days
            const delegationKeyMinutes = Math.max(expiryMinutes * 2, 60);
            const delegationKeyExpiry = new Date(delegationKeyStart.getTime() + delegationKeyMinutes * 60 * 1000);
            
            const userDelegationKey = await this.blobServiceClient.getUserDelegationKey(
                delegationKeyStart,
                delegationKeyExpiry
            );
            
            if (!userDelegationKey) {
                throw new Error('Failed to obtain user delegation key from Azure AD');
            }

            // Generate user delegation SAS
            const sasOptions = {
                containerName,
                blobName,
                permissions: BlobSASPermissions.parse(permissions),
                startsOn: new Date(),
                expiresOn: new Date(Date.now() + expiryMinutes * 60 * 1000),
                ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
            };

            const sasToken = generateBlobSASQueryParameters(sasOptions, userDelegationKey, this.accountName).toString();
            
            if (!sasToken) {
                throw new Error('Generated SAS token is empty');
            }
            
            if (context) {
                context.log('Successfully generated User Delegation SAS token');
            }

            return {
                sasToken,
                sasType: 'UserDelegation',
                expiresAt: new Date(Date.now() + expiryMinutes * 60 * 1000)
            };
            
        } catch (error) {
            if (context) {
                context.log.error('User Delegation SAS generation failed:', error.message);
            }
            throw new Error(`Failed to generate secure SAS token: ${error.message}`);
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
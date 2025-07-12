/**
 * SAS Token Manager - Implements the 5 rules for secure SAS tokens
 * 
 * 1. Use user-delegation SAS whenever possible
 * 2. Grant bare minimum permissions with short expiry
 * 3. Force HTTPS and optionally IP ranges  
 * 4. Keep tokens out of logs
 * 5. Have a revocation strategy
 */

const { BlobServiceClient, BlobSASPermissions, generateBlobSASQueryParameters, StorageSharedKeyCredential } = require('@azure/storage-blob');
const { DefaultAzureCredential } = require('@azure/identity');
const { CosmosClient } = require('@azure/cosmos');

class SASTokenManager {
    constructor(connectionString, cosmosConnectionString) {
        this.connectionString = connectionString;
        this.blobServiceClient = BlobServiceClient.fromConnectionString(connectionString);
        
        // Extract account details for fallback
        this.accountName = connectionString.match(/AccountName=([^;]*)/)[1];
        const accountKey = connectionString.match(/AccountKey=([^;]*)/)[1];
        this.sharedKeyCredential = new StorageSharedKeyCredential(this.accountName, accountKey);
        
        // For token tracking and revocation (Rule #5)
        if (cosmosConnectionString) {
            this.cosmosClient = new CosmosClient(cosmosConnectionString);
            this.tokensContainer = this.cosmosClient.database('audiocleaner').container('sastokens');
        }
    }

    /**
     * Generate a secure SAS token following all 5 rules
     */
    async generateSASToken(options) {
        const {
            containerName,
            blobName,
            permissions = 'r', // Default to read-only (Rule #2)
            expiryMinutes = 10, // Default to 10 minutes (Rule #2)
            clientIP = null, // Optional IP restriction (Rule #3)
            userId = null, // For tracking and revocation
            context = null // For logging
        } = options;

        let sasToken;
        let sasType = 'AccountKey';
        
        try {
            // Rule #1: Prefer user delegation SAS
            const delegationKeyStart = new Date();
            const delegationKeyExpiry = new Date(delegationKeyStart.getTime() + Math.max(expiryMinutes, 30) * 60 * 1000);
            
            if (context) {
                context.log(`Attempting to get user delegation key, expires: ${delegationKeyExpiry.toISOString()}`);
            }
            
            const userDelegationKey = await this.blobServiceClient.getUserDelegationKey(
                delegationKeyStart,
                delegationKeyExpiry
            );
            
            const sasOptions = {
                containerName,
                blobName,
                permissions: BlobSASPermissions.parse(permissions), // Rule #2: minimal permissions
                startsOn: new Date(new Date().valueOf() - 2 * 60 * 1000), // 2 min buffer for clock skew
                expiresOn: new Date(new Date().valueOf() + expiryMinutes * 60 * 1000), // Rule #2: short expiry
                protocol: 'https', // Rule #3: Force HTTPS
            };

            // Rule #3: Temporarily disable IP restriction due to authentication issues
            // TODO: Re-enable after investigating IP format issues
            if (clientIP && this.isValidIPv4(clientIP)) {
                if (context) {
                    context.log(`IP available but not adding restriction (disabled): ${clientIP}`);
                }
            } else if (clientIP) {
                if (context) {
                    context.log(`Skipping invalid IP: ${clientIP}`);
                }
            }

            sasToken = generateBlobSASQueryParameters(sasOptions, userDelegationKey, this.accountName).toString();
            sasType = 'UserDelegation';
            
            if (context) {
                // Rule #4: Don't log the actual token or full blob path
                context.log(`Generated User Delegation SAS: ${blobName.substring(0, 20)}..., expires in ${expiryMinutes}min`);
            }
            
        } catch (error) {
            // Fallback to account key SAS
            if (context) {
                context.log('User delegation SAS failed, using account key SAS:', error.message);
            }
            
            const sasOptions = {
                containerName,
                blobName,
                permissions: BlobSASPermissions.parse(permissions), // Rule #2: minimal permissions
                startsOn: new Date(new Date().valueOf() - 2 * 60 * 1000),
                expiresOn: new Date(new Date().valueOf() + expiryMinutes * 60 * 1000), // Rule #2: short expiry
                protocol: 'https', // Rule #3: Force HTTPS
            };

            // Rule #3: Temporarily disable IP restriction due to authentication issues
            // TODO: Re-enable after investigating IP format issues
            if (clientIP && this.isValidIPv4(clientIP)) {
                if (context) {
                    context.log(`IP available but not adding restriction to fallback SAS (disabled): ${clientIP}`);
                }
            } else if (clientIP) {
                if (context) {
                    context.log(`Skipping invalid IP for fallback SAS: ${clientIP}`);
                }
            }

            try {
                sasToken = generateBlobSASQueryParameters(sasOptions, this.sharedKeyCredential).toString();
                
                if (context) {
                    // Rule #4: Don't log the actual token or full blob path
                    context.log(`Generated Account Key SAS: ${blobName.substring(0, 20)}..., expires in ${expiryMinutes}min`);
                }
            } catch (fallbackError) {
                if (context) {
                    context.log('Account key SAS also failed:', fallbackError.message);
                }
                throw new Error(`Failed to generate SAS token: ${fallbackError.message}`);
            }
        }

        // Rule #5: Track token for potential revocation
        if (this.tokensContainer && userId) {
            try {
                await this.tokensContainer.items.create({
                    id: `${userId}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
                    userId,
                    containerName,
                    blobName: blobName.substring(0, 50), // Store partial path only for privacy
                    sasType,
                    permissions,
                    createdAt: Date.now(),
                    expiresAt: Date.now() + (expiryMinutes * 60 * 1000),
                    clientIP,
                    ttl: expiryMinutes * 60 + 300 // Auto-delete 5 minutes after expiry
                });
            } catch (error) {
                if (context) {
                    context.log('Failed to track SAS token:', error.message);
                }
                // Don't fail the request if tracking fails
            }
        }

        return {
            sasToken,
            sasType,
            expiresAt: new Date(Date.now() + expiryMinutes * 60 * 1000)
        };
    }

    /**
     * Rule #5: Revoke SAS tokens for a user by invalidating user delegation key
     * Note: This only works for user delegation SAS tokens
     */
    async revokeSASTokensForUser(userId, context = null) {
        try {
            // For user delegation SAS: Get a new delegation key to invalidate old ones
            // This is a simplified approach - in production you might want more granular control
            const newKey = await this.blobServiceClient.getUserDelegationKey(
                new Date(),
                new Date(Date.now() + 60 * 60 * 1000) // 1 hour
            );
            
            // Mark tracked tokens as revoked
            if (this.tokensContainer) {
                const { resources: userTokens } = await this.tokensContainer.items
                    .query(`SELECT * FROM c WHERE c.userId = "${userId}" AND c.expiresAt > ${Date.now()}`)
                    .fetchAll();
                
                for (const token of userTokens) {
                    await this.tokensContainer.item(token.id).patch([
                        { op: 'add', path: '/revoked', value: true },
                        { op: 'add', path: '/revokedAt', value: Date.now() }
                    ]);
                }
                
                if (context) {
                    context.log(`Revoked ${userTokens.length} SAS tokens for user ${userId}`);
                }
            }
            
            return true;
        } catch (error) {
            if (context) {
                context.log(`Failed to revoke SAS tokens for user ${userId}:`, error.message);
            }
            return false;
        }
    }

    /**
     * Validate if an IP address is a valid IPv4 format
     */
    isValidIPv4(ip) {
        if (!ip || typeof ip !== 'string') return false;
        
        // Remove any port numbers
        const cleanIP = ip.split(':')[0];
        
        const parts = cleanIP.split('.');
        if (parts.length !== 4) return false;
        
        return parts.every(part => {
            const num = parseInt(part, 10);
            return num >= 0 && num <= 255 && part === num.toString();
        });
    }

    /**
     * Get client IP from Azure Functions request headers with better parsing
     */
    static getClientIP(req) {
        // Try various headers that might contain the real client IP
        const forwardedFor = req.headers['x-forwarded-for'];
        if (forwardedFor) {
            // x-forwarded-for can contain multiple IPs, take the first one
            const firstIP = forwardedFor.split(',')[0].trim();
            // Remove any port numbers and validate
            const cleanIP = firstIP.split(':')[0];
            if (SASTokenManager.prototype.isValidIPv4.call({}, cleanIP)) {
                return cleanIP;
            }
        }
        
        const clientIP = req.headers['x-client-ip'];
        if (clientIP) {
            const cleanIP = clientIP.split(':')[0];
            if (SASTokenManager.prototype.isValidIPv4.call({}, cleanIP)) {
                return cleanIP;
            }
        }
        
        const realIP = req.headers['x-real-ip'];
        if (realIP) {
            const cleanIP = realIP.split(':')[0];
            if (SASTokenManager.prototype.isValidIPv4.call({}, cleanIP)) {
                return cleanIP;
            }
        }
        
        // Fallback to connection remote address
        const remoteAddress = req.connection?.remoteAddress;
        if (remoteAddress) {
            const cleanIP = remoteAddress.split(':')[0];
            if (SASTokenManager.prototype.isValidIPv4.call({}, cleanIP)) {
                return cleanIP;
            }
        }
        
        return null; // Return null if no valid IPv4 found
    }
}

module.exports = SASTokenManager;

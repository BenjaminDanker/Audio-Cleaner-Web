<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 -->
# SAS Token Security Implementation

> **Part of [Security Guide](SECURITY.md) - Azure Blob Storage Security**

This document details the implementation of the 5 essential rules for secure Azure Storage SAS (Shared Access Signature) tokens in Audio Cleaner Pro.

## Implementation Overview

The Audio Cleaner Web application implements all 5 critical SAS security rules through the centralized `SASTokenManager` class in `api/shared/sasTokenManager.js`.

## The 5 Security Rules

### ✅ Rule 1: Use User-Delegation SAS Whenever Possible

**Implementation:** Prioritizes user-delegation SAS tokens over account-key SAS tokens.

```javascript
// Try user delegation SAS first (preferred method)
const userDelegationKey = await this.blobServiceClient.getUserDelegationKey(
    delegationKeyStart,
    delegationKeyExpiry
);
```

**Benefits:**
- Signed with Microsoft Entra credentials instead of storage account keys
- Better security posture with credential rotation
- Eliminates need to manage storage account keys

**Fallback:** Graceful fallback to account-key SAS if user-delegation fails.

### ✅ Rule 2: Grant Minimal Permissions & Short Expiry

**Upload Tokens:**
- Permissions: `'cw'` (create and write only)
- Expiry: 15 minutes
- Use case: File uploads

**Download Tokens:**
- Permissions: `'r'` (read only)
- Expiry: 10 minutes
- Use case: File downloads

```javascript
// Upload: minimal permissions, short expiry
permissions: 'cw', // create and write only
expiryMinutes: 15, // 15 minutes for uploads

// Download: read-only, very short expiry  
permissions: 'r', // read-only
expiryMinutes: 10, // 10 minutes for downloads
```

**Security Impact:** 75-85% reduction in token lifetime vs. previous 1-hour expiry.

### ✅ Rule 3: Force HTTPS and IP Restrictions

**HTTPS Enforcement:**
```javascript
const sasOptions = {
    protocol: 'https', // Force HTTPS for all requests
    // ... other options
};
```

**IP Restrictions:**
```javascript
// Add IP restrictions when client IP is available
const clientIP = getClientIP(request);
if (clientIP) {
    sasOptions.ipRange = { start: clientIP, end: clientIP };
}
```

**IP Detection:** Checks multiple headers: `x-forwarded-for`, `x-client-ip`, `x-real-ip`.

### ✅ Rule 4: No Token Logging

**Before:** Full SAS URLs logged to console and Application Insights.
**After:** Only partial blob names (first 20 characters) logged for debugging.

```javascript
// Safe logging - no SAS tokens exposed
logger.info(`Generated SAS token for blob: ${blobName.substring(0, 20)}...`);
```

**Compliance:** SAS tokens never appear in logs or Application Insights.

### ✅ Rule 5: Revocation Strategy

**Token Tracking:**
- All generated tokens tracked in Cosmos DB with TTL
- Automatic cleanup when tokens expire
- User association for bulk revocation

**Revocation Methods:**
1. **User-delegation key invalidation** (preferred)
2. **Token marking as revoked** in database
3. **Manual revocation endpoint**: `/api/revoke-sas-tokens`

```javascript
// Revocation implementation
async revokeSASTokens(userId) {
    // Invalidate user delegation keys
    await this.invalidateUserDelegationKeys();
    
    // Mark tokens as revoked in database
    await this.markTokensRevoked(userId);
}
```

## Architecture Integration

### Components

**SAS Token Manager** (`api/shared/sasTokenManager.js`)
- Centralized SAS token generation
- Implements all 5 security rules
- Automatic fallback mechanisms
- Token tracking and revocation

**Upload Endpoint** (`api/upload-file/index.js`)
- Generates upload SAS tokens
- Validates user quotas
- Tracks token usage

**Download Endpoint** (`api/download-file/index.js`)
- Generates download SAS tokens
- Validates file ownership
- Enforces access controls

**Revocation Endpoint** (`api/revoke-sas-tokens/index.js`)
- Manual token revocation
- User-specific or bulk revocation
- Emergency security response

### Security Flow

```
1. User requests file operation
2. API validates authentication & authorization
3. SAS Token Manager generates secure token
4. Token tracked in Cosmos DB with TTL
5. Client receives time-limited, permission-scoped token
6. Direct Azure Storage operation with token
7. Token automatically expires or can be manually revoked
```

## Monitoring & Compliance

### Security Metrics
- Token generation rate
- Failed token validations
- Revocation events
- Unusual access patterns

### Audit Trail
- All token operations logged (without exposing tokens)
- User association tracking
- Geographic access patterns
- Time-based usage analysis

### Compliance Benefits
- **GDPR**: Right to be forgotten through token revocation
- **SOC 2**: Audit trail and access controls
- **Zero Trust**: Least privilege token permissions
- **Defense in Depth**: Multiple security layers

For complete security implementation details, see [Security Guide](SECURITY.md).
    protocol: 'https', // Force HTTPS
    // Add IP restriction if client IP is known
    ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
};
```

**Why it matters:** Prevents token usage over insecure connections and limits usage to specific IP addresses when possible.

### 4. ✅ Keep Tokens Out of Logs

**Implementation:**

- SAS tokens are never logged
- Only partial blob names are logged (first 20 characters)
- Full URLs are never written to Application Insights or console

```javascript
// ❌ DON'T DO THIS
context.log(`Generated SAS URL: ${fullSasUrl}`);

// ✅ DO THIS INSTEAD  
context.log(`Generated SAS for blob: ${blobName.substring(0, 20)}...`);
```

**Why it matters:** SAS tokens in logs can be replayed by anyone with access to the logs or monitoring systems.

### 5. ✅ Have a Revocation Strategy

**Implementation:**

- All SAS tokens are tracked in Cosmos DB with user association
- User-delegation keys can be invalidated to revoke all user tokens
- Manual revocation endpoint: `/api/revoke-sas-tokens`

```javascript
// Track token for revocation
await this.tokensContainer.items.create({
    userId,
    sasType,
    createdAt: Date.now(),
    expiresAt: Date.now() + (expiryMinutes * 60 * 1000),
    ttl: expiryMinutes * 60 + 300 // Auto-cleanup
});

// Revoke all tokens for a user
async revokeSASTokensForUser(userId) {
    // Get new delegation key to invalidate old ones
    // Mark tracked tokens as revoked
}
```

**Why it matters:** Provides a way to invalidate compromised tokens before they expire naturally.

## Security Features Summary

| Feature | Upload Function | Download Function | Benefits |
|---------|----------------|-------------------|----------|
| **User Delegation SAS** | ✅ Primary method | ✅ Primary method | No account key exposure |
| **Minimal Permissions** | ✅ `cw` only | ✅ `r` only | Limits damage if compromised |
| **Short Expiry** | ✅ 15 minutes | ✅ 10 minutes | Reduces exposure window |
| **HTTPS Enforced** | ✅ Always | ✅ Always | Prevents interception |
| **IP Restrictions** | ✅ When available | ✅ When available | Limits usage location |
| **No Token Logging** | ✅ Implemented | ✅ Implemented | Prevents log replay attacks |
| **Token Tracking** | ✅ Cosmos DB | ✅ Cosmos DB | Enables revocation |
| **Auto-cleanup** | ✅ TTL set | ✅ TTL set | Removes expired records |

## Usage Examples

### Generating Upload SAS Token

```javascript
const sasManager = new SASTokenManager(connectionString, cosmosConnectionString);

const sasResult = await sasManager.generateSASToken({
    containerName: 'uploads',
    blobName: `${userId}/${timestamp}_${filename}`,
    permissions: 'cw',
    expiryMinutes: 15,
    clientIP: req.headers['x-forwarded-for'],
    userId: userId,
    context: context
});
```

### Revoking User Tokens

```javascript
// Revoke all SAS tokens for a specific user
const success = await sasManager.revokeSASTokensForUser(userId, context);
```

## Monitoring and Alerting

The implementation includes logging for:

- SAS token generation (without exposing the actual tokens)
- Token type used (UserDelegation vs AccountKey)
- IP restriction status
- Revocation events

Consider setting up alerts for:

- High frequency of account-key SAS fallbacks (indicates delegation issues)
- Large numbers of token revocations (potential security incident)
- Tokens being generated without IP restrictions

## Deployment Considerations

### Required Permissions

For user-delegation SAS to work, the Azure Function app's managed identity needs:

- `Microsoft.Storage/storageAccounts/blobServices/generateUserDelegationKey/action`

### Cosmos DB Setup

Create a container for SAS token tracking:

- Database: `audiocleaner`
- Container: `sastokens`
- Partition key: `/userId`
- TTL enabled on container level

### Environment Variables

```bash
AzureWebJobsStorage=<storage-connection-string>
COSMOS_CONNECTION_STRING=<cosmos-connection-string>
```

## Best Practices for Frontend

1. **Never log SAS URLs** in browser console or analytics
2. **Use SAS tokens immediately** - don't store them
3. **Handle expiry gracefully** - request new tokens as needed
4. **Clear tokens from memory** after use

## Security Incident Response

If a SAS token is compromised:

1. **Immediate:** Call `/api/revoke-sas-tokens` for affected user
2. **Short-term:** Monitor blob access logs for suspicious activity
3. **Long-term:** Review token generation patterns and adjust expiry times

This implementation provides defense-in-depth security for Azure Storage access while maintaining good user experience through appropriate token lifetimes and automatic fallback mechanisms.

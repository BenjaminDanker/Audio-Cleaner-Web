# SAS Token Security Implementation

This document outlines how the Audio Cleaner Web application implements the 5 essential rules for secure Azure Storage SAS (Shared Access Signature) tokens.

## The 5 Rules Implemented

### 1. ✅ Use User-Delegation SAS Whenever Possible

**Implementation:** Our `SASTokenManager` class prioritizes user-delegation SAS tokens over account-key SAS tokens.

```javascript
// Try user delegation SAS first (preferred method)
const userDelegationKey = await this.blobServiceClient.getUserDelegationKey(
    delegationKeyStart,
    delegationKeyExpiry
);
```

**Why it matters:** User-delegation SAS tokens are signed with Microsoft Entra credentials instead of storage account keys, providing better security and eliminating the need to manage account keys.

**Fallback:** If user-delegation SAS fails (e.g., insufficient permissions), the system gracefully falls back to account-key SAS.

### 2. ✅ Grant Bare Minimum Permissions

**Implementation:**

- **Upload tokens:** `'cw'` (create and write only) - 15 minutes expiry
- **Download tokens:** `'r'` (read only) - 10 minutes expiry

```javascript
// Upload: minimal permissions, short expiry
permissions: 'cw', // create and write only
expiryMinutes: 15, // 15 minutes for uploads

// Download: read-only, very short expiry  
permissions: 'r', // read-only
expiryMinutes: 10, // 10 minutes for downloads
```

**Why it matters:** If a token is compromised, it can only perform limited operations for a very short time.

### 3. ✅ Force HTTPS and IP Restrictions

**Implementation:**

- All SAS tokens include `protocol: 'https'`
- IP restrictions are added when client IP is available

```javascript
const sasOptions = {
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

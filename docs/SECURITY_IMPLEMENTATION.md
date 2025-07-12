<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 -->
# Audio Cleaner Web - Security Implementation Guide

## Overview
This document outlines the comprehensive security measures implemented in the Audio Cleaner Web application to protect against various threats and ensure data security.

## Security Layers Implemented

### 1. Rate Limiting & Throttling
- **Implementation**: Custom SecurityMiddleware with sliding window rate limiting
- **Features**:
  - Per-endpoint rate limits (different limits for upload, download, job processing)
  - Burst protection (prevents rapid-fire requests)
  - IP-based and user-based tracking
  - Automatic cleanup of expired rate limit entries
- **Configuration**:
  - Upload: 10 requests/minute, 3 burst limit
  - Download: 50 requests/minute, 10 burst limit
  - Job Queue: 5 requests/minute, 2 burst limit
  - Auth: 20 requests/minute, 5 burst limit

### 2. Input Validation & Sanitization
- **Implementation**: InputValidator class with comprehensive validation schemas
- **Protection Against**:
  - XSS attacks (script injection detection)
  - SQL injection (pattern detection)
  - Command injection
  - Path traversal attacks
  - File type validation
  - Size limit enforcement
- **Features**:
  - Endpoint-specific validation schemas
  - HTML escaping
  - Filename sanitization
  - GUID and custom format validation

### 3. Authentication & Authorization
- **Azure AD Integration**: Secure authentication through Azure Static Web Apps
- **Enhanced Checks**:
  - Principal validation
  - Session integrity checking
  - User context tracking
  - Authentication state monitoring
- **Security Headers**: Proper authentication headers validation

### 4. File Security
- **Upload Security**:
  - File type whitelist (only video/audio formats)
  - File size limits (5GB max, 2GB warning threshold)
  - Filename sanitization
  - Virus scanning ready (extensible)
- **Download Security**:
  - User-specific access control
  - Secure SAS token generation
  - IP-restricted downloads
  - Short-lived tokens (5 minutes)

### 5. SAS Token Security (5 Rules Implementation)
- **Rule 1**: Prefer user-delegation SAS over account key SAS
- **Rule 2**: Minimal permissions with short expiry times
- **Rule 3**: HTTPS enforcement and IP restrictions
- **Rule 4**: Keep tokens out of logs (partial logging only)
- **Rule 5**: Token tracking and revocation capability

### 6. Threat Detection
- **Suspicious Activity Monitoring**:
  - Malicious user agent detection
  - Attack pattern recognition in URLs and headers
  - Excessive request size detection
  - Geographic restrictions (configurable)
- **Automatic Blocking**: Suspicious requests are automatically blocked

### 7. Security Headers
- **Response Headers Applied**:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Strict-Transport-Security: max-age=31536000`
  - `Content-Security-Policy`: Restrictive policy
  - `Permissions-Policy`: Camera, microphone, geolocation disabled

### 8. Infrastructure Security
- **Azure Functions Configuration**:
  - Reduced concurrent request limits
  - Health monitoring enabled
  - Retry policies with exponential backoff
  - Enhanced logging and monitoring
- **Key Vault Security**:
  - Premium tier with HSM support
  - Network restrictions
  - RBAC authorization
  - Soft delete and purge protection
  - Diagnostic logging enabled

### 9. Error Handling & Information Disclosure Prevention
- **Generic Error Messages**: Internal errors don't expose system details
- **Logging**: Comprehensive security event logging without sensitive data
- **Monitoring**: Security events are tracked in Cosmos DB for analysis

### 10. Automated Security Maintenance
- **Cleanup Functions**:
  - Rate limit data cleanup (every 15 minutes)
  - Security event cleanup (30-day retention)
  - Expired token cleanup
- **Health Monitoring**: Function health and performance monitoring

## Security Configuration

### Environment Variables Required
```
COSMOS_CONNECTION_STRING=<cosmos-db-connection>
AZURE_SERVICE_BUS_CONNECTION_STRING=<service-bus-connection>
AzureWebJobsStorage=<storage-connection>
JOB_SECURITY_SALT=<random-salt-for-job-hashing>
```

### Rate Limit Configuration
Rate limits can be adjusted in `SecurityMiddleware.js`:
```javascript
this.rateLimits = {
    '/api/upload-file': { requests: 10, windowMs: 60000, burstLimit: 3 },
    '/api/download-file': { requests: 50, windowMs: 60000, burstLimit: 10 },
    // ... other endpoints
};
```

### File Security Configuration
Allowed file types in `InputValidator.js`:
```javascript
this.allowedFileTypes = [
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v',
    '.mp3', '.wav', '.aac', '.ogg', '.flac', '.m4a'
];
```

## Security Monitoring

### Cosmos DB Collections
1. **ratelimits**: Tracks rate limiting data per user/IP
2. **securityevents**: Logs security incidents and threats
3. **sastokens**: Tracks SAS token usage for revocation

### Security Events Logged
- Rate limit violations
- Authentication failures
- Invalid input attempts
- Threat detection triggers
- Suspicious file uploads
- Failed authorization attempts

## Deployment Security Checklist

### Pre-Deployment
- [ ] Update all security configuration parameters
- [ ] Review and test rate limiting thresholds
- [ ] Validate Key Vault access policies
- [ ] Configure IP restrictions for Key Vault
- [ ] Set up security monitoring alerts

### Post-Deployment
- [ ] Verify security headers are applied
- [ ] Test rate limiting functionality
- [ ] Validate authentication flows
- [ ] Check security event logging
- [ ] Monitor for security alerts

## Security Best Practices for Operations

### 1. Regular Security Reviews
- Review security event logs weekly
- Analyze rate limiting patterns
- Monitor for failed authentication attempts
- Check for suspicious file upload patterns

### 2. Key Rotation
- Rotate storage account keys quarterly
- Update SAS token security salt annually
- Review and update IP allowlists

### 3. Monitoring & Alerting
- Set up alerts for high rate of security events
- Monitor for authentication failures
- Alert on suspicious file upload patterns
- Track unusual download patterns

### 4. Incident Response
- Investigate security event spikes
- Block suspicious IPs if necessary
- Revoke compromised user sessions
- Review and update security policies

## Security Testing

### Automated Tests
- Rate limiting functionality
- Input validation effectiveness
- Authentication bypass attempts
- File upload security

### Manual Security Tests
- XSS injection attempts
- SQL injection testing
- Path traversal testing
- Authentication token manipulation
- Rate limiting bypass attempts

## Compliance & Auditing

### Data Protection
- Personal data is encrypted in transit and at rest
- User data access is logged and auditable
- Data retention policies are enforced
- Right to deletion is supported

### Audit Trail
- All security events are logged with timestamps
- User actions are tracked and attributable
- System access is monitored and logged
- Configuration changes are auditable

## Future Security Enhancements

### Planned Improvements
1. **Advanced Threat Detection**: ML-based anomaly detection
2. **Geographic Restrictions**: Enhanced IP-based filtering
3. **Advanced Rate Limiting**: User behavior analysis
4. **File Content Scanning**: Malware detection integration
5. **Security Automation**: Automated threat response

### Security Metrics to Track
- Authentication success/failure rates
- Rate limiting effectiveness
- Security event frequency
- File upload security blocks
- Performance impact of security measures

---

**Note**: This security implementation follows industry best practices and Azure security recommendations. Regular security reviews and updates are essential to maintain effectiveness against evolving threats.

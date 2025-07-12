<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 -->
# Security Guide

## Overview

Audio Cleaner Pro implements defense-in-depth security across all layers of the application, following Azure Well-Architected Framework security principles.

## Security Architecture

### Authentication & Authorization

#### User Authentication
- **Provider**: Azure AD B2C
- **Flow**: OAuth 2.0 / OpenID Connect
- **Tokens**: JWT with configurable expiration
- **MFA**: Optional multi-factor authentication support

#### Service Authentication
- **Azure Services**: Managed Identity (System-assigned)
- **External APIs**: Client credentials with Key Vault storage
- **Database**: Connection strings in Key Vault
- **No secrets in code**: All credentials externalized

#### Authorization Model
```
User Groups:
├── Free Tier Users
│   ├── Upload limit: 100MB/month
│   ├── Processing time: 5 minutes/file
│   └── File retention: 7 days
├── Premium Users
│   ├── Upload limit: 1GB/month
│   ├── Processing time: Unlimited
│   └── File retention: 30 days
└── Enterprise Users
    ├── Upload limit: Unlimited
    ├── Processing time: Unlimited
    ├── File retention: 1 year
    └── Custom SLA options
```

### Data Protection

#### Encryption at Rest
- **Azure Storage**: AES-256 encryption (Microsoft-managed keys)
- **Cosmos DB**: Transparent Data Encryption (TDE)
- **Key Vault**: FIPS 140-2 Level 2 validated HSMs
- **Container Images**: Encrypted container registry storage

#### Encryption in Transit
- **HTTPS Everywhere**: TLS 1.2+ for all communications
- **API Endpoints**: Certificate pinning for critical endpoints
- **Database Connections**: SSL/TLS encrypted connections
- **Internal Traffic**: VNet encryption for service-to-service

#### Data Classification
```
Classification Levels:
├── Public: Marketing content, documentation
├── Internal: System logs, metadata
├── Confidential: User files, payment data
└── Restricted: Encryption keys, admin credentials
```

### Network Security

#### Network Isolation
- **Virtual Network**: Private subnets for all services
- **Private Endpoints**: Database and storage access
- **Network Security Groups**: Restrict traffic by port/protocol
- **Azure Firewall**: Centralized network filtering

#### Internet Access
- **Azure Front Door**: Global load balancer with DDoS protection
- **Web Application Firewall**: OWASP Top 10 protection
- **Rate Limiting**: API throttling and abuse prevention
- **Geographic Restrictions**: Optional country-based blocking

#### Internal Communication
```
Network Flow:
Internet → Front Door → Static Web App
                    ↓
Internet → Front Door → Functions (API)
                    ↓
Functions → Private Endpoint → Cosmos DB
Functions → Private Endpoint → Storage
Functions → Service Bus → Container Apps
```

### Storage Security

#### Azure Blob Storage
- **SAS Tokens**: Time-limited, permission-scoped access
- **User Delegation SAS**: Preferred over account key SAS
- **Network Access**: Private endpoint + firewall rules
- **Versioning**: Blob versioning with lifecycle management

#### SAS Token Security Rules

**Rule 1: Use User-Delegation SAS**
```javascript
// Implementation in sasTokenManager.js
const userDelegationKey = await blobServiceClient.getUserDelegationKey(
    delegationKeyStart,
    delegationKeyExpiry
);
```

**Rule 2: Minimal Permissions & Short Expiry**
- Upload tokens: `'cw'` (create/write) - 15 minutes
- Download tokens: `'r'` (read-only) - 10 minutes

**Rule 3: HTTPS & IP Restrictions**
```javascript
const sasOptions = {
    protocol: 'https',
    ipRange: clientIP ? { start: clientIP, end: clientIP } : undefined
};
```

**Rule 4: No Token Logging**
- Tokens excluded from Application Insights
- Only blob names (first 20 chars) logged for debugging

**Rule 5: Revocation Strategy**
- All tokens tracked in Cosmos DB with TTL
- Manual revocation endpoint: `/api/revoke-sas-tokens`
- User delegation key invalidation

### API Security

#### Input Validation
- **Request Size Limits**: 100MB max file size
- **Content Type Validation**: Only allowed video formats
- **Rate Limiting**: Per-user and per-IP throttling
- **SQL Injection Prevention**: Parameterized queries only

#### Authentication Flow
```
1. User authenticates with Azure AD B2C
2. Frontend receives JWT token
3. Token included in API requests (Authorization header)
4. Functions validate JWT signature and claims
5. User context extracted for authorization decisions
```

#### API Endpoints Security
| Endpoint | Authentication | Authorization | Rate Limit |
|----------|---------------|---------------|------------|
| `/api/auth` | Optional | Public | 10/min/IP |
| `/api/subscription` | Required | User-specific | 60/min/user |
| `/api/upload-file` | Required | User quota | 5/min/user |
| `/api/enqueue-job` | Required | User quota | 3/min/user |
| `/api/job-status` | Required | Job owner | 30/min/user |
| `/api/download-file` | Required | Job owner | 10/min/user |

### Container Security

#### Image Security
- **Base Images**: Microsoft-maintained minimal images
- **Vulnerability Scanning**: Trivy scanning in CI/CD
- **Image Signing**: Notary v2 for image authenticity
- **Private Registry**: Azure Container Registry with RBAC

#### Runtime Security
- **Non-root User**: Containers run as non-privileged user
- **Read-only Filesystem**: Immutable container filesystems
- **Resource Limits**: CPU/memory limits to prevent DoS
- **Health Checks**: Kubernetes liveness/readiness probes

#### Container Apps Security
```yaml
# Security context in deployment
securityContext:
  runAsNonRoot: true
  runAsUser: 1001
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
```

### Secrets Management

#### Azure Key Vault Integration
- **Secret Storage**: All passwords, API keys, connection strings
- **Access Policies**: Managed Identity with minimal permissions
- **Rotation**: Automatic secret rotation where supported
- **Auditing**: All secret access logged

#### Secret Categories
```
Secrets by Type:
├── Database Connections
│   ├── Cosmos DB connection string
│   └── Storage account connection string
├── External APIs
│   ├── Stripe API keys
│   ├── Azure AD B2C secrets
│   └── Notification service keys
├── Encryption Keys
│   ├── JWT signing keys
│   ├── Data encryption keys
│   └── Certificate private keys
└── System Secrets
    ├── Service principal credentials
    └── Container registry passwords
```

#### Key Vault Access Pattern
```javascript
// Functions access secrets via Managed Identity
const secretClient = new SecretClient(
    `https://${keyVaultName}.vault.azure.net/`,
    new DefaultAzureCredential()
);
const secret = await secretClient.getSecret("stripe-secret-key");
```

### Monitoring & Compliance

#### Security Monitoring
- **Azure Security Center**: Continuous security assessment
- **Application Insights**: Real-time security event tracking
- **Log Analytics**: Centralized security log analysis
- **Sentinel**: AI-powered threat detection (optional)

#### Audit Logging
```
Logged Security Events:
├── Authentication Events
│   ├── User login/logout
│   ├── Failed authentication attempts
│   └── Token validation failures
├── Authorization Events
│   ├── Permission denied events
│   ├── Privilege escalation attempts
│   └── Resource access patterns
├── Data Access Events
│   ├── File uploads/downloads
│   ├── Database queries
│   └── Key Vault access
└── Administrative Events
    ├── Configuration changes
    ├── User management actions
    └── System maintenance activities
```

#### Compliance Features
- **GDPR**: Right to be forgotten, data portability
- **SOC 2**: Azure compliance inheritance
- **ISO 27001**: Security management system
- **HIPAA**: Healthcare data protection (configurable)

### Incident Response

#### Security Incident Types
1. **Authentication Bypass**: Unauthorized access attempts
2. **Data Breach**: Unauthorized data access/exfiltration
3. **Service Abuse**: Resource exhaustion or misuse
4. **Infrastructure Compromise**: Malware or unauthorized changes

#### Response Procedures
```
Incident Response Flow:
1. Detection (Automated alerts)
2. Analysis (Security team investigation)
3. Containment (Isolate affected resources)
4. Eradication (Remove threats)
5. Recovery (Restore normal operations)
6. Lessons Learned (Update procedures)
```

#### Emergency Procedures
```bash
# Disable compromised user account
az ad user update --id user@domain.com --account-enabled false

# Rotate compromised secrets
az keyvault secret set --vault-name <vault> --name <secret> --value <new-value>

# Block suspicious IP addresses
az network nsg rule create \
  --name "Block-Suspicious-IP" \
  --nsg-name <nsg-name> \
  --priority 100 \
  --source-address-prefixes <suspicious-ip> \
  --access Deny
```

### Security Testing

#### Automated Security Testing
- **SAST**: Static code analysis with SonarCloud
- **DAST**: Dynamic application security testing
- **Dependency Scanning**: npm audit, Dependabot
- **Infrastructure Scanning**: Checkov for Bicep templates

#### Penetration Testing
- **Schedule**: Annual third-party penetration testing
- **Scope**: Full application stack and infrastructure
- **Remediation**: 30-day SLA for critical findings
- **Validation**: Re-testing of fixes within 60 days

### Security Configuration Checklist

#### Deployment Security
- [ ] All secrets stored in Key Vault
- [ ] Managed Identity configured for all services
- [ ] Private endpoints enabled for data services
- [ ] Network Security Groups configured
- [ ] Azure Security Center enabled
- [ ] Application Insights configured with security events
- [ ] Backup and disaster recovery tested

#### Application Security
- [ ] JWT token validation implemented
- [ ] Input validation on all endpoints
- [ ] Rate limiting configured
- [ ] CORS policies properly configured
- [ ] Error messages don't leak sensitive information
- [ ] Security headers implemented (CSP, HSTS, etc.)
- [ ] File upload restrictions enforced

#### Operational Security
- [ ] Security monitoring alerts configured
- [ ] Incident response procedures documented
- [ ] Security team access provisioned
- [ ] Regular security reviews scheduled
- [ ] Employee security training completed
- [ ] Third-party security assessments completed

For implementation details, see specific security configurations in the `/infra` directory Bicep templates.

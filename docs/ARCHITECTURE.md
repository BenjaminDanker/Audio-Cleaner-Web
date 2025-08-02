<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 MD009 MD024-->
# System Architecture

## Overview

Audio Cleaner Pro is a cloud-native application built on Azure that uses AI to remove background noise from video files. The system follows a microservices architecture with clear separation of concerns.

## Architecture Diagram

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Azure Front Door                         │
│                    (Global Load Balancer)                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Static Web App                                │
│  ┌─────────────────┐       ┌─────────────────────────────────┐   │
│  │   Frontend      │───────│    Managed Functions API       │   │
│  │    (SPA)        │       │      (Node.js)                 │   │
│  │                 │       │  ┌─────────────────────────────┐│   │
│  │                 │       │  │  Authentication (Azure AD)  ││   │
│  └─────────────────┘       │  └─────────────────────────────┘│   │
│                            └─────────────────────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
    ┌─────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Storage   │   │    Database     │   │    Messaging    │
    │Blob Storage │   │   Cosmos DB     │   │  Service Bus    │
    └─────────────┘   └─────────────────┘   └─────┬───────────┘
                                                   │
                                                   ▼
                                        ┌─────────────────┐
                                        │   AI Processor  │
                                        │ Container Apps  │
                                        │   (Python)      │
                                        └─────────────────┘
```

## Components

### Frontend Layer

**Technology**: React 18 + Vite
**Hosting**: Azure Static Web Apps (with integrated API)
**Responsibilities**:
- User interface for file upload and management
- Real-time job status monitoring
- User authentication flows
- Payment integration (Stripe)

**Key Features**:
- Progressive Web App (PWA) capabilities
- Responsive design for mobile/desktop
- Drag-and-drop file upload with progress tracking
- Integrated API communication (no CORS issues)
- Built-in authentication with Azure AD B2C

### API Layer

**Technology**: Azure Functions (Node.js v18) - Managed Functions
**Hosting**: Azure Static Web Apps (integrated)
**Authentication**: JWT tokens with Azure AD B2C

**Endpoints**:
- `POST /api/auth` - User authentication
- `GET /api/subscription` - Subscription management
- `POST /api/upload-file` - Generate SAS tokens for uploads
- `POST /api/enqueue-job` - Queue processing jobs
- `GET /api/job-status/{id}` - Job status monitoring
- `GET /api/download-file/{id}` - Secure file downloads
- `POST /api/webhook-stripe` - Stripe payment webhooks

**Security Features**:
- Managed Identity for Azure service authentication
- Integrated with Static Web App authentication
- Rate limiting and request validation
- SAS token generation with minimal permissions

### Processing Layer

**Technology**: Python 3.11 + DeepFilterNet3
**Hosting**: Azure Container Apps
**Scaling**: Automatic based on queue depth

**Components**:
- **AI Model**: DeepFilterNet3 for audio denoising
- **Video Processing**: FFmpeg for video/audio manipulation
- **Queue Consumer**: Service Bus message processing
- **Health Checks**: Kubernetes-style health endpoints

**Processing Flow**:
1. Receive job message from Service Bus
2. Download input file from Blob Storage
3. Extract audio track using FFmpeg
4. Apply AI denoising with DeepFilterNet3
5. Merge processed audio back to video
6. Upload result to Blob Storage
7. Update job status in Cosmos DB

### Data Layer

#### Cosmos DB (NoSQL Database)
**Purpose**: Metadata and job tracking
**Containers**:
- `users` - User profiles and subscriptions
- `jobs` - Processing job metadata and status
- `tokens` - SAS token tracking for security

#### Azure Blob Storage
**Purpose**: File storage
**Containers**:
- `uploads` - User-uploaded video files
- `processed` - AI-processed output files
- `temp` - Temporary processing files (auto-cleanup)

**Storage Tiers**:
- Hot tier for active files
- Cool tier for completed jobs (30+ days)
- Archive tier for long-term retention (1+ year)

#### Azure Service Bus
**Purpose**: Asynchronous job queuing
**Queues**:
- `audio-processing-queue` - Main processing queue
- `deadletter-queue` - Failed message handling

### Infrastructure Layer

**Infrastructure as Code**: Bicep templates
**Deployment**: Azure Developer CLI (azd)
**Monitoring**: Application Insights + Log Analytics

**Security**:
- Azure Key Vault for secrets management
- Managed Identity for service-to-service auth
- Private endpoints for database connections
- Network Security Groups for traffic filtering

## Data Flow

### Upload Process
1. User selects video file in frontend
2. Frontend requests SAS token from API
3. API generates time-limited SAS token
4. Frontend uploads directly to Blob Storage
5. Frontend notifies API of successful upload
6. API creates job record in Cosmos DB
7. API sends message to Service Bus queue

### Processing Flow
1. Container App receives message from queue
2. Downloads video file using Managed Identity
3. Extracts audio track with FFmpeg
4. Processes audio through DeepFilterNet3 AI model
5. Merges cleaned audio back to video
6. Uploads result to processed container
7. Updates job status in Cosmos DB
8. Sends notification (if configured)

### Download Process
1. User requests processed file
2. API validates user permissions
3. API generates SAS token for download
4. Frontend receives secure download URL
5. User downloads directly from Blob Storage

## Scalability & Performance

### Auto-scaling
- **Container Apps**: Scale 0-30 instances based on queue depth
- **Static Web Apps**: Managed functions with automatic scaling
- **Storage**: Automatic scaling with geo-redundancy

### Performance Optimizations
- **CDN**: Static assets served via Azure CDN
- **Caching**: Redis cache for frequent API responses
- **Parallel Processing**: Multiple container instances
- **Blob Storage**: Parallel upload/download chunks

### Cost Optimization
- **Consumption-based**: Pay only for actual usage
- **Reserved Instances**: For predictable workloads
- **Storage Lifecycle**: Automatic tier management
- **Development Environment**: Scaled-down resources

## Security Architecture

### Authentication & Authorization
- **Frontend**: Azure AD B2C for user authentication
- **API**: JWT token validation
- **Services**: Managed Identity between Azure services
- **RBAC**: Role-based access control

### Data Protection
- **Encryption at Rest**: All storage encrypted
- **Encryption in Transit**: HTTPS/TLS everywhere
- **Key Management**: Azure Key Vault
- **Secrets**: No hardcoded credentials

### Network Security
- **Private Endpoints**: Database and storage access
- **VNet Integration**: Isolated network traffic
- **Firewall Rules**: Restrict public access
- **DDoS Protection**: Azure-native protection

## Monitoring & Observability

### Application Monitoring
- **Application Insights**: Performance and usage analytics
- **Custom Metrics**: Business-specific KPIs
- **Distributed Tracing**: End-to-end request tracking
- **Dependency Tracking**: External service monitoring

### Infrastructure Monitoring
- **Azure Monitor**: Infrastructure metrics
- **Log Analytics**: Centralized log aggregation
- **Alerts**: Proactive issue detection
- **Dashboards**: Real-time operational views

### Health Checks
- **Container Health**: Kubernetes liveness/readiness probes
- **Function Health**: Built-in monitoring
- **Storage Health**: Availability monitoring
- **End-to-End**: Synthetic transaction testing

## Disaster Recovery

### Backup Strategy
- **Database**: Automatic Cosmos DB backups
- **Storage**: Geo-redundant storage with versioning
- **Infrastructure**: Source-controlled Bicep templates
- **Configuration**: Environment variable backups

### Recovery Procedures
- **RTO**: Recovery Time Objective < 4 hours
- **RPO**: Recovery Point Objective < 15 minutes
- **Multi-Region**: Standby region for failover
- **Data Consistency**: Eventually consistent across regions

## Development Workflow

### Environments
- **Development**: Personal Azure subscriptions
- **Staging**: Shared environment for integration testing
- **Production**: Live user-facing environment

### CI/CD Pipeline
- **Source Control**: GitHub with branch protection
- **Build**: GitHub Actions for automated builds
- **Testing**: Unit tests, integration tests, security scans
- **Deployment**: Blue-green deployment with rollback capability

### Quality Gates
- **Code Quality**: SonarCloud analysis
- **Security**: OWASP dependency scanning
- **Performance**: Load testing with Azure Load Testing
- **Compliance**: Policy enforcement with Azure Policy

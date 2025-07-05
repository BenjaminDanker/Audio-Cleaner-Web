# 🎵 Audio Cleaner Pro - Complete Azure Video Denoising Web App

## 🏗️ Architecture Overview

This is a complete, production-ready Azure-based video denoising web application with the following architecture:

### Frontend (React + Vite)
- **Technology**: React 18, Vite, Modern CSS
- **Features**: User authentication, file upload, job tracking, subscription management
- **Hosting**: Azure Static Web Apps
- **Auth**: Azure AD B2C integration

### API Layer (Azure Functions)
- **Technology**: Node.js, Azure Functions v4
- **Endpoints**: User subscriptions, job queueing, Stripe payments, file downloads
- **Authentication**: JWT validation with Azure AD B2C
- **Integration**: Cosmos DB, Storage, Service Bus

### AI Processing Service (Python Container)
- **Technology**: Python 3.11, DeepFilterNet3, OpenCV
- **Hosting**: Azure Container Apps
- **Processing**: Async video denoising with Service Bus queue
- **Storage**: Azure Blob Storage for input/output files

### Infrastructure
- **IaC**: Bicep templates with Azure Developer CLI
- **Data**: Cosmos DB for metadata, Azure Storage for files
- **Messaging**: Service Bus for job queuing
- **Security**: Managed Identity, RBAC, secure connections
- **CI/CD**: GitHub Actions workflows

## 📁 Project Structure

```
audio-cleaner-web/
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/           # Route pages
│   │   └── ...
│   ├── package.json
│   └── staticwebapp.config.json
├── api/                     # Azure Functions API
│   ├── get-subscription/    # User subscription endpoint
│   ├── enqueue-job/        # Job creation endpoint
│   ├── job-status/         # Job status tracking
│   ├── create-checkout-session/ # Stripe checkout
│   ├── webhook-stripe/     # Stripe webhooks
│   ├── download/           # File download endpoint
│   ├── create-portal-session/ # Stripe portal
│   └── utils/              # Shared utilities
├── infra/                  # Infrastructure as Code
│   ├── main.bicep          # Main Bicep template
│   ├── main.parameters.json # Deployment parameters
│   └── abbreviations.json  # Resource naming
├── scripts/                # Development scripts
│   ├── start-dev.ps1      # Local development setup
│   ├── stop-dev.ps1       # Stop local services
│   └── deploy-check.ps1   # Deployment verification
├── .github/workflows/      # CI/CD pipelines
│   ├── deploy.yml         # Main deployment workflow
│   └── docker.yml         # Container build workflow
├── models/DeepFilterNet3/  # AI model files
├── Dockerfile             # AI service container
├── processor_main.py      # Service Bus processor
├── processor_app.py       # Health check endpoint
├── azure.yaml            # Azure Developer CLI config
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- Docker Desktop
- Azure CLI
- Azure Developer CLI (azd)

### 1. Local Development
```powershell
# Start all local services
./scripts/start-dev.ps1

# Access the app
# Frontend: http://localhost:5173
# API: http://localhost:7071
```

### 2. Deploy to Azure
```powershell
# Verify deployment readiness
./scripts/deploy-check.ps1

# Login to Azure
az login

# Deploy everything
azd up
```

## 🔧 Configuration

### Environment Variables (.env.local)
```bash
# Azure Storage (use Azurite for local)
AZURE_STORAGE_CONNECTION_STRING=...

# Azure Service Bus
AZURE_SERVICE_BUS_CONNECTION_STRING=...

# Azure Cosmos DB (use emulator for local)
AZURE_COSMOS_CONNECTION_STRING=...

# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Azure AD B2C
AZURE_CLIENT_ID=...
AZURE_TENANT_ID=...
```

### Post-Deployment Setup
1. **Stripe Configuration**:
   - Set up webhook endpoints pointing to your Azure Functions
   - Configure product/price IDs
   - Update environment variables

2. **Azure AD B2C** (Optional for MVP):
   - Create B2C tenant
   - Configure application registration
   - Set up user flows

## 🎯 Features

### ✅ Implemented
- **Video Upload**: Drag & drop, progress tracking
- **AI Processing**: DeepFilterNet3 noise reduction
- **Job Management**: Status tracking, download links
- **User System**: Authentication, subscriptions
- **Payment Processing**: Stripe integration
- **Infrastructure**: Complete Azure deployment
- **CI/CD**: Automated build and deployment

### 🔄 Processing Flow
1. User uploads video → Azure Storage
2. Job queued → Service Bus
3. Container App processes → AI denoising
4. Result stored → Azure Storage
5. User notified → Download available

## 🛠️ Development

### Local Testing
```powershell
# Start Azurite (Azure Storage Emulator)
azurite --location c:\azurite

# Start Cosmos DB Emulator
# (Install from Microsoft)

# Start frontend
cd frontend && npm run dev

# Start API
cd api && npm run dev

# Run processor locally
python processor_app.py
```

### Build Container
```powershell
docker build -t audio-cleaner-processor .
docker run -e AZURE_STORAGE_CONNECTION_STRING="..." audio-cleaner-processor
```

## 📊 Monitoring & Logs

- **Application Insights**: Automatic telemetry
- **Container Logs**: Azure Monitor
- **Function Logs**: Azure Functions monitoring
- **Storage Metrics**: Azure Storage analytics

## 🔐 Security

- **Managed Identity**: Secure Azure resource access
- **RBAC**: Least privilege access
- **HTTPS**: All communications encrypted
- **Input Validation**: File type and size limits
- **Authentication**: Azure AD B2C integration

## 💰 Cost Optimization

- **Container Apps**: Pay-per-use scaling
- **Function Apps**: Consumption plan
- **Storage**: Hot/Cool tier management
- **Service Bus**: Standard tier for development

## 🚦 Status

### ✅ Complete
- Frontend React application with modern UI
- Azure Functions API with all endpoints
- Python AI processing service with Docker
- Complete Bicep infrastructure templates
- GitHub Actions CI/CD workflows
- Local development environment
- Deployment verification scripts

### 🔄 Next Steps
- Configure Stripe webhooks post-deployment
- Set up custom domain (optional)
- Configure monitoring alerts
- Load testing and optimization
- User acceptance testing

## 📚 Resources

- [Azure Developer CLI Docs](https://docs.microsoft.com/en-us/azure/developer/azure-developer-cli/)
- [Azure Static Web Apps](https://docs.microsoft.com/en-us/azure/static-web-apps/)
- [Azure Container Apps](https://docs.microsoft.com/en-us/azure/container-apps/)
- [DeepFilterNet Documentation](https://github.com/Rikorose/DeepFilterNet)

---

**Ready to deploy!** Run `azd up` to get started with your Azure video denoising application.

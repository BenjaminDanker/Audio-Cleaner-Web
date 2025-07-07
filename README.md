# Audio Cleaner Pro - Azure Video Denoising Web App

> **A complete Azure-based video denoising application using AI to clean audio tracks from video files.**

## 🏗️ Architecture

- **Frontend**: React + Vite (Azure Static Web Apps)
- **API**: Azure Functions (Node.js)
- **Processing**: Python Container App with DeepFilterNet3
- **Storage**: Azure Blob Storage, Cosmos DB
- **Messaging**: Azure Service Bus
- **Infrastructure**: Bicep templates with Azure Developer CLI

## 🚀 Quick Deploy to Azure

### Prerequisites
1. **Azure subscription** with Owner or Contributor access
2. **Azure Developer CLI** (`azd`) - [Install guide](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
3. **Azure CLI** - [Install guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
4. **Git** and **GitHub account**

### Deploy Steps

1. **Clone and setup**:
```bash
git clone <your-repo-url>
cd Audio-Cleaner-Web
```

2. **Login to Azure**:
```bash
azd auth login
az login
```

3. **Initialize environment**:
```bash
azd init
# Follow prompts to set environment name and region
```

4. **Deploy to Azure**:
```bash
azd up
```

This will:
- ✅ Provision all Azure resources using Bicep
- ✅ Build and deploy the React frontend
- ✅ Deploy Azure Functions API
- ✅ Build and deploy the Python processing container
- ✅ Configure networking, security, and monitoring

### Post-deployment
- Frontend URL: Check `azd show` output
- API URL: Check Azure Functions app in portal
- Monitor logs: `azd logs`

## 🛠️ Local Development

See [docs/LOCAL_DEV_GUIDE.md](docs/LOCAL_DEV_GUIDE.md) for detailed local development setup.

### Quick Start
```bash
# Frontend
cd frontend && npm install && npm run dev

# API
cd api && npm install && func start

# Processor (for testing)
pip install -r requirements.txt
python local-dev/local_processor.py
```

## 📚 Documentation

- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Local Development Guide](docs/LOCAL_DEV_GUIDE.md)
- [Deployment Secrets](docs/DEPLOYMENT_SECRETS.md)
- [Azure CLI Commands](docs/AZURE_CLI_COMMANDS.md)

## 🔧 Configuration

Key environment variables:
- `AZURE_ENV_NAME`: Your environment name
- `AZURE_LOCATION`: Azure region (e.g., eastus)
- `AZURE_SUBSCRIPTION_ID`: Your Azure subscription ID

## 🎯 Features

- ✅ Video upload with progress tracking
- ✅ AI-powered audio denoising using DeepFilterNet3
- ✅ Real-time job status monitoring
- ✅ User authentication and subscription management
- ✅ Stripe payment integration
- ✅ Fully scalable Azure infrastructure
- ✅ CI/CD with GitHub Actions

## 🔒 Security

- Managed Identity authentication
- RBAC for resource access
- Secure secrets management with Key Vault
- HTTPS everywhere with proper CORS

## 📊 Monitoring

- Application Insights for telemetry
- Log Analytics for centralized logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

#### Start Azurite (Azure Storage Emulator)
```bash
azurite --silent --location c:\azurite --debug c:\azurite\debug.log
```

#### Start Cosmos DB Emulator
Start the Cosmos DB Emulator application from Windows Start Menu

#### Start Service Bus (Optional for full local dev)
For Service Bus, you can either:
- Use a development Service Bus namespace in Azure
- Use a Service Bus emulator (experimental)

### 5. Frontend Development
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at `http://localhost:5173`

### 6. API Development
```bash
cd api
npm install
npm run dev
```
The API will be available at `http://localhost:7071`

### 7. AI Processing Service Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the processor locally (for testing)
python processor/src/processor_app.py
```

### 8. Docker Development
```bash
# Build the processing container
docker build -t audio-cleaner-processor .

# Run the container locally
docker run -e AZURE_STORAGE_CONNECTION_STRING="..." -e AZURE_SERVICE_BUS_CONNECTION_STRING="..." audio-cleaner-processor
```

## Testing the Application

1. Start all local services (Azurite, Cosmos DB Emulator)
2. Start the API: `cd api && npm run dev`
3. Start the frontend: `cd frontend && npm run dev`
4. Navigate to `http://localhost:5173`
5. Test video upload and processing

## Deployment

### Deploy Infrastructure
```bash
# Login to Azure
az login

# Initialize Azure Developer CLI
azd init

# Deploy infrastructure and application
azd up
```

### Manual Deployment Steps
```bash
# Deploy infrastructure
az deployment group create --resource-group rg-audio-cleaner --template-file infra/main.bicep --parameters @infra/main.parameters.json

# Deploy Static Web App
cd frontend
npm run build
# Use Azure CLI or GitHub Actions to deploy

# Deploy Functions
cd api
func azure functionapp publish your-function-app-name

# Deploy Container App
az containerapp update --name your-container-app --resource-group rg-audio-cleaner --image your-registry.azurecr.io/audio-cleaner:latest
```

## Environment Variables Reference

### Required for Local Development
- `AZURE_STORAGE_CONNECTION_STRING`: Connection to Azurite or Azure Storage
- `AZURE_SERVICE_BUS_CONNECTION_STRING`: Connection to Service Bus
- `COSMOS_CONNECTION_STRING`: Connection to Cosmos DB Emulator or Azure Cosmos DB
- `STRIPE_SECRET_KEY`: Your Stripe secret key (test mode for development)

### Required for Production
- All above variables pointing to production Azure resources
- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`: Azure AD B2C configuration
- `STRIPE_WEBHOOK_SECRET`: Stripe webhook endpoint secret

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the API is running with `--cors true` flag
2. **Storage Errors**: Check if Azurite is running and accessible
3. **Auth Errors**: Verify Azure AD B2C configuration
4. **Stripe Errors**: Check Stripe test keys are correctly configured

### Debugging

- Check Azure Functions logs: `func host start --verbose`
- Check browser network tab for API call failures
- Check Cosmos DB data explorer for data issues
- Check Docker logs: `docker logs container-name`

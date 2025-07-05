# Local Development Setup Guide

## Prerequisites

1. **Node.js** (v18 or later)
2. **Python** (3.11 or later)
3. **Azure Functions Core Tools**
4. **Docker Desktop**
5. **Azure CLI**

## Local Development Environment Setup

### 1. Install Azure Storage Emulator (Azurite)
```bash
npm install -g azurite
```

### 2. Install Azure Cosmos DB Emulator
Download and install from: https://docs.microsoft.com/en-us/azure/cosmos-db/local-emulator

### 3. Setup Environment Variables
```bash
# Copy the example environment file
cp .env.local.example .env.local

# Edit .env.local with your local values
# For development, you can use the provided Azurite and Cosmos DB emulator connection strings
```

### 4. Start Local Services

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
python processor_app.py
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
- `AZURE_COSMOS_CONNECTION_STRING`: Connection to Cosmos DB Emulator or Azure Cosmos DB
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

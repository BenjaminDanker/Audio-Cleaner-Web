# Azure CLI Commands for Infrastructure Deployment

## Create Resource Group
```bash
az group create --name rg-audio-cleaner-dev --location eastus
```

## Create Storage Account and Container
```bash
# Create storage account
az storage account create \
    --name studiocleanerdev \
    --resource-group rg-audio-cleaner-dev \
    --location eastus \
    --sku Standard_LRS \
    --kind StorageV2

# Create blob container for audio files
az storage container create \
    --name audio-files \
    --account-name studiocleanerdev \
    --auth-mode login
```

## Create Service Bus Namespace and Queue
```bash
# Create Service Bus namespace
az servicebus namespace create \
    --name sb-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --location eastus \
    --sku Standard

# Create queue
az servicebus queue create \
    --name audio-processing-queue \
    --namespace-name sb-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --max-size 1024
```

## Create Cosmos DB Account
```bash
# Create Cosmos DB account with free tier
az cosmosdb create \
    --name cosmos-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --locations regionName=eastus \
    --enable-free-tier true \
    --default-consistency-level Session

# Create database
az cosmosdb sql database create \
    --account-name cosmos-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --name audiocleaner

# Create containers
az cosmosdb sql container create \
    --account-name cosmos-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --database-name audiocleaner \
    --name subscriptions \
    --partition-key-path "/id" \
    --throughput 400

az cosmosdb sql container create \
    --account-name cosmos-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --database-name audiocleaner \
    --name jobs \
    --partition-key-path "/id" \
    --throughput 400
```

## Create Function App
```bash
# Create function app
az functionapp create \
    --name func-audio-cleaner-dev \
    --storage-account studiocleanerdev \
    --resource-group rg-audio-cleaner-dev \
    --consumption-plan-location eastus \
    --runtime node \
    --runtime-version 18 \
    --functions-version 4
```

## Create Static Web App
```bash
# Create static web app (requires GitHub repo)
az staticwebapp create \
    --name swa-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --source https://github.com/your-username/audio-cleaner-web \
    --location eastus2 \
    --branch main \
    --app-location "/frontend" \
    --api-location "/api" \
    --output-location "dist"
```

## Create Container Registry
```bash
# Create container registry
az acr create \
    --name craudiocleanerdev \
    --resource-group rg-audio-cleaner-dev \
    --sku Basic \
    --admin-enabled true
```

## Create Container Instance for AI Processing
```bash
# Get container registry credentials
ACR_PASSWORD=$(az acr credential show --name craudiocleanerdev --query passwords[0].value -o tsv)

# Create container instance
az container create \
    --name aci-audio-processor \
    --resource-group rg-audio-cleaner-dev \
    --image craudiocleanerdev.azurecr.io/audio-cleaner-processor:latest \
    --registry-login-server craudiocleanerdev.azurecr.io \
    --registry-username craudiocleanerdev \
    --registry-password $ACR_PASSWORD \
    --cpu 2 \
    --memory 4 \
    --environment-variables \
        AZURE_STORAGE_CONNECTION_STRING="$(az storage account show-connection-string --name studiocleanerdev --resource-group rg-audio-cleaner-dev --query connectionString -o tsv)" \
        AZURE_SERVICE_BUS_CONNECTION_STRING="$(az servicebus namespace authorization-rule keys list --namespace-name sb-audio-cleaner-dev --resource-group rg-audio-cleaner-dev --name RootManageSharedAccessKey --query primaryConnectionString -o tsv)" \
        COSMOS_CONNECTION_STRING="$(az cosmosdb keys list --name cosmos-audio-cleaner-dev --resource-group rg-audio-cleaner-dev --type connection-strings --query connectionStrings[0].connectionString -o tsv)"
```

## Configure CORS for Function App
```bash
az functionapp cors add \
    --name func-audio-cleaner-dev \
    --resource-group rg-audio-cleaner-dev \
    --allowed-origins https://your-static-web-app.azurestaticapps.net
```

## Build and Push Docker Image
```bash
# Build image
docker build -t audio-cleaner-processor .

# Tag for ACR
docker tag audio-cleaner-processor craudiocleanerdev.azurecr.io/audio-cleaner-processor:latest

# Login to ACR
az acr login --name craudiocleanerdev

# Push image
docker push craudiocleanerdev.azurecr.io/audio-cleaner-processor:latest
```

## Alternative: Use Azure Developer CLI (AZD)
```bash
# Initialize project
azd init --template azure-functions-typescript

# Deploy infrastructure and code
azd up
```

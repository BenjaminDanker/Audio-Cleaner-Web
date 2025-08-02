# PowerShell script to extract Azure connection strings for local development
# This script retrieves connection strings from your deployed Azure infrastructure

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroupName = "rg-audioclean-4a95209e"
)

Write-Host "🔍 Retrieving Azure connection strings for local development..." -ForegroundColor Green

# Get script directory and set up paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$terraformPath = Join-Path $rootDir "terraform"
$apiPath = Join-Path $rootDir "api"
$processorPath = Join-Path $rootDir "processor"

Write-Host "� Script directory: $scriptDir" -ForegroundColor Gray
Write-Host "📁 Root directory: $rootDir" -ForegroundColor Gray
Write-Host "📁 Terraform path: $terraformPath" -ForegroundColor Gray

# Check if Azure CLI is installed and user is logged in
try {
    $account = az account show 2>$null | ConvertFrom-Json
    if (-not $account) {
        Write-Host "❌ Please login to Azure CLI first: az login" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Logged in as: $($account.user.name)" -ForegroundColor Green
} catch {
    Write-Host "❌ Azure CLI not found. Please install Azure CLI first." -ForegroundColor Red
    exit 1
}

# Initialize connection string variables
$storageConnectionString = $null
$cosmosConnectionString = $null
$serviceBusConnectionString = $null

# Try to get connection strings from Terraform outputs first
if (Test-Path $terraformPath) {
    Write-Host "📁 Found Terraform directory, attempting to get outputs..." -ForegroundColor Yellow
    
    Push-Location $terraformPath
    try {
        $terraformOutput = terraform output -json 2>$null
        if ($terraformOutput) {
            $outputs = $terraformOutput | ConvertFrom-Json
            Write-Host "✅ Successfully retrieved Terraform outputs" -ForegroundColor Green
            
            $storageConnectionString = $outputs.storage_connection_string.value
            $cosmosConnectionString = $outputs.cosmos_connection_string.value
            $serviceBusConnectionString = $outputs.servicebus_connection_string.value
            
            Write-Host "✅ Retrieved connection strings from Terraform" -ForegroundColor Green
        } else {
            Write-Host "⚠️  No Terraform outputs found, falling back to Azure CLI..." -ForegroundColor Yellow
            throw "No terraform outputs"
        }
    } catch {
        Write-Host "⚠️  Failed to get Terraform outputs: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "🔄 Falling back to Azure CLI resource queries..." -ForegroundColor Yellow
        
        # Fallback to Azure CLI resource queries
        try {
            $resources = az resource list --resource-group $ResourceGroupName --output json | ConvertFrom-Json
            
            # Find storage account
            $storageAccount = $resources | Where-Object { $_.type -eq "Microsoft.Storage/storageAccounts" }
            if ($storageAccount) {
                $storageConnectionString = (az storage account show-connection-string --name $storageAccount.name --resource-group $ResourceGroupName --output json | ConvertFrom-Json).connectionString
                Write-Host "✅ Retrieved storage connection string from Azure CLI" -ForegroundColor Green
            }
            
            # Find Cosmos DB
            $cosmosAccount = $resources | Where-Object { $_.type -eq "Microsoft.DocumentDB/databaseAccounts" }
            if ($cosmosAccount) {
                $cosmosKeys = az cosmosdb keys list --name $cosmosAccount.name --resource-group $ResourceGroupName --output json | ConvertFrom-Json
                $cosmosConnectionString = "AccountEndpoint=https://$($cosmosAccount.name).documents.azure.com:443/;AccountKey=$($cosmosKeys.primaryMasterKey);"
                Write-Host "✅ Retrieved Cosmos DB connection string from Azure CLI" -ForegroundColor Green
            }
            
            # Find Service Bus
            $serviceBusNamespace = $resources | Where-Object { $_.type -eq "Microsoft.ServiceBus/namespaces" }
            if ($serviceBusNamespace) {
                $serviceBusConnectionString = (az servicebus namespace authorization-rule keys list --resource-group $ResourceGroupName --namespace-name $serviceBusNamespace.name --name RootManageSharedAccessKey --output json | ConvertFrom-Json).primaryConnectionString
                Write-Host "✅ Retrieved Service Bus connection string from Azure CLI" -ForegroundColor Green
            }
        } catch {
            Write-Host "❌ Failed to retrieve connection strings from Azure CLI: $($_.Exception.Message)" -ForegroundColor Red
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "📁 No Terraform directory found at $terraformPath" -ForegroundColor Yellow
    Write-Host "🔄 Using Azure CLI for resource discovery..." -ForegroundColor Yellow
    
    # Direct Azure CLI implementation
    try {
        $resources = az resource list --resource-group $ResourceGroupName --output json | ConvertFrom-Json
        Write-Host "✅ Found $($resources.Count) resources in resource group" -ForegroundColor Green
        
        # Continue with Azure CLI logic...
    } catch {
        Write-Host "❌ Failed to list resources: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

# Validate that we got the connection strings
if (-not $storageConnectionString -or -not $cosmosConnectionString -or -not $serviceBusConnectionString) {
    Write-Host "❌ Failed to retrieve all required connection strings" -ForegroundColor Red
    Write-Host "   Storage: $(if ($storageConnectionString) { '✅' } else { '❌' })" -ForegroundColor White
    Write-Host "   Cosmos: $(if ($cosmosConnectionString) { '✅' } else { '❌' })" -ForegroundColor White
    Write-Host "   Service Bus: $(if ($serviceBusConnectionString) { '✅' } else { '❌' })" -ForegroundColor White
    exit 1
}

# Create local development configuration
$localDevConfig = @{
    "FUNCTIONS_WORKER_RUNTIME" = "node"
    "AzureWebJobsSecretStorageType" = "Files"
    "AZURE_STORAGE_CONNECTION_STRING" = $storageConnectionString
    "COSMOS_CONNECTION_STRING" = $cosmosConnectionString
    "AZURE_SERVICE_BUS_CONNECTION_STRING" = $serviceBusConnectionString
    "STRIPE_SECRET_KEY" = "sk_test_your_stripe_test_key_here"
    "STRIPE_WEBHOOK_SECRET" = "whsec_your_webhook_secret_here"
    "FRONTEND_URL" = "http://localhost:5173"
    "LOCAL_DEV_MODE" = "false"  # Using cloud services
    # Fetch Azure account info dynamically
    "AZURE_CLIENT_ID" = (az account show --query "user.name" -o tsv)
    "AZURE_TENANT_ID" = (az account show --query "tenantId" -o tsv)
}

# Create the local.settings.cloud.json file
$localSettingsCloud = @{
    "IsEncrypted" = $false
    "Values" = $localDevConfig
    "Host" = @{
        "CORS" = "*"
        "CORSCredentials" = $false
    }
}

# Ensure directories exist
if (-not (Test-Path $apiPath)) {
    New-Item -ItemType Directory -Path $apiPath -Force | Out-Null
}
if (-not (Test-Path $processorPath)) {
    New-Item -ItemType Directory -Path $processorPath -Force | Out-Null
}

$outputPath = Join-Path $apiPath "local.settings.cloud.json"
$localSettingsCloud | ConvertTo-Json -Depth 3 | Out-File -FilePath $outputPath -Encoding UTF8

Write-Host "✅ Created $outputPath with cloud connection strings" -ForegroundColor Green

# Create .env file for Docker processor
$envContent = @"
# Cloud Azure Services for Local Development
AZURE_STORAGE_CONNECTION_STRING=$storageConnectionString
COSMOS_CONNECTION_STRING=$cosmosConnectionString
AZURE_SERVICE_BUS_CONNECTION_STRING=$serviceBusConnectionString
LOCAL_DEV_MODE=false
"@

$envPath = Join-Path $processorPath ".env.cloud"
$envContent | Out-File -FilePath $envPath -Encoding UTF8

Write-Host "✅ Created $envPath for Docker processor" -ForegroundColor Green

# Display summary
Write-Host "`n📋 Configuration Summary:" -ForegroundColor Cyan
Write-Host "  Storage Account: " -NoNewline; Write-Host ($storageConnectionString -replace "AccountKey=[^;]*", "AccountKey=***") -ForegroundColor Yellow
Write-Host "  Cosmos DB: " -NoNewline; Write-Host ($cosmosConnectionString -replace "AccountKey=[^;]*", "AccountKey=***") -ForegroundColor Yellow
Write-Host "  Service Bus: " -NoNewline; Write-Host ($serviceBusConnectionString -replace "SharedAccessKey=[^;]*", "SharedAccessKey=***") -ForegroundColor Yellow

Write-Host "`n🚀 Next steps:" -ForegroundColor Green
Write-Host "  1. Run: .\scripts\switch-dev-mode.ps1 -Mode cloud" -ForegroundColor White
Write-Host "  2. Run: docker-compose -f docker-compose.dev.yml up processor" -ForegroundColor White
Write-Host "  3. Run: swa start" -ForegroundColor White

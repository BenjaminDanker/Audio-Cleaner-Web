# PowerShell script to switch between local and cloud development configurations
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("local", "cloud")]
    [string]$Mode
)

# Get script directory and set up paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$apiPath = Join-Path $rootDir "api"

Write-Host "🔄 Switching to $Mode development mode..." -ForegroundColor Green

switch ($Mode) {
    "local" {
        # Copy local development settings
        $localSettingsLocal = Join-Path $apiPath "local.settings.local.json"
        $localSettingsActive = Join-Path $apiPath "local.settings.json"
        
        if (Test-Path $localSettingsLocal) {
            Copy-Item $localSettingsLocal $localSettingsActive -Force
            Write-Host "✅ Switched to local development (Azurite/local services)" -ForegroundColor Green
        } else {
            Write-Host "⚠️  local.settings.local.json not found, creating default local config..." -ForegroundColor Yellow
            
            $localConfig = @{
                "IsEncrypted" = $false
                "Values" = @{
                    "FUNCTIONS_WORKER_RUNTIME" = "node"
                    "AzureWebJobsSecretStorageType" = "Files"
                    "AZURE_STORAGE_CONNECTION_STRING" = "UseDevelopmentStorage=true"
                    "COSMOS_CONNECTION_STRING" = "file-based-for-local-dev"
                    "AZURE_SERVICE_BUS_CONNECTION_STRING" = "file-based-for-local-dev"
                    "STRIPE_SECRET_KEY" = "sk_test_your_stripe_test_key_here"
                    "STRIPE_WEBHOOK_SECRET" = "whsec_your_webhook_secret_here"
                    "FRONTEND_URL" = "http://localhost:5173"
                    "LOCAL_DEV_MODE" = "true"
                    # Fetch Azure account info dynamically
                    "AZURE_CLIENT_ID" = (az account show --query "user.name" -o tsv)
                    "AZURE_TENANT_ID" = (az account show --query "tenantId" -o tsv)
                }
                "Host" = @{
                    "CORS" = "*"
                    "CORSCredentials" = $false
                }
            }
            
            $localConfig | ConvertTo-Json -Depth 3 | Out-File -FilePath $localSettingsActive -Encoding UTF8
            $localConfig | ConvertTo-Json -Depth 3 | Out-File -FilePath $localSettingsLocal -Encoding UTF8
            Write-Host "✅ Created local development configuration" -ForegroundColor Green
        }
        
        Write-Host "`n🔧 Local development setup:" -ForegroundColor Cyan
        Write-Host "  • API Functions: Uses Azurite (local storage emulator)" -ForegroundColor White
        Write-Host "  • Cosmos DB: File-based simulation" -ForegroundColor White
        Write-Host "  • Service Bus: File-based simulation" -ForegroundColor White
        Write-Host "  • Run: swa start" -ForegroundColor Yellow
    }
    
    "cloud" {
        # Copy cloud development settings
        $localSettingsCloud = Join-Path $apiPath "local.settings.cloud.json"
        $localSettingsActive = Join-Path $apiPath "local.settings.json"
        
        if (Test-Path $localSettingsCloud) {
            Copy-Item $localSettingsCloud $localSettingsActive -Force
            Write-Host "✅ Switched to cloud development (Azure services)" -ForegroundColor Green
            
            Write-Host "`n🔧 Cloud development setup:" -ForegroundColor Cyan
            Write-Host "  • API Functions: Uses Azure Storage" -ForegroundColor White
            Write-Host "  • Cosmos DB: Uses Azure Cosmos DB" -ForegroundColor White
            Write-Host "  • Service Bus: Uses Azure Service Bus" -ForegroundColor White
            Write-Host "  • Processor: docker-compose -f docker-compose.dev.yml up processor" -ForegroundColor Yellow
            Write-Host "  • Frontend + API: swa start" -ForegroundColor Yellow
        } else {
            Write-Host "❌ cloud configuration not found!" -ForegroundColor Red
            Write-Host "Please run: .\scripts\get-cloud-config.ps1 first" -ForegroundColor Yellow
            exit 1
        }
    }
}

Write-Host "`n📝 Current configuration summary:" -ForegroundColor Green
if (Test-Path $localSettingsActive) {
    $config = Get-Content $localSettingsActive | ConvertFrom-Json
    $isLocal = $config.Values.LOCAL_DEV_MODE -eq "true"
    $storageMode = if ($isLocal) { "Local (Azurite)" } else { "Cloud (Azure)" }
    $cosmosMode = if ($config.Values.COSMOS_CONNECTION_STRING -eq "file-based-for-local-dev") { "Local (File-based)" } else { "Cloud (Azure)" }
    
    Write-Host "  Storage: $storageMode" -ForegroundColor White
    Write-Host "  Cosmos: $cosmosMode" -ForegroundColor White
    Write-Host "  Local Dev Mode: $($config.Values.LOCAL_DEV_MODE)" -ForegroundColor White
}

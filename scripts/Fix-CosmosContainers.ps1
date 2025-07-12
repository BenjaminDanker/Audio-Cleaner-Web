#!/usr/bin/env pwsh
<#
.SYNOPSIS
    CRITICAL FIX: Creates missing Cosmos DB containers to stop massive log costs

.DESCRIPTION
    This script creates the missing 'ratelimits' and 'securityevents' containers
    in Cosmos DB that are causing 14.5KB error logs for every SecurityMiddleware call.
    
    Root cause of $100/hour log costs:
    - Missing containers cause "Resource Not Found" errors
    - Each error generates 14.5KB logs with full Cosmos DB diagnostic data
    - Multiple functions × multiple calls = massive log explosion

.PARAMETER SubscriptionId
    Azure subscription ID (optional, will prompt if not provided)

.PARAMETER ResourceGroupName  
    Resource group name (optional, will auto-detect from azd)

.PARAMETER CosmosAccountName
    Cosmos DB account name (optional, will auto-detect from azd)

.EXAMPLE
    .\Fix-CosmosContainers.ps1
    Auto-detects all parameters from azd environment

.EXAMPLE
    .\Fix-CosmosContainers.ps1 -SubscriptionId "your-sub-id" -ResourceGroupName "rg-audiocleaner" -CosmosAccountName "cosmos-audiocleaner"
    Uses specific parameters
#>

param(
    [string]$SubscriptionId,
    [string]$ResourceGroupName, 
    [string]$CosmosAccountName
)

Write-Host "🔥 CRITICAL FIX: Creating missing Cosmos DB containers to stop log cost explosion" -ForegroundColor Red
Write-Host "💰 This should immediately reduce your $100/hour log costs" -ForegroundColor Yellow

# Auto-detect parameters from azd if not provided
if (-not $SubscriptionId -or -not $ResourceGroupName -or -not $CosmosAccountName) {
    Write-Host "🔍 Auto-detecting parameters from azd environment..." -ForegroundColor Cyan
    
    try {
        # Get azd environment info
        $azdEnv = azd env list --output json | ConvertFrom-Json
        if ($azdEnv.Length -eq 0) {
            throw "No azd environments found"
        }
        
        $envName = $azdEnv[0].Name
        Write-Host "📋 Using azd environment: $envName" -ForegroundColor Green
        
        # Get environment variables
        $envVars = azd env get-values --environment $envName --output json | ConvertFrom-Json
        
        if (-not $SubscriptionId) {
            $SubscriptionId = $envVars.AZURE_SUBSCRIPTION_ID
        }
        
        if (-not $ResourceGroupName) {
            $ResourceGroupName = $envVars.AZURE_RESOURCE_GROUP
        }
        
        if (-not $CosmosAccountName) {
            # Extract from Cosmos connection string
            $cosmosConnection = $envVars.COSMOS_CONNECTION_STRING
            if ($cosmosConnection -match "AccountEndpoint=https://([^.]+)\.") {
                $CosmosAccountName = $Matches[1]
            }
        }
        
        Write-Host "✅ Auto-detected parameters:" -ForegroundColor Green
        Write-Host "  Subscription: $SubscriptionId" -ForegroundColor Gray
        Write-Host "  Resource Group: $ResourceGroupName" -ForegroundColor Gray  
        Write-Host "  Cosmos Account: $CosmosAccountName" -ForegroundColor Gray
        
    } catch {
        Write-Host "❌ Failed to auto-detect parameters from azd: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please provide parameters manually or run 'azd env list' to check your environment" -ForegroundColor Yellow
        exit 1
    }
}

# Validate required parameters
if (-not $SubscriptionId -or -not $ResourceGroupName -or -not $CosmosAccountName) {
    Write-Host "❌ Missing required parameters:" -ForegroundColor Red
    Write-Host "  SubscriptionId: $SubscriptionId" -ForegroundColor Gray
    Write-Host "  ResourceGroupName: $ResourceGroupName" -ForegroundColor Gray
    Write-Host "  CosmosAccountName: $CosmosAccountName" -ForegroundColor Gray
    exit 1
}

# Set Azure subscription
Write-Host "🔧 Setting Azure subscription..." -ForegroundColor Cyan
az account set --subscription $SubscriptionId

# Check if Cosmos account exists
Write-Host "🔍 Checking Cosmos DB account..." -ForegroundColor Cyan
$cosmosExists = az cosmosdb show --name $CosmosAccountName --resource-group $ResourceGroupName 2>$null
if (-not $cosmosExists) {
    Write-Host "❌ Cosmos DB account '$CosmosAccountName' not found in resource group '$ResourceGroupName'" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Found Cosmos DB account: $CosmosAccountName" -ForegroundColor Green

# Check if database exists
Write-Host "🔍 Checking if 'audiocleaner' database exists..." -ForegroundColor Cyan
$dbExists = az cosmosdb sql database show --account-name $CosmosAccountName --resource-group $ResourceGroupName --name "audiocleaner" 2>$null
if (-not $dbExists) {
    Write-Host "📦 Creating 'audiocleaner' database..." -ForegroundColor Yellow
    az cosmosdb sql database create --account-name $CosmosAccountName --resource-group $ResourceGroupName --name "audiocleaner"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to create database" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Created 'audiocleaner' database" -ForegroundColor Green
} else {
    Write-Host "✅ 'audiocleaner' database already exists" -ForegroundColor Green
}

# Create the missing containers that are causing the log explosion
$containers = @(
    @{
        Name = "ratelimits"
        PartitionKey = "/clientIP"
        Description = "Rate limiting data - prevents massive SecurityMiddleware errors"
    },
    @{
        Name = "securityevents" 
        PartitionKey = "/eventType"
        Description = "Security events - prevents massive SecurityMiddleware errors"
    }
)

foreach ($container in $containers) {
    Write-Host "🔍 Checking container '$($container.Name)'..." -ForegroundColor Cyan
    
    $containerExists = az cosmosdb sql container show --account-name $CosmosAccountName --resource-group $ResourceGroupName --database-name "audiocleaner" --name $container.Name 2>$null
    
    if (-not $containerExists) {
        Write-Host "🛠️  Creating container '$($container.Name)' - $($container.Description)" -ForegroundColor Yellow
        
        az cosmosdb sql container create `
            --account-name $CosmosAccountName `
            --resource-group $ResourceGroupName `
            --database-name "audiocleaner" `
            --name $container.Name `
            --partition-key-path $container.PartitionKey
            
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to create container '$($container.Name)'" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✅ Created container '$($container.Name)'" -ForegroundColor Green
    } else {
        Write-Host "✅ Container '$($container.Name)' already exists" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🎉 SUCCESS: All missing Cosmos DB containers have been created!" -ForegroundColor Green
Write-Host ""
Write-Host "💰 COST SAVINGS:" -ForegroundColor Yellow
Write-Host "  • No more 'Resource Not Found' errors from SecurityMiddleware" -ForegroundColor Gray
Write-Host "  • No more 14.5KB error logs per function call" -ForegroundColor Gray  
Write-Host "  • Should reduce log costs from $100/hour to normal levels" -ForegroundColor Gray
Write-Host ""
Write-Host "🚀 NEXT STEPS:" -ForegroundColor Cyan
Write-Host "  1. Deploy the updated function code with reduced logging" -ForegroundColor Gray
Write-Host "  2. Monitor log costs in Azure portal" -ForegroundColor Gray
Write-Host "  3. Check Log Analytics workspace for reduced data ingestion" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠️  NOTE: It may take 5-10 minutes for changes to take effect" -ForegroundColor Yellow

# Display container information
Write-Host "📋 Created Containers:" -ForegroundColor Cyan
foreach ($container in $containers) {
    Write-Host "  • $($container.Name) (Partition: $($container.PartitionKey))" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🔧 To deploy the fixed code, run:" -ForegroundColor Cyan
Write-Host "    azd deploy" -ForegroundColor White

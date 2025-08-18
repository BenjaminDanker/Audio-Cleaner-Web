#!/usr/bin/env pwsh
# Build and push Audio Cleaner containers to ACR with ZERO retention (minimal costs)

param(
    [string]$AcrName = ""
)

# Colors for output
$Green = "`e[32m"
$Yellow = "`e[33m" 
$Red = "`e[31m"
$Reset = "`e[0m"

function Write-Info($message) { Write-Host "${Green}[INFO]${Reset} $message" }
function Write-Warn($message) { Write-Host "${Yellow}[WARN]${Reset} $message" }
function Write-Error($message) { Write-Host "${Red}[ERROR]${Reset} $message" }

Write-Info "Building Audio Cleaner containers..."

# Build images locally with ONLY latest tag
Write-Info "Building batch processor image..."
docker build -f Dockerfile.batch -t "audio-cleaner-processor:latest" .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build batch processor image"
    exit 1
}

Write-Info "Building streaming service image..."
docker build -f Dockerfile.streaming -t "audio-cleaner-streaming:latest" .
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to build streaming service image"
    exit 1
}

Write-Info "✅ Both images built successfully!"

# Auto-detect ACR name if not provided
if (-not $AcrName) {
    Write-Info "Auto-detecting ACR name..."
    
    # Find ACR in current subscription
    $registries = az acr list --query "[].name" -o tsv 2>$null
    if ($LASTEXITCODE -eq 0 -and $registries) {
        # Filter for ACRs with audio or cleaner in the name
        $acrList = $registries | Where-Object { $_ -like "*audio*" -or $_ -like "*cleaner*" }
        if ($acrList.Count -eq 1) {
            $AcrName = $acrList.Trim()
            Write-Info "Auto-detected ACR: $AcrName"
        } elseif ($acrList.Count -gt 1) {
            Write-Error "Multiple ACRs found: $($acrList -join ', '). Please specify with -AcrName"
            exit 1
        } else {
            Write-Warn "No ACRs found with 'audio' or 'cleaner' in name"
        }
    } else {
        Write-Warn "Could not list ACRs via Azure CLI"
    }
    
    if (-not $AcrName) {
        Write-Error "Could not auto-detect ACR name. Please specify with -AcrName parameter"
        exit 1
    }
}

Write-Info "Using ACR: $AcrName"

# Login to ACR
Write-Info "Logging into ACR: $AcrName"
az acr login --name $AcrName
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to login to ACR"
    exit 1
}

$AcrServer = "$AcrName.azurecr.io"

# Tag and push batch processor (OVERWRITES existing latest)
Write-Info "Pushing batch processor (overwriting latest)..."
docker tag "audio-cleaner-processor:latest" "$AcrServer/audio-cleaner-processor:latest"
docker push "$AcrServer/audio-cleaner-processor:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push batch processor image"
    exit 1
}
# Tag and push streaming service (OVERWRITES existing latest)
Write-Info "Pushing streaming service (overwriting latest)..."
docker tag "audio-cleaner-streaming:latest" "$AcrServer/audio-cleaner-streaming:latest"
docker push "$AcrServer/audio-cleaner-streaming:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to push streaming service image"
    exit 1
}

Write-Info "✅ All images pushed to ACR (old versions overwritten)!"
Write-Info "🐳 Build and push completed!"
Write-Info ""
Write-Info "💰 COST OPTIMIZED: Only 'latest' tags used - no version accumulation!"
Write-Info ""
Write-Info "Usage:"
Write-Info "  .\build-containers.ps1                    # Auto-detect ACR, build and push"
Write-Info "  .\build-containers.ps1 -AcrName 'myacr'   # Specify ACR name, build and push"

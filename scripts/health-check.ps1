# Audio Cleaner Pro - Development Environment Health Check
Write-Host "Audio Cleaner Pro - Development Environment Health Check" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Function to test if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to test if a port is in use
function Test-Port($port) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -InformationLevel Quiet -WarningAction SilentlyContinue
        return $connection
    } catch {
        return $false
    }
}

# Function to test HTTP endpoint
function Test-HttpEndpoint($url, $description) {
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host "✓ $description - Status: $($response.StatusCode)" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "✗ $description - Error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

Write-Host "`n1. CHECKING REQUIRED TOOLS" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Check Node.js
if (Test-Command "node") {
    $nodeVersion = node --version
    Write-Host "✓ Node.js installed: $nodeVersion" -ForegroundColor Green
    
    # Check if it's a supported version (18+ recommended)
    $versionNumber = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
    if ($versionNumber -ge 18) {
        Write-Host "  ✓ Node.js version is supported (18+)" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Node.js version may be too old. Recommend v18 or v20 LTS" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Node.js not found" -ForegroundColor Red
}

# Check npm
if (Test-Command "npm") {
    $npmVersion = npm --version
    Write-Host "✓ npm installed: v$npmVersion" -ForegroundColor Green
} else {
    Write-Host "✗ npm not found" -ForegroundColor Red
}

# Check Azure Functions Core Tools
if (Test-Command "func") {
    $funcVersion = func --version
    Write-Host "✓ Azure Functions Core Tools installed: v$funcVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Azure Functions Core Tools not found" -ForegroundColor Red
    Write-Host "  Install with: npm install -g azure-functions-core-tools@4" -ForegroundColor Yellow
}

# Check Docker (optional)
if (Test-Command "docker") {
    try {
        $dockerVersion = docker --version
        Write-Host "✓ Docker installed: $dockerVersion" -ForegroundColor Green
        
        # Check if Docker daemon is running
        $dockerInfo = docker info 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ Docker daemon is running" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ Docker daemon is not running" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ⚠ Docker installed but not accessible" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ Docker not found (optional for local development)" -ForegroundColor Yellow
}

Write-Host "`n2. CHECKING PROJECT STRUCTURE" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Check key directories
$requiredDirs = @("api", "frontend", "infra", "scripts")
foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-Host "✓ $dir directory exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $dir directory missing" -ForegroundColor Red
    }
}

# Check key files
$requiredFiles = @(
    "api/package.json",
    "api/host.json", 
    "frontend/package.json",
    "frontend/vite.config.js",
    "azure.yaml",
    "scripts/start-dev.ps1",
    "scripts/stop-dev.ps1"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✓ $file exists" -ForegroundColor Green
    } else {
        Write-Host "✗ $file missing" -ForegroundColor Red
    }
}

Write-Host "`n3. CHECKING DEPENDENCIES" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Check API dependencies
if (Test-Path "api/node_modules") {
    Write-Host "✓ API dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ API dependencies not installed - run 'npm install' in api directory" -ForegroundColor Yellow
}

# Check Frontend dependencies
if (Test-Path "frontend/node_modules") {
    Write-Host "✓ Frontend dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠ Frontend dependencies not installed - run 'npm install' in frontend directory" -ForegroundColor Yellow
}

Write-Host "`n4. CHECKING RUNNING SERVICES" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Check if ports are in use
$apiPortInUse = Test-Port 7071
$frontendPortInUse = Test-Port 5173

if ($apiPortInUse) {
    Write-Host "✓ API port (7071) is in use" -ForegroundColor Green
} else {
    Write-Host "⚠ API port (7071) is not in use - service may not be running" -ForegroundColor Yellow
}

if ($frontendPortInUse) {
    Write-Host "✓ Frontend port (5173) is in use" -ForegroundColor Green
} else {
    Write-Host "⚠ Frontend port (5173) is not in use - service may not be running" -ForegroundColor Yellow
}

Write-Host "`n5. TESTING SERVICE ENDPOINTS" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Test API endpoints
if ($apiPortInUse) {
    Test-HttpEndpoint "http://localhost:7071/api/auth" "API Auth Endpoint"
    Test-HttpEndpoint "http://localhost:7071/api/get-subscription" "API Subscription Endpoint"
    Test-HttpEndpoint "http://localhost:7071/api/enqueue-job" "API Job Queue Endpoint"
    Test-HttpEndpoint "http://localhost:7071/api/job-status" "API Job Status Endpoint"
} else {
    Write-Host "⚠ Skipping API endpoint tests - service not running" -ForegroundColor Yellow
}

# Test Frontend
if ($frontendPortInUse) {
    Test-HttpEndpoint "http://localhost:5173" "Frontend Application"
} else {
    Write-Host "⚠ Skipping Frontend test - service not running" -ForegroundColor Yellow
}

Write-Host "`n6. CHECKING BACKGROUND JOBS" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

# Check for background jobs
$jobs = Get-Job -ErrorAction SilentlyContinue
if ($jobs) {
    Write-Host "✓ Found $($jobs.Count) background job(s)" -ForegroundColor Green
    foreach ($job in $jobs) {
        $status = if ($job.State -eq "Running") { "✓" } else { "⚠" }
        $color = if ($job.State -eq "Running") { "Green" } else { "Yellow" }
        Write-Host "  $status Job $($job.Id): $($job.State)" -ForegroundColor $color
    }
} else {
    Write-Host "⚠ No background jobs found" -ForegroundColor Yellow
}

# Check process file
if (Test-Path "dev-processes.json") {
    Write-Host "✓ Process tracking file exists" -ForegroundColor Green
    try {
        $processes = Get-Content "dev-processes.json" | ConvertFrom-Json
        Write-Host "  Tracked processes: API($($processes.api)), Frontend($($processes.frontend))" -ForegroundColor Cyan
    } catch {
        Write-Host "  ⚠ Could not read process tracking file" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠ Process tracking file not found" -ForegroundColor Yellow
}

Write-Host "`n7. SUMMARY" -ForegroundColor Yellow
Write-Host "───────────────────────────────────────────────────────────────────────────────" -ForegroundColor Gray

if ($apiPortInUse -and $frontendPortInUse) {
    Write-Host "✓ Development environment appears to be running correctly!" -ForegroundColor Green
    Write-Host "  🌐 Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "  🔧 API: http://localhost:7071" -ForegroundColor Cyan
} elseif (-not $apiPortInUse -and -not $frontendPortInUse) {
    Write-Host "⚠ Development environment is not running" -ForegroundColor Yellow
    Write-Host "  Run 'scripts\start-dev.ps1' to start all services" -ForegroundColor Cyan
} else {
    Write-Host "⚠ Development environment is partially running" -ForegroundColor Yellow
    Write-Host "  Some services may need to be restarted" -ForegroundColor Cyan
}

Write-Host "`nHealth check completed!" -ForegroundColor Green

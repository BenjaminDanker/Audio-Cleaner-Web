# Audio Cleaner Pro - Stop Development Environment
Write-Host "Stopping Audio Cleaner Pro Development Environment..." -ForegroundColor Yellow

$processFile = "dev-processes.json"

# Function to safely stop a job
function Stop-BackgroundJob($jobId, $serviceName) {
    if ($jobId) {
        try {
            $job = Get-Job -Id $jobId -ErrorAction SilentlyContinue
            if ($job) {
                Write-Host "Stopping $serviceName (Job ID: $jobId)..." -ForegroundColor Yellow
                Stop-Job -Job $job -ErrorAction SilentlyContinue
                Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
                Write-Host "✓ $serviceName stopped" -ForegroundColor Green
            }
        } catch {
            Write-Host "⚠ Could not stop $serviceName gracefully" -ForegroundColor Yellow
        }
    }
}

# Function to kill processes by port
function Stop-ProcessByPort($port, $serviceName) {
    try {
        $processes = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
                    Select-Object -ExpandProperty OwningProcess | 
                    Sort-Object -Unique
        
        if ($processes) {
            foreach ($processId in $processes) {
                try {
                    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Host "Stopping $serviceName process (PID: $processId)..." -ForegroundColor Yellow
                        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
                        Write-Host "✓ $serviceName process stopped" -ForegroundColor Green
                    }
                } catch {
                    # Process might already be stopped
                }
            }
        }
    } catch {
        # Port might not be in use
    }
}

# Read and stop background jobs
if (Test-Path $processFile) {
    try {
        $processes = Get-Content $processFile | ConvertFrom-Json
        
        # Stop API job (Azure Functions)
        if ($processes.api) {
            Stop-BackgroundJob $processes.api "Azure Functions API"
        }
        
        # Stop SWA CLI job
        if ($processes.swa) {
            Stop-BackgroundJob $processes.swa "Azure Static Web Apps CLI"
        }
        
        # Stop Processor job
        if ($processes.processor) {
            Stop-BackgroundJob $processes.processor "Processor Service"
        }
        
        # Stop Frontend job (Vite)
        if ($processes.frontend) {
            Stop-BackgroundJob $processes.frontend "Frontend Vite Server"
        }
        
        # Remove process file
        Remove-Item $processFile -Force -ErrorAction SilentlyContinue
        
    } catch {
        Write-Host "Could not read process file, trying alternative cleanup..." -ForegroundColor Yellow
    }
}

# Force stop any remaining processes on known ports
Write-Host "Checking for remaining processes on development ports..." -ForegroundColor Yellow
Stop-ProcessByPort 7071 "Azure Functions API"
Stop-ProcessByPort 4280 "Azure Static Web Apps CLI"
Stop-ProcessByPort 8080 "Python Processor"
Stop-ProcessByPort 5173 "Vite Frontend"
Stop-ProcessByPort 3000 "Alternative Frontend"

# Stop any remaining background jobs from this session
$jobs = Get-Job | Where-Object { $_.Name -like "*dev*" -or $_.Command -like "*func*" -or $_.Command -like "*npm*" -or $_.Command -like "*swa*" }
if ($jobs) {
    Write-Host "Stopping remaining background jobs..." -ForegroundColor Yellow
    $jobs | Stop-Job -ErrorAction SilentlyContinue
    $jobs | Remove-Job -Force -ErrorAction SilentlyContinue
}

# Clean up any node/python processes that might be hanging
try {
    # Stop hanging node processes
    $nodeProcesses = Get-Process node -ErrorAction SilentlyContinue | 
                    Where-Object { $_.ProcessName -eq "node" }
    
    foreach ($process in $nodeProcesses) {
        $commandLine = ""
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
        } catch {
            # Can't get command line, skip
        }
        
        # Only kill processes that look like our dev servers
        if ($commandLine -match "func start|vite|npm run dev") {
            Write-Host "Stopping Node.js process (PID: $($process.Id))..." -ForegroundColor Yellow
            try {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                Write-Host "✓ Node.js process stopped" -ForegroundColor Green
            } catch {
                # Process might already be stopped
            }
        }
    }

    # Stop hanging python processes
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | 
                      Where-Object { $_.ProcessName -eq "python" }
    
    foreach ($process in $pythonProcesses) {
        $commandLine = ""
        try {
            $commandLine = (Get-WmiObject Win32_Process -Filter "ProcessId = $($process.Id)").CommandLine
        } catch {
            # Can't get command line, skip
        }
        
        # Only kill processes that look like our processor
        if ($commandLine -match "local_processor.py|processor_app.py") {
            Write-Host "Stopping Python process (PID: $($process.Id))..." -ForegroundColor Yellow
            try {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                Write-Host "✓ Python process stopped" -ForegroundColor Green
            } catch {
                # Process might already be stopped
            }
        }
    }
} catch {
    # No processes or permission issues
}

Write-Host "`n✓ Development environment stopped successfully!" -ForegroundColor Green
Write-Host "All services have been shut down." -ForegroundColor Cyan

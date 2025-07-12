#Requires -Version 5.1
<#
.SYNOPSIS
    Load Testing Script for Rate Limiting and Parallel Operations

.DESCRIPTION
    Simulates high-load scenarios to test rate limiting, parallel upload handling,
    and system stability under stress.

.PARAMETER BaseUrl
    Base URL of the deployed Function App

.PARAMETER TestDuration
    Duration of load test in seconds (default: 60)

.PARAMETER ConcurrentUsers
    Number of concurrent users to simulate (default: 10)

.PARAMETER TestType
    Type of load test: Basic, Parallel, Sustained, Burst
    Default: Basic

.EXAMPLE
    .\Test-LoadAndRateLimit.ps1 -BaseUrl "https://func-4ositsvdlpac6.azurewebsites.net" -TestType Parallel -ConcurrentUsers 20
#>

param(
    [string]$BaseUrl = "https://func-4ositsvdlpac6.azurewebsites.net",
    [int]$TestDuration = 60,
    [int]$ConcurrentUsers = 10,
    [ValidateSet("Basic", "Parallel", "Sustained", "Burst")]
    [string]$TestType = "Basic"
)

# Global variables for test tracking
$Global:TestResults = @{
    TotalRequests = 0
    SuccessfulRequests = 0
    RateLimitedRequests = 0
    ErrorRequests = 0
    AverageResponseTime = 0
    MaxResponseTime = 0
    MinResponseTime = [int]::MaxValue
}

$Global:ResponseTimes = @()

function Write-LoadTestHeader {
    param([string]$Title)
    Write-Host ""
    Write-Host "⚡ $Title" -ForegroundColor Cyan
    Write-Host ("=" * ($Title.Length + 4)) -ForegroundColor Cyan
}

function Test-SingleRequest {
    param(
        [string]$Endpoint,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [string]$Body = $null
    )
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    try {
        $requestParams = @{
            Uri = "$BaseUrl$Endpoint"
            Method = $Method
            Headers = $Headers
            ErrorAction = 'SilentlyContinue'
        }
        
        if ($Body) {
            $requestParams.Body = $Body
            $requestParams.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @requestParams
        $stopwatch.Stop()
        
        $Global:TestResults.TotalRequests++
        $responseTime = $stopwatch.ElapsedMilliseconds
        $Global:ResponseTimes += $responseTime
        
        # Update response time stats
        if ($responseTime -gt $Global:TestResults.MaxResponseTime) {
            $Global:TestResults.MaxResponseTime = $responseTime
        }
        if ($responseTime -lt $Global:TestResults.MinResponseTime) {
            $Global:TestResults.MinResponseTime = $responseTime
        }
        
        switch ($response.StatusCode) {
            200 { 
                $Global:TestResults.SuccessfulRequests++
                return @{ Success = $true; StatusCode = 200; ResponseTime = $responseTime; RateLimited = $false }
            }
            429 { 
                $Global:TestResults.RateLimitedRequests++
                return @{ Success = $false; StatusCode = 429; ResponseTime = $responseTime; RateLimited = $true }
            }
            default { 
                $Global:TestResults.ErrorRequests++
                return @{ Success = $false; StatusCode = $response.StatusCode; ResponseTime = $responseTime; RateLimited = $false }
            }
        }
        
    } catch {
        $stopwatch.Stop()
        $Global:TestResults.TotalRequests++
        
        if ($_.Exception.Response.StatusCode -eq 429) {
            $Global:TestResults.RateLimitedRequests++
            return @{ Success = $false; StatusCode = 429; ResponseTime = $stopwatch.ElapsedMilliseconds; RateLimited = $true }
        } else {
            $Global:TestResults.ErrorRequests++
            return @{ Success = $false; StatusCode = 0; ResponseTime = $stopwatch.ElapsedMilliseconds; RateLimited = $false }
        }
    }
}

function Test-BasicLoad {
    Write-LoadTestHeader "Basic Load Test - Sequential Requests"
    
    Write-Host "🧪 Sending sequential requests to trigger rate limiting..." -ForegroundColor Yellow
    Write-Host "Configuration: 50 requests with 200ms intervals" -ForegroundColor Gray
    
    $requestCount = 50
    $rateLimitTriggered = $false
    
    for ($i = 1; $i -le $requestCount; $i++) {
        $result = Test-SingleRequest -Endpoint "/api/index"
        
        if ($result.RateLimited) {
            $rateLimitTriggered = $true
            Write-Host "Rate limit triggered at request #$i" -ForegroundColor Red
        }
        
        # Progress indicator
        if ($i % 10 -eq 0) {
            Write-Host "Progress: $i/$requestCount requests completed" -ForegroundColor Gray
        }
        
        Start-Sleep -Milliseconds 200
    }
    
    return @{
        RateLimitTriggered = $rateLimitTriggered
        RequestsProcessed = $requestCount
    }
}

function Test-ParallelLoad {
    Write-LoadTestHeader "Parallel Load Test - Concurrent Requests"
    
    Write-Host "🧪 Testing parallel request handling..." -ForegroundColor Yellow
    Write-Host "Configuration: $ConcurrentUsers concurrent users, each sending 10 requests" -ForegroundColor Gray
    
    $jobs = @()
    $requestsPerUser = 10
    
    # Start concurrent jobs
    for ($user = 1; $user -le $ConcurrentUsers; $user++) {
        $jobs += Start-Job -ScriptBlock {
            param($BaseUrl, $RequestsPerUser, $UserId)
            
            $userResults = @{
                UserId = $UserId
                Successful = 0
                RateLimited = 0
                Errors = 0
                ResponseTimes = @()
            }
            
            for ($req = 1; $req -le $RequestsPerUser; $req++) {
                $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
                
                try {
                    $response = Invoke-WebRequest -Uri "$BaseUrl/api/index" -Method GET -ErrorAction SilentlyContinue
                    $stopwatch.Stop()
                    
                    $userResults.ResponseTimes += $stopwatch.ElapsedMilliseconds
                    
                    switch ($response.StatusCode) {
                        200 { $userResults.Successful++ }
                        429 { $userResults.RateLimited++ }
                        default { $userResults.Errors++ }
                    }
                } catch {
                    $stopwatch.Stop()
                    $userResults.ResponseTimes += $stopwatch.ElapsedMilliseconds
                    
                    if ($_.Exception.Response.StatusCode -eq 429) {
                        $userResults.RateLimited++
                    } else {
                        $userResults.Errors++
                    }
                }
                
                Start-Sleep -Milliseconds (Get-Random -Minimum 100 -Maximum 500)
            }
            
            return $userResults
            
        } -ArgumentList $BaseUrl, $requestsPerUser, $user
    }
    
    Write-Host "Waiting for all $ConcurrentUsers users to complete..." -ForegroundColor Gray
    
    $results = $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
    
    # Aggregate results
    $totalSuccessful = ($results | Measure-Object -Property Successful -Sum).Sum
    $totalRateLimited = ($results | Measure-Object -Property RateLimited -Sum).Sum
    $totalErrors = ($results | Measure-Object -Property Errors -Sum).Sum
    $allResponseTimes = $results | ForEach-Object { $_.ResponseTimes } | Where-Object { $_ -ne $null }
    
    $Global:TestResults.TotalRequests += ($totalSuccessful + $totalRateLimited + $totalErrors)
    $Global:TestResults.SuccessfulRequests += $totalSuccessful
    $Global:TestResults.RateLimitedRequests += $totalRateLimited
    $Global:TestResults.ErrorRequests += $totalErrors
    $Global:ResponseTimes += $allResponseTimes
    
    Write-Host "Parallel test completed!" -ForegroundColor Green
    Write-Host "Users: $ConcurrentUsers | Successful: $totalSuccessful | Rate Limited: $totalRateLimited | Errors: $totalErrors" -ForegroundColor Gray
    
    return @{
        TotalUsers = $ConcurrentUsers
        SuccessfulRequests = $totalSuccessful
        RateLimitedRequests = $totalRateLimited
        ErrorRequests = $totalErrors
        AverageResponseTime = if ($allResponseTimes.Count -gt 0) { ($allResponseTimes | Measure-Object -Average).Average } else { 0 }
    }
}

function Test-SustainedLoad {
    Write-LoadTestHeader "Sustained Load Test - Extended Duration"
    
    Write-Host "🧪 Testing sustained load over $TestDuration seconds..." -ForegroundColor Yellow
    Write-Host "Configuration: Continuous requests with random intervals" -ForegroundColor Gray
    
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($TestDuration)
    $requestCount = 0
    $rateLimitCount = 0
    
    while ((Get-Date) -lt $endTime) {
        $result = Test-SingleRequest -Endpoint "/api/index"
        $requestCount++
        
        if ($result.RateLimited) {
            $rateLimitCount++
        }
        
        # Progress update every 10 seconds
        $elapsed = ((Get-Date) - $startTime).TotalSeconds
        if ([math]::Floor($elapsed) % 10 -eq 0 -and $requestCount % 5 -eq 0) {
            $remaining = $TestDuration - $elapsed
            Write-Host "Elapsed: $([math]::Floor($elapsed))s | Remaining: $([math]::Floor($remaining))s | Requests: $requestCount | Rate Limited: $rateLimitCount" -ForegroundColor Gray
        }
        
        # Random delay to simulate real user behavior
        Start-Sleep -Milliseconds (Get-Random -Minimum 500 -Maximum 2000)
    }
    
    return @{
        Duration = $TestDuration
        TotalRequests = $requestCount
        RateLimitedRequests = $rateLimitCount
        RequestsPerSecond = [math]::Round($requestCount / $TestDuration, 2)
    }
}

function Test-BurstLoad {
    Write-LoadTestHeader "Burst Load Test - Traffic Spikes"
    
    Write-Host "🧪 Testing burst traffic patterns..." -ForegroundColor Yellow
    Write-Host "Configuration: 3 bursts of 20 requests each, with calm periods" -ForegroundColor Gray
    
    $burstResults = @()
    
    for ($burst = 1; $burst -le 3; $burst++) {
        Write-Host "Burst #$burst - Sending 20 rapid requests..." -ForegroundColor Yellow
        
        $burstStart = Get-Date
        $burstSuccessful = 0
        $burstRateLimited = 0
        
        # Rapid burst of requests
        for ($i = 1; $i -le 20; $i++) {
            $result = Test-SingleRequest -Endpoint "/api/index"
            
            if ($result.Success) {
                $burstSuccessful++
            } elseif ($result.RateLimited) {
                $burstRateLimited++
            }
            
            Start-Sleep -Milliseconds 50  # Very short delay for burst effect
        }
        
        $burstDuration = ((Get-Date) - $burstStart).TotalSeconds
        
        $burstResults += @{
            BurstNumber = $burst
            Successful = $burstSuccessful
            RateLimited = $burstRateLimited
            Duration = $burstDuration
            RequestsPerSecond = [math]::Round(20 / $burstDuration, 2)
        }
        
        Write-Host "Burst #$burst completed: $burstSuccessful successful, $burstRateLimited rate limited" -ForegroundColor Gray
        
        # Calm period between bursts
        if ($burst -lt 3) {
            Write-Host "Calm period - waiting 10 seconds..." -ForegroundColor Blue
            Start-Sleep -Seconds 10
        }
    }
    
    return $burstResults
}

function Test-UploadLoadTest {
    Write-LoadTestHeader "File Upload Load Test"
    
    Write-Host "🧪 Testing file upload rate limiting..." -ForegroundColor Yellow
    Write-Host "Configuration: Simulated file upload requests" -ForegroundColor Gray
    
    $uploadRequests = 15
    $uploadResults = @{
        Successful = 0
        RateLimited = 0
        Errors = 0
    }
    
    for ($i = 1; $i -le $uploadRequests; $i++) {
        $uploadBody = @{
            filename = "test-file-$i.mp3"
            contentType = "audio/mpeg"
            size = 1024000
        } | ConvertTo-Json
        
        $result = Test-SingleRequest -Endpoint "/api/upload-file" -Method "POST" -Body $uploadBody
        
        if ($result.Success) {
            $uploadResults.Successful++
        } elseif ($result.RateLimited) {
            $uploadResults.RateLimited++
        } else {
            $uploadResults.Errors++
        }
        
        Write-Host "Upload $i`: Status $($result.StatusCode) | Response Time: $($result.ResponseTime)ms" -ForegroundColor Gray
        
        Start-Sleep -Milliseconds 300
    }
    
    return $uploadResults
}

function Show-TestSummary {
    Write-LoadTestHeader "Load Test Summary"
    
    # Calculate final statistics
    if ($Global:ResponseTimes.Count -gt 0) {
        $Global:TestResults.AverageResponseTime = [math]::Round(($Global:ResponseTimes | Measure-Object -Average).Average, 2)
        if ($Global:TestResults.MinResponseTime -eq [int]::MaxValue) {
            $Global:TestResults.MinResponseTime = 0
        }
    }
    
    Write-Host "📊 Overall Test Results:" -ForegroundColor Cyan
    Write-Host "   Total Requests: $($Global:TestResults.TotalRequests)" -ForegroundColor White
    Write-Host "   Successful: $($Global:TestResults.SuccessfulRequests) ($([math]::Round(($Global:TestResults.SuccessfulRequests / $Global:TestResults.TotalRequests) * 100, 1))%)" -ForegroundColor Green
    Write-Host "   Rate Limited: $($Global:TestResults.RateLimitedRequests) ($([math]::Round(($Global:TestResults.RateLimitedRequests / $Global:TestResults.TotalRequests) * 100, 1))%)" -ForegroundColor Yellow
    Write-Host "   Errors: $($Global:TestResults.ErrorRequests) ($([math]::Round(($Global:TestResults.ErrorRequests / $Global:TestResults.TotalRequests) * 100, 1))%)" -ForegroundColor Red
    Write-Host ""
    Write-Host "⏱️  Response Time Statistics:" -ForegroundColor Cyan
    Write-Host "   Average: $($Global:TestResults.AverageResponseTime)ms" -ForegroundColor White
    Write-Host "   Minimum: $($Global:TestResults.MinResponseTime)ms" -ForegroundColor Green
    Write-Host "   Maximum: $($Global:TestResults.MaxResponseTime)ms" -ForegroundColor Red
    
    # Rate limiting effectiveness
    $rateLimitEffectiveness = [math]::Round(($Global:TestResults.RateLimitedRequests / $Global:TestResults.TotalRequests) * 100, 1)
    
    Write-Host ""
    Write-Host "🛡️  Rate Limiting Analysis:" -ForegroundColor Cyan
    if ($rateLimitEffectiveness -ge 10) {
        Write-Host "   ✅ Rate limiting is working effectively ($rateLimitEffectiveness% of requests limited)" -ForegroundColor Green
    } elseif ($rateLimitEffectiveness -ge 5) {
        Write-Host "   ⚠️  Rate limiting is active but may need tuning ($rateLimitEffectiveness% of requests limited)" -ForegroundColor Yellow
    } else {
        Write-Host "   ❌ Rate limiting may not be working properly ($rateLimitEffectiveness% of requests limited)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "📈 Recommendations:" -ForegroundColor Cyan
    
    if ($Global:TestResults.AverageResponseTime -gt 2000) {
        Write-Host "   ⚠️  High response times detected - consider optimizing performance" -ForegroundColor Yellow
    }
    
    if ($Global:TestResults.ErrorRequests -gt ($Global:TestResults.TotalRequests * 0.1)) {
        Write-Host "   ⚠️  High error rate detected - investigate server issues" -ForegroundColor Yellow
    }
    
    if ($rateLimitEffectiveness -lt 5) {
        Write-Host "   ⚠️  Consider adjusting rate limiting thresholds" -ForegroundColor Yellow
    }
    
    Write-Host "   💡 Monitor Application Insights for detailed metrics" -ForegroundColor Blue
    Write-Host "   💡 Run: .\Monitor-AzureLogs.ps1 -LogType Performance -Hours 1" -ForegroundColor Blue
}

# Main execution
Write-Host "⚡ Load Testing Suite - Audio Cleaner Web Application" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Test Configuration:" -ForegroundColor White
Write-Host "   Function App: $BaseUrl" -ForegroundColor Gray
Write-Host "   Test Type: $TestType" -ForegroundColor Gray
Write-Host "   Duration: $TestDuration seconds" -ForegroundColor Gray
Write-Host "   Concurrent Users: $ConcurrentUsers" -ForegroundColor Gray
Write-Host ""

$testStart = Get-Date

# Execute selected test type
switch ($TestType) {
    "Basic" {
        $basicResult = Test-BasicLoad
        Write-Host "Basic test completed: Rate limiting triggered = $($basicResult.RateLimitTriggered)" -ForegroundColor Green
    }
    "Parallel" {
        $parallelResult = Test-ParallelLoad
        Write-Host "Parallel test completed with $($parallelResult.TotalUsers) users" -ForegroundColor Green
    }
    "Sustained" {
        $sustainedResult = Test-SustainedLoad
        Write-Host "Sustained test completed: $($sustainedResult.RequestsPerSecond) requests/second average" -ForegroundColor Green
    }
    "Burst" {
        $burstResults = Test-BurstLoad
        Write-Host "Burst test completed with $($burstResults.Count) traffic spikes" -ForegroundColor Green
    }
}

# Always test upload rate limiting
$uploadResult = Test-UploadLoadTest
Write-Host "Upload test completed: $($uploadResult.RateLimited) rate limited uploads" -ForegroundColor Green

$testEnd = Get-Date
$totalTestDuration = ($testEnd - $testStart).TotalSeconds

Write-Host ""
Write-Host "Total test duration: $([math]::Round($totalTestDuration, 1)) seconds" -ForegroundColor Gray

Show-TestSummary

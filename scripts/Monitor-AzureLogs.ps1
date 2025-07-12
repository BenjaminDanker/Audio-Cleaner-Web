#Requires -Version 5.1
<#
.SYNOPSIS
    Comprehensive Azure Log Monitoring Script for Audio Cleaner Web Application

.DESCRIPTION
    A centralized script to monitor logs across all Azure resources with flexible filtering options.
    Supports Application Insights, Function Apps, Container Apps, and general Azure resource logs.

.PARAMETER SubscriptionId
    Azure subscription ID. Defaults to the Audio Cleaner Web subscription.

.PARAMETER ResourceGroup
    Azure resource group name. Defaults to rg-denoise-audio.

.PARAMETER Hours
    Number of hours to look back for logs. Default is 24.

.PARAMETER LogType
    Type of logs to retrieve. Options: All, Errors, Functions, Requests, Dependencies, Performance, Traces
    Default: Errors

.PARAMETER FunctionName
    Filter logs for a specific Azure Function name (e.g., 'upload-file', 'job-status')

.PARAMETER Severity
    Minimum severity level to show. Options: All, Information, Warning, Error, Critical
    Default: Warning

.PARAMETER MaxResults
    Maximum number of results to return per query. Default is 20.

.PARAMETER OutputFormat
    Output format. Options: Console, Json, Csv
    Default: Console

.PARAMETER ShowRaw
    Show raw log data without formatting

.PARAMETER Verbose
    Show detailed information and all available logs

.EXAMPLE
    .\Monitor-AzureLogs.ps1
    Show errors and warnings from the last 24 hours

.EXAMPLE
    .\Monitor-AzureLogs.ps1 -LogType All -Hours 1 -Verbose
    Show all logs from the last hour with detailed output

.EXAMPLE
    .\Monitor-AzureLogs.ps1 -LogType Functions -FunctionName "upload-file" -Severity Error
    Show only error logs for the upload-file function

.EXAMPLE
    .\Monitor-AzureLogs.ps1 -LogType Performance -Hours 12
    Show performance metrics from the last 12 hours

.EXAMPLE
    .\Monitor-AzureLogs.ps1 -LogType All -Severity All -OutputFormat Json -MaxResults 100
    Export all logs to JSON format
#>

param(
    [string]$SubscriptionId = "61d9a026-8bd4-41de-8aeb-44722187b1da",
    
    [string]$ResourceGroup = "rg-denoise-audio",
    
    [ValidateRange(1, 8760)]  # Max 1 year
    [int]$Hours = 24,
    
    [ValidateSet("All", "Errors", "Functions", "Requests", "Dependencies", "Performance", "Traces")]
    [string]$LogType = "Errors",
    
    [string]$FunctionName = "",
    
    [ValidateSet("All", "Information", "Warning", "Error", "Critical")]
    [string]$Severity = "Warning",
    
    [ValidateRange(1, 1000)]
    [int]$MaxResults = 20,
    
    [ValidateSet("Console", "Json", "Csv")]
    [string]$OutputFormat = "Console",
    
    [switch]$ShowRaw,
    
    [switch]$VerboseOutput
)

# Configuration
$AppInsightsId = "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.Insights/components/appi-4ositsvdlpac6"
$LogWorkspaceId = "/subscriptions/$SubscriptionId/resourceGroups/$ResourceGroup/providers/Microsoft.OperationalInsights/workspaces/log-4ositsvdlpac6"

# Color scheme
$Colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Dim = "Gray"
    Highlight = "Magenta"
}

# Severity level mappings
$SeverityLevels = @{
    "All" = -1
    "Information" = 1
    "Warning" = 2
    "Error" = 3
    "Critical" = 4
}

function Write-Header {
    param([string]$Title, [string]$Color = "Cyan")
    
    if ($OutputFormat -eq "Console") {
        Write-Host ""
        Write-Host "=" * 80 -ForegroundColor $Color
        Write-Host " $Title" -ForegroundColor $Color
        Write-Host "=" * 80 -ForegroundColor $Color
    }
}

function Write-Section {
    param([string]$Title, [string]$Color = "White")
    
    if ($OutputFormat -eq "Console") {
        Write-Host ""
        Write-Host "📊 $Title" -ForegroundColor $Color
        Write-Host ("-" * ($Title.Length + 4)) -ForegroundColor $Colors.Dim
    }
}

function Execute-Query {
    param(
        [string]$Query,
        [string]$Description,
        [string]$Icon = "📝"
    )
    
    try {
        Write-Section "$Icon $Description" $Colors.Info
        
        if ($VerboseOutput -and $OutputFormat -eq "Console") {
            Write-Host "Query: $Query" -ForegroundColor $Colors.Dim
            Write-Host ""
        }
        
        $resultJson = az monitor app-insights query --apps $AppInsightsId --analytics-query $Query --output json 2>&1
        
        if ($LASTEXITCODE -ne 0) {
            if ($OutputFormat -eq "Console") {
                Write-Host "❌ Query failed: $resultJson" -ForegroundColor $Colors.Error
            }
            return @()
        }
        
        $result = $resultJson | ConvertFrom-Json
        
        if (-not $result -or -not $result.tables -or $result.tables.Count -eq 0) {
            if ($OutputFormat -eq "Console") {
                Write-Host "⚠️  No data returned from query" -ForegroundColor $Colors.Warning
            }
            return @()
        }
        
        if ($result.tables[0].rows.Count -eq 0) {
            if ($OutputFormat -eq "Console") {
                Write-Host "✅ No entries found" -ForegroundColor $Colors.Success
            }
            return @()
        }
        
        $data = @()
        $columns = $result.tables[0].columns
        
        foreach ($row in $result.tables[0].rows) {
            $obj = @{}
            for ($i = 0; $i -lt $columns.Count; $i++) {
                $obj[$columns[$i].name] = $row[$i]
            }
            $data += $obj
        }
        
        if ($OutputFormat -eq "Console") {
            Write-Host "📈 Found $($data.Count) entries" -ForegroundColor $Colors.Info
            
            if ($ShowRaw) {
                $data | Format-Table -AutoSize
            } else {
                Format-LogOutput -Data $data -QueryType $Description
            }
        }
        
        return $data
        
    } catch {
        if ($OutputFormat -eq "Console") {
            Write-Host "❌ Error executing query: $($_.Exception.Message)" -ForegroundColor $Colors.Error
        }
        return @()
    }
}

function Convert-ToCDT {
    param([string]$Timestamp)
    
    try {
        # Parse the timestamp (Application Insights typically returns UTC)
        $utcTime = [DateTime]::Parse($Timestamp)
        
        # Get Central Time Zone (handles both CST and CDT automatically)
        $centralTimeZone = [TimeZoneInfo]::FindSystemTimeZoneById("Central Standard Time")
        
        # Convert to Central Time
        $centralTime = [TimeZoneInfo]::ConvertTimeFromUtc($utcTime, $centralTimeZone)
        
        # Format with timezone indicator
        $timezoneName = if ($centralTimeZone.IsDaylightSavingTime($centralTime)) { "CDT" } else { "CST" }
        
        return "$($centralTime.ToString('MM/dd/yyyy HH:mm:ss')) $timezoneName"
    } catch {
        # If conversion fails, return original timestamp
        return $Timestamp
    }
}

function Format-LogOutput {
    param($Data, $QueryType)
    
    $maxDisplay = if ($VerboseOutput) { $Data.Count } else { [Math]::Min($MaxResults, $Data.Count) }
    
    for ($i = 0; $i -lt $maxDisplay; $i++) {
        $entry = $Data[$i]
        
        # Handle both hashtables and PSCustomObjects for timestamp
        $timestamp = if ($entry -is [hashtable]) { $entry["timestamp"] } else { $entry.timestamp }
        
        # Convert timestamp to Central Daylight Time
        $convertedTimestamp = Convert-ToCDT -Timestamp $timestamp
        
        Write-Host "  📍 $convertedTimestamp" -ForegroundColor $Colors.Info
        
        switch -Wildcard ($QueryType) {
            "*Exception*" {
                Write-Host "     Type: $($entry.type)" -ForegroundColor $Colors.Dim
                Write-Host "     Message: $($entry.message)" -ForegroundColor $Colors.Error
                if ($entry.operation_Name) { Write-Host "     Operation: $($entry.operation_Name)" -ForegroundColor $Colors.Dim }
                if ($entry.cloud_RoleName) { Write-Host "     Service: $($entry.cloud_RoleName)" -ForegroundColor $Colors.Dim }
            }
            "*Request*" {
                $statusColor = if ($entry.success -eq "false" -or $entry.resultCode -ge 400) { $Colors.Error } else { $Colors.Success }
                Write-Host "     Request: $($entry.name)" -ForegroundColor $Colors.Dim
                Write-Host "     Status: $($entry.resultCode) | Duration: $($entry.duration)ms" -ForegroundColor $statusColor
                if ($entry.url) { Write-Host "     URL: $($entry.url)" -ForegroundColor $Colors.Dim }
            }
            "*Trace*" {
                $severityText = switch ($entry.severityLevel) {
                    0 { "Verbose" }
                    1 { "Information" }
                    2 { "Warning" }
                    3 { "Error" }
                    4 { "Critical" }
                    default { "Unknown" }
                }
                $severityColor = switch ($entry.severityLevel) {
                    0 { $Colors.Dim }
                    1 { $Colors.Info }
                    2 { $Colors.Warning }
                    3 { $Colors.Error }
                    4 { $Colors.Error }
                    default { $Colors.Dim }
                }
                Write-Host "     Severity: $severityText" -ForegroundColor $severityColor
                Write-Host "     Message: $($entry.message)" -ForegroundColor $Colors.Info
                if ($entry.operation_Name) { Write-Host "     Operation: $($entry.operation_Name)" -ForegroundColor $Colors.Dim }
            }
            "*Dependency*" {
                $statusColor = if ($entry.success -eq "false") { $Colors.Error } else { $Colors.Success }
                Write-Host "     Dependency: $($entry.name) ($($entry.type))" -ForegroundColor $Colors.Dim
                Write-Host "     Status: $($entry.resultCode) | Duration: $($entry.duration)ms" -ForegroundColor $statusColor
                if ($entry.data) { Write-Host "     Data: $($entry.data)" -ForegroundColor $Colors.Dim }
            }
            "*Performance*" {
                if ($entry -is [hashtable]) {
                    foreach ($key in $entry.Keys) {
                        if ($key -ne "timestamp") {
                            $value = $entry[$key]
                            $color = $Colors.Info
                            
                            # Color code based on performance metrics
                            if ($key -like "*Duration*" -and $value -is [string] -and $value -match "\d+") {
                                $numValue = [int]($value -replace "[^\d]", "")
                                $color = if ($numValue -gt 5000) { $Colors.Error } elseif ($numValue -gt 1000) { $Colors.Warning } else { $Colors.Success }
                            } elseif ($key -like "*Error*" -and $value -and [int]$value -gt 0) {
                                $color = $Colors.Error
                            } elseif ($key -like "*Warning*" -and $value -and [int]$value -gt 0) {
                                $color = $Colors.Warning
                            }
                            
                            Write-Host "     $($key): $value" -ForegroundColor $color
                        }
                    }
                } else {
                    foreach ($prop in $entry.PSObject.Properties) {
                        if ($prop.Name -ne "timestamp") {
                            $value = $prop.Value
                            $color = $Colors.Info
                            
                            # Color code based on performance metrics
                            if ($prop.Name -like "*Duration*" -and $value -is [int]) {
                                $color = if ($value -gt 5000) { $Colors.Error } elseif ($value -gt 1000) { $Colors.Warning } else { $Colors.Success }
                            } elseif ($prop.Name -like "*Error*" -and $value -gt 0) {
                                $color = $Colors.Error
                            } elseif ($prop.Name -like "*Warning*" -and $value -gt 0) {
                                $color = $Colors.Warning
                            }
                            
                            Write-Host "     $($prop.Name): $value" -ForegroundColor $color
                        }
                    }
                }
            }
            "*Function*" {
                if ($entry -is [hashtable]) {
                    foreach ($key in $entry.Keys) {
                        if ($key -ne "timestamp") {
                            $value = $entry[$key]
                            $color = $Colors.Info
                            
                            # Special formatting for function data
                            if ($key -eq "Status") {
                                $color = if ($value -eq "Success") { $Colors.Success } else { $Colors.Error }
                            } elseif ($key -eq "Duration" -and $value) {
                                if ($value -match "\d+") {
                                    $durationValue = [int]($value -replace "[^\d]", "")
                                    $color = if ($durationValue -gt 5000) { $Colors.Error } elseif ($durationValue -gt 1000) { $Colors.Warning } else { $Colors.Success }
                                    if ($value -notlike "*ms") { $value = "$value ms" }
                                }
                            }
                            
                            Write-Host "     $($key): $value" -ForegroundColor $color
                        }
                    }
                } else {
                    foreach ($prop in $entry.PSObject.Properties) {
                        if ($prop.Name -ne "timestamp") {
                            $value = $prop.Value
                            $color = $Colors.Info
                            
                            # Special formatting for function data
                            if ($prop.Name -eq "Status") {
                                $color = if ($value -eq "Success") { $Colors.Success } else { $Colors.Error }
                            } elseif ($prop.Name -eq "Duration" -and $value) {
                                $durationValue = [int]($value -replace "ms", "")
                                $color = if ($durationValue -gt 5000) { $Colors.Error } elseif ($durationValue -gt 1000) { $Colors.Warning } else { $Colors.Success }
                                $value = "$value ms"
                            }
                            
                            Write-Host "     $($prop.Name): $value" -ForegroundColor $color
                        }
                    }
                }
            }
            default {
                # Generic formatting - handle both hashtables and PSCustomObjects
                if ($entry -is [hashtable]) {
                    foreach ($key in $entry.Keys) {
                        if ($key -ne "timestamp" -and $entry[$key]) {
                            Write-Host "     $($key): $($entry[$key])" -ForegroundColor $Colors.Dim
                        }
                    }
                } else {
                    foreach ($prop in $entry.PSObject.Properties) {
                        if ($prop.Name -ne "timestamp" -and $prop.Value) {
                            Write-Host "     $($prop.Name): $($prop.Value)" -ForegroundColor $Colors.Dim
                        }
                    }
                }
            }
        }
        Write-Host ""
    }
    
    if (!$VerboseOutput -and $Data.Count -gt $MaxResults) {
        Write-Host "     ... and $($Data.Count - $MaxResults) more entries (use -VerboseOutput to see all)" -ForegroundColor $Colors.Dim
    }
}

function Get-BaseFilter {
    param([string]$TableType = "")
    
    $timeFilter = "timestamp > ago($($Hours)h)"
    
    $severityFilter = ""
    # Only apply severity filter to tables that have severityLevel column
    if ($Severity -ne "All" -and ($TableType -eq "traces" -or $TableType -eq "exceptions" -or $TableType -eq "")) {
        $severityFilter = " and severityLevel >= $($SeverityLevels[$Severity])"
    }
    
    $functionFilter = ""
    if ($FunctionName -ne "") {
        $functionFilter = " and operation_Name contains '$FunctionName'"
    }
    
    return "$timeFilter$severityFilter$functionFilter"
}

function Get-AllLogs {
    $allData = @()
    
    # Get all exceptions
    $baseFilter = Get-BaseFilter -TableType "exceptions"
    $query = "exceptions | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, type, message, operation_Name, cloud_RoleName"
    $allData += Execute-Query -Query $query -Description "Exceptions" -Icon "🚨"
    
    # Get all requests
    $baseFilter = Get-BaseFilter -TableType "requests"
    $successFilter = if ($Severity -eq "All") { "" } else { " and success == false" }
    $query = "requests | where $baseFilter$successFilter | order by timestamp asc | take $MaxResults | project timestamp, name, resultCode, duration, url, success"
    $allData += Execute-Query -Query $query -Description "Requests" -Icon "🌐"
    
    # Get all traces
    $baseFilter = Get-BaseFilter -TableType "traces"
    $query = "traces | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, message, severityLevel, operation_Name, cloud_RoleName"
    $allData += Execute-Query -Query $query -Description "Traces" -Icon "📝"
    
    # Get all dependencies
    $baseFilter = Get-BaseFilter -TableType "dependencies"
    $successFilter = if ($Severity -eq "All") { "" } else { " and success == false" }
    $query = "dependencies | where $baseFilter$successFilter | order by timestamp asc | take $MaxResults | project timestamp, name, type, data, resultCode, duration, success"
    $allData += Execute-Query -Query $query -Description "Dependencies" -Icon "🔗"
    
    # Get custom events
    $baseFilter = Get-BaseFilter -TableType "customEvents"
    $query = "customEvents | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, name, customDimensions"
    $allData += Execute-Query -Query $query -Description "Custom Events" -Icon "⚡"
    
    return $allData
}

function Get-ErrorLogs {
    $allData = @()
    
    # Exceptions
    $baseFilter = Get-BaseFilter -TableType "exceptions"
    $query = "exceptions | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, type, message, operation_Name, cloud_RoleName, outerMessage"
    $allData += Execute-Query -Query $query -Description "Exceptions" -Icon "🚨"
    
    # Failed requests
    $baseFilter = Get-BaseFilter -TableType "requests"
    $query = "requests | where $baseFilter and success == false | order by timestamp asc | take $MaxResults | project timestamp, name, resultCode, duration, url, success"
    $allData += Execute-Query -Query $query -Description "Failed Requests" -Icon "❌"
    
    # Error traces
    $errorSeverity = [Math]::Max(3, $SeverityLevels[$Severity])
    $baseFilter = Get-BaseFilter -TableType "traces"
    $query = "traces | where $baseFilter and severityLevel >= $errorSeverity | order by timestamp asc | take $MaxResults | project timestamp, message, severityLevel, operation_Name, cloud_RoleName"
    $allData += Execute-Query -Query $query -Description "Error Traces" -Icon "🔥"
    
    # Failed dependencies
    $baseFilter = Get-BaseFilter -TableType "dependencies"
    $query = "dependencies | where $baseFilter and success == false | order by timestamp asc | take $MaxResults | project timestamp, name, type, data, resultCode, duration, success"
    $allData += Execute-Query -Query $query -Description "Failed Dependencies" -Icon "💥"
    
    return $allData
}

function Get-FunctionLogs {
    $allData = @()
    $functionFilter = if ($FunctionName -ne "") { " and operation_Name contains '$FunctionName'" } else { " and cloud_RoleName contains 'func-'" }
    
    # Function traces
    $baseFilter = Get-BaseFilter -TableType "traces"
    $query = "traces | where $baseFilter$functionFilter | order by timestamp asc | take $MaxResults | project timestamp, message, severityLevel, operation_Name, cloud_RoleName"
    $allData += Execute-Query -Query $query -Description "Function Traces" -Icon "⚙️"
    
    # Function requests
    $baseFilter = Get-BaseFilter -TableType "requests"
    $query = "requests | where $baseFilter$functionFilter | order by timestamp asc | take $MaxResults | project timestamp, name, resultCode, duration, url, success"
    $allData += Execute-Query -Query $query -Description "Function Requests" -Icon "🔧"
    
    # Function exceptions
    $baseFilter = Get-BaseFilter -TableType "exceptions"
    $query = "exceptions | where $baseFilter$functionFilter | order by timestamp asc | take $MaxResults | project timestamp, type, message, operation_Name, cloud_RoleName"
    $allData += Execute-Query -Query $query -Description "Function Exceptions" -Icon "⚠️"
    
    return $allData
}

function Get-PerformanceLogs {
    
    # Check if we have any request data first
    $baseFilter = Get-BaseFilter -TableType "requests"
    $requestCheckQuery = "requests | where $baseFilter | count"
    
    try {
        $requestCheckResult = az monitor app-insights query --apps $AppInsightsId --analytics-query $requestCheckQuery --output json 2>&1
        if ($LASTEXITCODE -eq 0) {
            $requestCheck = $requestCheckResult | ConvertFrom-Json
            $hasRequests = $requestCheck.tables[0].rows[0][0] -gt 0
        } else {
            $hasRequests = $false
        }
    } catch {
        $hasRequests = $false
    }
    
    if ($hasRequests) {
        # Traditional request-based performance metrics
        $query = @"
requests 
| where $baseFilter
| summarize 
    TotalRequests = count(),
    SuccessfulRequests = countif(success == true),
    FailedRequests = countif(success == false),
    AvgDuration = round(avg(duration), 2),
    MaxDuration = round(max(duration), 2),
    P95Duration = round(percentile(duration, 95), 2),
    P99Duration = round(percentile(duration, 99), 2)
"@
        $perfData = Execute-Query -Query $query -Description "Performance Summary" -Icon "📈"
        
        # Slow requests
        $query = "requests | where $baseFilter and duration > 5000 | order by duration desc | take $MaxResults | project timestamp, name, duration, resultCode, url"
        $slowData = Execute-Query -Query $query -Description "Slow Requests (>5s)" -Icon "🐌"
        
        # Performance by operation
        $query = @"
requests 
| where $baseFilter
| summarize 
    Count = count(),
    AvgDuration = round(avg(duration), 2),
    ErrorRate = round(100.0 * countif(success == false) / count(), 2)
by operation_Name
| order by Count desc
| take 20
"@
        $opData = Execute-Query -Query $query -Description "Performance by Operation" -Icon "📊"
        
        return $perfData + $slowData + $opData
    } else {
        # Function-based performance metrics using traces
        if ($OutputFormat -eq "Console") {
            Write-Section "📊 Function Performance Analysis (Trace-based)" $Colors.Info
            Write-Host "ℹ️  No HTTP requests found. Analyzing Azure Function performance from traces..." -ForegroundColor $Colors.Info
        }
        
        # Function execution summary
        $traceFilter = Get-BaseFilter -TableType "traces"
        $query = @"
traces 
| where $traceFilter
| where message contains "Executed 'Functions." and message contains "Succeeded"
| extend FunctionName = extract(@"Executed 'Functions\.([^']+)'", 1, message)
| extend Duration = extract(@"Duration=(\d+)ms", 1, message)
| where isnotempty(FunctionName) and isnotempty(Duration)
| extend DurationMs = toint(Duration)
| summarize 
    ExecutionCount = count(),
    AvgDuration = round(avg(DurationMs), 2),
    MaxDuration = max(DurationMs),
    MinDuration = min(DurationMs),
    P95Duration = round(percentile(DurationMs, 95), 2)
by FunctionName
| order by ExecutionCount desc
"@
        $funcPerfData = Execute-Query -Query $query -Description "Function Execution Performance" -Icon "⚙️"
        
        # Recent function executions
        $query = @"
traces 
| where $traceFilter
| where message contains "Executed 'Functions." and (message contains "Succeeded" or message contains "Failed")
| extend FunctionName = extract(@"Executed 'Functions\.([^']+)'", 1, message)
| extend Duration = extract(@"Duration=(\d+)ms", 1, message)
| extend Status = case(message contains "Succeeded", "Success", "Failed")
| where isnotempty(FunctionName)
| project timestamp, FunctionName, Status, Duration, message
| order by timestamp asc
| take $MaxResults
"@
        $recentExecData = Execute-Query -Query $query -Description "Recent Function Executions" -Icon "🕐"
        
        # Function errors and warnings
        $query = @"
traces 
| where $traceFilter and severityLevel >= 2
| summarize 
    ErrorCount = countif(severityLevel >= 3),
    WarningCount = countif(severityLevel == 2),
    LastError = max(timestamp)
by operation_Name
| where ErrorCount > 0 or WarningCount > 0
| order by ErrorCount desc, WarningCount desc
"@
        $errorSummaryData = Execute-Query -Query $query -Description "Function Error Summary" -Icon "⚠️"
        
        return $funcPerfData + $recentExecData + $errorSummaryData
    }
}

# Main execution
try {
    Write-Header "Azure Log Monitor - Audio Cleaner Web Application"
    
    if ($OutputFormat -eq "Console") {
        Write-Host "🔧 Configuration:" -ForegroundColor $Colors.Header
        Write-Host "   Subscription: $SubscriptionId" -ForegroundColor $Colors.Dim
        Write-Host "   Resource Group: $ResourceGroup" -ForegroundColor $Colors.Dim
        Write-Host "   Time Range: Last $Hours hours" -ForegroundColor $Colors.Dim
        Write-Host "   Log Type: $LogType" -ForegroundColor $Colors.Dim
        Write-Host "   Severity: $Severity" -ForegroundColor $Colors.Dim
        if ($FunctionName -ne "") {
            Write-Host "   Function Filter: $FunctionName" -ForegroundColor $Colors.Dim
        }
        Write-Host "   Max Results: $MaxResults per query" -ForegroundColor $Colors.Dim
    }
    
    $allLogData = @()
    
    switch ($LogType) {
        "All" { $allLogData = Get-AllLogs }
        "Errors" { $allLogData = Get-ErrorLogs }
        "Functions" { $allLogData = Get-FunctionLogs }
        "Requests" { 
            $baseFilter = Get-BaseFilter -TableType "requests"
            $query = "requests | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, name, resultCode, duration, url, success"
            $allLogData = Execute-Query -Query $query -Description "All Requests" -Icon "🌐"
        }
        "Dependencies" { 
            $baseFilter = Get-BaseFilter -TableType "dependencies"
            $query = "dependencies | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, name, type, data, resultCode, duration, success"
            $allLogData = Execute-Query -Query $query -Description "All Dependencies" -Icon "🔗"
        }
        "Performance" { $allLogData = Get-PerformanceLogs }
        "Traces" { 
            $baseFilter = Get-BaseFilter -TableType "traces"
            $query = "traces | where $baseFilter | order by timestamp asc | take $MaxResults | project timestamp, message, severityLevel, operation_Name, cloud_RoleName"
            $allLogData = Execute-Query -Query $query -Description "All Traces" -Icon "📝"
        }
    }
    
    # Output in requested format
    if ($OutputFormat -eq "Json") {
        $allLogData | ConvertTo-Json -Depth 10
    } elseif ($OutputFormat -eq "Csv") {
        $allLogData | ConvertTo-Csv -NoTypeInformation
    }
    
    if ($OutputFormat -eq "Console") {
        Write-Header "Quick Links and Commands" $Colors.Header
        Write-Host "🔗 Application Insights:" -ForegroundColor $Colors.Info
        Write-Host "   https://portal.azure.com/#@silverjunk08gmail.onmicrosoft.com/resource$AppInsightsId/overview" -ForegroundColor $Colors.Highlight
        Write-Host ""
        Write-Host "💡 Usage Examples:" -ForegroundColor $Colors.Info
        Write-Host "   .\Monitor-AzureLogs.ps1 -LogType All -Severity All -VerboseOutput" -ForegroundColor $Colors.Dim
        Write-Host "   .\Monitor-AzureLogs.ps1 -LogType Functions -FunctionName 'upload-file'" -ForegroundColor $Colors.Dim
        Write-Host "   .\Monitor-AzureLogs.ps1 -LogType Performance -Hours 1" -ForegroundColor $Colors.Dim
        Write-Host "   .\Monitor-AzureLogs.ps1 -LogType Errors -OutputFormat Json > errors.json" -ForegroundColor $Colors.Dim
    }
    
} catch {
    Write-Host "❌ Script execution failed: $($_.Exception.Message)" -ForegroundColor $Colors.Error
    exit 1
}

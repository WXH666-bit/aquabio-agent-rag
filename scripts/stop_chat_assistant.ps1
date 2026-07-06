$Root = Split-Path -Parent $PSScriptRoot
$ProcessFile = Join-Path $Root "data\mrag\logs\chat_processes.json"

if (-not (Test-Path $ProcessFile)) {
    Write-Host "No process record found; checking ports 8000 and 8510."
    $Processes = $null
} else {
    $Processes = Get-Content $ProcessFile -Raw | ConvertFrom-Json
}

$Ids = @()
if ($Processes) {
    $Ids += @(
        $Processes.api_pid,
        $Processes.ui_pid,
        $Processes.api_launcher_pid,
        $Processes.ui_launcher_pid,
        $Processes.mcp_pid,
        $Processes.mcp_launcher_pid
    )
}
$Ids += @(
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 8000, 8510, 8765 } |
    Select-Object -ExpandProperty OwningProcess
)
$Ids += @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match "^python" -and
        (
            $_.CommandLine -like "*api_app:app*" -or
            $_.CommandLine -like "*mrag_app.py*" -or
            $_.CommandLine -like "*aquabio_raganything.mcp_server*"
        )
    } |
    Select-Object -ExpandProperty ProcessId
)

foreach ($Id in ($Ids | Where-Object { $_ } | Select-Object -Unique)) {
    if (-not $Id) {
        continue
    }
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$Id" `
        -ErrorAction SilentlyContinue
    $CommandLine = if ($ProcessInfo) { [string]$ProcessInfo.CommandLine } else { "" }
    $IsAquaBioService = (
        $CommandLine -like "*AquaBio-AgentRAG*" -or
        $CommandLine -like "*api_app:app*" -or
        $CommandLine -like "*mrag_app.py*" -or
        $CommandLine -like "*aquabio_raganything.mcp_server*" -or
        $CommandLine -like "*uvicorn*api_app*" -or
        $CommandLine -like "*streamlit*run*mrag_app.py*"
    )
    if (-not $IsAquaBioService) {
        Write-Host "Skipped non-AquaBio process $Id"
        continue
    }
    $Process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if ($Process) {
        try {
            Stop-Process -Id $Id -Force
            Write-Host "Stopped process $Id"
        } catch {
            Write-Host "Could not stop process ${Id}: $($_.Exception.Message)"
        }
    }
}

if (Test-Path $ProcessFile) {
    try {
        Remove-Item -LiteralPath $ProcessFile -Force
    } catch {
        Write-Host "Could not remove process record: $($_.Exception.Message)"
    }
}

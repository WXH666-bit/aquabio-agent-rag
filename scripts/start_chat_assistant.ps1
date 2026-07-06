param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8510,
    [int]$McpPort = 8765,
    [switch]$Visible
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RagPython = Join-Path $Root ".venv-raganything\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Project virtual environment not found: $Python"
}
if (-not (Test-Path $RagPython)) {
    throw "RAG-Anything virtual environment not found: $RagPython"
}

$LogDir = Join-Path $Root "data\mrag\logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Get-CommandLineForPid {
    param([int]$ProcessId)
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" `
        -ErrorAction SilentlyContinue
    if ($ProcessInfo) {
        return [string]$ProcessInfo.CommandLine
    }
    return ""
}

function Get-ExecutablePathForPid {
    param([int]$ProcessId)
    $ProcessInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" `
        -ErrorAction SilentlyContinue
    if ($ProcessInfo) {
        return [string]$ProcessInfo.ExecutablePath
    }
    return ""
}

function Stop-StaleAquaBioPortProcess {
    param(
        [int]$Port,
        [string]$ExpectedRoot
    )
    $Connections = Get-NetTCPConnection `
        -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($Connection in $Connections) {
        $OwnerPid = [int]$Connection.OwningProcess
        if (-not $OwnerPid) {
            continue
        }
        $CommandLine = Get-CommandLineForPid $OwnerPid
        $LooksLikeAquaBio = (
            $CommandLine -like "*api_app:app*" -or
            $CommandLine -like "*mrag_app.py*" -or
            $CommandLine -like "*aquabio_raganything.mcp_server*" -or
            $CommandLine -like "*AquaBio-AgentRAG*"
        )
        if (-not $LooksLikeAquaBio) {
            throw "Port $Port is occupied by a non-AquaBio process PID $OwnerPid. Command: $CommandLine"
        }
        if ($CommandLine -notlike "*$ExpectedRoot*") {
            Write-Warning "Stopping stale AquaBio process on port $Port, PID $OwnerPid. Command: $CommandLine"
        } else {
            Write-Warning "Restarting existing AquaBio process on port $Port, PID $OwnerPid."
        }
        Stop-Process -Id $OwnerPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

$ExistingApi = Get-NetTCPConnection `
    -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
$ExistingUi = Get-NetTCPConnection `
    -LocalPort $UiPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
$ExistingMcp = Get-NetTCPConnection `
    -LocalPort $McpPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($ExistingApi -and $ExistingUi -and $ExistingMcp) {
    try {
        $Health = Invoke-RestMethod `
            "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 5
    } catch {
        $Health = $null
    }
    $ApiCommandLine = Get-CommandLineForPid ([int]$ExistingApi.OwningProcess)
    $UiCommandLine = Get-CommandLineForPid ([int]$ExistingUi.OwningProcess)
    $McpCommandLine = Get-CommandLineForPid ([int]$ExistingMcp.OwningProcess)
    $RunningCurrentRoot = (
        $ApiCommandLine -like "*$Root*" -and
        $UiCommandLine -like "*$Root*" -and
        $McpCommandLine -like "*$Root*"
    )
    if ($Health.status -eq "ok" -and $RunningCurrentRoot) {
        Write-Host "AquaBio chat assistant is already running."
        Write-Host "API PID: $($ExistingApi.OwningProcess)"
        Write-Host "UI PID:  $($ExistingUi.OwningProcess)"
        Write-Host "MCP PID: $($ExistingMcp.OwningProcess)"
        Write-Host "API: http://127.0.0.1:$ApiPort/docs"
        Write-Host "UI:  http://127.0.0.1:$UiPort"
        Write-Host "MCP: http://127.0.0.1:$McpPort/mcp"
        exit 0
    } elseif ($Health.status -eq "ok") {
        Write-Warning "AquaBio-like services are running, but they are not rooted at $Root. Restarting them."
    }
}
if ($ExistingApi -or $ExistingUi -or $ExistingMcp) {
    Stop-StaleAquaBioPortProcess -Port $ApiPort -ExpectedRoot $Root
    Stop-StaleAquaBioPortProcess -Port $UiPort -ExpectedRoot $Root
    Stop-StaleAquaBioPortProcess -Port $McpPort -ExpectedRoot $Root
    Start-Sleep -Seconds 2
}

if ($Visible) {
    $McpCommand = (
        "title AquaBio RAG-Anything MCP && " +
        "cd /d `"$Root`" && " +
        "set `"AQUABIO_RAG_MCP_TRANSPORT=streamable-http`" && " +
        "set `"AQUABIO_RAG_MCP_PORT=$McpPort`" && " +
        "set `"PYTHONPATH=$Root\src`" && " +
        "`"$RagPython`" -m aquabio_raganything.mcp_server"
    )
    $McpLauncher = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/k", $McpCommand) `
        -WorkingDirectory $Root `
        -PassThru
} else {
    $env:AQUABIO_RAG_MCP_TRANSPORT = "streamable-http"
    $env:AQUABIO_RAG_MCP_PORT = "$McpPort"
    $env:PYTHONPATH = (Join-Path $Root "src")
    $env:HF_HOME = if ($env:HF_HOME) { $env:HF_HOME } else { "F:\huggingface" }
    $env:HUGGINGFACE_HUB_CACHE = if ($env:HUGGINGFACE_HUB_CACHE) {
        $env:HUGGINGFACE_HUB_CACHE
    } else {
        Join-Path $env:HF_HOME "hub"
    }
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $McpLauncher = Start-Process `
        -FilePath $RagPython `
        -ArgumentList @("-m", "aquabio_raganything.mcp_server") `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "mcp.stdout.log") `
        -RedirectStandardError (Join-Path $LogDir "mcp.stderr.log") `
        -PassThru
}

if ($Visible) {
    $ApiCommand = (
        "title AquaBio FastAPI Backend && " +
        "cd /d `"$Root`" && " +
        "`"$Python`" -m uvicorn api_app:app " +
        "--host 127.0.0.1 --port $ApiPort"
    )
    $ApiLauncher = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/k", $ApiCommand) `
        -WorkingDirectory $Root `
        -PassThru
} else {
    $ApiLauncher = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "uvicorn", "api_app:app",
            "--host", "127.0.0.1",
            "--port", "$ApiPort"
        ) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "api.stdout.log") `
        -RedirectStandardError (Join-Path $LogDir "api.stderr.log") `
        -PassThru
}

$env:AQUABIO_API_URL = "http://127.0.0.1:$ApiPort"
if ($Visible) {
    $UiCommand = (
        "title AquaBio Streamlit Frontend && " +
        "cd /d `"$Root`" && " +
        "set `"AQUABIO_API_URL=http://127.0.0.1:$ApiPort`" && " +
        "`"$Python`" -m streamlit run mrag_app.py " +
        "--server.port $UiPort --server.headless true " +
        "--browser.gatherUsageStats false"
    )
    $UiLauncher = Start-Process `
        -FilePath "cmd.exe" `
        -ArgumentList @("/k", $UiCommand) `
        -WorkingDirectory $Root `
        -PassThru
} else {
    $UiLauncher = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "streamlit", "run", "mrag_app.py",
            "--server.port", "$UiPort",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ) `
        -WorkingDirectory $Root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $LogDir "ui.stdout.log") `
        -RedirectStandardError (Join-Path $LogDir "ui.stderr.log") `
        -PassThru
}

$Deadline = (Get-Date).AddSeconds(90)
do {
    Start-Sleep -Milliseconds 500
    $ApiConnection = Get-NetTCPConnection `
        -LocalPort $ApiPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $UiConnection = Get-NetTCPConnection `
        -LocalPort $UiPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $McpConnection = Get-NetTCPConnection `
        -LocalPort $McpPort -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
} while (
    (
        -not $ApiConnection -or
        -not $UiConnection -or
        -not $McpConnection
    ) -and
    (Get-Date) -lt $Deadline
)

if (-not $ApiConnection -or -not $UiConnection) {
    throw "API or UI did not start within 90 seconds. Check data\mrag\logs."
}
if (-not $McpConnection) {
    Write-Warning (
        "RAG-Anything MCP did not open port $McpPort within 90 seconds. " +
        "API/UI will continue to run; graph/PDF-image MCP retrieval will be marked unavailable."
    )
}

try {
    $Health = Invoke-RestMethod `
        "http://127.0.0.1:$ApiPort/api/health" -TimeoutSec 10
} catch {
    throw "FastAPI opened port $ApiPort but health check failed: $($_.Exception.Message)"
}
if ($Health.status -ne "ok") {
    throw "FastAPI health check returned an unexpected response."
}

function Invoke-RestMethodWithRetry {
    param(
        [string]$Uri,
        [int]$TimeoutSec = 10,
        [int]$TotalWaitSec = 90
    )
    $Deadline = (Get-Date).AddSeconds($TotalWaitSec)
    $LastError = $null
    do {
        try {
            return Invoke-RestMethod $Uri -TimeoutSec $TimeoutSec
        } catch {
            $LastError = $_.Exception.Message
            Start-Sleep -Seconds 2
        }
    } while ((Get-Date) -lt $Deadline)
    throw "Request timed out after $TotalWaitSec seconds: $Uri. Last error: $LastError"
}

$RuntimeStatus = Invoke-RestMethodWithRetry `
    -Uri "http://127.0.0.1:$ApiPort/api/system/status" `
    -TimeoutSec 10 `
    -TotalWaitSec 90
if (-not $RuntimeStatus.python_runtime.uses_project_venv) {
    throw (
        "FastAPI is not using the project virtual environment. " +
        "Expected prefix $Root\.venv; actual prefix " +
        "$($RuntimeStatus.python_runtime.prefix); executable " +
        "$($RuntimeStatus.python_runtime.executable)."
    )
}

if ($McpConnection) {
    Write-Host "Checking RAG-Anything MCP tools..."
    try {
        $McpWarmupScript = @"
from pathlib import Path
from aquabio_mrag.mcp_client import project_mcp_client
client = project_mcp_client(Path(r"$Root"))
tools = client.list_tools_sync("raganything")
print("ready", ",".join(tool["name"] for tool in tools))
"@
        $McpWarmupOutput = $McpWarmupScript |
            & $Python -
        Write-Host "RAG-Anything MCP check completed: $McpWarmupOutput"
    } catch {
        Write-Warning (
            "RAG-Anything MCP check failed: $($_.Exception.Message). " +
            "The assistant will still run and fall back to Chroma/local image retrieval."
        )
    }
}

Write-Host "Warming LangGraph, BGE-M3 and Chroma. This can take several minutes on the first start..."
try {
    $Warmup = Invoke-RestMethod `
        -Method Post `
        "http://127.0.0.1:$ApiPort/api/system/warmup" `
        -TimeoutSec 300
} catch {
    throw "Backend warmup failed: $($_.Exception.Message)"
}
if ($Warmup.status -ne "ready") {
    throw "Backend warmup did not complete: $($Warmup.detail)"
}
Write-Host "Backend warmup completed in $($Warmup.elapsed_seconds) seconds."

@{
    api_pid = $ApiConnection.OwningProcess
    ui_pid = $UiConnection.OwningProcess
    api_launcher_pid = $ApiLauncher.Id
    ui_launcher_pid = $UiLauncher.Id
    mcp_pid = $McpConnection.OwningProcess
    mcp_launcher_pid = $McpLauncher.Id
    api_url = "http://127.0.0.1:$ApiPort"
    ui_url = "http://127.0.0.1:$UiPort"
    mcp_url = "http://127.0.0.1:$McpPort/mcp"
} | ConvertTo-Json | Set-Content `
    (Join-Path $LogDir "chat_processes.json") -Encoding UTF8

Write-Host "AquaBio chat assistant started."
Write-Host "Backend and frontend consoles are visible: $Visible"
Write-Host "API: http://127.0.0.1:$ApiPort/docs"
Write-Host "UI:  http://127.0.0.1:$UiPort"
Write-Host "MCP: http://127.0.0.1:$McpPort/mcp"

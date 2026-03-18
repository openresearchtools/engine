param(
    [string]$WindowsBundleDir = "",
    [string]$WindowsCargoTargetDir = "",
    [string]$WindowsRuntimeDir = "",
    [string]$MacHost = "user@192.168.8.108",
    [string]$MacBundleDir = "/Users/user/ENGINEbuilds/engine-controller-macos-metal/bundle",
    [string]$MacRuntimeDir = "/Users/user/ENGINEbuilds/engine-controller-macos-metal/bundle",
    [string]$MacRepoRoot = "/Users/user/ENGINE",
    [string]$MacCargoTargetDir = "/Users/user/ENGINEbuilds/engine-controller-macos-metal/cargo-target",
    [string]$WindowsModel4B = "D:\QWEN 3.5\gguf\Qwen3.5-4B\Qwen3.5-4B-Q4_K_M.gguf",
    [string]$WindowsModel35B = "D:\QWEN 3.5\gguf\Qwen3.5-35B-A3B\Qwen3.5-35B-A3B-Q8_0.gguf",
    [string]$MacModel2B = "/Users/user/ENGINEbuilds/Qwen3.5-2B-Q4_K_M.gguf",
    [string]$BuildRoot = "",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [string]$BasePath
    )
    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $PathValue))
}

function Invoke-LoggedCommand {
    param(
        [string]$Label,
        [string]$LogPath,
        [scriptblock]$Command
    )

    Write-Host "==> $Label"
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Command 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }

    if ($exitCode -ne 0) {
        throw "$Label failed (see $LogPath)"
    }
}

function Clear-ClusterEnv {
    $names = @(
        'CLUSTER_AGENT_ADDR',
        'CLUSTER_ACTION',
        'CLUSTER_MODEL_PATH',
        'CLUSTER_GROUP_ID',
        'CLUSTER_RPC_SERVERS',
        'CLUSTER_PROMPT',
        'CLUSTER_INSTANCE_NAME',
        'CLUSTER_RETENTION_MODE',
        'CLUSTER_POST_CHAT_SLEEP_SECONDS',
        'CLUSTER_SKIP_CLEANUP',
        'CLUSTER_N_CTX',
        'CLUSTER_N_BATCH',
        'CLUSTER_N_UBATCH',
        'CLUSTER_N_THREADS',
        'CLUSTER_N_THREADS_BATCH',
        'CLUSTER_N_GPU_LAYERS',
        'CLUSTER_N_PREDICT',
        'CLUSTER_TEMPERATURE',
        'CLUSTER_TOP_P',
        'CLUSTER_TOP_K',
        'CLUSTER_MIN_P',
        'CLUSTER_REPEAT_LAST_N',
        'CLUSTER_REPEAT_PENALTY'
    )

    foreach ($name in $names) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"

if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BuildRoot = Join-Path $buildsRoot "cluster-matrix-$stamp"
}
$BuildRoot = Resolve-AbsolutePath -PathValue $BuildRoot -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($WindowsBundleDir)) {
    $WindowsBundleDir = Join-Path $buildsRoot "engine-controller-final-win-manual\bundle"
}
$WindowsBundleDir = Resolve-AbsolutePath -PathValue $WindowsBundleDir -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($WindowsCargoTargetDir)) {
    $WindowsCargoTargetDir = Join-Path $buildsRoot "engine-controller-final-win-manual\cargo-target"
}
$WindowsCargoTargetDir = Resolve-AbsolutePath -PathValue $WindowsCargoTargetDir -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($WindowsRuntimeDir)) {
    $WindowsRuntimeDir = $WindowsBundleDir
}
$WindowsRuntimeDir = Resolve-AbsolutePath -PathValue $WindowsRuntimeDir -BasePath $repoRoot

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

$windowsHarness = Join-Path $WindowsCargoTargetDir "release\cluster-agent-harness.exe"
$windowsUiExe = Join-Path $WindowsBundleDir "Engine.exe"
$windowsCtlExe = $windowsUiExe
$windowsLogsDir = Join-Path $WindowsBundleDir "logs"
New-Item -ItemType Directory -Force -Path $windowsLogsDir | Out-Null

if (-not $SkipBuild) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq 'Engine.exe' -and $_.ExecutablePath -eq $windowsUiExe } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

    Invoke-LoggedCommand -Label "Sync cluster sources to macOS repo" -LogPath (Join-Path $BuildRoot "sync-macos-sources.log") -Command {
        ssh $MacHost "mkdir -p $MacRepoRoot/clusterui/src/bin $MacRepoRoot/bridge"
        & scp `
            (Join-Path $repoRoot "clusterui\src\agent.rs") `
            (Join-Path $repoRoot "clusterui\src\cluster_api.rs") `
            (Join-Path $repoRoot "clusterui\src\main.rs") `
            (Join-Path $repoRoot "clusterui\src\protocol.rs") `
            (Join-Path $repoRoot "clusterui\src\tray.rs") `
            "${MacHost}:$MacRepoRoot/clusterui/src/"
        & scp `
            (Join-Path $repoRoot "bridge\llama_server_cluster.cpp") `
            (Join-Path $repoRoot "bridge\llama_server_cluster.h") `
            "${MacHost}:$MacRepoRoot/bridge/"
    }

    Invoke-LoggedCommand -Label "Build Windows cluster controller" -LogPath (Join-Path $BuildRoot "build-windows-controller.log") -Command {
        & (Join-Path $repoRoot "build\build_cluster_controller.ps1") -Backend cuda -BuildRoot (Split-Path -Parent $WindowsBundleDir) -BundleDir $WindowsBundleDir -CargoTargetDir $WindowsCargoTargetDir
    }

    Invoke-LoggedCommand -Label "Build Windows cluster harnesses" -LogPath (Join-Path $BuildRoot "build-windows-harnesses.log") -Command {
        cargo build --release --manifest-path (Join-Path $repoRoot "clusterui\Cargo.toml") --bin cluster-agent-harness --bin cluster-direct-harness --target-dir $WindowsCargoTargetDir
    }

    Invoke-LoggedCommand -Label "Build macOS cluster controller" -LogPath (Join-Path $BuildRoot "build-macos-controller.log") -Command {
        ssh $MacHost "cd $MacRepoRoot && bash build/build_cluster_controller_macos.sh"
    }

    Invoke-LoggedCommand -Label "Build macOS cluster harnesses" -LogPath (Join-Path $BuildRoot "build-macos-harnesses.log") -Command {
        ssh $MacHost "cd $MacRepoRoot && cargo build --release --manifest-path clusterui/Cargo.toml --bin cluster-agent-harness --bin cluster-direct-harness --target-dir $MacCargoTargetDir"
    }
}

Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'Engine.exe' -and $_.ExecutablePath -eq $windowsUiExe } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Start-Sleep -Seconds 2

Start-Process -FilePath $windowsUiExe `
    -ArgumentList @("--agent", "--runtime-dir", $WindowsRuntimeDir, "--bind", "0.0.0.0:46211") `
    -RedirectStandardOutput (Join-Path $windowsLogsDir "agent.stdout.log") `
    -RedirectStandardError (Join-Path $windowsLogsDir "agent.stderr.log") `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

$windowsReady = $false
for ($i = 0; $i -lt 8; $i++) {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $windowsCtlExe --dump-state --bind 127.0.0.1:46211 *> $null
        if ($LASTEXITCODE -eq 0) {
            $windowsReady = $true
            break
        }
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    Start-Sleep -Seconds 2
}

if (-not $windowsReady) {
    throw "Windows agent did not become ready"
}

Invoke-LoggedCommand -Label "Start macOS agent" -LogPath (Join-Path $BuildRoot "start-macos-agent.log") -Command {
    ssh $MacHost 'bundle=/Users/user/ENGINEbuilds/engine-controller-macos-metal/bundle; pkill -f "engine-controller-macos-metal/bundle/Engine --agent" || true; pkill -f "engine-controller-macos-metal/bundle/rpc-server" || true; nohup "$bundle/Engine" --agent --runtime-dir "$bundle" --bind 0.0.0.0:46211 > "$bundle/logs/agent.stdout.log" 2> "$bundle/logs/agent.stderr.log" < /dev/null &'
}

$macReady = $false
for ($i = 0; $i -lt 8; $i++) {
    $oldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        ssh $MacHost "$MacBundleDir/Engine --dump-state --bind 127.0.0.1:46211" *> $null
        if ($LASTEXITCODE -eq 0) {
            $macReady = $true
            break
        }
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    Start-Sleep -Seconds 2
}

if (-not $macReady) {
    throw "macOS agent did not become ready"
}

Invoke-LoggedCommand -Label "Windows snapshot" -LogPath (Join-Path $BuildRoot "snapshot-windows.log") -Command {
    & $windowsCtlExe --dump-state --bind 127.0.0.1:46211
}

Invoke-LoggedCommand -Label "Mac snapshot" -LogPath (Join-Path $BuildRoot "snapshot-macos.log") -Command {
    ssh $MacHost "$MacBundleDir/Engine --dump-state --bind 127.0.0.1:46211"
}

Invoke-LoggedCommand -Label "Add Mac peer on Windows" -LogPath (Join-Path $BuildRoot "peer-add-windows.log") -Command {
    & $windowsCtlExe --add-peer 192.168.8.108:46211 --bind 127.0.0.1:46211
}

$tunnelLogDir = $windowsLogsDir
Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq 'ssh.exe' -and $_.CommandLine -like '*55211:127.0.0.1:46211*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Start-Process -FilePath ssh.exe `
        -ArgumentList @('-N', '-R', '55211:127.0.0.1:46211', '-R', '56214:127.0.0.1:46214', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30', $MacHost) `
    -RedirectStandardOutput (Join-Path $tunnelLogDir 'ssh-tunnel.stdout.log') `
    -RedirectStandardError (Join-Path $tunnelLogDir 'ssh-tunnel.stderr.log') `
    -WindowStyle Hidden

Start-Sleep -Seconds 3

Invoke-LoggedCommand -Label "Add Windows tunneled peer on Mac" -LogPath (Join-Path $BuildRoot "peer-add-macos.log") -Command {
    ssh $MacHost "$MacBundleDir/Engine --add-peer 127.0.0.1:55211 --bind 127.0.0.1:46211"
}

Invoke-LoggedCommand -Label "Mac local 2B" -LogPath (Join-Path $BuildRoot "mac-local-2b.log") -Command {
    ssh $MacHost "env CLUSTER_AGENT_ADDR=127.0.0.1:46211 CLUSTER_MODEL_PATH='$MacModel2B' CLUSTER_GROUP_ID=device:0 CLUSTER_PROMPT=OK CLUSTER_INSTANCE_NAME=mac-local-2b CLUSTER_N_CTX=2048 CLUSTER_N_BATCH=256 CLUSTER_N_UBATCH=256 CLUSTER_N_THREADS=4 CLUSTER_N_THREADS_BATCH=4 CLUSTER_N_GPU_LAYERS=999 $MacCargoTargetDir/release/cluster-agent-harness"
}

Invoke-LoggedCommand -Label "Windows local 4B" -LogPath (Join-Path $BuildRoot "win-local-4b.log") -Command {
    Clear-ClusterEnv
    $env:CLUSTER_AGENT_ADDR = '127.0.0.1:46211'
    $env:CLUSTER_MODEL_PATH = $WindowsModel4B
    $env:CLUSTER_GROUP_ID = 'device:1'
    $env:CLUSTER_PROMPT = 'OK'
    $env:CLUSTER_INSTANCE_NAME = 'win-local-4b'
    $env:CLUSTER_N_CTX = '2048'
    $env:CLUSTER_N_BATCH = '256'
    $env:CLUSTER_N_UBATCH = '256'
    $env:CLUSTER_N_THREADS = '8'
    $env:CLUSTER_N_THREADS_BATCH = '8'
    $env:CLUSTER_N_GPU_LAYERS = '999'
    & $windowsHarness
}

Invoke-LoggedCommand -Label "Windows load_on_demand grace" -LogPath (Join-Path $BuildRoot "win-grace.log") -Command {
    Clear-ClusterEnv
    $env:CLUSTER_AGENT_ADDR = '127.0.0.1:46211'
    $env:CLUSTER_MODEL_PATH = $WindowsModel4B
    $env:CLUSTER_GROUP_ID = 'device:1'
    $env:CLUSTER_PROMPT = 'OK'
    $env:CLUSTER_INSTANCE_NAME = 'win-grace'
    $env:CLUSTER_RETENTION_MODE = 'load_on_demand'
    $env:CLUSTER_POST_CHAT_SLEEP_SECONDS = '5'
    $env:CLUSTER_SKIP_CLEANUP = '1'
    $env:CLUSTER_N_CTX = '2048'
    $env:CLUSTER_N_BATCH = '256'
    $env:CLUSTER_N_UBATCH = '256'
    $env:CLUSTER_N_THREADS = '8'
    $env:CLUSTER_N_THREADS_BATCH = '8'
    $env:CLUSTER_N_GPU_LAYERS = '999'
    & $windowsHarness
}

Start-Sleep -Seconds 35

Invoke-LoggedCommand -Label "Windows post-grace snapshot" -LogPath (Join-Path $BuildRoot "win-post-grace-snapshot.log") -Command {
    & $windowsCtlExe --dump-state --bind 127.0.0.1:46211
}

Invoke-LoggedCommand -Label "Windows hybrid 4B" -LogPath (Join-Path $BuildRoot "win-hybrid-4b.log") -Command {
    Clear-ClusterEnv
    $env:CLUSTER_AGENT_ADDR = '127.0.0.1:46211'
    $env:CLUSTER_MODEL_PATH = $WindowsModel4B
    $env:CLUSTER_GROUP_ID = 'cluster-split-gpu-all'
$env:CLUSTER_RPC_SERVERS = '192.168.8.108:46214'
    $env:CLUSTER_PROMPT = 'OK'
    $env:CLUSTER_INSTANCE_NAME = 'win-hybrid-4b'
    $env:CLUSTER_N_CTX = '2048'
    $env:CLUSTER_N_BATCH = '512'
    $env:CLUSTER_N_UBATCH = '512'
    $env:CLUSTER_N_THREADS = '8'
    $env:CLUSTER_N_THREADS_BATCH = '8'
    $env:CLUSTER_N_GPU_LAYERS = '999'
    & $windowsHarness
}

Invoke-LoggedCommand -Label "Windows hybrid 35B" -LogPath (Join-Path $BuildRoot "win-hybrid-35b.log") -Command {
    Clear-ClusterEnv
    $env:CLUSTER_AGENT_ADDR = '127.0.0.1:46211'
    $env:CLUSTER_MODEL_PATH = $WindowsModel35B
    $env:CLUSTER_GROUP_ID = 'cluster-split-gpu-all'
$env:CLUSTER_RPC_SERVERS = '192.168.8.108:46214'
    $env:CLUSTER_PROMPT = 'OK'
    $env:CLUSTER_INSTANCE_NAME = 'win-hybrid-35b'
    $env:CLUSTER_N_CTX = '2048'
    $env:CLUSTER_N_BATCH = '512'
    $env:CLUSTER_N_UBATCH = '512'
    $env:CLUSTER_N_THREADS = '8'
    $env:CLUSTER_N_THREADS_BATCH = '8'
    $env:CLUSTER_N_GPU_LAYERS = '999'
    & $windowsHarness
}

Invoke-LoggedCommand -Label "Mac controls Windows hybrid via tunnel" -LogPath (Join-Path $BuildRoot "mac-controls-win-hybrid.log") -Command {
ssh $MacHost "CLUSTER_AGENT_ADDR=127.0.0.1:55211 CLUSTER_MODEL_PATH='$WindowsModel4B' CLUSTER_GROUP_ID=cluster-split-gpu-all CLUSTER_RPC_SERVERS=192.168.8.108:46214 CLUSTER_PROMPT=OK CLUSTER_INSTANCE_NAME=mac-controls-win-hybrid CLUSTER_N_CTX=2048 CLUSTER_N_BATCH=512 CLUSTER_N_UBATCH=512 CLUSTER_N_THREADS=8 CLUSTER_N_THREADS_BATCH=8 CLUSTER_N_GPU_LAYERS=999 CLUSTER_N_PREDICT=16 $MacCargoTargetDir/release/cluster-agent-harness"
}

Write-Host "Cluster matrix complete."
Write-Host "Logs: $BuildRoot"

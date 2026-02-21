param(
    [string]$CmakeExe = "cmake",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Backend = "cuda",
    [string]$LlamaCppDir = "",
    [string]$BuildRoot = "",
    [string]$BuildDir = "",
    [string]$BridgeSrcDir = "",
    [string]$BridgeRelativeDir = "MARKDOWN\\bridge",
    [bool]$ApplyBridgeOverlay = $true,
    [bool]$ApplyDiarizeOverlay = $true,
    [switch]$BuildLlamaServerCli,
    [switch]$BuildPyannoteCli,
    [string[]]$OverlaySearchRoots = @(),
    [switch]$EnableFfmpeg,
    [bool]$EnableBackendDl = $false,
    [bool]$EnableCpuAllVariants = $false,
    [bool]$DisableGgmlNative = $false,
    [string]$FfmpegRoot = "",
    [ValidateSet("off", "openssl", "boringssl", "libressl")]
    [string]$HttpsBackend = "boringssl",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Test-IsUnderPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    $fullPath = [System.IO.Path]::GetFullPath($PathValue).TrimEnd('\') + '\'
    $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    return $fullPath.StartsWith($fullBase, [System.StringComparison]::OrdinalIgnoreCase)
}

function Resolve-OverlayRoot {
    param(
        [string[]]$SearchRoots,
        [string]$RepoRoot
    )

    $candidates = @()
    if ($SearchRoots -and $SearchRoots.Count -gt 0) {
        foreach ($root in $SearchRoots) {
            if ([string]::IsNullOrWhiteSpace($root)) {
                continue
            }
            $candidate = Resolve-AbsolutePath -PathValue $root -RepoRoot $RepoRoot
            $candidates += $candidate
        }
    } else {
        $candidates += (Join-Path $RepoRoot "diarize\\addons\\overlay\\llama.cpp")
        $candidates += (Join-Path $RepoRoot "diarize\\overlay\\llama.cpp")
        $candidates += (Join-Path $RepoRoot "addons\\overlay\\llama.cpp")
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

function Copy-OverlayTree {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot
    )

    $files = Get-ChildItem -Path $SourceRoot -Recurse -File
    foreach ($src in $files) {
        $rel = $src.FullName.Substring($SourceRoot.Length + 1)
        $dst = Join-Path $DestinationRoot $rel
        $dstDir = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        Copy-Item -LiteralPath $src.FullName -Destination $dst -Force
    }
}

function Invoke-CmakeBuildTargets {
    param(
        [string]$Cmake,
        [string]$BuildDirPath,
        [string]$ConfigName,
        [string[]]$Targets,
        [int]$ParallelJobs
    )

    $args = @("--build", $BuildDirPath, "--config", $ConfigName)
    if ($ParallelJobs -gt 0) {
        $args += @("--parallel", $ParallelJobs)
    }
    $args += @("--target")
    $args += $Targets

    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Cmake @args 2>&1 | ForEach-Object { Write-Host $_ }
        $cmakeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($cmakeExitCode -ne 0) {
        throw "CMake build failed for targets: $($Targets -join ', ')"
    }
}

function Resolve-CmakeExecutable {
    param(
        [string]$ToolValue
    )

    if (-not [string]::IsNullOrWhiteSpace($ToolValue) -and [System.IO.Path]::IsPathRooted($ToolValue)) {
        if (Test-Path -LiteralPath $ToolValue) {
            return [System.IO.Path]::GetFullPath($ToolValue)
        }
        throw "CMake executable not found at: $ToolValue"
    }

    if (-not [string]::IsNullOrWhiteSpace($ToolValue)) {
        $cmd = Get-Command $ToolValue -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }

    $vsCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )

    foreach ($candidate in $vsCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "CMake executable not found. Install cmake or pass -CmakeExe with full path."
}

function Ensure-BridgeCmakeHooks {
    param(
        [string]$LlamaRoot
    )

    $cmakePath = Join-Path $LlamaRoot "CMakeLists.txt"
    if (-not (Test-Path -LiteralPath $cmakePath)) {
        throw "Cannot patch bridge hooks. Missing file: $cmakePath"
    }

    $text = Get-Content -Raw -LiteralPath $cmakePath
    $newline = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
    $updated = $text

    if ($updated -notmatch 'LLAMA_BUILD_MARKDOWN_BRIDGE') {
        $optionLine = 'option(LLAMA_BUILD_MARKDOWN_BRIDGE "llama: build markdown in-process bridge library" OFF)'
        $serverOptionPattern = 'option\(LLAMA_BUILD_SERVER\s+"[^"]+"\s+\$\{LLAMA_STANDALONE\}\)'
        $serverOptionMatch = [regex]::Match($updated, $serverOptionPattern)
        if (-not $serverOptionMatch.Success) {
            throw "Failed to locate LLAMA_BUILD_SERVER option in $cmakePath to inject bridge option."
        }
        $insertText = $serverOptionMatch.Value + $newline + $optionLine
        $updated = $updated.Remove($serverOptionMatch.Index, $serverOptionMatch.Length).Insert($serverOptionMatch.Index, $insertText)
    }

    if ($updated -notmatch 'add_subdirectory\(MARKDOWN/bridge\)') {
        $toolsBlockPattern = 'if \(LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS\)\s*add_subdirectory\(tools\)\s*endif\(\)'
        $toolsBlockMatch = [regex]::Match($updated, $toolsBlockPattern)
        if (-not $toolsBlockMatch.Success) {
            throw "Failed to locate tools block in $cmakePath to inject MARKDOWN/bridge subdirectory."
        }
        $bridgeBlock = @(
            "",
            "if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_MARKDOWN_BRIDGE)",
            "    add_subdirectory(MARKDOWN/bridge)",
            "endif()"
        ) -join $newline
        $insertText = $toolsBlockMatch.Value + $bridgeBlock
        $updated = $updated.Remove($toolsBlockMatch.Index, $toolsBlockMatch.Length).Insert($toolsBlockMatch.Index, $insertText)
    }

    if ($updated -ne $text) {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($cmakePath, $updated, $encoding)
        Write-Host "Patched llama.cpp CMakeLists bridge hooks: $cmakePath"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$CmakeExe = Resolve-CmakeExecutable -ToolValue $CmakeExe

if ([string]::IsNullOrWhiteSpace($LlamaCppDir)) {
    $LlamaCppDir = Join-Path $repoRoot "third_party\\llama.cpp"
}
$LlamaCppDir = Resolve-AbsolutePath -PathValue $LlamaCppDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $buildsRoot "llama"
}
$BuildRoot = Resolve-AbsolutePath -PathValue $BuildRoot -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $BuildRoot ("cmake-" + $Backend + "-" + $Config.ToLowerInvariant())
}
$BuildDir = Resolve-AbsolutePath -PathValue $BuildDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $BuildDir -BasePath $repoRoot) {
    throw "BuildDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $BuildDir"
}

if ([string]::IsNullOrWhiteSpace($BridgeSrcDir)) {
    $BridgeSrcDir = Join-Path $repoRoot "bridge"
}
$BridgeSrcDir = Resolve-AbsolutePath -PathValue $BridgeSrcDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($FfmpegRoot)) {
    $FfmpegRoot = Join-Path $buildsRoot "runtime-deps\\ffmpeg"
}
$FfmpegRoot = Resolve-AbsolutePath -PathValue $FfmpegRoot -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $FfmpegRoot -BasePath $repoRoot) {
    throw "FfmpegRoot must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $FfmpegRoot"
}

$cmakeLists = Join-Path $LlamaCppDir "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeLists)) {
    throw "llama.cpp source not found at: $LlamaCppDir"
}

$cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
if ($cmakeText -match 'project\("whisper\.cpp"') {
    throw "Expected llama.cpp at '$LlamaCppDir' but found whisper.cpp sources. Fix third_party/llama.cpp first."
}
if ($cmakeText -notmatch 'project\("llama\.cpp"') {
    throw "Unable to verify llama.cpp source root at '$LlamaCppDir'."
}

if ($ApplyBridgeOverlay) {
    Ensure-BridgeCmakeHooks -LlamaRoot $LlamaCppDir
}

if ($ApplyBridgeOverlay) {
    if (-not (Test-Path -LiteralPath $BridgeSrcDir)) {
        throw "Bridge source dir not found: $BridgeSrcDir"
    }
    $bridgeDstDir = Join-Path $LlamaCppDir $BridgeRelativeDir
    New-Item -ItemType Directory -Force -Path $bridgeDstDir | Out-Null
    Copy-Item -Path (Join-Path $BridgeSrcDir "*") -Destination $bridgeDstDir -Recurse -Force
}

if ($ApplyDiarizeOverlay) {
    $overlayRoot = Resolve-OverlayRoot -SearchRoots $OverlaySearchRoots -RepoRoot $repoRoot
    if (-not $overlayRoot) {
        throw "Could not locate diarize overlay root. Checked default candidates and provided OverlaySearchRoots."
    }
    Copy-OverlayTree -SourceRoot $overlayRoot -DestinationRoot $LlamaCppDir
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

$cmakeArgs = @(
    "-S", $LlamaCppDir,
    "-B", $BuildDir,
    "-DBUILD_SHARED_LIBS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_MARKDOWN_BRIDGE=ON",
    "-DLLAMA_HTTPLIB=ON"
)

switch ($HttpsBackend) {
    "off" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "openssl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=ON"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "boringssl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=ON"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "libressl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=ON"
    }
}

switch ($Backend) {
    "cuda" {
        $cmakeArgs += "-DGGML_CUDA=ON"
        $cmakeArgs += "-DGGML_CUDA_CUB_3DOT2=ON"
    }
    "vulkan" {
        $cmakeArgs += "-DGGML_VULKAN=ON"
    }
    default { }
}

if ($EnableCpuAllVariants) {
    $EnableBackendDl = $true
}
if ($EnableBackendDl) {
    $cmakeArgs += "-DGGML_BACKEND_DL=ON"
}
if ($EnableCpuAllVariants) {
    $cmakeArgs += "-DGGML_CPU_ALL_VARIANTS=ON"
}
if ($DisableGgmlNative) {
    $cmakeArgs += "-DGGML_NATIVE=OFF"
}

if ($EnableFfmpeg) {
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON"
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_FFMPEG_ROOT=$FfmpegRoot"
}

$oldErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $CmakeExe @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
    $cmakeConfigureExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $oldErrorActionPreference
}
if ($cmakeConfigureExitCode -ne 0) {
    throw "CMake configure failed for build dir: $BuildDir"
}

$buildTargets = @("llama-server-bridge")
if ($BuildLlamaServerCli) {
    $buildTargets += "llama-server"
}
Invoke-CmakeBuildTargets -Cmake $CmakeExe -BuildDirPath $BuildDir -ConfigName $Config -Targets $buildTargets -ParallelJobs $Jobs

if ($BuildPyannoteCli) {
    if (-not $ApplyDiarizeOverlay) {
        throw "BuildPyannoteCli requires ApplyDiarizeOverlay."
    }
    Invoke-CmakeBuildTargets -Cmake $CmakeExe -BuildDirPath $BuildDir -ConfigName $Config -Targets @("llama-pyannote-align", "llama-pyannote-diarize", "llama-pyannote-inspect") -ParallelJobs $Jobs
}

Write-Host "Bridge build completed."
Write-Host "CMake build dir: $BuildDir"
Write-Host "Runtime output dir (expected): $(Join-Path $BuildDir 'bin')"
Write-Host "HTTPS backend: $HttpsBackend"

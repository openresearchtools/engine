param(
    [string]$LlamaSourceDir = "third_party/llama.cpp",
    [string]$LlamaBuildDir = "",
    [string]$WhisperSourceDir = "third_party/whisper.cpp",
    [string]$WhisperBuildDir = "",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",
    [int]$Jobs = 8,
    [string]$CmakeExe = "cmake",
    [bool]$ApplyOverlay = $true,
    [bool]$BuildWhisperCli = $true
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Test-IsUnderPath {
    param(
        [string]$PathValue,
        [string]$BasePath
    )

    $fullPath = [System.IO.Path]::GetFullPath($PathValue).TrimEnd('\') + '\'
    $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    return $fullPath.StartsWith($fullBase, [System.StringComparison]::OrdinalIgnoreCase)
}

function Invoke-CmakeConfigure {
    param(
        [string]$Cmake,
        [string]$SourceDir,
        [string]$BuildDir,
        [string]$ConfigName,
        [string[]]$ExtraArgs
    )

    $args = @(
        "-S", $SourceDir,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=$ConfigName"
    )
    $args += $ExtraArgs

    & $Cmake @args
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed for source '$SourceDir'."
    }
}

function Invoke-CmakeBuild {
    param(
        [string]$Cmake,
        [string]$BuildDir,
        [string]$ConfigName,
        [string[]]$Targets,
        [int]$ParallelJobs
    )

    $args = @("--build", $BuildDir, "--config", $ConfigName, "--target")
    $args += $Targets
    if ($ParallelJobs -gt 0) {
        $args += @("--parallel", $ParallelJobs)
    }

    & $Cmake @args
    if ($LASTEXITCODE -ne 0) {
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

$engineRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$buildsRoot = Join-Path (Split-Path -Parent $engineRoot) "ENGINEbuilds"
$CmakeExe = Resolve-CmakeExecutable -ToolValue $CmakeExe

$LlamaSourceDir = Resolve-AbsolutePath -PathValue $LlamaSourceDir -RepoRoot $engineRoot
$WhisperSourceDir = Resolve-AbsolutePath -PathValue $WhisperSourceDir -RepoRoot $engineRoot

if ([string]::IsNullOrWhiteSpace($LlamaBuildDir)) {
    $LlamaBuildDir = Join-Path $buildsRoot ("audio\\llama-cuda-" + $Config.ToLowerInvariant())
}
if ([string]::IsNullOrWhiteSpace($WhisperBuildDir)) {
    $WhisperBuildDir = Join-Path $buildsRoot ("audio\\whisper-cuda-" + $Config.ToLowerInvariant())
}

$LlamaBuildDir = Resolve-AbsolutePath -PathValue $LlamaBuildDir -RepoRoot $engineRoot
$WhisperBuildDir = Resolve-AbsolutePath -PathValue $WhisperBuildDir -RepoRoot $engineRoot

if (Test-IsUnderPath -PathValue $LlamaBuildDir -BasePath $engineRoot) {
    throw "LlamaBuildDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $LlamaBuildDir"
}
if (Test-IsUnderPath -PathValue $WhisperBuildDir -BasePath $engineRoot) {
    throw "WhisperBuildDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $WhisperBuildDir"
}

$llamaCmakeLists = Join-Path $LlamaSourceDir "CMakeLists.txt"
$whisperCmakeLists = Join-Path $WhisperSourceDir "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $llamaCmakeLists)) {
    throw "llama source dir not found: $LlamaSourceDir"
}
if (-not (Test-Path -LiteralPath $whisperCmakeLists)) {
    throw "whisper source dir not found: $WhisperSourceDir"
}

$llamaText = Get-Content -Raw -LiteralPath $llamaCmakeLists
if ($llamaText -match 'project\("whisper\.cpp"') {
    throw "Expected llama.cpp at '$LlamaSourceDir' but found whisper.cpp sources. Fix third_party/llama.cpp first."
}

if ($ApplyOverlay) {
    $overlayScript = Join-Path $PSScriptRoot "apply_llama_overlay.ps1"
    if (-not (Test-Path -LiteralPath $overlayScript)) {
        throw "Overlay script not found: $overlayScript"
    }
    & $overlayScript -LlamaRoot $LlamaSourceDir
    if ($LASTEXITCODE -ne 0) {
        throw "Overlay apply failed for: $LlamaSourceDir"
    }
}

New-Item -ItemType Directory -Force -Path $LlamaBuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $WhisperBuildDir | Out-Null

Invoke-CmakeConfigure -Cmake $CmakeExe -SourceDir $LlamaSourceDir -BuildDir $LlamaBuildDir -ConfigName $Config -ExtraArgs @("-DGGML_CUDA=ON", "-DLLAMA_BUILD_SERVER=ON")
Invoke-CmakeBuild -Cmake $CmakeExe -BuildDir $LlamaBuildDir -ConfigName $Config -Targets @("llama-server", "llama-pyannote-align", "llama-pyannote-diarize", "llama-pyannote-inspect", "llama-quantize") -ParallelJobs $Jobs

if ($BuildWhisperCli) {
    Invoke-CmakeConfigure -Cmake $CmakeExe -SourceDir $WhisperSourceDir -BuildDir $WhisperBuildDir -ConfigName $Config -ExtraArgs @("-DGGML_CUDA=ON")
    Invoke-CmakeBuild -Cmake $CmakeExe -BuildDir $WhisperBuildDir -ConfigName $Config -Targets @("whisper-cli") -ParallelJobs $Jobs
}

Write-Host "Unified CUDA audio build completed."
Write-Host "llama build dir: $LlamaBuildDir"
Write-Host "whisper build dir: $WhisperBuildDir"

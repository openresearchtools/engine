param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$CmakeConfig = "Release",
    [ValidateSet("cuda", "vulkan")]
    [string]$Backend = "vulkan",
    [ValidateSet("Debug", "Release")]
    [string]$CargoProfile = "Release",
    [string]$CmakeExe = "cmake",
    [string]$CargoExe = "cargo",
    [string]$LlamaCppDir = "",
    [string]$WhisperCppDir = "third_party/whisper.cpp",
    [bool]$PrepareLlamaSource = $true,
    [string]$BuildRoot = "",
    [string]$LlamaBuildDir = "",
    [string]$WhisperBuildDir = "",
    [string]$CargoTargetDir = "",
    [string]$BundleDir = "",
    [bool]$FetchRuntimeDeps = $true,
    [bool]$StageCudaRuntime = $false,
    [switch]$ForceDependencyRefresh,
    [bool]$StageWhisperSource = $true,
    [bool]$BuildWhisperCli = $false,
    [switch]$BuildLlamaServerCli,
    [switch]$EnableFfmpeg,
    [bool]$EnableCpuAllVariants = $false,
    [ValidateSet("off", "openssl", "boringssl", "libressl")]
    [string]$HttpsBackend = "boringssl",
    [string]$FfmpegRoot = "",
    [string]$FfmpegBinDir = "",
    [string]$FfmpegReleaseApiUrl = "",
    [string]$FfmpegAssetPattern = "",
    [string]$PdfiumTag = "latest",
    [string]$PdfiumDll = "",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [string]$RepoRoot
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return ""
    }
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

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $false
    }
    $fullPath = [System.IO.Path]::GetFullPath($PathValue).TrimEnd('\') + '\'
    $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    return $fullPath.StartsWith($fullBase, [System.StringComparison]::OrdinalIgnoreCase)
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

function Invoke-WhisperBuild {
    param(
        [string]$Cmake,
        [string]$SourceDir,
        [string]$BuildDir,
        [string]$ConfigName,
        [int]$ParallelJobs
    )

    $cmakeLists = Join-Path $SourceDir "CMakeLists.txt"
    if (-not (Test-Path -LiteralPath $cmakeLists)) {
        throw "whisper.cpp source not found at: $SourceDir"
    }

    $cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
    if ($cmakeText -notmatch 'project\("whisper\.cpp"') {
        throw "Unable to verify whisper.cpp source root at '$SourceDir'."
    }

    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Cmake -S $SourceDir -B $BuildDir "-DCMAKE_BUILD_TYPE=$ConfigName" "-DGGML_CUDA=ON" "-DWHISPER_FFMPEG=OFF" 2>&1 | ForEach-Object { Write-Host $_ }
        $whisperConfigureExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($whisperConfigureExitCode -ne 0) {
        throw "Failed to configure whisper.cpp build."
    }

    $args = @("--build", $BuildDir, "--config", $ConfigName, "--target", "whisper-cli")
    if ($ParallelJobs -gt 0) {
        $args += @("--parallel", $ParallelJobs)
    }
    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Cmake @args 2>&1 | ForEach-Object { Write-Host $_ }
        $whisperBuildExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($whisperBuildExitCode -ne 0) {
        throw "Failed to build whisper-cli."
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$CmakeExe = Resolve-CmakeExecutable -ToolValue $CmakeExe
$repoDefaultLlamaPatch = Join-Path $repoRoot "diarize\\addons\\patches\\0300-llama-unified-audio.patch"
if (-not (Test-Path -LiteralPath $repoDefaultLlamaPatch)) {
    throw "Required llama patch is missing: $repoDefaultLlamaPatch"
}

if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $buildsRoot ("full-stack-" + $Backend)
}
$BuildRoot = Resolve-AbsolutePath -PathValue $BuildRoot -RepoRoot $repoRoot

# Default CUDA runtime staging ON for CUDA backend unless explicitly overridden.
if (-not $PSBoundParameters.ContainsKey("StageCudaRuntime") -and $Backend -eq "cuda") {
    $StageCudaRuntime = $true
}

if ([string]::IsNullOrWhiteSpace($LlamaBuildDir)) {
    $LlamaBuildDir = Join-Path $BuildRoot "llama-build"
}
if ([string]::IsNullOrWhiteSpace($WhisperBuildDir)) {
    $WhisperBuildDir = Join-Path $BuildRoot "whisper-build"
}
if ([string]::IsNullOrWhiteSpace($CargoTargetDir)) {
    $CargoTargetDir = Join-Path $BuildRoot "cargo-target"
}
if ([string]::IsNullOrWhiteSpace($BundleDir)) {
    $BundleDir = Join-Path $BuildRoot "bundle"
}

$LlamaBuildDir = Resolve-AbsolutePath -PathValue $LlamaBuildDir -RepoRoot $repoRoot
$WhisperBuildDir = Resolve-AbsolutePath -PathValue $WhisperBuildDir -RepoRoot $repoRoot
$CargoTargetDir = Resolve-AbsolutePath -PathValue $CargoTargetDir -RepoRoot $repoRoot
$BundleDir = Resolve-AbsolutePath -PathValue $BundleDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $LlamaBuildDir -BasePath $repoRoot) {
    throw "LlamaBuildDir must be outside the repo. Current: $LlamaBuildDir"
}
if (Test-IsUnderPath -PathValue $WhisperBuildDir -BasePath $repoRoot) {
    throw "WhisperBuildDir must be outside the repo. Current: $WhisperBuildDir"
}
if (Test-IsUnderPath -PathValue $CargoTargetDir -BasePath $repoRoot) {
    throw "CargoTargetDir must be outside the repo. Current: $CargoTargetDir"
}
if (Test-IsUnderPath -PathValue $BundleDir -BasePath $repoRoot) {
    throw "BundleDir must be outside the repo. Current: $BundleDir"
}

if (-not $PrepareLlamaSource) {
    throw "Patch-only source mode is required. Re-run with -PrepareLlamaSource `$true."
}

$prepareLlamaScript = Join-Path $PSScriptRoot "prepare_llama_source_from_patch.ps1"
if (-not (Test-Path -LiteralPath $prepareLlamaScript)) {
    throw "Missing llama source prep script: $prepareLlamaScript"
}

if ([string]::IsNullOrWhiteSpace($LlamaCppDir) -or $LlamaCppDir -eq "third_party/llama.cpp") {
    $LlamaCppDir = Join-Path $BuildRoot "llama-src"
}

$prepareArgs = @{
    OutDir = $LlamaCppDir
    Force = $true
}

& $prepareLlamaScript @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to prepare external llama source."
}

if ([string]::IsNullOrWhiteSpace($LlamaCppDir)) {
    $LlamaCppDir = Join-Path $repoRoot "third_party\\llama.cpp"
}
$LlamaCppDir = Resolve-AbsolutePath -PathValue $LlamaCppDir -RepoRoot $repoRoot
$WhisperCppDir = Resolve-AbsolutePath -PathValue $WhisperCppDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $LlamaCppDir -BasePath $repoRoot) {
    throw "LlamaCppDir must be outside the repo for full-stack build. Use repo-source patch prep or pass an external path."
}

if ([string]::IsNullOrWhiteSpace($FfmpegRoot)) {
    $FfmpegRoot = Join-Path $buildsRoot "runtime-deps\\ffmpeg"
}
$FfmpegRoot = Resolve-AbsolutePath -PathValue $FfmpegRoot -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($FfmpegBinDir)) {
    $FfmpegBinDir = Join-Path $FfmpegRoot "bin"
}
$FfmpegBinDir = Resolve-AbsolutePath -PathValue $FfmpegBinDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($PdfiumDll)) {
    $PdfiumDll = Join-Path $buildsRoot "runtime-deps\\pdfium\\bin\\pdfium.dll"
}
$PdfiumDll = Resolve-AbsolutePath -PathValue $PdfiumDll -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $FfmpegRoot -BasePath $repoRoot) {
    throw "FfmpegRoot must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $FfmpegRoot"
}
if (Test-IsUnderPath -PathValue $FfmpegBinDir -BasePath $repoRoot) {
    throw "FfmpegBinDir must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $FfmpegBinDir"
}
if (Test-IsUnderPath -PathValue $PdfiumDll -BasePath $repoRoot) {
    throw "PdfiumDll must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $PdfiumDll"
}

$bridgeScript = Join-Path $PSScriptRoot "build_bridge.ps1"
$engineScript = Join-Path $PSScriptRoot "build_engine.ps1"
$downloadPdfiumScript = Join-Path $PSScriptRoot "download-pdfium-win-x64.ps1"
$downloadFfmpegScript = Join-Path $PSScriptRoot "download-ffmpeg-lgpl-win-x64.ps1"

if (-not (Test-Path -LiteralPath $bridgeScript)) {
    throw "Missing build script: $bridgeScript"
}
if (-not (Test-Path -LiteralPath $engineScript)) {
    throw "Missing build script: $engineScript"
}
if ($FetchRuntimeDeps) {
    if (-not (Test-Path -LiteralPath $downloadPdfiumScript)) {
        throw "Missing dependency fetch script: $downloadPdfiumScript"
    }
    if (-not (Test-Path -LiteralPath $downloadFfmpegScript)) {
        throw "Missing dependency fetch script: $downloadFfmpegScript"
    }
}

if ($FetchRuntimeDeps) {
    $pdfiumDestination = Split-Path -Parent $PdfiumDll
    if ((Split-Path -Leaf $pdfiumDestination).Equals("bin", [System.StringComparison]::OrdinalIgnoreCase)) {
        $pdfiumDestination = Split-Path -Parent $pdfiumDestination
    }

    $pdfiumFetchArgs = @{
        Destination = $pdfiumDestination
    }
    if (-not [string]::IsNullOrWhiteSpace($PdfiumTag)) {
        $pdfiumFetchArgs["Tag"] = $PdfiumTag
    }
    if ($ForceDependencyRefresh) {
        $pdfiumFetchArgs["Force"] = $true
    }
    & $downloadPdfiumScript @pdfiumFetchArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to fetch PDFium runtime."
    }

    if ($EnableFfmpeg) {
        $ffmpegFetchArgs = @{
            OutDir = $FfmpegRoot
        }
        if (-not [string]::IsNullOrWhiteSpace($FfmpegReleaseApiUrl)) {
            $ffmpegFetchArgs["ReleaseApiUrl"] = $FfmpegReleaseApiUrl
        }
        if (-not [string]::IsNullOrWhiteSpace($FfmpegAssetPattern)) {
            $ffmpegFetchArgs["AssetPattern"] = $FfmpegAssetPattern
        }
        if ($ForceDependencyRefresh) {
            $ffmpegFetchArgs["Force"] = $true
        }
        & $downloadFfmpegScript @ffmpegFetchArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to fetch FFmpeg LGPL runtime."
        }
    }
}

$bridgeArgs = @{
    CmakeExe = $CmakeExe
    Config = $CmakeConfig
    Backend = $Backend
    HttpsBackend = $HttpsBackend
    LlamaCppDir = $LlamaCppDir
    BuildDir = $LlamaBuildDir
    StageBridgeSources = $true
    StageWhisperSource = $StageWhisperSource
    BuildLlamaServerCli = $BuildLlamaServerCli.IsPresent
    Jobs = $Jobs
}
if ($EnableCpuAllVariants) {
    $bridgeArgs["EnableBackendDl"] = $true
    $bridgeArgs["EnableCpuAllVariants"] = $true
    $bridgeArgs["DisableGgmlNative"] = $true
}
if ($EnableFfmpeg) {
    $bridgeArgs["EnableFfmpeg"] = $true
    $bridgeArgs["FfmpegRoot"] = $FfmpegRoot
}

& $bridgeScript @bridgeArgs
if ($LASTEXITCODE -ne 0) {
    throw "Bridge/llama build failed for backend '$Backend'."
}

if ($BuildWhisperCli) {
    if ($Backend -ne "cuda") {
        throw "BuildWhisperCli currently supports only Backend=cuda."
    }
    Invoke-WhisperBuild -Cmake $CmakeExe -SourceDir $WhisperCppDir -BuildDir $WhisperBuildDir -ConfigName $CmakeConfig -ParallelJobs $Jobs
}

$bridgeBinDir = Join-Path $LlamaBuildDir "bin"
$engineArgs = @{
    Profile = $CargoProfile
    CargoExe = $CargoExe
    CargoTargetDir = $CargoTargetDir
    OutDir = $BundleDir
    CmakeBuildDir = $LlamaBuildDir
    BridgeBinDir = $bridgeBinDir
    PdfiumDll = $PdfiumDll
    FfmpegBinDir = $FfmpegBinDir
    StageCmakeRuntime = $true
    StageFfmpegRuntime = $EnableFfmpeg.IsPresent
    StageCudaRuntime = $StageCudaRuntime
    LicenseProfile = $Backend
}

& $engineScript @engineArgs
if ($LASTEXITCODE -ne 0) {
    throw "Engine build/stage failed."
}

Write-Host "Full stack build completed for backend: $Backend"
Write-Host "llama/bridge build: $LlamaBuildDir"
if ($BuildWhisperCli) {
    Write-Host "whisper build: $WhisperBuildDir"
}
Write-Host "cargo target: $CargoTargetDir"
Write-Host "bundle: $BundleDir"

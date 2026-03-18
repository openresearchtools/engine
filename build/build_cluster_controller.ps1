param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$CmakeConfig = "Release",
    [ValidateSet("Debug", "Release")]
    [string]$CargoProfile = "Release",
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Backend = "cuda",
    [string]$BuildRoot = "",
    [string]$LlamaBuildDir = "",
    [string]$CargoTargetDir = "",
    [string]$BundleDir = "",
    [string]$RuntimeDir = "",
    [switch]$InstallRuntime,
    [string]$CmakeExe = "cmake",
    [string]$CmakeGenerator = "",
    [string]$CmakeArch = "",
    [string]$CargoExe = "cargo",
    [switch]$EnableFfmpeg,
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$logicalCpuCount = [Environment]::ProcessorCount
if ($Jobs -le 0) {
    $Jobs = $logicalCpuCount
}

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

function Sync-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir,
        [Parameter(Mandatory = $true)]
        [string]$DestDir
    )

    $preserveFiles = @("settings.json", "cluster-public-api.json")
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
    Get-ChildItem -LiteralPath $DestDir -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($preserveFiles -contains $_.Name) {
            return
        }
        Remove-Item -LiteralPath $_.FullName -Recurse -Force
    }
    Get-ChildItem -LiteralPath $SourceDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $DestDir -Recurse -Force
    }
}

if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $buildsRoot ("engine-controller-" + $Backend)
}
$BuildRoot = Resolve-AbsolutePath -PathValue $BuildRoot -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($LlamaBuildDir)) {
    $LlamaBuildDir = Join-Path $BuildRoot "llama-build"
}
$LlamaBuildDir = Resolve-AbsolutePath -PathValue $LlamaBuildDir -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($CargoTargetDir)) {
    $CargoTargetDir = Join-Path $BuildRoot "cargo-target"
}
$CargoTargetDir = Resolve-AbsolutePath -PathValue $CargoTargetDir -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($BundleDir)) {
    $BundleDir = Join-Path $BuildRoot "bundle"
}
$BundleDir = Resolve-AbsolutePath -PathValue $BundleDir -BasePath $repoRoot

if ([string]::IsNullOrWhiteSpace($RuntimeDir)) {
    $appData = [Environment]::GetFolderPath("ApplicationData")
    $RuntimeDir = Join-Path $appData "OpenResearchTools\\engine"
}
$RuntimeDir = Resolve-AbsolutePath -PathValue $RuntimeDir -BasePath $repoRoot

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

$bridgeArgs = @{
    Config = $CmakeConfig
    Backend = $Backend
    CmakeExe = $CmakeExe
    BuildRoot = $BuildRoot
    BuildDir = $LlamaBuildDir
    CargoTargetDir = $CargoTargetDir
    Jobs = $Jobs
}
if (-not [string]::IsNullOrWhiteSpace($CmakeGenerator)) {
    $bridgeArgs["CmakeGenerator"] = $CmakeGenerator
}
if (-not [string]::IsNullOrWhiteSpace($CmakeArch)) {
    $bridgeArgs["CmakeArch"] = $CmakeArch
}
if ($EnableFfmpeg.IsPresent) {
    $bridgeArgs["EnableFfmpeg"] = $true
}

Write-Host "Building native runtime targets into $LlamaBuildDir"
& (Join-Path $repoRoot "build\\build_bridge.ps1") @bridgeArgs

$engineArgs = @{
    Profile = $CargoProfile
    CargoExe = $CargoExe
    CargoTargetDir = $CargoTargetDir
    OutDir = $BundleDir
    CmakeBuildDir = $LlamaBuildDir
    Jobs = $Jobs
}

Write-Host "Staging cluster controller bundle into $BundleDir"
& (Join-Path $repoRoot "build\\build_engine.ps1") @engineArgs

if ($InstallRuntime.IsPresent) {
    Write-Host "Installing bundle into runtime directory $RuntimeDir"
    Sync-DirectoryContents -SourceDir $BundleDir -DestDir $RuntimeDir
}

Write-Host "Engine controller build complete."
Write-Host "Bundle:  $BundleDir"
Write-Host "Runtime: $RuntimeDir"
Write-Host "Parallel jobs: $Jobs"

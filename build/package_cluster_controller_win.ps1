param(
  [string]$BundleDir = "",
  [string]$CargoTargetDir = "",
  [string]$TargetTriple = "",
  [switch]$Locked = $true
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
  param(
    [string]$PathValue,
    [Parameter(Mandatory = $true)]
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

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"

if ([string]::IsNullOrWhiteSpace($BundleDir)) {
  $BundleDir = Join-Path $buildsRoot "engine-controller-win-x64-bundle"
}
if ([string]::IsNullOrWhiteSpace($CargoTargetDir)) {
  $CargoTargetDir = Join-Path $buildsRoot "engine-controller-win-x64-target"
}

$BundleDir = Resolve-AbsolutePath -PathValue $BundleDir -BasePath $repoRoot
$CargoTargetDir = Resolve-AbsolutePath -PathValue $CargoTargetDir -BasePath $repoRoot

if (Test-IsUnderPath -PathValue $BundleDir -BasePath $repoRoot) {
  throw "BundleDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $BundleDir"
}
if (Test-IsUnderPath -PathValue $CargoTargetDir -BasePath $repoRoot) {
  throw "CargoTargetDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $CargoTargetDir"
}

if (Test-Path $BundleDir) {
  Remove-Item -Recurse -Force $BundleDir
}
New-Item -ItemType Directory -Force -Path $BundleDir | Out-Null
New-Item -ItemType Directory -Force -Path $CargoTargetDir | Out-Null

$env:CARGO_TARGET_DIR = $CargoTargetDir
$buildArgs = @("build", "--release", "-p", "clusterui", "--bin", "Engine")
if ($Locked) {
  $buildArgs += "--locked"
}
if (-not [string]::IsNullOrWhiteSpace($TargetTriple)) {
  $buildArgs += @("--target", $TargetTriple)
}

Push-Location $repoRoot
try {
  cargo @buildArgs | Out-Host
} finally {
  Pop-Location
}

$targetRelease = if ([string]::IsNullOrWhiteSpace($TargetTriple)) {
  Join-Path $CargoTargetDir "release"
} else {
  Join-Path (Join-Path $CargoTargetDir $TargetTriple) "release"
}

$mainExe = Join-Path $targetRelease "Engine.exe"
if (!(Test-Path $mainExe)) {
  throw "Missing built controller exe: $mainExe"
}

$outExe = Join-Path $BundleDir "Engine.exe"
Copy-Item -Force $mainExe $outExe

Write-Host "Standalone controller ready: $outExe"

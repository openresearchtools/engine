param(
    [string]$Destination = "",
    [string]$Tag = "latest",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

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

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"

if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $buildsRoot "runtime-deps\\pdfium"
}
$dest = Resolve-AbsolutePath -PathValue $Destination -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $dest -BasePath $repoRoot) {
    throw "Destination must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $dest"
}

$dllPath = Join-Path $dest "pdfium.dll"
$nestedDllPath = Join-Path $dest "bin\\pdfium.dll"

if (((Test-Path $dllPath) -or (Test-Path $nestedDllPath)) -and -not $Force) {
    $existingDll = if (Test-Path $dllPath) { $dllPath } else { $nestedDllPath }
    Write-Host "pdfium.dll already exists at $existingDll"
    Write-Host "Use -Force to re-download."
    exit 0
}

if (-not (Test-Path $dest)) {
    New-Item -Path $dest -ItemType Directory -Force | Out-Null
}

$releaseUrl = if ($Tag -eq "latest") {
    "https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest"
}
else {
    "https://api.github.com/repos/bblanchon/pdfium-binaries/releases/tags/$Tag"
}

Write-Host "Fetching release metadata from $releaseUrl"
$headers = @{
    "User-Agent" = "ENGINE-pdfium-fetch"
    "Accept" = "application/vnd.github+json"
}
$token = $env:GH_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
    $token = $env:GITHUB_TOKEN
}
if (-not [string]::IsNullOrWhiteSpace($token)) {
    $headers["Authorization"] = "Bearer $token"
}

$release = Invoke-RestMethod -Headers $headers -Uri $releaseUrl

$asset = $release.assets | Where-Object { $_.name -eq "pdfium-win-x64.tgz" } | Select-Object -First 1
if (-not $asset) {
    throw "Could not find asset pdfium-win-x64.tgz in release $($release.tag_name)"
}

$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) ("pdfium-win-x64-" + [guid]::NewGuid().ToString() + ".tgz")

Write-Host "Downloading $($asset.name) from release $($release.tag_name)"
Invoke-WebRequest -Headers $headers -Uri $asset.browser_download_url -OutFile $archivePath

Write-Host "Extracting archive to $dest"
tar -xzf $archivePath -C $dest
Remove-Item $archivePath -Force

if (Test-Path $dllPath) {
    Write-Host "Ready: $dllPath"
    exit 0
}

if (Test-Path $nestedDllPath) {
    Write-Host "Ready: $nestedDllPath"
    exit 0
}

throw "Download/extract completed but pdfium.dll was not found under $dest"

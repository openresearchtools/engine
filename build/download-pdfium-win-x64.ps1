param(
    [string]$Destination = (Join-Path $PSScriptRoot "..\\third_party\\pdfium"),
    [string]$Tag = "latest",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-DestinationPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path (Get-Location) $PathValue)
}

$dest = Resolve-DestinationPath -PathValue $Destination
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
$release = Invoke-RestMethod -Uri $releaseUrl

$asset = $release.assets | Where-Object { $_.name -eq "pdfium-win-x64.tgz" } | Select-Object -First 1
if (-not $asset) {
    throw "Could not find asset pdfium-win-x64.tgz in release $($release.tag_name)"
}

$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) ("pdfium-win-x64-" + [guid]::NewGuid().ToString() + ".tgz")

Write-Host "Downloading $($asset.name) from release $($release.tag_name)"
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archivePath

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


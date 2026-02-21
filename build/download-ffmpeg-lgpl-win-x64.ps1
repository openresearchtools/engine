param(
    [string]$OutDir = "",
    [string]$AssetPattern = "*win64-lgpl-shared*.zip"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $root "third_party\\ffmpeg"
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ffmpeg-lgpl-" + [Guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tmp "ffmpeg.zip"
$extractDir = Join-Path $tmp "extract"

try {
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    $headers = @{
        "User-Agent" = "ENGINE-ffmpeg-fetch"
        "Accept" = "application/vnd.github+json"
    }

    $release = Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest"
    if ($null -eq $release -or $null -eq $release.assets) {
        throw "Failed to retrieve release assets from BtbN/FFmpeg-Builds"
    }

    $asset = $release.assets | Where-Object { $_.name -like $AssetPattern } | Select-Object -First 1
    if ($null -eq $asset) {
        throw "No asset matched pattern '$AssetPattern'"
    }

    Invoke-WebRequest -Headers $headers -Uri $asset.browser_download_url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    $payloadRoot = Get-ChildItem -Path $extractDir -Directory | Select-Object -First 1
    if ($null -eq $payloadRoot) {
        throw "Expanded archive did not contain a top-level directory"
    }

    Copy-Item -Path (Join-Path $payloadRoot.FullName "*") -Destination $OutDir -Recurse -Force

    Write-Host "FFmpeg LGPL shared build downloaded:"
    Write-Host "  Source: $($asset.browser_download_url)"
    Write-Host "  Destination: $OutDir"
    Write-Host "  Expected DLL location: $(Join-Path $OutDir 'bin')"
}
finally {
    if ([System.IO.Directory]::Exists($tmp)) {
        [System.IO.Directory]::Delete($tmp, $true)
    }
}

param(
    [string]$OutDir = "",
    [string]$AssetPattern = "*win64-lgpl-shared*.zip",
    [string]$ReleaseApiUrl = "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest",
    [switch]$Force
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

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $buildsRoot "runtime-deps\\ffmpeg"
}
$OutDir = Resolve-AbsolutePath -PathValue $OutDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $OutDir -BasePath $repoRoot) {
    throw "OutDir must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $OutDir"
}

$expectedBin = Join-Path $OutDir "bin"
$expectedLib = Join-Path $OutDir "lib"
$expectedInclude = Join-Path $OutDir "include"
if (-not $Force -and (Test-Path -LiteralPath $expectedBin) -and (Test-Path -LiteralPath $expectedLib) -and (Test-Path -LiteralPath $expectedInclude)) {
    Write-Host "FFmpeg LGPL shared build already present at $OutDir"
    Write-Host "Use -Force to re-download."
    exit 0
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
    $token = $env:GH_TOKEN
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = $env:GITHUB_TOKEN
    }
    if (-not [string]::IsNullOrWhiteSpace($token)) {
        $headers["Authorization"] = "Bearer $token"
    }

    $release = Invoke-RestMethod -Headers $headers -Uri $ReleaseApiUrl
    if ($null -eq $release -or $null -eq $release.assets) {
        throw "Failed to retrieve release assets from: $ReleaseApiUrl"
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
    Write-Host "  Release API: $ReleaseApiUrl"
    Write-Host "  Source: $($asset.browser_download_url)"
    Write-Host "  Destination: $OutDir"
    Write-Host "  Expected DLL location: $(Join-Path $OutDir 'bin')"
}
finally {
    if ([System.IO.Directory]::Exists($tmp)) {
        [System.IO.Directory]::Delete($tmp, $true)
    }
}

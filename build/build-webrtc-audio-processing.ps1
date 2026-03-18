param(
    [string]$OutDir = "",
    [string]$RepoUrl = "https://github.com/cross-platform/webrtc-audio-processing.git",
    [string]$Ref = "907015852bc78d8e3ac0e8fbb93c93e76110192a",
    [string]$GitExe = "git",
    [string]$MesonExe = "meson",
    [switch]$Force
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

function Find-WebRtcStaticLibrary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildDir
    )

    $patterns = @("webrtc.lib", "libwebrtc.a", "libwebrtc.lib")
    foreach ($pattern in $patterns) {
        $match = Get-ChildItem -Path $BuildDir -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }
    return ""
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $buildsRoot "runtime-deps\\webrtc-audio-processing"
}
$OutDir = Resolve-AbsolutePath -PathValue $OutDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $OutDir -BasePath $repoRoot) {
    throw "OutDir must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $OutDir"
}

$sourceDir = Join-Path $OutDir "src"
$buildDir = Join-Path $OutDir "build"

$existingLib = Find-WebRtcStaticLibrary -BuildDir $buildDir
if (-not $Force -and -not [string]::IsNullOrWhiteSpace($existingLib) -and (Test-Path -LiteralPath $sourceDir)) {
    Write-Host "WebRTC AudioProcessing dependency already prepared."
    Write-Host "Source dir: $sourceDir"
    Write-Host "Library: $existingLib"
    exit 0
}

if (Test-Path -LiteralPath $OutDir) {
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

& $GitExe clone --depth 1 $RepoUrl $sourceDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to clone WebRTC AudioProcessing source from $RepoUrl"
}

if (-not [string]::IsNullOrWhiteSpace($Ref)) {
    Push-Location $sourceDir
    try {
        & $GitExe fetch --depth 1 origin $Ref
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to fetch WebRTC AudioProcessing ref '$Ref'"
        }
        & $GitExe checkout --force FETCH_HEAD
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to checkout WebRTC AudioProcessing ref '$Ref'"
        }
    }
    finally {
        Pop-Location
    }
}

& $MesonExe setup $buildDir $sourceDir --default-library=static --buildtype=release
if ($LASTEXITCODE -ne 0) {
    throw "Failed to configure WebRTC AudioProcessing build."
}

& $MesonExe compile -C $buildDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build WebRTC AudioProcessing."
}

$builtLib = Find-WebRtcStaticLibrary -BuildDir $buildDir
if ([string]::IsNullOrWhiteSpace($builtLib)) {
    throw "Could not locate built WebRTC AudioProcessing static library under $buildDir"
}

Write-Host "Built WebRTC AudioProcessing."
Write-Host "Source dir: $sourceDir"
Write-Host "Build dir: $buildDir"
Write-Host "Library: $builtLib"

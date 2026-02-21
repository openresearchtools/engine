param(
    [string]$UpstreamRepo = "https://github.com/ggerganov/llama.cpp.git",
    [string]$UpstreamRef = "master",
    [string]$CacheDir = "",
    [string]$OutDir = "",
    [string]$PatchFile = "",
    [switch]$SkipPatch,
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

function Invoke-GitOrThrow {
    param(
        [string]$WorkingDir,
        [string[]]$GitArgs,
        [string]$FailureMessage
    )

    & git -C $WorkingDir @GitArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"

if ([string]::IsNullOrWhiteSpace($CacheDir)) {
    $CacheDir = Join-Path $buildsRoot "upstream\\llama-clean"
}
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $buildsRoot "sources\\llama.cpp"
}

$CacheDir = Resolve-AbsolutePath -PathValue $CacheDir -RepoRoot $repoRoot
$OutDir = Resolve-AbsolutePath -PathValue $OutDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $CacheDir -BasePath $repoRoot) {
    throw "CacheDir must be outside the repo. Current: $CacheDir"
}
if (Test-IsUnderPath -PathValue $OutDir -BasePath $repoRoot) {
    throw "OutDir must be outside the repo. Current: $OutDir"
}

if (-not $SkipPatch) {
    if ([string]::IsNullOrWhiteSpace($PatchFile)) {
        $patchRoot = Join-Path $buildsRoot "patches"
        $latestPatch = Get-ChildItem -Path $patchRoot -File -Filter "llama-working-overlay-*.patch" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($null -eq $latestPatch) {
            throw "No patch found in $patchRoot. Pass -PatchFile or generate one first."
        }
        $PatchFile = $latestPatch.FullName
    } else {
        $PatchFile = Resolve-AbsolutePath -PathValue $PatchFile -RepoRoot $repoRoot
    }

    if (-not (Test-Path -LiteralPath $PatchFile)) {
        throw "Patch file not found: $PatchFile"
    }
}

$cacheParent = Split-Path -Parent $CacheDir
if (-not [string]::IsNullOrWhiteSpace($cacheParent)) {
    New-Item -ItemType Directory -Force -Path $cacheParent | Out-Null
}

if (-not (Test-Path -LiteralPath $CacheDir)) {
    & git clone $UpstreamRepo $CacheDir | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone upstream llama.cpp from $UpstreamRepo"
    }
} else {
    Invoke-GitOrThrow -WorkingDir $CacheDir -GitArgs @("fetch", "origin", "--tags", "--prune") -FailureMessage "Failed to fetch upstream updates for cache."
}

Invoke-GitOrThrow -WorkingDir $CacheDir -GitArgs @("checkout", "--force", $UpstreamRef) -FailureMessage "Failed to checkout upstream ref '$UpstreamRef' in cache."
Invoke-GitOrThrow -WorkingDir $CacheDir -GitArgs @("reset", "--hard", $UpstreamRef) -FailureMessage "Failed to reset cache to '$UpstreamRef'."
Invoke-GitOrThrow -WorkingDir $CacheDir -GitArgs @("clean", "-xfd") -FailureMessage "Failed to clean cache."

if (Test-Path -LiteralPath $OutDir) {
    if (-not $Force) {
        throw "OutDir already exists: $OutDir. Re-run with -Force to replace it."
    }
    Remove-Item -LiteralPath $OutDir -Recurse -Force
}

$outParent = Split-Path -Parent $OutDir
if (-not [string]::IsNullOrWhiteSpace($outParent)) {
    New-Item -ItemType Directory -Force -Path $outParent | Out-Null
}

& git clone --local --no-hardlinks $CacheDir $OutDir | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create build source clone at $OutDir"
}

Invoke-GitOrThrow -WorkingDir $OutDir -GitArgs @("checkout", "--force", $UpstreamRef) -FailureMessage "Failed to checkout '$UpstreamRef' in build source."

if (-not $SkipPatch) {
    & git -C $OutDir apply --check $PatchFile | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Patch does not apply cleanly: $PatchFile"
    }

    & git -C $OutDir apply --whitespace=nowarn $PatchFile | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to apply patch: $PatchFile"
    }
}

$head = (& git -C $OutDir rev-parse HEAD).Trim()
$statusCount = ((& git -C $OutDir status --porcelain) | Measure-Object).Count

Write-Host "Prepared llama source:"
Write-Host "  OutDir: $OutDir"
Write-Host "  Upstream ref: $UpstreamRef"
Write-Host "  Upstream HEAD: $head"
if (-not $SkipPatch) {
    Write-Host "  Patch: $PatchFile"
}
Write-Host "  Working tree entries after patch: $statusCount"

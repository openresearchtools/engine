param(
    [string]$OutDir = "",
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

function Copy-DirectoryTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot
    )

    if (-not (Test-Path -LiteralPath $SourceRoot)) {
        throw "Source directory not found: $SourceRoot"
    }

    if (Test-Path -LiteralPath $DestinationRoot) {
        Remove-Item -LiteralPath $DestinationRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    $items = Get-ChildItem -LiteralPath $SourceRoot -Force -ErrorAction SilentlyContinue
    foreach ($item in $items) {
        Copy-Item -LiteralPath $item.FullName -Destination $DestinationRoot -Recurse -Force
    }
}

function Stage-ExtraToolFilesNoOverwrite {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$LlamaOutDir
    )

    $extraRoot = Join-Path $RepoRoot "diarize\\addons\\overlay\\llama.cpp\\tools"
    $relativeFiles = @(
        "pyannote\\CMakeLists.txt",
        "pyannote\\pyannote-align.cpp",
        "pyannote\\pyannote-diarize.cpp",
        "pyannote\\pyannote-entrypoints.h",
        "pyannote\\pyannote-inspect.cpp",
        "server\\inproc-pipeline-repro.cpp",
        "whisper\\whisper-cli-entrypoint.cpp",
        "whisper\\whisper-cli-entrypoint.h",
        "whisper\\whisper-common-audio.cpp"
    )

    foreach ($rel in $relativeFiles) {
        $src = Join-Path $extraRoot $rel
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing required add-on source file: $src"
        }
    }

    $stagedCount = 0
    foreach ($rel in $relativeFiles) {
        $src = Join-Path $extraRoot $rel
        $dst = Join-Path (Join-Path $LlamaOutDir "tools") $rel
        $dstDir = Split-Path -Parent $dst
        if (-not (Test-Path -LiteralPath $dstDir)) {
            New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        }
        if (Test-Path -LiteralPath $dst) {
            throw "Refusing to overwrite existing llama.cpp file: $dst"
        }
        Copy-Item -LiteralPath $src -Destination $dst -Force
        $stagedCount++
    }

    return $stagedCount
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$repoLlamaDir = Join-Path $repoRoot "third_party\\llama.cpp"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $buildsRoot "sources\\llama.cpp"
}
$OutDir = Resolve-AbsolutePath -PathValue $OutDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $OutDir -BasePath $repoRoot) {
    throw "OutDir must be outside the repo. Current: $OutDir"
}

if (-not (Test-Path -LiteralPath $repoLlamaDir)) {
    throw "Required repo source not found: $repoLlamaDir"
}

$cmakeLists = Join-Path $repoLlamaDir "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeLists)) {
    throw "Missing llama.cpp CMakeLists.txt in repo source: $cmakeLists"
}
$cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
if ($cmakeText -match 'project\("whisper\.cpp"') {
    throw "Expected llama.cpp sources at '$repoLlamaDir' but found whisper.cpp content."
}
if ($cmakeText -notmatch 'project\("llama\.cpp"') {
    throw "Unable to verify llama.cpp source root at '$repoLlamaDir'."
}

$PatchFile = Join-Path $repoRoot "diarize\\addons\\patches\\0300-llama-unified-audio.patch"
if (-not (Test-Path -LiteralPath $PatchFile)) {
    throw "Required patch not found: $PatchFile"
}

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

Copy-DirectoryTree -SourceRoot $repoLlamaDir -DestinationRoot $OutDir
$extraFilesStaged = Stage-ExtraToolFilesNoOverwrite -RepoRoot $repoRoot -LlamaOutDir $OutDir

& git -C $OutDir init | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize git workspace at: $OutDir"
}

& git -C $OutDir apply --check --ignore-space-change --ignore-whitespace $PatchFile | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Patch does not apply cleanly: $PatchFile"
}

& git -C $OutDir apply --whitespace=nowarn --ignore-space-change --ignore-whitespace $PatchFile | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Failed to apply patch: $PatchFile"
}

$statusCount = ((& git -C $OutDir status --porcelain) | Measure-Object).Count

Write-Host "Prepared llama source from repo snapshot:"
Write-Host "  Source: $repoLlamaDir"
Write-Host "  OutDir: $OutDir"
Write-Host "  Extra add-on files staged: $extraFilesStaged"
Write-Host "  Patch: $PatchFile"
Write-Host "  Working tree entries after patch: $statusCount"

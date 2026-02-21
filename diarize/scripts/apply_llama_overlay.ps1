param(
    [string]$LlamaRoot = "third_party/llama.cpp",
    [string]$OverlayRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Resolve-OverlayRoot {
    param(
        [string]$ExplicitOverlayRoot,
        [string]$DiarizeRootPath,
        [string]$RepoRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitOverlayRoot)) {
        $candidate = Resolve-AbsolutePath -PathValue $ExplicitOverlayRoot -RepoRoot $RepoRoot
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
        throw "Explicit overlay root not found: $candidate"
    }

    $candidates = @(
        (Join-Path $DiarizeRootPath "addons/overlay/llama.cpp"),
        (Join-Path $DiarizeRootPath "overlay/llama.cpp"),
        (Join-Path $RepoRoot "diarize/addons/overlay/llama.cpp"),
        (Join-Path $RepoRoot "diarize/overlay/llama.cpp")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

$diarizeRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$engineRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

$LlamaRoot = Resolve-AbsolutePath -PathValue $LlamaRoot -RepoRoot $engineRoot
$overlayRootResolved = Resolve-OverlayRoot -ExplicitOverlayRoot $OverlayRoot -DiarizeRootPath $diarizeRoot -RepoRoot $engineRoot
if (-not $overlayRootResolved) {
    throw "Overlay root not found. Checked diarize/addons/overlay/llama.cpp and diarize/overlay/llama.cpp."
}

if (-not (Test-Path -LiteralPath $LlamaRoot)) {
    throw "llama root not found: $LlamaRoot"
}
$targetRoot = [System.IO.Path]::GetFullPath($LlamaRoot)

$cmakeLists = Join-Path $targetRoot "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeLists)) {
    throw "Target does not look like llama.cpp source root (missing CMakeLists.txt): $targetRoot"
}

$cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
if ($cmakeText -match 'project\("whisper\.cpp"') {
    throw "Expected llama.cpp at '$targetRoot' but found whisper.cpp sources. Fix third_party/llama.cpp first."
}

$files = Get-ChildItem -Path $overlayRootResolved -Recurse -File | Sort-Object FullName
if ($files.Count -eq 0) {
    throw "No overlay files found in: $overlayRootResolved"
}

Write-Host "Applying overlay from: $overlayRootResolved"
Write-Host "Target llama.cpp root: $targetRoot"
Write-Host "Files: $($files.Count)"

foreach ($src in $files) {
    $rel = $src.FullName.Substring($overlayRootResolved.Length + 1)
    $dst = Join-Path $targetRoot $rel
    $dstDir = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $dstDir)) {
        if ($DryRun) {
            Write-Host "[dry-run] mkdir $dstDir"
        } else {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
    }
    if ($DryRun) {
        Write-Host "[dry-run] copy $rel"
    } else {
        Copy-Item -LiteralPath $src.FullName -Destination $dst -Force
        Write-Host "[ok] $rel"
    }
}

if ($DryRun) {
    Write-Host "Dry run complete."
} else {
    Write-Host "Overlay applied successfully."
}

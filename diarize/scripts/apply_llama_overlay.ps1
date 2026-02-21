param(
    [string]$LlamaRoot = "third_party/llama.cpp",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$diarizeRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$engineRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$overlayRoot = Join-Path $diarizeRoot "addons/overlay/llama.cpp"

if (-not [System.IO.Path]::IsPathRooted($LlamaRoot)) {
    $LlamaRoot = Join-Path $engineRoot $LlamaRoot
}
$targetRoot = Resolve-Path $LlamaRoot

if (-not (Test-Path -LiteralPath $overlayRoot)) {
    throw "Overlay root not found: $overlayRoot"
}
if (-not (Test-Path -LiteralPath $targetRoot)) {
    throw "llama root not found: $targetRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $targetRoot "tools/server/server-context.cpp"))) {
    throw "Target does not look like llama.cpp source root: $targetRoot"
}

$files = Get-ChildItem -Path $overlayRoot -Recurse -File | Sort-Object FullName
if ($files.Count -eq 0) {
    throw "No overlay files found in: $overlayRoot"
}

Write-Host "Applying overlay from: $overlayRoot"
Write-Host "Target llama.cpp root: $targetRoot"
Write-Host "Files: $($files.Count)"

foreach ($src in $files) {
    $rel = $src.FullName.Substring($overlayRoot.Length + 1)
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

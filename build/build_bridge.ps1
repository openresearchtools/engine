param(
    [string]$CmakeExe = "cmake",
    [string]$Config = "Release",
    [string]$LlamaCppDir = "",
    [string]$BuildDir = "",
    [bool]$ApplyDiarizeOverlay = $true,
    [switch]$EnableFfmpeg,
    [string]$FfmpegRoot = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($LlamaCppDir)) {
    $LlamaCppDir = Join-Path $root "third_party\\llama.cpp"
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $root "out\\llama-build"
}
if ([string]::IsNullOrWhiteSpace($FfmpegRoot)) {
    $FfmpegRoot = Join-Path $root "third_party\\ffmpeg"
}

$bridgeSrcDir = Join-Path $root "bridge"
$bridgeDstDir = Join-Path $LlamaCppDir "MARKDOWN\\bridge"
$diarizeOverlayDir = Join-Path $root "diarize\\addons\\overlay\\llama.cpp"

if (-not (Test-Path (Join-Path $LlamaCppDir "CMakeLists.txt"))) {
    throw "llama.cpp source not found at: $LlamaCppDir"
}

New-Item -ItemType Directory -Force -Path $bridgeDstDir | Out-Null
Copy-Item -Path (Join-Path $bridgeSrcDir "*") -Destination $bridgeDstDir -Force

if ($ApplyDiarizeOverlay -and (Test-Path $diarizeOverlayDir)) {
    $overlayFiles = Get-ChildItem -Path $diarizeOverlayDir -Recurse -File
    foreach ($src in $overlayFiles) {
        $rel = $src.FullName.Substring($diarizeOverlayDir.Length + 1)
        $dst = Join-Path $LlamaCppDir $rel
        $dstDir = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
        Copy-Item -Path $src.FullName -Destination $dst -Force
    }
}

$cmakeArgs = @(
    "-S", $LlamaCppDir,
    "-B", $BuildDir,
    "-DBUILD_SHARED_LIBS=ON",
    "-DLLAMA_BUILD_SERVER=ON"
)
if ($EnableFfmpeg) {
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON"
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_FFMPEG_ROOT=$FfmpegRoot"
}

& $CmakeExe @cmakeArgs
& $CmakeExe --build $BuildDir --config $Config --target llama-server llama-server-bridge

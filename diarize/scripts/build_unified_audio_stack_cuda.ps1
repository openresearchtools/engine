param(
    [string]$LlamaSourceDir = "third_party/llama.cpp",
    [string]$LlamaBuildDir = "",
    [string]$WhisperSourceDir = "third_party/whisper.cpp",
    [string]$WhisperBuildDir = "",
    [ValidateSet("vulkan","cuda")]
    [string]$Backend = "vulkan",
    [int]$Jobs = 8,
    [string]$VsDevCmd = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\Tools\\VsDevCmd.bat",
    [string]$CmakeExe = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe"
)

$ErrorActionPreference = "Stop"

$engineRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not [System.IO.Path]::IsPathRooted($LlamaSourceDir)) {
    $LlamaSourceDir = Join-Path $engineRoot $LlamaSourceDir
}
if (-not [System.IO.Path]::IsPathRooted($WhisperSourceDir)) {
    $WhisperSourceDir = Join-Path $engineRoot $WhisperSourceDir
}

if (-not (Test-Path -LiteralPath $VsDevCmd)) {
    throw "VsDevCmd.bat not found: $VsDevCmd"
}
if (-not (Test-Path -LiteralPath $CmakeExe)) {
    throw "cmake.exe not found: $CmakeExe"
}
if (-not (Test-Path -LiteralPath $LlamaSourceDir)) {
    throw "llama source dir not found: $LlamaSourceDir"
}
if (-not (Test-Path -LiteralPath $WhisperSourceDir)) {
    throw "whisper source dir not found: $WhisperSourceDir"
}

$backendLower = $Backend.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($LlamaBuildDir)) {
    $LlamaBuildDir = if ($backendLower -eq "vulkan") { Join-Path $LlamaSourceDir "bvk" } else { Join-Path $LlamaSourceDir "bcu" }
}
if ([string]::IsNullOrWhiteSpace($WhisperBuildDir)) {
    $WhisperBuildDir = if ($backendLower -eq "vulkan") { Join-Path $WhisperSourceDir "bvk" } else { Join-Path $WhisperSourceDir "bcu" }
}

$llamaBackendFlags = switch ($backendLower) {
    "vulkan" { "-DGGML_VULKAN=ON" }
    "cuda"   { "-DGGML_CUDA=ON" }
}
$whisperBackendFlags = $llamaBackendFlags

$cmdParts = @()
$cmdParts += "`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul"
$cmdParts += "`"$CmakeExe`" -S `"$LlamaSourceDir`" -B `"$LlamaBuildDir`" -G Ninja $llamaBackendFlags -DCMAKE_BUILD_TYPE=Release"
$cmdParts += "`"$CmakeExe`" --build `"$LlamaBuildDir`" --config Release --target llama-server llama-pyannote-align llama-pyannote-diarize llama-pyannote-inspect llama-quantize -j $Jobs"
$cmdParts += "`"$CmakeExe`" -S `"$WhisperSourceDir`" -B `"$WhisperBuildDir`" -G Ninja $whisperBackendFlags -DCMAKE_BUILD_TYPE=Release"
$cmdParts += "`"$CmakeExe`" --build `"$WhisperBuildDir`" --config Release --target whisper-cli -j $Jobs"

$cmd = ($cmdParts -join " && ")
cmd /d /c $cmd
if ($LASTEXITCODE -ne 0) {
    throw "Unified $backendLower build failed with exit code $LASTEXITCODE"
}

Write-Host "Unified $backendLower build completed."
Write-Host "llama-server: $LlamaBuildDir/bin/llama-server.exe"
Write-Host "llama-pyannote-align: $LlamaBuildDir/bin/llama-pyannote-align.exe"
Write-Host "llama-pyannote-diarize: $LlamaBuildDir/bin/llama-pyannote-diarize.exe"
Write-Host "whisper-cli: $WhisperBuildDir/bin/whisper-cli.exe"

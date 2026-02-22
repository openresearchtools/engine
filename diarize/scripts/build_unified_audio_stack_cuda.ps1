param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",
    [ValidateSet("cuda", "vulkan")]
    [string]$Backend = "cuda",
    [int]$Jobs = 8,
    [switch]$EnableFfmpeg,
    [switch]$BuildWhisperCli
)

$ErrorActionPreference = "Stop"

$engineRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$buildScript = Join-Path $engineRoot "build\build_full_stack_cuda.ps1"

if (-not (Test-Path -LiteralPath $buildScript)) {
    throw "Missing build script: $buildScript"
}

$args = @(
    "-Backend", $Backend,
    "-CmakeConfig", $Config,
    "-PrepareLlamaSource", $true,
    "-BuildWhisperCli", $BuildWhisperCli.IsPresent,
    "-Jobs", $Jobs
)

if ($EnableFfmpeg) {
    $args += "-EnableFfmpeg"
}

& $buildScript @args
if ($LASTEXITCODE -ne 0) {
    throw "Unified audio stack build failed."
}

Write-Host "Unified audio stack build completed using repo-source + patch mode."

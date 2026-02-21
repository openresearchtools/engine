param(
    [string]$PythonExe = ".venv_pyannote312/Scripts/python.exe",
    [string]$Converter = "convert_pyannote_checkpoint_to_gguf.py",
    [string]$NpzConverter = "convert_pyannote_npz_to_gguf.py",
    [string]$PipelineRoot = "hf_models/pyannote-speaker-diarization-community-1",
    [string]$OutDir = "gguf",
    [string]$ArtifactsDir = "artifacts",
    [ValidateSet("f32","f16","bf16","auto")]
    [string]$OutType = "f16"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$scriptRoot = Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)
$engineRoot = Resolve-Path (Join-Path $scriptRoot "..")

if (-not [System.IO.Path]::IsPathRooted($PythonExe)) {
    $PythonExe = Join-Path $engineRoot $PythonExe
}
if (-not [System.IO.Path]::IsPathRooted($Converter)) {
    $Converter = Join-Path $scriptRoot $Converter
}
if (-not [System.IO.Path]::IsPathRooted($NpzConverter)) {
    $NpzConverter = Join-Path $scriptRoot $NpzConverter
}
if (-not [System.IO.Path]::IsPathRooted($PipelineRoot)) {
    $PipelineRoot = Join-Path $engineRoot $PipelineRoot
}
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
    $OutDir = Join-Path $engineRoot $OutDir
}
if (-not [System.IO.Path]::IsPathRooted($ArtifactsDir)) {
    $ArtifactsDir = Join-Path $engineRoot $ArtifactsDir
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Converter)) {
    throw "Converter not found: $Converter"
}
if (-not (Test-Path -LiteralPath $NpzConverter)) {
    throw "NPZ converter not found: $NpzConverter"
}
if (-not (Test-Path -LiteralPath $PipelineRoot)) {
    throw "Pipeline root not found: $PipelineRoot"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $ArtifactsDir | Out-Null

$segOut = Join-Path $OutDir ("pyannote_segmentation_stub_{0}.gguf" -f $OutType)
$embOut = Join-Path $OutDir ("pyannote_embedding_stub_{0}.gguf" -f $OutType)
$segModel = Join-Path $PipelineRoot "segmentation"
$embModel = Join-Path $PipelineRoot "embedding"
$segLog = Join-Path $ArtifactsDir "pyannote_segmentation_stub.log"
$embLog = Join-Path $ArtifactsDir "pyannote_embedding_stub.log"

$segCmd = "`"$PythonExe`" `"$Converter`" --outtype $OutType --outfile `"$segOut`" `"$segModel`" > `"$segLog`" 2>&1"
cmd /c $segCmd
if ($LASTEXITCODE -ne 0) {
    throw "Segmentation conversion failed. See: $segLog"
}

$embCmd = "`"$PythonExe`" `"$Converter`" --outtype $OutType --outfile `"$embOut`" `"$embModel`" > `"$embLog`" 2>&1"
cmd /c $embCmd
if ($LASTEXITCODE -ne 0) {
    throw "Embedding conversion failed. See: $embLog"
}

$pldaNpz = Join-Path $PipelineRoot "plda/plda.npz"
$xvecNpz = Join-Path $PipelineRoot "plda/xvec_transform.npz"
$pldaOut = Join-Path $OutDir "pyannote_plda_f32.gguf"
$xvecOut = Join-Path $OutDir "pyannote_xvec_transform_f32.gguf"
$npzLog = Join-Path $ArtifactsDir "pyannote_npz_to_gguf.log"

$npzCmd = "`"$PythonExe`" `"$NpzConverter`" --plda-npz `"$pldaNpz`" --xvec-npz `"$xvecNpz`" --plda-out `"$pldaOut`" --xvec-out `"$xvecOut`" > `"$npzLog`" 2>&1"
cmd /c $npzCmd
if ($LASTEXITCODE -ne 0) {
    throw "PLDA/xvec NPZ conversion failed. See: $npzLog"
}

Write-Host "Wrote: $segOut"
Write-Host "Wrote: $embOut"
Write-Host "Wrote: $pldaOut"
Write-Host "Wrote: $xvecOut"
Write-Host "Logs:  $segLog, $embLog, $npzLog"

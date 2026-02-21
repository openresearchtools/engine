param(
    [ValidateSet("Debug", "Release")]
    [string]$Profile = "Release",
    [string]$CargoExe = "cargo",
    [string]$PdfiumDll = "",
    [string]$FfmpegBinDir = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    if ($Profile -eq "Release") {
        & $CargoExe build --release -p pdfvlm
        & $CargoExe build --release -p pdf
        & $CargoExe build --release -p engine
    } else {
        & $CargoExe build -p pdfvlm
        & $CargoExe build -p pdf
        & $CargoExe build -p engine
    }

    $targetProfile = if ($Profile -eq "Release") { "release" } else { "debug" }
    if ([string]::IsNullOrWhiteSpace($OutDir)) {
        $OutDir = Join-Path $root ("out\\bundle-" + $targetProfile)
    }
    if ([string]::IsNullOrWhiteSpace($PdfiumDll)) {
        $PdfiumDll = Join-Path $root "third_party\\pdfium\\bin\\pdfium.dll"
    }
    if ([string]::IsNullOrWhiteSpace($FfmpegBinDir)) {
        $FfmpegBinDir = Join-Path $root "third_party\\ffmpeg\\bin"
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

    $engineExe = Join-Path $root ("target\\" + $targetProfile + "\\engine.exe")
    $pdfDll = Join-Path $root ("target\\" + $targetProfile + "\\pdf.dll")
    $pdfvlmDll = Join-Path $root ("target\\" + $targetProfile + "\\pdfvlm.dll")
    $targetDir = Join-Path $root ("target\\" + $targetProfile)

    if (Test-Path $engineExe) {
        Copy-Item -Path $engineExe -Destination (Join-Path $OutDir "engine.exe") -Force
    }
    if (Test-Path $pdfDll) {
        Copy-Item -Path $pdfDll -Destination (Join-Path $OutDir "pdf.dll") -Force
    }
    if (Test-Path $pdfvlmDll) {
        Copy-Item -Path $pdfvlmDll -Destination (Join-Path $OutDir "pdfvlm.dll") -Force
    }
    if (Test-Path $PdfiumDll) {
        Copy-Item -Path $PdfiumDll -Destination (Join-Path $OutDir "pdfium.dll") -Force
        if (Test-Path $targetDir) {
            Copy-Item -Path $PdfiumDll -Destination (Join-Path $targetDir "pdfium.dll") -Force
        }
    } else {
        Write-Warning "pdfium.dll not found at '$PdfiumDll' (skipping copy)"
    }

    if (Test-Path $FfmpegBinDir) {
        $ffmpegDlls = Get-ChildItem -Path $FfmpegBinDir -Filter "*.dll" -File -ErrorAction SilentlyContinue
        foreach ($dll in $ffmpegDlls) {
            Copy-Item -Path $dll.FullName -Destination (Join-Path $OutDir $dll.Name) -Force
            if (Test-Path $targetDir) {
                Copy-Item -Path $dll.FullName -Destination (Join-Path $targetDir $dll.Name) -Force
            }
        }
    } else {
        Write-Warning "FFmpeg bin dir not found at '$FfmpegBinDir' (skipping FFmpeg DLL copy)"
    }
}
finally {
    Pop-Location
}

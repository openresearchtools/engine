param(
    [ValidateSet("Debug", "Release")]
    [string]$Profile = "Release",
    [string]$CargoExe = "cargo",
    [string]$CargoTargetDir = "",
    [string]$OutDir = "",
    [string]$CmakeBuildDir = "",
    [string]$BridgeBinDir = "",
    [string]$BridgeLibDir = "",
    [string]$PdfiumDll = "",
    [string]$FfmpegBinDir = "",
    [string]$CudaBinDir = "",
    [bool]$StageCmakeRuntime = $true,
    [bool]$StageFfmpegRuntime = $true,
    [bool]$StageCudaRuntime = $true,
    [bool]$StageRepoLicenseFiles = $true,
    [ValidateSet("default", "cuda", "vulkan")]
    [string]$LicenseProfile = "default"
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
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
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    $fullPath = [System.IO.Path]::GetFullPath($PathValue).TrimEnd('\') + '\'
    $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    return $fullPath.StartsWith($fullBase, [System.StringComparison]::OrdinalIgnoreCase)
}

function Prepend-EnvPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$VariableName,
        [Parameter(Mandatory = $true)]
        [string]$PrefixValue
    )

    if ([string]::IsNullOrWhiteSpace($PrefixValue)) {
        return
    }
    $current = [Environment]::GetEnvironmentVariable($VariableName, "Process")
    if ([string]::IsNullOrWhiteSpace($current)) {
        [Environment]::SetEnvironmentVariable($VariableName, $PrefixValue, "Process")
    } else {
        [Environment]::SetEnvironmentVariable($VariableName, "$PrefixValue;$current", "Process")
    }
}

function Copy-IfExists {
    param(
        [string]$SourcePath,
        [string]$DestPath
    )

    if (Test-Path -LiteralPath $SourcePath) {
        Copy-Item -LiteralPath $SourcePath -Destination $DestPath -Force
        return $true
    }
    return $false
}

function Resolve-PdfiumRoot {
    param(
        [string]$PdfiumDllPath
    )

    if ([string]::IsNullOrWhiteSpace($PdfiumDllPath)) {
        return ""
    }

    $dllFullPath = [System.IO.Path]::GetFullPath($PdfiumDllPath)
    $binDir = Split-Path -Parent $dllFullPath
    if ([string]::IsNullOrWhiteSpace($binDir)) {
        return ""
    }

    if ((Split-Path -Leaf $binDir).Equals("bin", [System.StringComparison]::OrdinalIgnoreCase)) {
        return (Split-Path -Parent $binDir)
    }
    return $binDir
}

function Copy-LicenseFiles {
    param(
        [string]$SourceRoot,
        [string]$DestinationRoot,
        [string]$ComponentName
    )

    if ([string]::IsNullOrWhiteSpace($SourceRoot) -or -not (Test-Path -LiteralPath $SourceRoot)) {
        Write-Warning "$ComponentName runtime root not found at '$SourceRoot' (skipping license copy)"
        return
    }

    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    $patterns = @(
        "LICENSE*",
        "LICENCE*",
        "COPYING*",
        "COPYRIGHT*",
        "NOTICE*",
        "PATENTS*",
        "EULA*"
    )

    $licenseFiles = @()
    foreach ($pattern in $patterns) {
        $licenseFiles += Get-ChildItem -Path $SourceRoot -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue
    }

    $licenseFiles = $licenseFiles | Sort-Object -Property FullName -Unique
    if (-not $licenseFiles -or $licenseFiles.Count -eq 0) {
        Write-Warning "No license files found under '$SourceRoot' for $ComponentName"
        return
    }

    foreach ($file in $licenseFiles) {
        $relativePath = $file.FullName.Substring($SourceRoot.Length).TrimStart('\', '/')
        if ([string]::IsNullOrWhiteSpace($relativePath)) {
            $relativePath = $file.Name
        }
        $destinationFile = Join-Path $DestinationRoot $relativePath
        $destinationDir = Split-Path -Parent $destinationFile
        if (-not [string]::IsNullOrWhiteSpace($destinationDir)) {
            New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
        }
        Copy-Item -LiteralPath $file.FullName -Destination $destinationFile -Force
    }
}

function Stage-RepoLicenseFiles {
    param(
        [string]$RepoRoot,
        [string]$BundleOutDir,
        [string]$BundleLicenseRoot,
        [string]$LicenseProfile
    )

    $sourceDir = Join-Path $RepoRoot "third_party\\licenses"
    if (-not (Test-Path -LiteralPath $sourceDir)) {
        Write-Warning "Repo license source folder not found at '$sourceDir' (skipping static license staging)"
        return
    }

    $destDir = Join-Path $BundleLicenseRoot "third_party"
    New-Item -ItemType Directory -Force -Path $destDir | Out-Null

    $excludedTopLevelFiles = @(
        "torch-LICENSE.txt",
        "torch-NOTICE.txt",
        "torchaudio-LICENSE.txt",
        "numpy-LICENSE.txt",
        "ffmpeg-SOURCE.txt"
    )
    if ($LicenseProfile -eq "vulkan") {
        $excludedTopLevelFiles += @(
            "nvidia-cuda-EULA.txt",
            "nvidia-cuda-runtime-NOTICE.txt"
        )
    }

    # Copy curated, top-level license files only. Skip tooling-only files.
    $topLevelFiles = Get-ChildItem -Path $sourceDir -File -ErrorAction SilentlyContinue
    foreach ($file in $topLevelFiles) {
        if ($excludedTopLevelFiles -contains $file.Name) {
            continue
        }
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $destDir $file.Name) -Force
    }

    $repoLicenseFile = Join-Path $RepoRoot "LICENSE"
    if (Test-Path -LiteralPath $repoLicenseFile) {
        Copy-Item -LiteralPath $repoLicenseFile -Destination (Join-Path $BundleOutDir "LICENSE-ENGINE.txt") -Force
    }

    $bundleKeyLicenses = Join-Path $BundleOutDir "LICENSES.txt"
    $licenseCandidates = @()
    if (-not [string]::IsNullOrWhiteSpace($LicenseProfile) -and $LicenseProfile -ne "default") {
        $licenseCandidates += "LICENSES-$LicenseProfile.txt"
    }
    $licenseCandidates += "LICENSES.txt"

    $selectedLicensePath = ""
    foreach ($candidateName in $licenseCandidates) {
        $candidatePath = Join-Path $sourceDir $candidateName
        if (Test-Path -LiteralPath $candidatePath) {
            $selectedLicensePath = $candidatePath
            break
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($selectedLicensePath)) {
        Copy-Item -LiteralPath $selectedLicensePath -Destination $bundleKeyLicenses -Force
    } else {
        Write-Warning "Key release licenses file not found (candidates: $($licenseCandidates -join ', '))"
    }

    $noticePath = Join-Path $BundleOutDir "THIRD_PARTY_NOTICES.md"
    @"
# Third-Party Notices

This bundle includes third-party software.

Key combined license text:

- ./LICENSES.txt

Bundled license folders:

- ./licenses/third_party/
- ./licenses/pdfium/
- ./licenses/ffmpeg/

Full license inventory (including transitive/tooling exports):

- https://github.com/openresearchtools/engine/tree/main/third_party/licenses
"@ | Set-Content -Path $noticePath -Encoding ASCII
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$targetProfile = if ($Profile -eq "Release") { "release" } else { "debug" }

if ([string]::IsNullOrWhiteSpace($CargoTargetDir)) {
    $CargoTargetDir = Join-Path $buildsRoot "cargo-target"
}
$CargoTargetDir = Resolve-AbsolutePath -PathValue $CargoTargetDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $buildsRoot ("bundle-" + $targetProfile)
}
$OutDir = Resolve-AbsolutePath -PathValue $OutDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $CargoTargetDir -BasePath $repoRoot) {
    throw "CargoTargetDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $CargoTargetDir"
}
if (Test-IsUnderPath -PathValue $OutDir -BasePath $repoRoot) {
    throw "OutDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $OutDir"
}

if ([string]::IsNullOrWhiteSpace($PdfiumDll)) {
    $PdfiumDll = Join-Path $buildsRoot "runtime-deps\\pdfium\\bin\\pdfium.dll"
}
$PdfiumDll = Resolve-AbsolutePath -PathValue $PdfiumDll -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($FfmpegBinDir)) {
    $FfmpegBinDir = Join-Path $buildsRoot "runtime-deps\\ffmpeg\\bin"
}
$FfmpegBinDir = Resolve-AbsolutePath -PathValue $FfmpegBinDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $PdfiumDll -BasePath $repoRoot) {
    throw "PdfiumDll must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $PdfiumDll"
}
if (Test-IsUnderPath -PathValue $FfmpegBinDir -BasePath $repoRoot) {
    throw "FfmpegBinDir must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $FfmpegBinDir"
}

if (-not [string]::IsNullOrWhiteSpace($CudaBinDir)) {
    $CudaBinDir = Resolve-AbsolutePath -PathValue $CudaBinDir -RepoRoot $repoRoot
}

$CmakeBuildDir = Resolve-AbsolutePath -PathValue $CmakeBuildDir -RepoRoot $repoRoot
$BridgeBinDir = Resolve-AbsolutePath -PathValue $BridgeBinDir -RepoRoot $repoRoot
$BridgeLibDir = Resolve-AbsolutePath -PathValue $BridgeLibDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($BridgeBinDir) -and -not [string]::IsNullOrWhiteSpace($CmakeBuildDir)) {
    $candidateBin = Join-Path $CmakeBuildDir "bin"
    if (Test-Path -LiteralPath $candidateBin) {
        $BridgeBinDir = $candidateBin
    }
}

if (-not [string]::IsNullOrWhiteSpace($BridgeBinDir) -and (Test-Path -LiteralPath $BridgeBinDir)) {
    $bridgeDll = Get-ChildItem -Path $BridgeBinDir -Recurse -Filter "llama-server-bridge.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bridgeDll) {
        $BridgeBinDir = $bridgeDll.DirectoryName
    }
}
elseif ([string]::IsNullOrWhiteSpace($BridgeBinDir) -and -not [string]::IsNullOrWhiteSpace($CmakeBuildDir) -and (Test-Path -LiteralPath $CmakeBuildDir)) {
    $bridgeDll = Get-ChildItem -Path $CmakeBuildDir -Recurse -Filter "llama-server-bridge.dll" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bridgeDll) {
        $BridgeBinDir = $bridgeDll.DirectoryName
    }
}

if ([string]::IsNullOrWhiteSpace($BridgeLibDir) -and -not [string]::IsNullOrWhiteSpace($CmakeBuildDir)) {
    $bridgeLib = Get-ChildItem -Path $CmakeBuildDir -Recurse -Filter "llama-server-bridge.lib" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bridgeLib) {
        $BridgeLibDir = $bridgeLib.DirectoryName
    }
}

if ([string]::IsNullOrWhiteSpace($BridgeLibDir) -and -not [string]::IsNullOrWhiteSpace($BridgeBinDir)) {
    $bridgeLib = Get-ChildItem -Path $BridgeBinDir -Recurse -Filter "llama-server-bridge.lib" -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bridgeLib) {
        $BridgeLibDir = $bridgeLib.DirectoryName
    }
}

if (-not [string]::IsNullOrWhiteSpace($BridgeLibDir)) {
    Prepend-EnvPath -VariableName "LIB" -PrefixValue $BridgeLibDir
}
if (-not [string]::IsNullOrWhiteSpace($BridgeBinDir)) {
    Prepend-EnvPath -VariableName "PATH" -PrefixValue $BridgeBinDir
}

New-Item -ItemType Directory -Force -Path $CargoTargetDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$originalCargoTargetDir = [Environment]::GetEnvironmentVariable("CARGO_TARGET_DIR", "Process")
[Environment]::SetEnvironmentVariable("CARGO_TARGET_DIR", $CargoTargetDir, "Process")

Push-Location $repoRoot
try {
    $cargoArgs = @("build")
    if ($Profile -eq "Release") {
        $cargoArgs += "--release"
    }
    $cargoArgs += @("-p", "pdfvlm", "-p", "pdf", "-p", "engine")

    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $CargoExe @cargoArgs 2>&1 | ForEach-Object { Write-Host $_ }
        $cargoExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($cargoExitCode -ne 0) {
        throw "Cargo build failed (profile: $Profile)."
    }
}
finally {
    Pop-Location
    [Environment]::SetEnvironmentVariable("CARGO_TARGET_DIR", $originalCargoTargetDir, "Process")
}

$profileDir = Join-Path $CargoTargetDir $targetProfile
$engineExe = Join-Path $profileDir "engine.exe"
$pdfDll = Join-Path $profileDir "pdf.dll"
$pdfvlmDll = Join-Path $profileDir "pdfvlm.dll"

if (-not (Copy-IfExists -SourcePath $engineExe -DestPath (Join-Path $OutDir "engine.exe"))) {
    throw "engine.exe not found at expected path: $engineExe"
}
if (-not (Copy-IfExists -SourcePath $pdfDll -DestPath (Join-Path $OutDir "pdf.dll"))) {
    Write-Warning "pdf.dll not found at expected path: $pdfDll"
}
if (-not (Copy-IfExists -SourcePath $pdfvlmDll -DestPath (Join-Path $OutDir "pdfvlm.dll"))) {
    Write-Warning "pdfvlm.dll not found at expected path: $pdfvlmDll"
}

if (Test-Path -LiteralPath $PdfiumDll) {
    Copy-Item -LiteralPath $PdfiumDll -Destination (Join-Path $OutDir "pdfium.dll") -Force
} else {
    Write-Warning "pdfium.dll not found at '$PdfiumDll' (skipping copy)"
}

$bundleLicenseRoot = Join-Path $OutDir "licenses"
if ($StageRepoLicenseFiles) {
    Stage-RepoLicenseFiles -RepoRoot $repoRoot -BundleOutDir $OutDir -BundleLicenseRoot $bundleLicenseRoot -LicenseProfile $LicenseProfile
}

$pdfiumRoot = Resolve-PdfiumRoot -PdfiumDllPath $PdfiumDll
if (-not [string]::IsNullOrWhiteSpace($pdfiumRoot)) {
    Copy-LicenseFiles -SourceRoot $pdfiumRoot -DestinationRoot (Join-Path $bundleLicenseRoot "pdfium") -ComponentName "PDFium"
}

if ($StageFfmpegRuntime) {
    if (Test-Path -LiteralPath $FfmpegBinDir) {
        $ffmpegPatterns = @(
            "avcodec*.dll",
            "avformat*.dll",
            "avutil*.dll",
            "swresample*.dll",
            "swscale*.dll"
        )
        foreach ($pattern in $ffmpegPatterns) {
            $ffmpegDlls = Get-ChildItem -Path $FfmpegBinDir -Filter $pattern -File -ErrorAction SilentlyContinue
            foreach ($dll in $ffmpegDlls) {
                Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $OutDir $dll.Name) -Force
            }
        }

        $ffmpegRoot = Split-Path -Parent $FfmpegBinDir
        Copy-LicenseFiles -SourceRoot $ffmpegRoot -DestinationRoot (Join-Path $bundleLicenseRoot "ffmpeg") -ComponentName "FFmpeg"
    } else {
        Write-Warning "FFmpeg bin dir not found at '$FfmpegBinDir' (skipping FFmpeg DLL copy)"
    }
}

if ($StageCmakeRuntime -and -not [string]::IsNullOrWhiteSpace($BridgeBinDir) -and (Test-Path -LiteralPath $BridgeBinDir)) {
    $patterns = @(
        "llama-server-bridge.dll",
        "llama.dll",
        "ggml*.dll",
        "mtmd.dll"
    )
    foreach ($pattern in $patterns) {
        $matches = Get-ChildItem -Path $BridgeBinDir -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($item in $matches) {
            Copy-Item -LiteralPath $item.FullName -Destination (Join-Path $OutDir $item.Name) -Force
        }
    }
}

if ($StageCudaRuntime) {
    $cudaCandidateDirs = @()
    if (-not [string]::IsNullOrWhiteSpace($CudaBinDir)) {
        $cudaCandidateDirs += $CudaBinDir
        $cudaCandidateDirs += (Join-Path $CudaBinDir "x64")
    }
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        $cudaCandidateDirs += (Join-Path $env:CUDA_PATH "bin")
        $cudaCandidateDirs += (Join-Path $env:CUDA_PATH "bin\\x64")
    }

    $cudaCandidateDirs = $cudaCandidateDirs |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { [System.IO.Path]::GetFullPath($_) } |
        Select-Object -Unique

    $cudaPatterns = @(
        "cublas64_*.dll",
        "cublasLt64_*.dll"
    )

    foreach ($dir in $cudaCandidateDirs) {
        if (-not (Test-Path -LiteralPath $dir)) {
            continue
        }
        foreach ($pattern in $cudaPatterns) {
            $cudaDlls = Get-ChildItem -Path $dir -Filter $pattern -File -ErrorAction SilentlyContinue
            foreach ($dll in $cudaDlls) {
                Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $OutDir $dll.Name) -Force
            }
        }
    }
}

Write-Host "Engine build and bundle staging completed."
Write-Host "Cargo target dir: $CargoTargetDir"
Write-Host "Bundle dir: $OutDir"
Write-Host "Bundle runtime licenses: $(Join-Path $OutDir 'licenses')"

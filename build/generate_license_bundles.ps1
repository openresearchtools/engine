param(
    [string]$RepoRoot = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

$licenseRoot = Join-Path $RepoRoot "third_party\licenses"
$projectLicensePath = Join-Path $RepoRoot "LICENSE"

if (-not (Test-Path -LiteralPath $licenseRoot)) {
    throw "License root not found: $licenseRoot"
}

if (-not (Test-Path -LiteralPath $projectLicensePath)) {
    throw "Project license not found: $projectLicensePath"
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = $licenseRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$curatedLicenseFiles = @(
    "llama.cpp-LICENSE.txt",
    "cpp-httplib-LICENSE.txt",
    "jsonhpp-LICENSE.txt",
    "boringssl-LICENSE.txt",
    "ggml-LICENSE.txt",
    "miniaudio-LICENSE.txt",
    "yarn-LICENSE.txt",
    "llamafile-sgemm-LICENSE.txt",
    "kleidiai-LICENSE.txt",
    "openvino-LICENSE.txt",
    "arm-optimized-routines-LICENSE.txt",
    "ggllm.cpp-LICENSE.txt",
    "string-algorithms-LICENSE.txt",
    "koboldcpp-LICENSE.txt",
    "llvm-project-LICENSE.TXT",
    "whisper.cpp-LICENSE.txt",
    "voxtral-cpp-LICENSE.txt",
    "nvidia-nemo-LICENSE.txt",
    "parakeet-cpp-LICENSE.txt",
    "vllm-LICENSE.txt",
    "voxtral.c-LICENSE.txt",
    "voxtral-mini-realtime-rs-LICENSE.txt",
    "mlx-audio-LICENSE.txt",
    "docling-LICENSE.txt",
    "pdfium-render-LICENSE.md",
    "pdfium-LICENSE.txt",
    "pdfium-binaries-LICENSE.txt",
    "ffmpeg-builds-LICENSE.txt",
    "ffmpeg-LGPL-2.1.txt"
)

$rustDirectLicenseFiles = @(
    "serde_json-LICENSE-MIT.txt",
    "serde_json-LICENSE-APACHE.txt",
    "anyhow-LICENSE-MIT.txt",
    "anyhow-LICENSE-APACHE.txt",
    "clap-LICENSE-MIT.txt",
    "clap-LICENSE-APACHE.txt",
    "once_cell-LICENSE-MIT.txt",
    "once_cell-LICENSE-APACHE.txt",
    "regex-LICENSE-MIT.txt",
    "regex-LICENSE-APACHE.txt",
    "walkdir-UNLICENSE.txt",
    "walkdir-LICENSE-MIT.txt",
    "walkdir-COPYING.txt",
    "image-LICENSE-MIT.txt",
    "image-LICENSE-APACHE.txt",
    "encoding_rs-LICENSE-MIT.txt",
    "encoding_rs-LICENSE-APACHE.txt",
    "encoding_rs-LICENSE-WHATWG.txt"
)

$cudaOnlyLicenseFiles = @(
    "nvidia-cuda-EULA.txt",
    "nvidia-cuda-runtime-NOTICE.txt"
)

$profiles = @(
    @{
        Output = "LICENSES.txt"
        Title = "Openresearchtools-Engine Release Key Licenses"
        Subtitle = $null
        IncludeCuda = $false
    },
    @{
        Output = "LICENSES-vulkan.txt"
        Title = "Openresearchtools-Engine Release Key Licenses (Vulkan)"
        Subtitle = "Target runtime profile: Vulkan bundle (CPU + Vulkan backends)."
        IncludeCuda = $false
    },
    @{
        Output = "LICENSES-metal.txt"
        Title = "Openresearchtools-Engine Release Key Licenses (Metal)"
        Subtitle = "Target runtime profile: macOS/Metal bundle."
        IncludeCuda = $false
    },
    @{
        Output = "LICENSES-cuda.txt"
        Title = "Openresearchtools-Engine Release Key Licenses (CUDA)"
        Subtitle = "Target runtime profile: CUDA bundle (CPU + CUDA backends)."
        IncludeCuda = $true
    },
    @{
        Output = "LICENSES-ubuntu-vulkan.txt"
        Title = "Openresearchtools-Engine Release Key Licenses (Ubuntu x64 Vulkan)"
        Subtitle = "Target runtime profile: Ubuntu x64 Vulkan bundle (includes CPU + Vulkan backends)."
        IncludeCuda = $false
    }
)

function Get-LicenseBlockText {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required license file: $Path"
    }

    return [System.IO.File]::ReadAllText($Path)
}

function Add-LicenseSection {
    param(
        [System.Text.StringBuilder]$Builder,
        [string]$DisplayName,
        [string]$SourceLabel,
        [string]$Body
    )

    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine("BEGIN: $DisplayName [$SourceLabel]")
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine()
    $null = $Builder.Append($Body.TrimEnd("`r", "`n"))
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine("END: $DisplayName")
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine()
}

foreach ($profile in $profiles) {
    $title = [string]$profile.Title
    $subtitle = [string]$profile.Subtitle
    $includeCuda = [bool]$profile.IncludeCuda
    $outputPath = Join-Path $OutputRoot ([string]$profile.Output)

    $builder = New-Object System.Text.StringBuilder
    $null = $builder.AppendLine($title)
    $null = $builder.AppendLine(("=" * $title.Length))
    $null = $builder.AppendLine()
    $null = $builder.AppendLine("This file aggregates key shipped/runtime/reference license texts from third_party/licenses/README.md.")
    if (-not [string]::IsNullOrWhiteSpace($subtitle)) {
        $null = $builder.AppendLine($subtitle)
    }
    $null = $builder.AppendLine("Excluded from this combined file: tooling-only torch/numpy/PyYAML entries and non-license provenance note ffmpeg-SOURCE-*.txt.")
    $null = $builder.AppendLine()
    $null = $builder.AppendLine("Full license inventory (including transitive/tooling exports): https://github.com/openresearchtools/engine/tree/main/third_party/licenses")
    $null = $builder.AppendLine()
    $null = $builder.AppendLine("Runtime-staged folders in build outputs also include:")
    $null = $builder.AppendLine("- ../vendor/pdfium/*")
    $null = $builder.AppendLine("- ../vendor/ffmpeg/*")
    $null = $builder.AppendLine()
    $null = $builder.AppendLine()

    Add-LicenseSection -Builder $builder -DisplayName "Openresearchtools-Engine" -SourceLabel "LICENSE" -Body (Get-LicenseBlockText -Path $projectLicensePath)

    foreach ($fileName in $curatedLicenseFiles) {
        $filePath = Join-Path $licenseRoot $fileName
        $displayName = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
        Add-LicenseSection -Builder $builder -DisplayName $displayName -SourceLabel $fileName -Body (Get-LicenseBlockText -Path $filePath)
    }

    if ($includeCuda) {
        foreach ($fileName in $cudaOnlyLicenseFiles) {
            $filePath = Join-Path $licenseRoot $fileName
            $displayName = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
            Add-LicenseSection -Builder $builder -DisplayName $displayName -SourceLabel $fileName -Body (Get-LicenseBlockText -Path $filePath)
        }
    }

    foreach ($fileName in $rustDirectLicenseFiles) {
        $filePath = Join-Path $licenseRoot $fileName
        $displayName = [System.IO.Path]::GetFileNameWithoutExtension($fileName)
        Add-LicenseSection -Builder $builder -DisplayName $displayName -SourceLabel $fileName -Body (Get-LicenseBlockText -Path $filePath)
    }

    $null = $builder.AppendLine("END OF AGGREGATED KEY LICENSES")
    $null = $builder.AppendLine("=============================")
    $null = $builder.AppendLine()
    $null = $builder.AppendLine("This file is intentionally limited to key shipped/runtime/reference licenses.")
    $null = $builder.AppendLine("Tooling-only Python dependencies are not included in this combined file.")
    $null = $builder.AppendLine()
    $null = $builder.AppendLine("For full dependency, subdependency, and external-reference license details, see:")
    $null = $builder.AppendLine("- https://github.com/openresearchtools/engine/tree/main/third_party/licenses")
    $null = $builder.AppendLine("- https://github.com/openresearchtools/engine/blob/main/third_party/licenses/README.md")
    $null = $builder.AppendLine("- https://github.com/openresearchtools/engine/tree/main/third_party/licenses/rust-full")
    $null = $builder.AppendLine("- https://github.com/openresearchtools/engine/tree/main/third_party/licenses/tooling-full")

    [System.IO.File]::WriteAllText($outputPath, $builder.ToString(), [System.Text.UTF8Encoding]::new($false))
    Write-Host "Wrote $outputPath"
}

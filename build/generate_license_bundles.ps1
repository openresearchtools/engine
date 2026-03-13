param(
    [string]$RepoRoot = "",
    [string]$OutputPath = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
} elseif (-not [System.IO.Path]::IsPathRooted($RepoRoot)) {
    $RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
} else {
    $RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
}

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
        $OutputPath = Join-Path $RepoRoot "third_party\LICENSES.md"
    } else {
        if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
            $OutputRoot = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputRoot))
        } else {
            $OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
        }
        $OutputPath = Join-Path $OutputRoot "LICENSES.md"
    }
} elseif (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $OutputPath))
} else {
    $OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
}

$licenseRoot = Join-Path $RepoRoot "third_party\licenses"
$rustFullRoot = Join-Path $licenseRoot "rust-full"
$projectLicensePath = Join-Path $RepoRoot "LICENSE"

if (-not (Test-Path -LiteralPath $licenseRoot)) {
    throw "License root not found: $licenseRoot"
}

if (-not (Test-Path -LiteralPath $rustFullRoot)) {
    throw "Rust full license root not found: $rustFullRoot"
}

if (-not (Test-Path -LiteralPath $projectLicensePath)) {
    throw "Project license not found: $projectLicensePath"
}

$outputDir = Split-Path -Parent $OutputPath
if (-not [string]::IsNullOrWhiteSpace($outputDir)) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

function Get-Text {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing required license source: $Path"
    }

    return [System.IO.File]::ReadAllText($Path)
}

function Get-TopLevelSectionMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    $base = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    if ($base -match '^(?<name>.+?)-(?<label>(?:LICENSE|NOTICE|COPYING|UNLICENSE|EULA)(?:-[A-Za-z0-9._+]+)?|SOURCE(?:-[A-Za-z0-9._+]+)?|LGPL-[A-Za-z0-9._+]+)$') {
        return [PSCustomObject]@{
            DisplayName = $matches.name
            SourceLabel = $matches.label
        }
    }

    return [PSCustomObject]@{
        DisplayName = $base
        SourceLabel = $base
    }
}

function Get-SectionText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,
        [Parameter(Mandatory = $true)]
        [string]$Body,
        [string]$Note = ""
    )

    $trimmedBody = $Body.TrimStart("`r", "`n")
    $bodyAlreadyHasHeading = $false
    if ($trimmedBody.StartsWith($DisplayName + "`r`n", [System.StringComparison]::Ordinal) -or
        $trimmedBody.StartsWith($DisplayName + "`n", [System.StringComparison]::Ordinal)) {
        $bodyAlreadyHasHeading = $true
    }

    $builder = New-Object System.Text.StringBuilder
    if (-not $bodyAlreadyHasHeading) {
        $null = $builder.AppendLine($DisplayName)
        $null = $builder.AppendLine(("=" * $DisplayName.Length))
        $null = $builder.AppendLine()
    }
    if (-not [string]::IsNullOrWhiteSpace($Note)) {
        $null = $builder.AppendLine($Note.Trim())
        $null = $builder.AppendLine()
    }
    $null = $builder.Append($Body.TrimEnd("`r", "`n"))
    return $builder.ToString()
}

function Add-LicenseSection {
    param(
        [Parameter(Mandatory = $true)]
        [System.Text.StringBuilder]$Builder,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,
        [Parameter(Mandatory = $true)]
        [string]$SourceLabel,
        [Parameter(Mandatory = $true)]
        [string]$Body,
        [string]$Note = ""
    )

    $sectionText = Get-SectionText -DisplayName $DisplayName -Body $Body -Note $Note
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine("BEGIN: $DisplayName [$SourceLabel]")
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine()
    $null = $Builder.Append($sectionText)
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine("END: $DisplayName")
    $null = $Builder.AppendLine("-------------------------------------------------------------------------------")
    $null = $Builder.AppendLine()
    $null = $Builder.AppendLine()
}

function Split-LicenseExpression {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Expression
    )

    $normalized = $Expression.Trim()
    $normalized = $normalized -replace '[()]', ''
    $parts = $normalized -split '\s+OR\s+'
    return $parts |
        ForEach-Object { $_.Trim() } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
}

function Is-StubLicenseFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FileName,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    if ($FileName -like 'LICENSE-EXPRESSION*') {
        return $true
    }

    if ($Content.Length -ge 500) {
        return $false
    }

    if ($Content -match 'License expression:') {
        return $true
    }

    if ($Content -match '^Licensed under either of') {
        return $true
    }

    return $false
}

function Resolve-StubLicenseIds {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    if ($Content -match 'License expression:\s*(.+)') {
        return Split-LicenseExpression -Expression $matches[1]
    }

    $resolved = New-Object System.Collections.Generic.List[string]
    if ($Content -match 'Apache License, Version 2\.0') {
        $resolved.Add("Apache-2.0")
    }
    if ($Content -match '\bMIT License\b' -or $Content -match 'opensource\.org/licenses/MIT') {
        $resolved.Add("MIT")
    }
    if ($Content -match '\bZlib\b' -or $Content -match '\bzlib\b') {
        $resolved.Add("Zlib")
    }

    $resolved = $resolved | Select-Object -Unique
    if (-not $resolved -or $resolved.Count -eq 0) {
        throw "Unable to resolve license expression from stub text."
    }
    return $resolved
}

$standardLicenseTexts = @{
    "MIT" = Get-Text -Path (Join-Path $licenseRoot "anyhow-LICENSE-MIT.txt")
    "Apache-2.0" = Get-Text -Path (Join-Path $licenseRoot "anyhow-LICENSE-APACHE.txt")
    "Zlib" = Get-Text -Path (Join-Path $rustFullRoot "zune-core-0.5.1\LICENSE-ZLIB")
}

function Add-SectionFromFileOrStub {
    param(
        [Parameter(Mandatory = $true)]
        [System.Text.StringBuilder]$Builder,
        [Parameter(Mandatory = $true)]
        [string]$DisplayName,
        [Parameter(Mandatory = $true)]
        [string]$SourceLabel,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $content = Get-Text -Path $Path
    $fileName = [System.IO.Path]::GetFileName($Path)
    if (Is-StubLicenseFile -FileName $fileName -Content $content) {
        $licenseIds = Resolve-StubLicenseIds -Content $content
        foreach ($licenseId in $licenseIds) {
            if (-not $standardLicenseTexts.ContainsKey($licenseId)) {
                throw "No full-text fallback is configured for license id '$licenseId' used by '$Path'."
            }

            $note = "The local source '$SourceLabel' only carries a short license expression note. The full text below expands the $licenseId option referenced by that source."
            Add-LicenseSection -Builder $Builder -DisplayName $DisplayName -SourceLabel "$SourceLabel -> $licenseId" -Body $standardLicenseTexts[$licenseId] -Note $note
        }
        return
    }

    Add-LicenseSection -Builder $Builder -DisplayName $DisplayName -SourceLabel $SourceLabel -Body $content
}

$cudaPreferredOrder = @(
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
    "ffmpeg-LGPL-2.1.txt",
    "nvidia-cuda-EULA.txt",
    "nvidia-cuda-runtime-NOTICE.txt",
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

$bundleExcludedTopLevelFiles = @(
    "torch-LICENSE.txt",
    "torch-NOTICE.txt",
    "numpy-LICENSE.txt",
    "PyYAML-LICENSE.txt"
)

$topLevelFiles = Get-ChildItem -Path $licenseRoot -File -ErrorAction Stop |
    Where-Object {
        $_.Name -ne "README.md" -and
        $_.Name -notmatch '^LICENSES(\.|-|$)' -and
        $_.Name -notin $bundleExcludedTopLevelFiles
    }

$topLevelByName = @{}
foreach ($file in $topLevelFiles) {
    $topLevelByName[$file.Name] = $file
}

$orderedTopLevelFiles = New-Object System.Collections.Generic.List[System.IO.FileInfo]
foreach ($fileName in $cudaPreferredOrder) {
    if ($topLevelByName.ContainsKey($fileName)) {
        $orderedTopLevelFiles.Add($topLevelByName[$fileName])
        $null = $topLevelByName.Remove($fileName)
    }
}

foreach ($remaining in ($topLevelByName.Values | Sort-Object Name)) {
    $orderedTopLevelFiles.Add($remaining)
}

$builder = New-Object System.Text.StringBuilder
$title = "Openresearchtools-Engine Consolidated Licenses"
$null = $builder.AppendLine($title)
$null = $builder.AppendLine(("=" * $title.Length))
$null = $builder.AppendLine()
$null = $builder.AppendLine("This app may contain the following dependencies.")
$null = $builder.AppendLine("Dependencies may differ between builds, for example CUDA and non-CUDA, Vulkan, Metal, Windows, macOS, Linux, CPU-only, and other packaged variants.")
$null = $builder.AppendLine("This file intentionally over-includes third-party license, notice, source-provenance, and EULA texts so one LICENSES.md covers all distributed application bundles.")
$null = $builder.AppendLine("Bundle-time third-party notices are staged separately from third_party/README.md as Third-Party-Notices.md.")
$null = $builder.AppendLine()

Add-LicenseSection -Builder $builder -DisplayName "Openresearchtools-Engine" -SourceLabel "LICENSE" -Body (Get-Text -Path $projectLicensePath)

foreach ($file in $orderedTopLevelFiles) {
    $metadata = Get-TopLevelSectionMetadata -FileName $file.Name
    Add-SectionFromFileOrStub -Builder $builder -DisplayName $metadata.DisplayName -SourceLabel $metadata.SourceLabel -Path $file.FullName
}

$crateDirs = Get-ChildItem -Path $rustFullRoot -Directory -ErrorAction Stop | Sort-Object Name
foreach ($crateDir in $crateDirs) {
    $crateDisplayName = $crateDir.Name
    $licenseFiles = Get-ChildItem -Path $crateDir.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE|PATENTS|UNLICENSE)(\.|-|$)' } |
        Sort-Object FullName

    if (-not $licenseFiles -or $licenseFiles.Count -eq 0) {
        throw "No license files found for rust-full crate directory: $($crateDir.FullName)"
    }

    $hasNonStubFile = $false
    foreach ($licenseFile in $licenseFiles) {
        $content = Get-Text -Path $licenseFile.FullName
        if (-not (Is-StubLicenseFile -FileName $licenseFile.Name -Content $content)) {
            $hasNonStubFile = $true
            break
        }
    }

    foreach ($licenseFile in $licenseFiles) {
        $content = Get-Text -Path $licenseFile.FullName
        if ($hasNonStubFile -and (Is-StubLicenseFile -FileName $licenseFile.Name -Content $content)) {
            continue
        }

        $sourceLabel = $licenseFile.FullName.Substring($crateDir.FullName.Length).TrimStart('\', '/')
        $sourceLabel = $sourceLabel -replace '\\', '/'
        Add-SectionFromFileOrStub -Builder $builder -DisplayName $crateDisplayName -SourceLabel $sourceLabel -Path $licenseFile.FullName
    }
}

[System.IO.File]::WriteAllText($OutputPath, $builder.ToString(), [System.Text.UTF8Encoding]::new($false))
Write-Host "Wrote $OutputPath"

param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
} elseif (-not [System.IO.Path]::IsPathRooted($RepoRoot)) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $RepoRoot))
} else {
    $RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
}

$licenseRoot = Join-Path $RepoRoot "third_party\licenses"
$rustFullRoot = Join-Path $licenseRoot "rust-full"
if (-not (Test-Path -LiteralPath $licenseRoot)) {
    throw "License root not found: $licenseRoot"
}

$cargoHome = if ([string]::IsNullOrWhiteSpace($env:CARGO_HOME)) {
    Join-Path $env:USERPROFILE ".cargo"
} else {
    $env:CARGO_HOME
}
$registrySrcRoot = Join-Path $cargoHome "registry\src"
if (-not (Test-Path -LiteralPath $registrySrcRoot)) {
    throw "Cargo registry source root not found: $registrySrcRoot"
}

$standardMitText = [System.IO.File]::ReadAllText((Join-Path $licenseRoot "anyhow-LICENSE-MIT.txt"))
$standardApacheText = [System.IO.File]::ReadAllText((Join-Path $licenseRoot "anyhow-LICENSE-APACHE.txt"))

$curatedRustTopLevelCrates = @(
    "axum",
    "base64",
    "bincode",
    "eframe",
    "egui",
    "egui_commonmark",
    "reqwest",
    "rfd",
    "sha2",
    "sysinfo",
    "tar",
    "time",
    "tokio",
    "tower-http",
    "tray-icon",
    "windows-sys",
    "zip"
)

$licenseFilePattern = '^(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE|PATENTS|UNLICENSE)(\.|-|$)'

function Invoke-Cargo {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & cargo @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "cargo $($Arguments -join ' ') failed.`n$output"
    }
    return $output
}

function Get-GraphCrateKeys {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $lines = Invoke-Cargo -Arguments $Arguments
    $keys = New-Object System.Collections.Generic.HashSet[string]
    foreach ($line in $lines) {
        if ($line -isnot [string]) {
            continue
        }
        if ($line -match '^([A-Za-z0-9_.+-]+) v([^ ]+)') {
            $null = $keys.Add("$($matches[1])-$($matches[2])")
        }
    }
    return $keys
}

function Get-LicenseFiles {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceDir
    )

    if (-not (Test-Path -LiteralPath $SourceDir)) {
        return @()
    }

    return @(Get-ChildItem -Path $SourceDir -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $licenseFilePattern } |
        Sort-Object FullName)
}

function Get-RegistrySourceDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CrateName,
        [Parameter(Mandatory = $true)]
        [string]$Version
    )

    $crateDirName = "$CrateName-$Version"
    foreach ($registryRoot in (Get-ChildItem -Path $registrySrcRoot -Directory -ErrorAction Stop)) {
        $candidate = Join-Path $registryRoot.FullName $crateDirName
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Content, [System.Text.UTF8Encoding]::new($false))
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    $baseUri = New-Object System.Uri(($BasePath.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar))
    $targetUri = New-Object System.Uri($TargetPath)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    return [System.Uri]::UnescapeDataString($relativeUri.ToString()) -replace '/', '\'
}

function Write-CuratedTopLevelLicenseFiles {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Package
    )

    $crateName = $Package.name
    $licenseExpr = ""
    if ($null -ne $Package.license) {
        $licenseExpr = ([string]$Package.license).Trim()
    }
    if ([string]::IsNullOrWhiteSpace($licenseExpr)) {
        return
    }

    $existingPatterns = @(
        "$crateName-LICENSE.txt",
        "$crateName-LICENSE-MIT.txt",
        "$crateName-LICENSE-APACHE.txt"
    )
    foreach ($pattern in $existingPatterns) {
        $path = Join-Path $licenseRoot $pattern
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }

    switch -Regex ($licenseExpr) {
        '^(MIT|Apache-2\.0 OR MIT|MIT OR Apache-2\.0)$' {
            if ($licenseExpr -eq "MIT") {
                Write-Utf8NoBom -Path (Join-Path $licenseRoot "$crateName-LICENSE.txt") -Content $standardMitText
            } else {
                Write-Utf8NoBom -Path (Join-Path $licenseRoot "$crateName-LICENSE-MIT.txt") -Content $standardMitText
                Write-Utf8NoBom -Path (Join-Path $licenseRoot "$crateName-LICENSE-APACHE.txt") -Content $standardApacheText
            }
            return
        }
    }

    $sourceDir = Get-RegistrySourceDir -CrateName $crateName -Version $Package.version
    $licenseFiles = if ($null -ne $sourceDir) { Get-LicenseFiles -SourceDir $sourceDir } else { @() }
    if ($licenseFiles.Count -eq 0) {
        throw "No top-level license strategy for curated crate '$crateName' with expression '$licenseExpr'."
    }

    foreach ($file in $licenseFiles) {
        $destination = Join-Path $licenseRoot "$crateName-$($file.Name)"
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
}

Write-Host "Fetching current Rust dependency graph metadata..."
Invoke-Cargo -Arguments @("fetch", "--locked", "--target", "x86_64-pc-windows-msvc") | Out-Null
Invoke-Cargo -Arguments @("fetch", "--locked", "--target", "aarch64-apple-darwin") | Out-Null

$metadata = Invoke-Cargo -Arguments @("metadata", "--format-version", "1") | Out-String | ConvertFrom-Json
$packageByKey = @{}
foreach ($package in $metadata.packages) {
    $key = "$($package.name)-$($package.version)"
    $packageByKey[$key] = $package
}

$windowsGraphKeys = Get-GraphCrateKeys -Arguments @(
    "tree",
    "--workspace",
    "--edges", "normal",
    "--target", "x86_64-pc-windows-msvc",
    "--prefix", "none"
)
$macClusterUiGraphKeys = Get-GraphCrateKeys -Arguments @(
    "tree",
    "-p", "clusterui",
    "--edges", "normal",
    "--target", "aarch64-apple-darwin",
    "--prefix", "none"
)

$requiredRegistryPackages = New-Object System.Collections.Generic.List[object]
$seenRegistryKeys = New-Object System.Collections.Generic.HashSet[string]
foreach ($key in @($windowsGraphKeys + $macClusterUiGraphKeys)) {
    if (-not $packageByKey.ContainsKey($key)) {
        continue
    }

    $package = $packageByKey[$key]
    if ([string]::IsNullOrWhiteSpace($package.source)) {
        continue
    }
    if (-not $package.source.StartsWith("registry+", [System.StringComparison]::Ordinal)) {
        continue
    }
    if ($seenRegistryKeys.Add($key)) {
        $requiredRegistryPackages.Add($package)
    }
}

Write-Host "Refreshing third_party/licenses/rust-full ..."
if (-not (Test-Path -LiteralPath $rustFullRoot)) {
    New-Item -ItemType Directory -Force -Path $rustFullRoot | Out-Null
}
Get-ChildItem -Path $rustFullRoot -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force

foreach ($package in ($requiredRegistryPackages | Sort-Object name, version)) {
    $key = "$($package.name)-$($package.version)"
    $destinationDir = Join-Path $rustFullRoot $key
    New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null

    $sourceDir = Get-RegistrySourceDir -CrateName $package.name -Version $package.version
    $licenseFiles = if ($null -ne $sourceDir) { Get-LicenseFiles -SourceDir $sourceDir } else { @() }

    if ($licenseFiles.Count -eq 0) {
        if ([string]::IsNullOrWhiteSpace($package.license)) {
            throw "No license files or license expression found for crate '$key'."
        }
        Write-Utf8NoBom -Path (Join-Path $destinationDir "LICENSE-EXPRESSION.txt") -Content "License expression: $($package.license)`n"
        continue
    }

    foreach ($licenseFile in $licenseFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $sourceDir -TargetPath $licenseFile.FullName
        $destination = Join-Path $destinationDir $relativePath
        $destinationParent = Split-Path -Parent $destination
        if (-not [string]::IsNullOrWhiteSpace($destinationParent)) {
            New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
        }
        Copy-Item -LiteralPath $licenseFile.FullName -Destination $destination -Force
    }
}

Write-Host "Refreshing curated top-level Rust license files ..."
foreach ($crateName in $curatedRustTopLevelCrates) {
    $package = $requiredRegistryPackages |
        Where-Object { $_.name -eq $crateName } |
        Sort-Object version -Descending |
        Select-Object -First 1
    if ($null -eq $package) {
        throw "Curated Rust crate '$crateName' is not present in the current dependency graph."
    }
    Write-CuratedTopLevelLicenseFiles -Package $package
}

Write-Host "Rust license snapshot refreshed."

param(
    [string[]]$Scopes = @(
        "engine",
        "bridge",
        "diarize",
        "pdf",
        "pdfvlm",
        "extras",
        "third_party"
    ),
    [string[]]$Extensions = @(
        ".c",".cc",".cpp",".cxx",".h",".hpp",".hxx",".hh",".m",".mm",".java",".js",".ts",".py",".sh",".ps1",".bat",".cmake",".in",
        ".txt",".md",".yml",".yaml",".toml",".json",".ini",".cfg",".config",".cmake.in",".cpp.in",".h.in",
        ".cu",".cuh",".swift",".kt",".gradle",".rs",".go",".rb",".php",".pl",".cs",".vb",".fs",".dart",".r",".scala",
        ".sln",".vcxproj",".props",".gradle",".props"
    ),
    [string[]]$ExcludeDirs = @(
        ".git",
        ".github",
        "node_modules",
        ".gradle",
        "build",
        "target",
        "dist",
        "out"
    ),
    [string[]]$VendorOnlyPaths = @("third_party"),
    [string]$OutputDir = "..\ENGINEbuilds\license-audits",
    [string]$Pattern = "(?im)spdx-license-identifier|copyright|all rights reserved|permission is hereby granted|licensed under|license:|apache license|gnu general public license|bsd [0-9]+-clause|mit license|mozilla public|bsd clause",
    [switch]$NoThirdParty,
    [switch]$Strict,
    [switch]$OnlyFirstParty,
    [switch]$FailOnFirstParty,
    [switch]$FailOnUnknown,
    [switch]$Json,
    [int]$MaxSampleBytes = 262144
)

$ErrorActionPreference = "Stop"

$binaryExtensions = New-Object 'System.Collections.Generic.HashSet[string]'
@(
    ".png",".jpg",".jpeg",".gif",".bmp",".tiff",".webp",".ico",".pdf",".zip",".gz",".bz2",".xz",".7z",".jar",".class",".exe",".dll",".so",
    ".a",".dylib",".lib",".obj",".o",".pyd",".mp3",".mp4",".wav",".woff",".woff2",".eot",".ttf",".otf",".webm",".mkv",".avi",".mov",
    ".parquet",".onnx",".pb",".pt",".safetensors",".npz",".npy",".bin",".sqlite",".db",".wasm",".lo"
) | ForEach-Object { [void]$binaryExtensions.Add($_) }

$extensionlessNames = @(
    "license",
    "license.txt",
    "license.md",
    "license.md.txt",
    "copying",
    "copyright",
    "notice",
    "readme"
)

$licensePatterns = @(
    [pscustomobject]@{ Name = "spdx"; Pattern = '(?im)^\s*(?://|#|\*|/\*|<!--)?\s*SPDX-License-Identifier\s*:\s*([^\r\n*]+)' },
    [pscustomobject]@{ Name = "license-statement"; Pattern = '(?im)license\s+under\s+the\s+([a-z0-9 .-]+license(?:\s*version\s*\d+(?:\.\d+)*)?)' },
    [pscustomobject]@{ Name = "apache"; Pattern = '(?im)Apache\s+License(?:,?\s*Version)?\s*2\.0' },
    [pscustomobject]@{ Name = "mit"; Pattern = '(?im)\bMIT\b' },
    [pscustomobject]@{ Name = "gpl"; Pattern = '(?im)GNU\s+(?:Lesser\s+)?General\s+Public\s+License' },
    [pscustomobject]@{ Name = "bsd"; Pattern = '(?im)\bBSD(?:\s*[- ]\s*\d+\s*[- ]\s*clause)' },
    [pscustomobject]@{ Name = "isc"; Pattern = '(?im)\bISC\b' },
    [pscustomobject]@{ Name = "copyright"; Pattern = '(?im)\b(?:copyright|©)\b' },
    [pscustomobject]@{ Name = "all-rights"; Pattern = '(?im)\ball rights reserved\b' },
    [pscustomobject]@{ Name = "permission"; Pattern = '(?im)\bpermission is hereby granted\b' }
)

$knownLicenseTokens = @(
    "Apache-2.0","Apache License 2.0","MIT","BSD-2-Clause","BSD-3-Clause","GPL","GPL-2.0","GPL-3.0",
    "LGPL","LGPL-2.1","LGPL-3.0","MPL-2.0","ISC","CC0-1.0","Boost-1.0","Zlib","Unlicense","Python","OpenSSL"
)

function Normalize-Path {
    param([string]$Path)
    return ($Path -replace "\\", "/")
}

function Is-KnownLicenseHint {
    param([string]$Hint)
    if ([string]::IsNullOrWhiteSpace($Hint)) { return $false }
    $lowerHint = $Hint.ToLowerInvariant()
    foreach ($token in $knownLicenseTokens) {
        if ($lowerHint -like "*$($token.ToLowerInvariant())*") { return $true }
    }
    return $false
}

function Get-Component {
    param([string]$Path, [string]$RepoRoot)
    $norm = ($Path -replace [regex]::Escape($RepoRoot), "").TrimStart("\", "/")
    if ($norm -notmatch '^third_party[/\\]') {
        return "repo_code"
    }
    $parts = $norm -split "[/\\]"
    if ($parts.Length -ge 2) { return $parts[1] }
    return "third_party_root"
}

function Is-TextTarget {
    param([System.IO.FileInfo]$File)
    if ($binaryExtensions.Contains($File.Extension.ToLowerInvariant())) { return $false }
    $extension = $File.Extension.ToLowerInvariant()
    if ($Extensions -contains $extension) { return $true }
    $name = $File.Name.ToLowerInvariant()
    return ($extensionlessNames -contains $name)
}

function Detect-License {
    param([string]$Text)

    $hits = New-Object System.Collections.Generic.List[string]
    $matchedPatternNames = New-Object System.Collections.Generic.List[string]

    foreach ($entry in $licensePatterns) {
        $regex = [regex]::new($entry.Pattern)
        $matches = $regex.Matches($Text)
        if ($matches.Count -gt 0) {
            $matchedPatternNames.Add($entry.Name)
            foreach ($match in $matches) {
                if ($match.Groups.Count -gt 1 -and $match.Groups[1].Success) {
                    $value = $match.Groups[1].Value.Trim()
                    if (-not [string]::IsNullOrWhiteSpace($value)) { $hits.Add($value) }
                } else {
                    $hits.Add($match.Value.Trim())
                }
            }
        }
    }

    $licenseHints = $hits | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique
    $hasKnownLicense = $false
    foreach ($hint in $licenseHints) {
        if (Is-KnownLicenseHint -Hint $hint) {
            $hasKnownLicense = $true
            break
        }
    }

    return [PSCustomObject]@{
        MatchedPatterns = ($matchedPatternNames | Sort-Object -Unique) -join "; "
        LicenseHints    = $licenseHints
        HasCopyright   = [regex]::IsMatch($Text, '(?im)\b(?:copyright|©)\b')
        KnownLicense   = $hasKnownLicense
    }
}

function Read-TextSample {
    param([string]$Path, [int]$MaxBytes = 262144)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -gt $MaxBytes) {
        $bytes = $bytes[0..($MaxBytes - 1)]
    }

    foreach ($b in $bytes) {
        if ($b -eq 0) { return $null }
    }

    return [System.Text.Encoding]::UTF8.GetString($bytes)
}

$repoRoot = (Resolve-Path ".").Path.TrimEnd('\')

$resolvedScopes = @()
$rawScopes = @()
foreach ($scope in $Scopes) {
    $rawScopes += ($scope -split "[,;]" | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

if ($NoThirdParty) {
    $rawScopes = $rawScopes | Where-Object { $_ -ne "third_party" }
}

foreach ($scope in $rawScopes) {
    if (Test-Path $scope) {
        $resolvedScopes += (Resolve-Path $scope).Path
    }
}

if ($resolvedScopes.Count -eq 0) {
    throw "No valid scope paths found. Check -Scopes and working directory."
}

$files = @()
foreach ($scope in $resolvedScopes) {
    $files += Get-ChildItem -LiteralPath $scope -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $relative = $_.FullName.Substring($repoRoot.Length).TrimStart('\','/')
            $parts = $relative -split "[/\\]"
            $skip = $false
            foreach ($ex in $ExcludeDirs) {
                if ($parts -contains $ex) {
                    $skip = $true
                    break
                }
            }
            if ($skip) { return $false }
            if (-not (Is-TextTarget $_)) { return $false }
            return $true
        }
}

$entries = New-Object System.Collections.Generic.List[PSObject]
foreach ($file in $files) {
    $text = Read-TextSample -Path $file.FullName -MaxBytes $MaxSampleBytes
    if ($null -eq $text) { continue }
    if ($text -notmatch $Pattern) { continue }

    $analysis = Detect-License -Text $text
    if (-not $analysis.HasCopyright -and [string]::IsNullOrWhiteSpace($analysis.MatchedPatterns)) {
        continue
    }

    $isVendor = $file.FullName -match "[/\\]third_party[/\\]"
    if ($OnlyFirstParty -and $isVendor) { continue }

    $unknown = $false
    if ($Strict -and -not $analysis.KnownLicense -and $analysis.HasCopyright) { $unknown = $true }

    $licenseHintText = if ($analysis.LicenseHints.Count -gt 0) { [string]::Join("; ", $analysis.LicenseHints) } else { "" }
    $entries.Add([PSCustomObject]@{
        File = Normalize-Path ($file.FullName.Substring($repoRoot.Length + 1))
        Type = if ($isVendor) { "vendor" } else { "first-party" }
        Category = if ($isVendor) { "third_party" } else { "repo_code" }
        Component = Get-Component -Path $file.FullName -RepoRoot $repoRoot
        MatchedPatterns = $analysis.MatchedPatterns
        LicenseHint = $licenseHintText
        HasCopyright = $analysis.HasCopyright
        KnownLicense = $analysis.KnownLicense
        UnknownLicense = $unknown
        SizeBytes = $file.Length
    })
}

$firstParty = $entries | Where-Object { $_.Type -eq "first-party" }
$vendor = $entries | Where-Object { $_.Type -eq "vendor" }
$unknowns = $entries | Where-Object { $_.UnknownLicense }
$firstPartyCount = if ($null -eq $firstParty) { 0 } else { ($firstParty | Measure-Object).Count }
$vendorCount = if ($null -eq $vendor) { 0 } else { ($vendor | Measure-Object).Count }
$unknownCount = if ($null -eq $unknowns) { 0 } else { ($unknowns | Measure-Object).Count }

$vendorByComponent = $vendor | Sort-Object Component | Group-Object Component | ForEach-Object {
    [PSCustomObject]@{ Component = $_.Name; Hits = $_.Count }
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
$outCsv = Join-Path $OutputDir "license-audit-$stamp.csv"

$entries | Export-Csv -NoTypeInformation -Path $outCsv -Encoding UTF8

Write-Host "License audit complete"
Write-Host "Scanned files: $($files.Count)"
Write-Host "Header hits: $($entries.Count)"
Write-Host "First-party hits: $firstPartyCount"
Write-Host "Third-party hits: $vendorCount"
if ($Strict) { Write-Host "Unknown/Unrecognized license markers: $unknownCount" }

if ($vendorByComponent.Count -gt 0) {
    Write-Host "Third-party component buckets:"
    foreach ($row in $vendorByComponent) {
        Write-Host "  $($row.Component): $($row.Hits)"
    }
}

Write-Host "Report: $outCsv"

if ($Json) {
    $outJson = Join-Path $OutputDir "license-audit-$stamp.json"
    @{
        scanned = $files.Count
        hits = $entries.Count
        firstPartyHits = $firstPartyCount
        vendorHits = $vendorCount
        unknownLicenseHits = $unknownCount
        vendorByComponent = $vendorByComponent
        entries = $entries
    } | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $outJson
    Write-Host "JSON report: $outJson"
}

if ($FailOnFirstParty -and $firstPartyCount -gt 0) {
    Write-Error "First-party files with license text were found. Review these files before release."
    exit 2
}

if ($Strict -and $FailOnUnknown -and $unknownCount -gt 0) {
    Write-Error "Unknown license markers found in strict mode. Review before release."
    exit 3
}

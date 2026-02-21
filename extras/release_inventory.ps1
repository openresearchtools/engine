param(
    [Parameter(Mandatory = $true)]
    [string]$ArtifactDir,
    [string]$OutputRoot = "..\ENGINEbuilds\release-inventory",
    [string]$BaselineName = "default",
    [switch]$WriteBaseline,
    [switch]$CompareBaseline,
    [switch]$FailOnDiff
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$BasePath,
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $baseUri = [System.Uri]::new(($BasePath.TrimEnd('\') + '\'))
    $targetUri = [System.Uri]::new($TargetPath)
    return [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString().Replace('/', '\'))
}

$repoRoot = Resolve-AbsolutePath -BasePath $PSScriptRoot -Path ".."
$artifactPath = Resolve-AbsolutePath -BasePath $repoRoot -Path $ArtifactDir
if (-not (Test-Path $artifactPath -PathType Container)) {
    throw "Artifact directory not found: $artifactPath"
}

$outputRootAbs = Resolve-AbsolutePath -BasePath $repoRoot -Path $OutputRoot
$stamp = (Get-Date).ToString("yyyyMMdd-HHmmss")
$runDir = Join-Path $outputRootAbs ("run-" + $stamp)
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$fileRows = New-Object System.Collections.Generic.List[object]
$files = Get-ChildItem -Path $artifactPath -Recurse -File | Sort-Object FullName

foreach ($file in $files) {
    $hash = Get-FileHash -Path $file.FullName -Algorithm SHA256
    $rel = Get-RelativePath -BasePath $artifactPath -TargetPath $file.FullName
    $fileRows.Add([PSCustomObject]@{
        RelativePath = $rel
        SizeBytes = $file.Length
        SHA256 = $hash.Hash
    })
}

$manifestCsv = Join-Path $runDir "artifact-files.csv"
$allowlistTxt = Join-Path $runDir "allowlist.txt"

$fileRows | Export-Csv -NoTypeInformation -Path $manifestCsv -Encoding UTF8
($fileRows | Select-Object -ExpandProperty RelativePath) | Set-Content -Path $allowlistTxt -Encoding UTF8

$depsTxt = Join-Path $runDir "binary-dependents.txt"
$binaryExts = @(".dll", ".exe")
$binaryFiles = $files | Where-Object { $binaryExts -contains $_.Extension.ToLowerInvariant() }
$dumpbinCmd = Get-Command dumpbin -ErrorAction SilentlyContinue

if ($null -eq $dumpbinCmd) {
    Set-Content -Path $depsTxt -Value "dumpbin not found on PATH. Run from a Visual Studio Developer PowerShell to capture dependency lists." -Encoding UTF8
} else {
    $depLines = New-Object System.Collections.Generic.List[string]
    foreach ($bin in $binaryFiles) {
        $rel = Get-RelativePath -BasePath $artifactPath -TargetPath $bin.FullName
        $depLines.Add("### $rel")
        $raw = & $dumpbinCmd.Source /DEPENDENTS $bin.FullName 2>&1
        foreach ($line in $raw) {
            $depLines.Add([string]$line)
        }
        $depLines.Add("")
    }
    $depLines | Set-Content -Path $depsTxt -Encoding UTF8
}

$summaryTxt = Join-Path $runDir "summary.txt"
$summaryLines = @(
    "Release inventory summary",
    "Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"),
    "ArtifactDir: $artifactPath",
    "FileCount: $($fileRows.Count)",
    "BinaryCount: $($binaryFiles.Count)",
    "Manifest: $manifestCsv",
    "Allowlist: $allowlistTxt",
    "Dependents: $depsTxt"
)
$summaryLines | Set-Content -Path $summaryTxt -Encoding UTF8

$baselineDir = Join-Path $outputRootAbs (Join-Path "baselines" $BaselineName)
$compareTxt = Join-Path $runDir "baseline-compare.txt"
$hasDiff = $false

if ($WriteBaseline) {
    New-Item -ItemType Directory -Path $baselineDir -Force | Out-Null
    Copy-Item -Path $manifestCsv -Destination (Join-Path $baselineDir "artifact-files.csv") -Force
    Copy-Item -Path $allowlistTxt -Destination (Join-Path $baselineDir "allowlist.txt") -Force
    Copy-Item -Path $depsTxt -Destination (Join-Path $baselineDir "binary-dependents.txt") -Force
}

if ($CompareBaseline) {
    $baselineManifest = Join-Path $baselineDir "artifact-files.csv"
    if (-not (Test-Path $baselineManifest -PathType Leaf)) {
        Set-Content -Path $compareTxt -Value "Baseline missing: $baselineManifest" -Encoding UTF8
        if ($FailOnDiff) {
            throw "Baseline missing for compare."
        }
    } else {
        $baseRows = Import-Csv $baselineManifest
        $baseRel = $baseRows | Select-Object -ExpandProperty RelativePath
        $currRel = $fileRows | Select-Object -ExpandProperty RelativePath
        $diffRel = Compare-Object -ReferenceObject $baseRel -DifferenceObject $currRel

        $baseHash = @{}
        foreach ($r in $baseRows) {
            $baseHash[$r.RelativePath] = $r.SHA256
        }
        $currHash = @{}
        foreach ($r in $fileRows) {
            $currHash[$r.RelativePath] = $r.SHA256
        }

        $hashChanges = New-Object System.Collections.Generic.List[string]
        foreach ($k in $currHash.Keys) {
            if ($baseHash.ContainsKey($k) -and $baseHash[$k] -ne $currHash[$k]) {
                $hashChanges.Add("$k : baseline=$($baseHash[$k]) current=$($currHash[$k])")
            }
        }

        $lines = New-Object System.Collections.Generic.List[string]
        $lines.Add("Baseline compare")
        $lines.Add("Baseline: $baselineManifest")
        $lines.Add("Current : $manifestCsv")
        $lines.Add("")
        $lines.Add("Path differences:")
        if ($diffRel.Count -eq 0) {
            $lines.Add("none")
        } else {
            $hasDiff = $true
            foreach ($d in $diffRel) {
                $side = if ($d.SideIndicator -eq "=>") { "added" } else { "removed" }
                $lines.Add("$side : $($d.InputObject)")
            }
        }
        $lines.Add("")
        $lines.Add("Hash differences:")
        if ($hashChanges.Count -eq 0) {
            $lines.Add("none")
        } else {
            $hasDiff = $true
            foreach ($h in $hashChanges) {
                $lines.Add($h)
            }
        }
        $lines | Set-Content -Path $compareTxt -Encoding UTF8
    }
}

Write-Host "Inventory complete"
Write-Host "Run directory: $runDir"
Write-Host "Manifest: $manifestCsv"
Write-Host "Allowlist: $allowlistTxt"
Write-Host "Dependents: $depsTxt"
if ($CompareBaseline) {
    Write-Host "Compare: $compareTxt"
}
if ($FailOnDiff -and $hasDiff) {
    throw "Differences found vs baseline."
}

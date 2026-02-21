param(
    [string[]]$Components = @("llama.cpp", "whisper.cpp"),
    [string]$ThirdPartyRoot = "third_party",
    [string]$Destination = "third_party\\licenses\\cpp_licenses",
    [switch]$IncludeNested
)

$ErrorActionPreference = "Stop"

function Normalize-Path {
    param([string]$Path)
    return ($Path -replace "\\", "/")
}

$root = (Get-Location).ProviderPath.TrimEnd('\')
$sourceRoot = Join-Path $root $ThirdPartyRoot
$destRoot = Join-Path $root $Destination

New-Item -ItemType Directory -Path $destRoot -Force | Out-Null

$fileNames = @(
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
    "NOTICE",
    "NOTICE.txt"
)

foreach ($component in $Components) {
    $componentRoot = Join-Path $sourceRoot $component
    if (-not (Test-Path $componentRoot)) {
        Write-Warning "Component missing: $componentRoot"
        continue
    }

    $outputDir = Join-Path $destRoot $component
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    $topLicense = Join-Path $componentRoot "LICENSE"
    if (Test-Path $topLicense) {
        Copy-Item -Path $topLicense -Destination (Join-Path $outputDir "LICENSE.txt") -Force
    }
    else {
        $topCandidate = Get-ChildItem -LiteralPath $componentRoot -File |
            Where-Object { $fileNames -contains $_.Name } |
            Select-Object -First 1
        if ($null -ne $topCandidate) {
            Copy-Item -LiteralPath $topCandidate.FullName -Destination (Join-Path $outputDir $topCandidate.Name) -Force
        }
    }

    if ($IncludeNested) {
        Get-ChildItem -LiteralPath $componentRoot -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object {
                $name = $_.Name
                return $fileNames -contains $name
            } |
            ForEach-Object {
                $relative = $_.FullName.Substring($componentRoot.Length).TrimStart('\','/')
                $safeName = $relative -replace "[/\\]", "__"
                Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $outputDir $safeName) -Force
            }
    }
}

Write-Host "C++ license dump written to:"
Write-Host (Normalize-Path $Destination)

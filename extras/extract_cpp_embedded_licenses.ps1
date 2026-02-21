param(
    [string[]]$Components = @("llama.cpp", "whisper.cpp"),
    [string]$ThirdPartyRoot = "third_party",
    [string]$OutputDir = "..\ENGINEbuilds\cpp-license-snippets",
    [string]$Pattern = '(?im)\bSPDX-License-Identifier\b|\bcopyright\b|\ball rights reserved\b|\bpermission is hereby granted\b|\blicensed under\b|\bApache License\b|\bGNU General Public License\b|\bGNU Lesser General Public License\b|\bBSD-[0-9]+-Clause\b|\bMIT License\b|\bMozilla Public License\b|\bboost software\b|\bGNU AGPL\b|\bGNU LESSER GENERAL PUBLIC LICENSE\b|\bBoost Software License\b|\bCC0\b|\bISC License\b|\bMPL\b|\bZlib\b',
    [int]$ContextLines = 5
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path.TrimEnd('\')
$outputRoot = Join-Path $repoRoot $OutputDir
$out = if (Test-Path $outputRoot) { (Resolve-Path $outputRoot).Path } else { [System.IO.Path]::GetFullPath($outputRoot) }

New-Item -ItemType Directory -Path $out -Force | Out-Null

$binaryExtensions = New-Object 'System.Collections.Generic.HashSet[string]'
@(
    ".png",".jpg",".jpeg",".gif",".bmp",".tiff",".webp",".ico",".pdf",".zip",".gz",".bz2",".xz",".7z",".jar",".class",".exe",".dll",".so",
    ".a",".dylib",".lib",".obj",".o",".pyd",".mp3",".mp4",".wav",".woff",".woff2",".eot",".ttf",".otf",".webm",".mkv",".avi",".mov",
    ".parquet",".onnx",".pb",".pt",".safetensors",".npz",".npy",".bin",".sqlite",".db",".wasm",".lo"
) | ForEach-Object { [void]$binaryExtensions.Add($_) }

function Is-BinaryFile {
    param([string]$Path, [int]$SampleBytes = 32768)

    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $count = [Math]::Min($bytes.Length, $SampleBytes)
    for ($i = 0; $i -lt $count; $i++) {
        if ($bytes[$i] -eq 0) { return $true }
    }
    return $false
}

function Is-LicenseContextLine {
    param([string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) { return $true }
    $trimmed = $Line.Trim()
    if ($trimmed -match '^(//|/\*|\*|#|<!--|-->|\*/)') { return $true }
    return $false
}

function Extract-LicenseSnippets {
    param(
        [string[]]$Lines,
        [string]$Pattern,
        [int]$ContextLines
    )

    $regex = [regex]::new($Pattern, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
    $hitIndexes = @()
    $matchedPatterns = @()

    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $line = $Lines[$i]
        $matches = $regex.Matches($line)
        if ($matches.Count -eq 0) { continue }

        $hitIndexes += $i
        foreach ($match in $matches) {
            if (-not $match.Success) { continue }
            $token = ($match.Value -replace "\s+", " ").Trim()
            if (-not [string]::IsNullOrWhiteSpace($token)) {
                $matchedPatterns += $token.ToLowerInvariant()
            }
        }
    }

    if ($hitIndexes.Count -eq 0) {
        return [PSCustomObject]@{
            HasMatch = $false
            FileSnippets = @()
            MatchedPatterns = @()
        }
    }

    $uniquePatterns = $matchedPatterns | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique

    $ranges = @()
    foreach ($hit in $hitIndexes) {
        $start = $hit
        $end = $hit

        while ($start -gt 0 -and $start -ge ($hit - $ContextLines) -and (Is-LicenseContextLine -Line $Lines[$start - 1])) {
            $start -= 1
        }
        while ($end -lt ($Lines.Count - 1) -and $end -le ($hit + $ContextLines) -and (Is-LicenseContextLine -Line $Lines[$end + 1])) {
            $end += 1
        }

        $ranges += [PSCustomObject]@{ Start = $start; End = $end }
    }

    $merged = @()
    if ($ranges.Count -gt 0) {
        $sorted = $ranges | Sort-Object Start
        $current = $sorted[0]
        foreach ($r in $sorted | Select-Object -Skip 1) {
            if ($r.Start -le ($current.End + 1)) {
                $current = [PSCustomObject]@{
                    Start = $current.Start
                    End = [Math]::Max($current.End, $r.End)
                }
            } else {
                $merged += $current
                $current = $r
            }
        }
        $merged += $current
    }

    $snippetList = @()
    foreach ($m in $merged) {
        $snippetLines = @()
        for ($i = $m.Start; $i -le $m.End; $i++) {
            $snippetLines += ("{0,6}: {1}" -f ($i + 1), $Lines[$i])
        }
        $snippetList += ($snippetLines -join "`r`n")
    }

    return [PSCustomObject]@{
        HasMatch = $true
        FileSnippets = $snippetList
        MatchedPatterns = $uniquePatterns
    }
}

$collected = @()

foreach ($component in $Components) {
    $componentRoot = Join-Path $repoRoot (Join-Path $ThirdPartyRoot $component)
    if (-not (Test-Path $componentRoot)) {
        Write-Warning "Component not found: $componentRoot"
        continue
    }

    $files = Get-ChildItem -Path $componentRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { -not $binaryExtensions.Contains($_.Extension.ToLowerInvariant()) } |
        Where-Object { -not (Is-BinaryFile -Path $_.FullName) }

    foreach ($file in $files) {
        $lines = [System.IO.File]::ReadAllLines($file.FullName)
        $result = Extract-LicenseSnippets -Lines $lines -Pattern $Pattern -ContextLines $ContextLines
        if (-not $result.HasMatch) { continue }

        $relative = $file.FullName.Substring($repoRoot.Length).TrimStart('\','/')
            $collected += [PSCustomObject]@{
                Component = $component
                Source = $file.FullName
                RelativePath = $relative
                SizeBytes = $file.Length
                MatchedPatterns = ($result.MatchedPatterns -join "; ")
                MatchCount = $result.FileSnippets.Count
                Snippets = $result.FileSnippets
            }
        }
}

$manifest = Join-Path $out "cpp-license-embedded-manifest.csv"
$bundle = Join-Path $out "cpp-license-snippets.txt"
$filesOutDir = Join-Path $out "files"
New-Item -ItemType Directory -Path $filesOutDir -Force | Out-Null
$count = $collected.Count

$collected |
    Sort-Object Source |
    Select-Object Component,Source,RelativePath,SizeBytes,MatchCount,MatchedPatterns |
    Export-Csv -NoTypeInformation -Path $manifest -Encoding UTF8

Set-Content -Path $bundle -Value @(
    "CPP LICENSE SNIPPETS (embedded in source files)",
    "Generated: " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"),
    "Total files: " + $count,
    "=============================================================="
) -Encoding UTF8

foreach ($entry in ($collected | Sort-Object Source)) {
    $safeName = $entry.RelativePath -replace "[\\/]", "__"
    $safeName = [Regex]::Replace($safeName, "[`:\*\?\|""<> ]", "_")
    $fileOutPath = Join-Path $filesOutDir "$safeName.license-snippets.txt"

    Add-Content -Path $bundle -Value @(
        "",
        "",
        "--- SOURCE: $($entry.Source)",
        "--- COMPONENT: $($entry.Component)",
        "--- MATCHED: $($entry.MatchedPatterns)",
        "=============================================================="
    )

    $entryLines = @(
        "SOURCE: $($entry.Source)",
        "COMPONENT: $($entry.Component)",
        "MATCHED: $($entry.MatchedPatterns)",
        "FILE SIZE BYTES: $($entry.SizeBytes)",
        "SNIPPET COUNT: $($entry.MatchCount)",
        ("=" * 80)
    )

    $entrySnippets = @($entry.Snippets)
    if ($entrySnippets.Count -gt 0) {
        foreach ($snippet in $entrySnippets) {
            Add-Content -Path $bundle -Value $snippet
            Add-Content -Path $bundle -Value ("-" * 60)
            $entryLines += ""
            $entryLines += $snippet
            $entryLines += ("-" * 60)
        }
    }

    Set-Content -Path $fileOutPath -Value $entryLines -Encoding UTF8
}

Write-Host "CPP license files with embedded license snippets: $count"
Write-Host "Manifest: $manifest"
Write-Host "Bundle: $bundle"
Write-Host "Per-file snippets: $filesOutDir"

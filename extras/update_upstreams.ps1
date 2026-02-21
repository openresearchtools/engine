param(
    [string]$LlamaRepo = "https://github.com/ggerganov/llama.cpp.git",
    [string]$WhisperRepo = "https://github.com/ggerganov/whisper.cpp.git",
    [string]$LlamaRef = "master",
    [string]$WhisperRef = "master",
    [string]$LlamaPrefix = "third_party/llama.cpp",
    [string]$WhisperPrefix = "third_party/whisper.cpp",
    [string]$CacheRoot = "",
    [switch]$LlamaOnly,
    [switch]$WhisperOnly,
    [switch]$SkipValidation,
    [switch]$Commit,
    [string]$CommitMessage = "Update upstream third_party sources",
    [switch]$Push
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
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

function Invoke-GitOrThrow {
    param(
        [string]$WorkingDir,
        [string[]]$GitArgs,
        [string]$FailureMessage
    )

    & git -C $WorkingDir @GitArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Sync-UpstreamFolder {
    param(
        [string]$Name,
        [string]$RepoUrl,
        [string]$Ref,
        [string]$DestinationPath,
        [string]$CacheBase,
        [string]$ProjectRegex,
        [bool]$ValidateProject
    )

    $cachePath = Join-Path $CacheBase $Name
    if (-not (Test-Path -LiteralPath $cachePath)) {
        Write-Host "Cloning $Name cache..."
        Invoke-GitOrThrow -WorkingDir $CacheBase -GitArgs @("clone", "--origin", "origin", $RepoUrl, $cachePath) -FailureMessage "Failed to clone $Name from $RepoUrl"
    } else {
        Write-Host "Fetching $Name cache..."
        Invoke-GitOrThrow -WorkingDir $cachePath -GitArgs @("fetch", "origin", "--tags", "--prune") -FailureMessage "Failed to fetch $Name from origin"
    }

    Invoke-GitOrThrow -WorkingDir $cachePath -GitArgs @("checkout", "--force", "--quiet", $Ref) -FailureMessage "Failed to checkout ref '$Ref' for $Name"
    Invoke-GitOrThrow -WorkingDir $cachePath -GitArgs @("reset", "--hard", "--quiet") -FailureMessage "Failed to reset cache for $Name"
    Invoke-GitOrThrow -WorkingDir $cachePath -GitArgs @("clean", "-xfd") -FailureMessage "Failed to clean cache for $Name"

    $head = (& git -C $cachePath rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to resolve HEAD for $Name"
    }

    if ($ValidateProject) {
        $cmakeLists = Join-Path $cachePath "CMakeLists.txt"
        if (-not (Test-Path -LiteralPath $cmakeLists)) {
            throw "$Name cache missing CMakeLists.txt at $cmakeLists"
        }
        $cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
        if ($cmakeText -notmatch $ProjectRegex) {
            throw "$Name validation failed for ref '$Ref'. Expected project regex '$ProjectRegex'."
        }
    }

    New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
    Write-Host "Syncing $Name to $DestinationPath"

    $roboLog = Join-Path $CacheBase ("robocopy-" + $Name + ".log")
    & robocopy $cachePath $DestinationPath /MIR /XD ".git" /NFL /NDL /NP /R:2 /W:1 /LOG:$roboLog | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -gt 7) {
        throw "robocopy failed for $Name with exit code $rc (log: $roboLog)"
    }

    return [pscustomobject]@{
        Name = $Name
        Ref = $Ref
        Commit = $head
        Destination = $DestinationPath
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    $CacheRoot = Join-Path $buildsRoot "upstream-cache"
}
$CacheRoot = Resolve-AbsolutePath -PathValue $CacheRoot -RepoRoot $repoRoot

if ($LlamaOnly -and $WhisperOnly) {
    throw "Use only one of -LlamaOnly or -WhisperOnly."
}

New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null

Set-Location $repoRoot
& git -C $repoRoot rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not a git repository: $repoRoot"
}

$results = @()
$validate = -not $SkipValidation.IsPresent

if (-not $WhisperOnly) {
    $llamaDest = Resolve-AbsolutePath -PathValue $LlamaPrefix -RepoRoot $repoRoot
    $results += Sync-UpstreamFolder `
        -Name "llama.cpp" `
        -RepoUrl $LlamaRepo `
        -Ref $LlamaRef `
        -DestinationPath $llamaDest `
        -CacheBase $CacheRoot `
        -ProjectRegex 'project\("llama\.cpp"' `
        -ValidateProject $validate
}

if (-not $LlamaOnly) {
    $whisperDest = Resolve-AbsolutePath -PathValue $WhisperPrefix -RepoRoot $repoRoot
    $results += Sync-UpstreamFolder `
        -Name "whisper.cpp" `
        -RepoUrl $WhisperRepo `
        -Ref $WhisperRef `
        -DestinationPath $whisperDest `
        -CacheBase $CacheRoot `
        -ProjectRegex 'project\("whisper\.cpp"' `
        -ValidateProject $validate
}

Write-Host ""
Write-Host "Upstream sync complete:"
foreach ($item in $results) {
    Write-Host ("- {0} @ {1} ({2}) -> {3}" -f $item.Name, $item.Ref, $item.Commit, $item.Destination)
}

if ($Commit) {
    $pathsToAdd = @()
    if (-not $WhisperOnly) { $pathsToAdd += $LlamaPrefix }
    if (-not $LlamaOnly) { $pathsToAdd += $WhisperPrefix }

    & git add -- @pathsToAdd
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to stage updated upstream folders."
    }

    & git commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to commit upstream updates."
    }
    Write-Host "Committed updates."
}

if ($Push) {
    if (-not $Commit) {
        Write-Warning "Push requested but -Commit was not provided; nothing pushed."
    } else {
        & git push
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to push commits."
        }
        Write-Host "Push complete."
    }
}

param(
    [string]$LlamaRepo = "https://github.com/ggerganov/llama.cpp.git",
    [string]$WhisperRepo = "https://github.com/ggerganov/whisper.cpp.git",
    [string]$LlamaBranch = "master",
    [string]$WhisperBranch = "master",
    [string]$LlamaPrefix = "third_party/llama.cpp",
    [string]$WhisperPrefix = "third_party/whisper.cpp",
    [string]$CommitMessage = "Update upstream third_party subtrees",
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = "Stop"

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Invoke-SubtreeUpdate {
    param(
        [string]$Name,
        [string]$Prefix,
        [string]$Repository,
        [string]$Branch
    )

    if (-not (Test-Path $Prefix)) {
        throw "Subtree path '$Prefix' does not exist."
    }

    Write-Host "Updating $Name from $Repository ($Branch)..."
    git subtree pull --prefix=$Prefix $Repository $Branch --squash
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to update $Name from $Repository."
    }
}

$repoRoot = Get-RepoRoot
Set-Location $repoRoot

git rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not a git repository: $repoRoot"
}

$dirty = git status --porcelain
if ($dirty) {
    Write-Error "Working tree is dirty. Commit or stash your changes first."
    $dirty | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Invoke-SubtreeUpdate -Name "llama.cpp" -Prefix $LlamaPrefix -Repository $LlamaRepo -Branch $LlamaBranch
Invoke-SubtreeUpdate -Name "whisper.cpp" -Prefix $WhisperPrefix -Repository $WhisperRepo -Branch $WhisperBranch

$changes = git status --short
if (-not $changes) {
    Write-Host "No upstream updates to apply."
    exit 0
}

if ($Commit) {
    git add $LlamaPrefix $WhisperPrefix
    git commit -m $CommitMessage
    Write-Host "Committed updates."
} else {
    Write-Host "Updates applied. Use -Commit to create a commit."
}

if ($Push) {
    if (-not $Commit) {
        Write-Warning "Push requested but no commit was created; nothing to push."
    } else {
        git push
        Write-Host "Push complete."
    }
}

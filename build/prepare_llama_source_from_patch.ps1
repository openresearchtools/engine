param(
    [string]$OutDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-PythonInvocation {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        return @($py.Source, "-3")
    }

    $python3 = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3) {
        return @($python3.Source)
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }

    throw "Python 3 was not found. Install Python or ensure py/python3/python is on PATH."
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonScript = Join-Path $PSScriptRoot "prepare_llama_source_from_patch.py"
if (-not (Test-Path -LiteralPath $pythonScript)) {
    throw "Missing script: $pythonScript"
}

$pythonInvocation = Resolve-PythonInvocation
$pythonExe = $pythonInvocation[0]
$pythonPrefix = @()
if ($pythonInvocation.Count -gt 1) {
    $pythonPrefix = $pythonInvocation[1..($pythonInvocation.Count - 1)]
}

$args = @()
$args += $pythonPrefix
$args += $pythonScript
$args += "--repo-root"
$args += $repoRoot
if (-not [string]::IsNullOrWhiteSpace($OutDir)) {
    if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
        $OutDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutDir))
    } else {
        $OutDir = [System.IO.Path]::GetFullPath($OutDir)
    }
    $args += "--out-dir"
    $args += $OutDir
}
if ($Force) {
    $args += "--force"
}

& $pythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "prepare_llama_source_from_patch.py failed."
}

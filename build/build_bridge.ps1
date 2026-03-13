param(
    [string]$CmakeExe = "cmake",
    [string]$CmakeGenerator = "Visual Studio 17 2022",
    [string]$CmakeArch = "x64",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Config = "Release",
    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$Backend = "vulkan",
    [string]$LlamaCppDir = "",
    [string]$WhisperCppDir = "",
    [string]$BuildRoot = "",
    [string]$BuildDir = "",
    [string]$BridgeSrcDir = "",
    [string]$BridgeRelativeDir = "MARKDOWN\\bridge",
    [bool]$StageBridgeSources = $true,
    [bool]$StageWhisperSource = $true,
    [switch]$BuildLlamaServerCli,
    [switch]$BuildRealtimeSmoke,
    [switch]$BuildRealtimeBridgeSmoke,
    [switch]$EnableFfmpeg,
    [bool]$EnableBackendDl = $false,
    [bool]$EnableCpuAllVariants = $false,
    [bool]$DisableGgmlNative = $false,
    [string]$FfmpegRoot = "",
    [ValidateSet("off", "openssl", "boringssl", "libressl")]
    [string]$HttpsBackend = "boringssl",
    [int]$Jobs = 0
)

$ErrorActionPreference = "Stop"

function Resolve-CmakeGenerator {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfiguredGenerator,
        [Parameter(Mandatory = $true)]
        [bool]$GeneratorExplicit
    )

    if (-not $GeneratorExplicit -and -not [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR)) {
        return $env:CMAKE_GENERATOR
    }
    return $ConfiguredGenerator
}

function Resolve-CmakeArch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ConfiguredArch,
        [Parameter(Mandatory = $true)]
        [bool]$ArchExplicit
    )

    if (-not $ArchExplicit -and -not [string]::IsNullOrWhiteSpace($env:CMAKE_GENERATOR_PLATFORM)) {
        return $env:CMAKE_GENERATOR_PLATFORM
    }
    return $ConfiguredArch
}

function Test-CmakeGeneratorSupportsPlatform {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Generator
    )

    $platformlessGenerators = @(
        "Ninja",
        "Ninja Multi-Config",
        "NMake Makefiles",
        "NMake Makefiles JOM",
        "Unix Makefiles",
        "MinGW Makefiles"
    )

    return -not ($platformlessGenerators -contains $Generator)
}

function Resolve-AbsolutePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $PathValue))
}

function Test-IsUnderPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    $fullPath = [System.IO.Path]::GetFullPath($PathValue).TrimEnd('\') + '\'
    $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\') + '\'
    return $fullPath.StartsWith($fullBase, [System.StringComparison]::OrdinalIgnoreCase)
}

function Copy-DirectoryTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourceRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationRoot
    )

    if (-not (Test-Path -LiteralPath $SourceRoot)) {
        throw "Source directory not found: $SourceRoot"
    }
    if ([System.IO.Path]::GetFullPath($SourceRoot).TrimEnd('\') -eq [System.IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')) {
        throw "Refusing to copy a directory onto itself: $SourceRoot"
    }

    if (Test-Path -LiteralPath $DestinationRoot) {
        Remove-Item -LiteralPath $DestinationRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    $items = Get-ChildItem -LiteralPath $SourceRoot -Force -ErrorAction SilentlyContinue
    foreach ($item in $items) {
        Copy-Item -LiteralPath $item.FullName -Destination $DestinationRoot -Recurse -Force
    }
}

function Assert-AudioPatchApplied {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LlamaRoot
    )

    $checks = @(
        @{
            Path = Join-Path $LlamaRoot "tools\\server\\server-context.h"
            Pattern = "post_audio_transcriptions"
            Name = "server audio route field"
        },
        @{
            Path = Join-Path $LlamaRoot "tools\\server\\server-context.cpp"
            Pattern = "this->post_audio_transcriptions"
            Name = "server audio transcription handler"
        },
        @{
            Path = Join-Path $LlamaRoot "tools\\server\\server.cpp"
            Pattern = "/v1/audio/transcriptions"
            Name = "server audio HTTP endpoint"
        },
        @{
            Path = Join-Path $LlamaRoot "tools\\server\\server.cpp"
            Pattern = "LLAMA_SERVER_AUDIO_ONLY"
            Name = "audio-only server mode flag"
        },
        @{
            Path = Join-Path $LlamaRoot "tools\\whisper\\whisper-cli-entrypoint.cpp"
            Pattern = "whisper_cli_inproc_main"
            Name = "whisper in-process entrypoint"
        },
        @{
            Path = Join-Path $LlamaRoot "whisper.cpp\\examples\\grammar-parser.cpp"
            Pattern = "namespace grammar_parser"
            Name = "whisper grammar parser source"
        },
        @{
            Path = Join-Path $LlamaRoot "whisper.cpp\\src\\whisper.cpp"
            Pattern = "whisper"
            Name = "whisper core source"
        }
    )

    foreach ($check in $checks) {
        if (-not (Test-Path -LiteralPath $check.Path)) {
            throw "Audio patch verification failed: missing file for $($check.Name): $($check.Path)"
        }

        $match = Select-String -LiteralPath $check.Path -Pattern $check.Pattern -SimpleMatch -ErrorAction SilentlyContinue
        if (-not $match) {
            throw "Audio patch verification failed: missing marker '$($check.Pattern)' for $($check.Name) in $($check.Path)"
        }
    }

    Write-Host "Verified audio patch markers in llama source."
}

function Invoke-CmakeBuildTargets {
    param(
        [string]$Cmake,
        [string]$BuildDirPath,
        [string]$ConfigName,
        [string[]]$Targets,
        [int]$ParallelJobs
    )

    $args = @("--build", $BuildDirPath, "--config", $ConfigName)
    if ($ParallelJobs -gt 0) {
        $args += @("--parallel", $ParallelJobs)
    }
    $args += @("--target")
    $args += $Targets

    $oldErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Cmake @args 2>&1 | ForEach-Object { Write-Host $_ }
        $cmakeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($cmakeExitCode -ne 0) {
        throw "CMake build failed for targets: $($Targets -join ', ')"
    }
}

function Resolve-CmakeExecutable {
    param(
        [string]$ToolValue
    )

    if (-not [string]::IsNullOrWhiteSpace($ToolValue) -and [System.IO.Path]::IsPathRooted($ToolValue)) {
        if (Test-Path -LiteralPath $ToolValue) {
            return [System.IO.Path]::GetFullPath($ToolValue)
        }
        throw "CMake executable not found at: $ToolValue"
    }

    if (-not [string]::IsNullOrWhiteSpace($ToolValue)) {
        $cmd = Get-Command $ToolValue -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }

    $vsCandidates = @(
        "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    )

    foreach ($candidate in $vsCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "CMake executable not found. Install cmake or pass -CmakeExe with full path."
}

function Ensure-BridgeCmakeHooks {
    param(
        [string]$LlamaRoot
    )

    $cmakePath = Join-Path $LlamaRoot "CMakeLists.txt"
    if (-not (Test-Path -LiteralPath $cmakePath)) {
        throw "Cannot patch bridge hooks. Missing file: $cmakePath"
    }

    $text = Get-Content -Raw -LiteralPath $cmakePath
    $newline = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }
    $updated = $text

    if ($updated -notmatch 'LLAMA_BUILD_MARKDOWN_BRIDGE') {
        $optionLine = 'option(LLAMA_BUILD_MARKDOWN_BRIDGE "llama: build markdown in-process bridge library" OFF)'
        $serverOptionPattern = 'option\(LLAMA_BUILD_SERVER\s+"[^"]+"\s+\$\{LLAMA_STANDALONE\}\)'
        $serverOptionMatch = [regex]::Match($updated, $serverOptionPattern)
        if (-not $serverOptionMatch.Success) {
            throw "Failed to locate LLAMA_BUILD_SERVER option in $cmakePath to inject bridge option."
        }
        $insertText = $serverOptionMatch.Value + $newline + $optionLine
        $updated = $updated.Remove($serverOptionMatch.Index, $serverOptionMatch.Length).Insert($serverOptionMatch.Index, $insertText)
    }

    if ($updated -notmatch 'add_subdirectory\(MARKDOWN/bridge\)') {
        $toolsBlockPattern = 'if \(LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS\)\s*add_subdirectory\(tools\)\s*endif\(\)'
        $toolsBlockMatch = [regex]::Match($updated, $toolsBlockPattern)
        if (-not $toolsBlockMatch.Success) {
            throw "Failed to locate tools block in $cmakePath to inject MARKDOWN/bridge subdirectory."
        }
        $bridgeBlock = @(
            "",
            "if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS AND LLAMA_BUILD_MARKDOWN_BRIDGE)",
            "    add_subdirectory(MARKDOWN/bridge)",
            "endif()"
        ) -join $newline
        $insertText = $toolsBlockMatch.Value + $bridgeBlock
        $updated = $updated.Remove($toolsBlockMatch.Index, $toolsBlockMatch.Length).Insert($toolsBlockMatch.Index, $insertText)
    }

    if ($updated -ne $text) {
        $encoding = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::WriteAllText($cmakePath, $updated, $encoding)
        Write-Host "Patched llama.cpp CMakeLists bridge hooks: $cmakePath"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$buildsRoot = Join-Path (Split-Path -Parent $repoRoot) "ENGINEbuilds"
$CmakeExe = Resolve-CmakeExecutable -ToolValue $CmakeExe
$CmakeGenerator = Resolve-CmakeGenerator -ConfiguredGenerator $CmakeGenerator -GeneratorExplicit $PSBoundParameters.ContainsKey("CmakeGenerator")
$CmakeArch = Resolve-CmakeArch -ConfiguredArch $CmakeArch -ArchExplicit $PSBoundParameters.ContainsKey("CmakeArch")
$repoLlamaCppDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "third_party\\llama.cpp"))
$repoWhisperCppDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "third_party\\whisper.cpp"))
$autoPreparedSource = $false

if ([string]::IsNullOrWhiteSpace($LlamaCppDir)) {
    $LlamaCppDir = Join-Path $buildsRoot "sources\\llama.cpp"
    $autoPreparedSource = $true
}
$LlamaCppDir = Resolve-AbsolutePath -PathValue $LlamaCppDir -RepoRoot $repoRoot

if ([System.IO.Path]::GetFullPath($LlamaCppDir).TrimEnd('\') -eq $repoLlamaCppDir.TrimEnd('\')) {
    $LlamaCppDir = Join-Path $buildsRoot "sources\\llama.cpp"
    $LlamaCppDir = Resolve-AbsolutePath -PathValue $LlamaCppDir -RepoRoot $repoRoot
    $autoPreparedSource = $true
}

if ([string]::IsNullOrWhiteSpace($WhisperCppDir)) {
    $WhisperCppDir = Join-Path $repoRoot "third_party\\whisper.cpp"
}
$WhisperCppDir = Resolve-AbsolutePath -PathValue $WhisperCppDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($BuildRoot)) {
    $BuildRoot = Join-Path $buildsRoot "llama"
}
$BuildRoot = Resolve-AbsolutePath -PathValue $BuildRoot -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $BuildRoot ("cmake-" + $Backend + "-" + $Config.ToLowerInvariant())
}
$BuildDir = Resolve-AbsolutePath -PathValue $BuildDir -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $BuildDir -BasePath $repoRoot) {
    throw "BuildDir must be outside the repo. Use a path under ..\\ENGINEbuilds. Current: $BuildDir"
}

if (Test-IsUnderPath -PathValue $LlamaCppDir -BasePath $repoRoot) {
    $llamaInsideRepo = [System.IO.Path]::GetFullPath($LlamaCppDir).TrimEnd('\')
    if ($llamaInsideRepo -ne $repoLlamaCppDir.TrimEnd('\')) {
        throw "LlamaCppDir must be outside the repo. Current: $LlamaCppDir"
    }
}

if ($autoPreparedSource) {
    $prepareScript = Join-Path $PSScriptRoot "prepare_llama_source_from_patch.ps1"
    if (-not (Test-Path -LiteralPath $prepareScript)) {
        throw "Missing llama source prep script: $prepareScript"
    }
    & $prepareScript -OutDir $LlamaCppDir -Force
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare external llama source."
    }
    $preparedSiblingWhisper = Join-Path (Split-Path -Parent $LlamaCppDir) "whisper.cpp"
    if ([string]::IsNullOrWhiteSpace($WhisperCppDir) -or [System.IO.Path]::GetFullPath($WhisperCppDir).TrimEnd('\') -eq $repoWhisperCppDir.TrimEnd('\')) {
        $WhisperCppDir = $preparedSiblingWhisper
    }
    $StageBridgeSources = $false
    $StageWhisperSource = $false
}

if ([string]::IsNullOrWhiteSpace($BridgeSrcDir)) {
    $BridgeSrcDir = Join-Path $repoRoot "bridge"
}
$BridgeSrcDir = Resolve-AbsolutePath -PathValue $BridgeSrcDir -RepoRoot $repoRoot

if ([string]::IsNullOrWhiteSpace($FfmpegRoot)) {
    $FfmpegRoot = Join-Path $buildsRoot "runtime-deps\\ffmpeg"
}
$FfmpegRoot = Resolve-AbsolutePath -PathValue $FfmpegRoot -RepoRoot $repoRoot

if (Test-IsUnderPath -PathValue $FfmpegRoot -BasePath $repoRoot) {
    throw "FfmpegRoot must be outside the repo. Use a path under ..\\ENGINEbuilds\\runtime-deps. Current: $FfmpegRoot"
}

$cmakeLists = Join-Path $LlamaCppDir "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeLists)) {
    throw "llama.cpp source not found at: $LlamaCppDir"
}

$cmakeText = Get-Content -Raw -LiteralPath $cmakeLists
if ($cmakeText -match 'project\("whisper\.cpp"') {
    throw "Expected llama.cpp at '$LlamaCppDir' but found whisper.cpp sources. Fix third_party/llama.cpp first."
}
if ($cmakeText -notmatch 'project\("llama\.cpp"') {
    throw "Unable to verify llama.cpp source root at '$LlamaCppDir'."
}

if ($StageBridgeSources) {
    if (Test-IsUnderPath -PathValue $LlamaCppDir -BasePath $repoRoot) {
        throw "StageBridgeSources cannot target repo sources. Use an external prepared llama source."
    }
    Ensure-BridgeCmakeHooks -LlamaRoot $LlamaCppDir
}

if ($StageBridgeSources) {
    if (-not (Test-Path -LiteralPath $BridgeSrcDir)) {
        throw "Bridge source dir not found: $BridgeSrcDir"
    }
    $bridgeDstDir = Join-Path $LlamaCppDir $BridgeRelativeDir
    New-Item -ItemType Directory -Force -Path $bridgeDstDir | Out-Null
    Copy-Item -Path (Join-Path $BridgeSrcDir "*") -Destination $bridgeDstDir -Recurse -Force
}

if ($StageWhisperSource) {
    if (Test-IsUnderPath -PathValue $LlamaCppDir -BasePath $repoRoot) {
        throw "StageWhisperSource cannot target repo sources. Use an external prepared llama source."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $WhisperCppDir "CMakeLists.txt"))) {
        $llamaParentDir = Split-Path -Parent $LlamaCppDir
        $fallbackCandidates = @(
            (Join-Path $llamaParentDir "whisper.cpp"),
            (Join-Path $LlamaCppDir "whisper.cpp")
        )
        foreach ($candidate in $fallbackCandidates) {
            if (Test-Path -LiteralPath (Join-Path $candidate "CMakeLists.txt")) {
                $WhisperCppDir = $candidate
                break
            }
        }
    }
    $whisperCmake = Join-Path $WhisperCppDir "CMakeLists.txt"
    if (-not (Test-Path -LiteralPath $whisperCmake)) {
        throw "whisper.cpp source not found at: $WhisperCppDir"
    }
    $llamaParentDir = Split-Path -Parent $LlamaCppDir
    $whisperDestInTree = Join-Path $LlamaCppDir "whisper.cpp"
    if ([System.IO.Path]::GetFullPath($WhisperCppDir).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($whisperDestInTree).TrimEnd('\')) {
        Copy-DirectoryTree -SourceRoot $WhisperCppDir -DestinationRoot $whisperDestInTree
        Write-Host "Staged whisper.cpp source tree: $whisperDestInTree"
    } else {
        Write-Host "whisper.cpp source tree already staged in llama source: $whisperDestInTree"
    }

    # Some legacy CMake paths resolve whisper.cpp as a sibling of the llama source root.
    # Stage a second copy there to preserve compatibility with legacy relative paths.
    $whisperDestSibling = Join-Path $llamaParentDir "whisper.cpp"
    if ([System.IO.Path]::GetFullPath($WhisperCppDir).TrimEnd('\') -ne [System.IO.Path]::GetFullPath($whisperDestSibling).TrimEnd('\')) {
        Copy-DirectoryTree -SourceRoot $WhisperCppDir -DestinationRoot $whisperDestSibling
        Write-Host "Staged whisper.cpp sibling source tree: $whisperDestSibling"
    } else {
        Write-Host "whisper.cpp sibling source already staged: $whisperDestSibling"
    }
}

if ($StageWhisperSource) {
    Assert-AudioPatchApplied -LlamaRoot $LlamaCppDir
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

if (-not (Test-CmakeGeneratorSupportsPlatform -Generator $CmakeGenerator) -and -not [string]::IsNullOrWhiteSpace($CmakeArch)) {
    Write-Host "Skipping CMake platform selection '-A $CmakeArch' for generator '$CmakeGenerator'."
    $CmakeArch = ""
}

$cmakeArgs = @(
    "-S", $LlamaCppDir,
    "-B", $BuildDir,
    "-G", $CmakeGenerator,
    "-DBUILD_SHARED_LIBS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_MARKDOWN_BRIDGE=ON",
    "-DLLAMA_HTTPLIB=ON"
)

if (-not [string]::IsNullOrWhiteSpace($CmakeArch)) {
    $cmakeArgs += @("-A", $CmakeArch)
}

switch ($HttpsBackend) {
    "off" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "openssl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=ON"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "boringssl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=ON"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=OFF"
    }
    "libressl" {
        $cmakeArgs += "-DLLAMA_OPENSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_BORINGSSL=OFF"
        $cmakeArgs += "-DLLAMA_BUILD_LIBRESSL=ON"
    }
}

switch ($Backend) {
    "cuda" {
        $cmakeArgs += "-DGGML_CUDA=ON"
        $cmakeArgs += "-DGGML_CUDA_CUB_3DOT2=ON"
        if (-not [string]::IsNullOrWhiteSpace($env:CUDACXX)) {
            $cmakeArgs += "-DCMAKE_CUDA_COMPILER=$env:CUDACXX"
        }
        if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
            $cmakeArgs += "-DCUDAToolkit_ROOT=$env:CUDA_PATH"
        }
    }
    "vulkan" {
        $cmakeArgs += "-DGGML_VULKAN=ON"
    }
    default { }
}

if ($EnableCpuAllVariants) {
    $EnableBackendDl = $true
}
if ($EnableBackendDl) {
    $cmakeArgs += "-DGGML_BACKEND_DL=ON"
}
if ($EnableCpuAllVariants) {
    $cmakeArgs += "-DGGML_CPU_ALL_VARIANTS=ON"
}
if ($DisableGgmlNative) {
    $cmakeArgs += "-DGGML_NATIVE=OFF"
}

if ($EnableFfmpeg) {
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_ENABLE_FFMPEG=ON"
    $cmakeArgs += "-DLLAMA_SERVER_BRIDGE_FFMPEG_ROOT=$FfmpegRoot"
}

$oldErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $CmakeExe @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
    $cmakeConfigureExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $oldErrorActionPreference
}
if ($cmakeConfigureExitCode -ne 0) {
    throw "CMake configure failed for build dir: $BuildDir"
}

$buildTargets = @("llama-server-bridge")
if ($BuildLlamaServerCli) {
    $buildTargets += "llama-server"
}
if ($BuildRealtimeSmoke) {
    $buildTargets += "llama-realtime-smoke"
}
if ($BuildRealtimeBridgeSmoke) {
    $buildTargets += "llama-server-bridge-realtime-smoke"
}
Invoke-CmakeBuildTargets -Cmake $CmakeExe -BuildDirPath $BuildDir -ConfigName $Config -Targets $buildTargets -ParallelJobs $Jobs

Write-Host "Bridge build completed."
Write-Host "CMake build dir: $BuildDir"
Write-Host "Runtime output dir (expected): $(Join-Path $BuildDir 'bin')"
Write-Host "HTTPS backend: $HttpsBackend"

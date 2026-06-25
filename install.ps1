<#
.SYNOPSIS
Installs a native Windows ggrun app home from a release bundle.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\install.ps1

.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Backend cuda

.EXAMPLE
iwr -useb https://raw.githubusercontent.com/raketenkater/ggrun/main/install.ps1 | iex
#>
[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:USERPROFILE 'ggrun'),
    [string]$Release = 'latest',
    [ValidateSet('cpu', 'cuda')]
    [string]$Backend = 'cpu',
    [string]$ReleaseDir = '',
    [string]$LlamaCppRepo = 'https://github.com/ggml-org/llama.cpp.git',
    [string]$LlamaCppRef = 'master',
    [string]$CudaArchitectures = '',
    [switch]$NoPath,
    [switch]$AssumeYes
)

$ErrorActionPreference = 'Stop'
$Repo = 'raketenkater/ggrun'
$Asset = "ggrun-windows-x86_64-$Backend.zip"
$CpuAsset = 'ggrun-windows-x86_64-cpu.zip'
$ReleaseInfo = $null

function Say($Message) { Write-Host $Message }
function Ok($Message) { Write-Host "  OK $Message" -ForegroundColor Green }
function Warn($Message) { Write-Host "  WARN $Message" -ForegroundColor Yellow }
function Fail($Message) { throw $Message }

function Test-Command($Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Require-Command($Name, $Hint) {
    if (!(Test-Command $Name)) { Fail "$Name was not found. $Hint" }
}

function Confirm-Install([string]$What) {
    # Consent gate before installing third-party software. Non-interactive when
    # -AssumeYes or LLM_INSTALL_NONINTERACTIVE=1 (CI / piped installs).
    if ($AssumeYes -or $env:LLM_INSTALL_NONINTERACTIVE -eq '1') { return $true }
    $reply = Read-Host "Install $What now? [Y/n]"
    return ($reply -eq '' -or $reply -match '^(y|yes)$')
}

function Get-ReleaseInfo {
    if ($script:ReleaseInfo) { return $script:ReleaseInfo }
    try {
        if ($Release -eq 'latest') {
            $script:ReleaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
        } else {
            $script:ReleaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Release"
        }
    } catch {
        Fail "Could not query GitHub release '$Release'. Check internet/proxy access to api.github.com. Details: $($_.Exception.Message)"
    }
    return $script:ReleaseInfo
}

function Get-AssetUrl($Info, [string]$Name) {
    foreach ($item in $Info.assets) {
        if ($item.name -eq $Name) { return $item.browser_download_url }
    }
    return ''
}

function Save-ReleaseAsset([string]$Name, [bool]$Required) {
    $archive = Join-Path $tmp $Name
    if ($ReleaseDir) {
        $localAsset = Join-Path $ReleaseDir $Name
        if (!(Test-Path $localAsset)) {
            if ($Required) { Fail "Release asset not found: $localAsset" }
            return ''
        }
        Copy-Item $localAsset $archive
        $sumsPath = Join-Path $ReleaseDir 'SHA256SUMS'
        if ((Test-Path $sumsPath) -and !(Test-Path (Join-Path $tmp 'SHA256SUMS'))) {
            Copy-Item $sumsPath (Join-Path $tmp 'SHA256SUMS')
        }
    } else {
        $info = Get-ReleaseInfo
        $url = Get-AssetUrl $info $Name
        if (!$url) {
            if ($Required) { Fail "No native Windows release asset found: $Name" }
            return ''
        }
        Say "Downloading: $Name"
        try {
            Invoke-WebRequest -Uri $url -OutFile $archive
        } catch {
            Fail "Could not download $Name. Check internet/proxy access to github.com. Details: $($_.Exception.Message)"
        }
        $sumsUrl = Get-AssetUrl $info 'SHA256SUMS'
        if ($sumsUrl -and !(Test-Path (Join-Path $tmp 'SHA256SUMS'))) {
            try {
                Invoke-WebRequest -Uri $sumsUrl -OutFile (Join-Path $tmp 'SHA256SUMS')
            } catch {
                Fail "Could not download SHA256SUMS; refusing an unverified install. Details: $($_.Exception.Message)"
            }
        }
    }

    $sums = Join-Path $tmp 'SHA256SUMS'
    $allowUnverified = ($env:LLM_INSTALL_ALLOW_UNVERIFIED -eq '1')
    if (Test-Path $sums) {
        $line = Get-Content $sums | Where-Object { $_ -match "\s$([regex]::Escape($Name))$" } | Select-Object -First 1
        if ($line) {
            $expected = ($line -split '\s+')[0].ToLowerInvariant()
            $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
            if ($actual -ne $expected) { Fail "Checksum mismatch for $Name" }
            Ok "Verified checksum for $Name"
        } elseif ($allowUnverified) {
            Warn "SHA256SUMS did not include $Name; LLM_INSTALL_ALLOW_UNVERIFIED=1 set — continuing"
        } else {
            Fail "SHA256SUMS did not include $Name; refusing to install unverified. Set LLM_INSTALL_ALLOW_UNVERIFIED=1 to override."
        }
    } elseif ($allowUnverified) {
        Warn "No SHA256SUMS found; LLM_INSTALL_ALLOW_UNVERIFIED=1 set — installing UNVERIFIED bundle"
    } else {
        Fail "No SHA256SUMS found; refusing to install an unverified bundle. Set LLM_INSTALL_ALLOW_UNVERIFIED=1 to override."
    }
    return $archive
}

function Install-ReleaseBundle([string]$Name, [bool]$Required) {
    $archive = Save-ReleaseAsset $Name $Required
    if (!$archive) { return $false }

    $payload = Join-Path $tmp ("payload-" + [IO.Path]::GetFileNameWithoutExtension($Name))
    try {
        Expand-Archive -Path $archive -DestinationPath $payload -Force
    } catch {
        Fail "Could not unpack release archive $Name. Delete the downloaded file and retry. Details: $($_.Exception.Message)"
    }
    $root = Get-ChildItem -Path $payload -Directory | Select-Object -First 1
    if (!$root) { Fail 'Release archive did not contain a payload directory' }

    Copy-Item -Path (Join-Path $root.FullName 'bin\*') -Destination $bin -Recurse -Force
    foreach ($file in @('LICENSE', 'README.md', 'CHANGELOG.md')) {
        $src = Join-Path $root.FullName $file
        if (Test-Path $src) { Copy-Item $src (Join-Path $InstallDir $file) -Force }
    }
    return $true
}

function Write-CmdWrapper([string]$Path, [string]$ArgsLine) {
    $content = @"
@echo off
set "LLM_APP_HOME=%~dp0"
set "PATH=%~dp0.bin;%PATH%"
"%~dp0.bin\ggrun.exe" $ArgsLine %*
"@
    Set-Content -Path $Path -Value $content -Encoding ASCII
}

function Build-CudaBackend {
    Require-Command 'git' 'Install Git for Windows and run this installer again.'
    Require-Command 'cmake' 'Install CMake or Visual Studio 2022 with C++ CMake tools.'
    Require-Command 'nvcc' 'Install the NVIDIA CUDA Toolkit and make sure nvcc is on PATH.'
    if (!(Test-Command 'nvidia-smi')) {
        Warn 'nvidia-smi was not found. Install or repair the NVIDIA driver before launching CUDA models.'
    }

    $srcRoot = Join-Path $InstallDir '.src'
    $repoDir = Join-Path $srcRoot 'llama.cpp'
    $buildDir = Join-Path $repoDir 'build-ggrun-cuda'
    New-Item -ItemType Directory -Force -Path $srcRoot | Out-Null

    if (Test-Path (Join-Path $repoDir '.git')) {
        Say "Updating llama.cpp source: $repoDir"
        & git -C $repoDir fetch --depth 1 origin $LlamaCppRef
        if ($LASTEXITCODE -ne 0) { Fail 'git fetch failed for llama.cpp' }
        & git -C $repoDir checkout FETCH_HEAD
        if ($LASTEXITCODE -ne 0) { Fail 'git checkout failed for llama.cpp' }
    } else {
        Say "Cloning llama.cpp: $LlamaCppRepo ($LlamaCppRef)"
        & git clone --depth 1 --branch $LlamaCppRef $LlamaCppRepo $repoDir
        if ($LASTEXITCODE -ne 0) { Fail 'git clone failed for llama.cpp' }
    }

    $cmakeArgs = @(
        '-S', $repoDir,
        '-B', $buildDir,
        '-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_CUDA=ON',
        '-DGGML_NATIVE=OFF',
        '-DBUILD_SHARED_LIBS=OFF'
    )
    if ($CudaArchitectures) { $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" }

    Say 'Configuring llama.cpp CUDA backend'
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        Fail 'CMake could not configure the CUDA backend. Install Visual Studio 2022 Build Tools with the Desktop development with C++ workload, CMake tools, and a Windows SDK; then rerun.'
    }

    Say 'Building llama-server CUDA backend'
    & cmake --build $buildDir --config Release --target llama-server --parallel
    if ($LASTEXITCODE -ne 0) {
        Fail 'The CUDA backend build failed. Review the CMake output above; verify that the NVIDIA CUDA Toolkit and Visual Studio C++ toolset are compatible.'
    }

    $server = Get-ChildItem -Path $buildDir -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
    if (!$server) { Fail 'llama-server.exe was not produced by the CUDA build' }
    Copy-Item $server.FullName (Join-Path $bin 'llama-server.exe') -Force
    $serverDir = Split-Path -Parent $server.FullName
    Get-ChildItem -Path $serverDir -Filter '*.dll' -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $bin $_.Name) -Force
    }
    Ok "Built native Windows CUDA backend from $($server.FullName)"
}

function Install-PrebuiltCudaBackend([string]$BinDir) {
    # Fetch upstream prebuilt llama.cpp CUDA binaries (server + cudart) so a GPU
    # backend works with no CUDA Toolkit / MSVC / CMake — the from-source build
    # (Build-CudaBackend) stays as a fallback.
    Say 'Fetching prebuilt llama.cpp CUDA backend from ggml-org/llama.cpp releases...'
    try {
        $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers @{ 'User-Agent' = 'ggrun-installer' }
    } catch {
        Warn "Could not query llama.cpp releases: $($_.Exception.Message)"
        return $false
    }
    $assets = $rel.assets
    # Prefer the cuda-12.4 build for broad driver compatibility, else any win-cuda x64.
    $server = $assets | Where-Object { $_.name -match 'bin-win-cuda-12\.4-x64\.zip$' } | Select-Object -First 1
    if (-not $server) { $server = $assets | Where-Object { $_.name -match 'bin-win-cuda-.*-x64\.zip$' } | Select-Object -First 1 }
    if (-not $server) { Warn 'No prebuilt win-cuda asset in the latest llama.cpp release.'; return $false }
    # Match the cudart redistributable to the server asset's CUDA version.
    $cudaVer = if ($server.name -match 'cuda-([0-9.]+)-x64') { $Matches[1] } else { '' }
    $cudart = $assets | Where-Object { $_.name -match "cudart-.*win-cuda-$([regex]::Escape($cudaVer))-x64\.zip$" } | Select-Object -First 1
    if (-not $cudart) { $cudart = $assets | Where-Object { $_.name -match 'cudart-.*win-cuda-.*-x64\.zip$' } | Select-Object -First 1 }
    try {
        foreach ($a in @($server, $cudart)) {
            if (-not $a) { continue }
            $zip = Join-Path $tmp $a.name
            Say "  downloading $($a.name)"
            Invoke-WebRequest -Uri $a.browser_download_url -OutFile $zip
            Expand-Archive -Path $zip -DestinationPath $BinDir -Force
        }
    } catch {
        Warn "Prebuilt CUDA download/extract failed: $($_.Exception.Message)"
        return $false
    }
    if (!(Test-Path (Join-Path $BinDir 'llama-server.exe'))) {
        Warn 'Prebuilt archive did not contain llama-server.exe.'
        return $false
    }
    Ok "Installed prebuilt CUDA backend (llama.cpp $($rel.tag_name), CUDA $cudaVer)"
    return $true
}

function Resolve-Python {
    # Returns @{ Exe; Pre } for a verified Python 3, or $null. Checking the
    # interpreter avoids mistaking the Microsoft Store execution alias for an
    # installed Python. `py` needs -3 to select Python 3 explicitly.
    #
    # Order MUST match the runtime resolver (go/pkg/download.pythonCommand:
    # python3 -> python -> py) so the deps we pip-install land in the very
    # interpreter ggrun later runs download_any_gguf.py with. A mismatch silently
    # installs huggingface_hub into a different interpreter than the one used at
    # runtime -> ModuleNotFoundError despite a "successful" install.
    $candidates = @(
        @{ Exe = 'python3'; Pre = @() },
        @{ Exe = 'python'; Pre = @() },
        @{ Exe = 'py'; Pre = @('-3') }
    )
    foreach ($candidate in $candidates) {
        if (!(Test-Command $candidate.Exe)) { continue }
        try {
            & $candidate.Exe @($candidate.Pre) -c 'import sys; raise SystemExit(0 if sys.version_info.major == 3 else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    return $null
}

function Test-PythonDownloadDeps($Py) {
    try {
        & $Py.Exe @($Py.Pre) -c 'import huggingface_hub, tqdm' 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Ensure-Python {
    # Model downloads (recommender) shell out to a Python helper that uses
    # huggingface_hub. ggrun runs local models without Python, so this is
    # best-effort and never fatal.
    $py = Resolve-Python
    if (!$py -and (Test-Command 'winget') -and (Confirm-Install 'Python 3.12 via winget (needed for model downloads)')) {
        Say 'Installing Python 3 via winget...'
        try { & winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements --silent } catch { }
        # winget updates the persistent PATH, not the current PowerShell
        # process. Refresh it before looking for the newly installed runtime.
        $env:Path = @(
            [Environment]::GetEnvironmentVariable('Path', 'Machine'),
            [Environment]::GetEnvironmentVariable('Path', 'User'),
            $env:Path
        ) -join ';'
        $py = Resolve-Python
    }
    if (!$py) {
        Warn 'Python 3 not found (needed to download models via the recommender).'
        Warn 'Install from https://www.python.org/downloads/ (tick "Add python.exe to PATH"), then run:'
        Warn '  python -m pip install --user huggingface_hub tqdm'
        return
    }
    if (Test-PythonDownloadDeps $py) {
        Ok 'Python download dependencies already installed'
        return
    }
    Say 'Installing Python download dependencies (huggingface_hub, tqdm)...'
    try {
        & $py.Exe @($py.Pre) -m pip install --user --upgrade huggingface_hub tqdm
        if ($LASTEXITCODE -eq 0 -and (Test-PythonDownloadDeps $py)) {
            Ok 'Python download dependencies ready'
        } else {
            Warn 'Could not install huggingface_hub/tqdm. Run: python -m pip install --user huggingface_hub tqdm'
        }
    } catch {
        Warn 'Could not install huggingface_hub/tqdm. Run: python -m pip install --user huggingface_hub tqdm'
    }
}

Say '=== ggrun native Windows installer ==='
Say "Install dir: $InstallDir"
Say "Backend:     $Backend"
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($arch -ne 'X64') {
    Fail "This installer currently supports Windows x86_64 only; detected architecture: $arch"
}
if ($Backend -eq 'cuda') {
    Say 'CUDA mode:   native Windows NVIDIA via llama.cpp GGML_CUDA'
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("ggrun-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
try {
    $bin = Join-Path $InstallDir '.bin'
    $models = Join-Path $InstallDir 'models'
    $config = Join-Path $InstallDir '.config'
    $cache = Join-Path $InstallDir '.cache'
    $logs = Join-Path $InstallDir '.logs'
    New-Item -ItemType Directory -Force -Path $bin, $models, $config, $cache, $logs | Out-Null

    if ($Backend -eq 'cuda') {
        $installed = Install-ReleaseBundle $Asset $false
        if (!$installed) {
            Say 'Installing the CPU Windows bundle for the launcher, then adding a CUDA backend.'
            [void](Install-ReleaseBundle $CpuAsset $true)
            $cudaOk = $false
            if (Confirm-Install 'prebuilt llama.cpp CUDA backend from ggml-org (no toolchain needed)') {
                $cudaOk = Install-PrebuiltCudaBackend $bin
            }
            if (!$cudaOk) {
                Warn 'Falling back to building llama.cpp CUDA from source (needs git + cmake + nvcc + MSVC).'
                Build-CudaBackend
            }
        }
    } else {
        [void](Install-ReleaseBundle $Asset $true)
    }

    $cfgPath = Join-Path $config 'config'
    $llamaServer = Join-Path $bin 'llama-server.exe'
    $cfg = @(
        '# ggrun Go config. Loaded when LLM_APP_HOME points at this app home.',
        "LLM_APP_HOME=`"$InstallDir`"",
        "LLM_MODEL_DIR=`"$models`"",
        "LLM_CACHE_DIR=`"$cache`"",
        "LLM_LOG_DIR=`"$logs`"",
        'LLM_BACKEND="llama"',
        "LLAMA_SERVER=`"$llamaServer`""
    )
    Set-Content -Path $cfgPath -Value $cfg -Encoding UTF8

    Write-CmdWrapper (Join-Path $InstallDir 'ggrun.cmd') ''

    if (!$NoPath) {
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $parts = @($userPath -split ';' | Where-Object { $_ })
        if ($parts -notcontains $InstallDir) {
            [Environment]::SetEnvironmentVariable('Path', (($parts + $InstallDir) -join ';'), 'User')
            Warn "Added $InstallDir to the user PATH. Open a new terminal to use ggrun.cmd directly."
        }
    }

    Ensure-Python

    if (!(Test-Path (Join-Path $bin 'ggrun.exe'))) { Fail 'ggrun.exe was not installed' }
    if (!(Test-Path $llamaServer)) { Fail 'llama-server.exe was not installed' }

    $oldAppHome = $env:LLM_APP_HOME
    try {
        $env:LLM_APP_HOME = $InstallDir
        & (Join-Path $bin 'ggrun.exe') version | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail 'Installed ggrun failed its version check' }
        & (Join-Path $bin 'ggrun.exe') detect | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail 'Installed ggrun failed hardware detection' }
        & $llamaServer --version | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Fail 'Installed llama-server could not start. The bundle may be incompatible or a required Visual C++ runtime/DLL may be missing.'
        }
    } finally {
        $env:LLM_APP_HOME = $oldAppHome
    }

    Ok 'Installed native Windows ggrun'
    Ok 'CLI, hardware detection, and backend startup checks passed'
    Say "CLI/GUI: $InstallDir\ggrun.cmd   (no arguments opens the GUI)"
    Say "Models:  $models"
    Say "Config:  $cfgPath"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

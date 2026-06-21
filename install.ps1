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
    [switch]$NoPath
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

function Get-ReleaseInfo {
    if ($script:ReleaseInfo) { return $script:ReleaseInfo }
    if ($Release -eq 'latest') {
        $script:ReleaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
    } else {
        $script:ReleaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$Release"
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
        Invoke-WebRequest -Uri $url -OutFile $archive
        $sumsUrl = Get-AssetUrl $info 'SHA256SUMS'
        if ($sumsUrl -and !(Test-Path (Join-Path $tmp 'SHA256SUMS'))) {
            Invoke-WebRequest -Uri $sumsUrl -OutFile (Join-Path $tmp 'SHA256SUMS')
        }
    }

    $sums = Join-Path $tmp 'SHA256SUMS'
    if (Test-Path $sums) {
        $line = Get-Content $sums | Where-Object { $_ -match "\s$([regex]::Escape($Name))$" } | Select-Object -First 1
        if ($line) {
            $expected = ($line -split '\s+')[0].ToLowerInvariant()
            $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
            if ($actual -ne $expected) { Fail "Checksum mismatch for $Name" }
            Ok "Verified checksum for $Name"
        } else {
            Warn "SHA256SUMS did not include $Name"
        }
    }
    return $archive
}

function Install-ReleaseBundle([string]$Name, [bool]$Required) {
    $archive = Save-ReleaseAsset $Name $Required
    if (!$archive) { return $false }

    $payload = Join-Path $tmp ("payload-" + [IO.Path]::GetFileNameWithoutExtension($Name))
    Expand-Archive -Path $archive -DestinationPath $payload -Force
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
    if ($LASTEXITCODE -ne 0) { Fail 'cmake configure failed for llama.cpp CUDA backend' }

    Say 'Building llama-server CUDA backend'
    & cmake --build $buildDir --config Release --target llama-server --parallel
    if ($LASTEXITCODE -ne 0) { Fail 'cmake build failed for llama.cpp CUDA backend' }

    $server = Get-ChildItem -Path $buildDir -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
    if (!$server) { Fail 'llama-server.exe was not produced by the CUDA build' }
    Copy-Item $server.FullName (Join-Path $bin 'llama-server.exe') -Force
    $serverDir = Split-Path -Parent $server.FullName
    Get-ChildItem -Path $serverDir -Filter '*.dll' -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $bin $_.Name) -Force
    }
    Ok "Built native Windows CUDA backend from $($server.FullName)"
}

Say '=== ggrun native Windows installer ==='
Say "Install dir: $InstallDir"
Say "Backend:     $Backend"
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
            Warn "No native Windows CUDA release asset found: $Asset"
            Say 'Installing the CPU Windows bundle for the launcher, then building llama.cpp CUDA locally.'
            [void](Install-ReleaseBundle $CpuAsset $true)
            Build-CudaBackend
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

    if (!(Test-Path (Join-Path $bin 'ggrun.exe'))) { Fail 'ggrun.exe was not installed' }
    if (!(Test-Path $llamaServer)) { Fail 'llama-server.exe was not installed' }

    Ok 'Installed native Windows ggrun'
    Say "CLI/GUI: $InstallDir\ggrun.cmd   (no arguments opens the GUI)"
    Say "Models:  $models"
    Say "Config:  $cfgPath"
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

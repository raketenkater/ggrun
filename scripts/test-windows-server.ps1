<#
.SYNOPSIS
Install into an isolated app home, download/serve a smoke model, reinstall and
relaunch, then serve/relaunch an optional large model. Retain all evidence.
.EXAMPLE
.\scripts\test-windows-server.ps1 -ModelPath D:\models\large.gguf -MinWeightDevices 2
#>
[CmdletBinding()]
param(
    [string]$WorkDir = (Join-Path $env:TEMP ("ggrun-server-check-" + [guid]::NewGuid().ToString('N'))),
    [string]$Release = 'latest',
    [string]$ReleaseDir = '',
    [ValidateSet('cpu', 'cuda')][string]$Backend = 'cuda',
    [string]$ModelPath = '',
    [string]$ModelRepo = '',
    [string]$Quant = 'Q4_K_M',
    [string]$SmokeRepo = 'ggml-org/Qwen3.5-0.8B-GGUF',
    [string]$SmokeQuant = 'Q4_0',
    [ValidateRange(128, 4194304)][int]$Context = 32768,
    [ValidateRange(1, 86400)][int]$StartupTimeout = 2400,
    [ValidateRange(1, 86400)][int]$RequestTimeout = 300,
    [ValidateRange(0, 64)][int]$MinWeightDevices = 1,
    [ValidateRange(1024, 65535)][int]$Port = 18855
)
$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Run this script on native Windows.' }
if ($ModelPath -and $ModelRepo) { throw 'Choose ModelPath or ModelRepo, not both.' }
if ($Backend -eq 'cpu' -and $MinWeightDevices -ne 0) { throw 'CPU checks require -MinWeightDevices 0.' }
if (Test-Path $WorkDir) { throw 'WorkDir must not exist; preserve previous evidence.' }
if ($ModelPath) { $ModelPath = (Resolve-Path $ModelPath).Path }
if ($ReleaseDir) { $ReleaseDir = (Resolve-Path $ReleaseDir).Path }
New-Item -ItemType Directory -Path $WorkDir | Out-Null
$WorkDir = (Resolve-Path $WorkDir).Path
$App = Join-Path $WorkDir 'app'
$Evidence = Join-Path $WorkDir 'evidence'
New-Item -ItemType Directory -Path $Evidence | Out-Null
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Installer = Join-Path $RepoRoot 'install.ps1'
$Checker = Join-Path $PSScriptRoot 'verify-gpu-install.py'
$Summary = [ordered]@{ passed = $false; backend = $Backend; release = $Release; releaseDir = $ReleaseDir; app = $App; stages = @() }
$SavedHome = $env:LLM_APP_HOME
$SavedNonInteractive = $env:LLM_INSTALL_NONINTERACTIVE
$SavedModelDir = $env:LLM_MODEL_DIR

function Invoke-NativeCheck([string]$Program, [string[]]$Arguments, [string]$Log) {
    & $Program @Arguments > $Log
    if ($LASTEXITCODE -ne 0) { throw "$Program failed ($LASTEXITCODE); see $Log" }
}
function Install-App([string]$Stage) {
    $params = @{ InstallDir = $App; Release = $Release; Backend = $Backend; NoPath = $true; AssumeYes = $true }
    if ($ReleaseDir) { $params.ReleaseDir = $ReleaseDir }
    & $Installer @params *> (Join-Path $Evidence "$Stage.log")
    $Summary.stages += $Stage
}
function Run-Serving([string]$Stage, [string[]]$SourceArgs, [int]$Ctx, [int]$Devices) {
    $directory = Join-Path $Evidence $Stage
    $arguments = @($Checker, '--launcher', $Launcher, '--output', $directory,
        '--ctx', "$Ctx", '--timeout', "$StartupTimeout", '--request-timeout', "$RequestTimeout",
        '--min-weight-devices', "$Devices", '--port', "$Port") + $SourceArgs
    & $Python @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Stage failed; see $directory" }
    $Summary.stages += $Stage
    return (Get-Content (Join-Path $directory 'selected-model.json') -Raw | ConvertFrom-Json).model
}

try {
    $env:LLM_APP_HOME = $App
    $env:LLM_INSTALL_NONINTERACTIVE = '1'
    $env:LLM_MODEL_DIR = Join-Path $App 'models'
    Install-App 'install'
    $Launcher = Join-Path $App 'ggrun.cmd'
    $Python = (Get-Command python -ErrorAction Stop).Source
    Invoke-NativeCheck $Python @('--version') (Join-Path $Evidence 'python-version.txt')
    Invoke-NativeCheck $Launcher @('--version') (Join-Path $Evidence 'launcher-version.txt')
    Invoke-NativeCheck $Launcher @('detect') (Join-Path $Evidence 'detect.txt')
    $Servers = @(Get-ChildItem (Join-Path $App '.bin') -Filter 'llama-server*.exe')
    if (-not $Servers) { throw 'No installed llama-server binary.' }
    foreach ($server in $Servers) {
        Invoke-NativeCheck $server.FullName @('--version') (Join-Path $Evidence ($server.Name + '-version.txt'))
    }
    if ($Backend -eq 'cuda') {
        Invoke-NativeCheck 'nvidia-smi' @('-q') (Join-Path $Evidence 'gpu-before.txt')
    }
    $SmokeDevices = 0
    if ($Backend -eq 'cuda') { $SmokeDevices = 1 }
    $smoke = Run-Serving 'smoke-download' @('--repo', $SmokeRepo, '--quant', $SmokeQuant) 2048 $SmokeDevices
    $config = Join-Path $App '.config/config'
    Add-Content $config "`nLLM_PORT=19249"
    $before = (Get-FileHash $config).Hash
    Install-App 'reinstall'
    if ((Get-FileHash $config).Hash -ne $before) { throw 'Reinstall changed user configuration.' }
    $null = Run-Serving 'smoke-relaunch' @('--model', $smoke) 2048 $SmokeDevices
    if ($ModelPath -or $ModelRepo) {
        $source = @('--model', $ModelPath)
        if ($ModelRepo) { $source = @('--repo', $ModelRepo, '--quant', $Quant) }
        $large = Run-Serving 'large-first' $source $Context $MinWeightDevices
        $null = Run-Serving 'large-relaunch' @('--model', $large) $Context $MinWeightDevices
    }
    if ($Backend -eq 'cuda') {
        Invoke-NativeCheck 'nvidia-smi' @('-q') (Join-Path $Evidence 'gpu-after.txt')
    }
    $Summary.passed = $true
} catch {
    $Summary.error = $_.Exception.Message
    throw
} finally {
    $Summary | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $Evidence 'summary.json')
    if (Test-Path (Join-Path $App '.logs')) {
        Copy-Item (Join-Path $App '.logs') (Join-Path $Evidence 'app-logs') -Recurse -Force
    }
    $env:LLM_APP_HOME = $SavedHome
    $env:LLM_INSTALL_NONINTERACTIVE = $SavedNonInteractive
    $env:LLM_MODEL_DIR = $SavedModelDir
    Write-Host "Evidence: $Evidence (downloaded weights are retained locally)"
}

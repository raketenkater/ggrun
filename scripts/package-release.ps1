[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$AssetName,
    [Parameter(Mandatory = $true)][string]$ServerBin,
    [string]$OutDir = 'dist',
    [string]$LlmServerBin = ''
)

$ErrorActionPreference = 'Stop'
if (!(Test-Path $ServerBin)) { throw "llama-server binary not found: $ServerBin" }

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir '..')
if (!$LlmServerBin) {
    $candidate = Join-Path $Root 'go\llm-server.exe'
    if (Test-Path $candidate) { $LlmServerBin = $candidate }
}
if (!$LlmServerBin -or !(Test-Path $LlmServerBin)) { throw 'llm-server.exe not found; pass -LlmServerBin' }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$work = Join-Path ([IO.Path]::GetTempPath()) ("llm-server-package-" + [guid]::NewGuid().ToString('N'))
$payloadName = $AssetName -replace '\.zip$', ''
$payload = Join-Path $work $payloadName
$bin = Join-Path $payload 'bin'

try {
    New-Item -ItemType Directory -Force -Path $bin | Out-Null

    foreach ($file in @('LICENSE', 'README.md', 'CHANGELOG.md', 'install.ps1')) {
        $src = Join-Path $Root $file
        if (Test-Path $src) { Copy-Item $src (Join-Path $payload $file) -Force }
    }

    Copy-Item $LlmServerBin (Join-Path $bin 'llm-server.exe') -Force
    Copy-Item $ServerBin (Join-Path $bin 'llama-server.exe') -Force

    foreach ($spec in @(
        @('tools\gguf\parse_gguf.py', 'parse_gguf.py'),
        @('tools\models\model_index.py', 'model_index.py'),
        @('tools\download\download_any_gguf.py', 'download_any_gguf.py')
    )) {
        $src = Join-Path $Root $spec[0]
        if (Test-Path $src) { Copy-Item $src (Join-Path $bin $spec[1]) -Force }
    }

    $serverDir = Split-Path -Parent (Resolve-Path $ServerBin)
    Get-ChildItem -Path $serverDir -Filter '*.dll' -File -ErrorAction SilentlyContinue | ForEach-Object {
        Copy-Item $_.FullName (Join-Path $bin $_.Name) -Force
    }

    Set-Content -Path (Join-Path $payload 'llm-server.cmd') -Encoding ASCII -Value @'
@echo off
set "LLM_APP_HOME=%~dp0"
set "PATH=%~dp0bin;%PATH%"
"%~dp0bin\llm-server.exe" %*
'@
    Set-Content -Path (Join-Path $payload 'llm-server-gui.cmd') -Encoding ASCII -Value @'
@echo off
set "LLM_APP_HOME=%~dp0"
set "PATH=%~dp0bin;%PATH%"
"%~dp0bin\llm-server.exe" gui %*
'@
    Set-Content -Path (Join-Path $bin 'llm-server-gui.cmd') -Encoding ASCII -Value @'
@echo off
"%~dp0llm-server.exe" gui %*
'@

    $zip = Join-Path $OutDir $AssetName
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $payload -DestinationPath $zip -Force
    Write-Output $zip
} finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

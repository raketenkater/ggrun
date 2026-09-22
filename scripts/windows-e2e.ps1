<#
.SYNOPSIS
Run ggrun's Windows end-to-end check on THIS machine, from any branch.

.DESCRIPTION
`scripts/test-windows-server.ps1` already does the real work: it installs into an
isolated app home, downloads and serves a smoke model, reinstalls, relaunches,
and (with -ModelPath) serves a large model while asserting real device
allocations. What it could not do is run against a BRANCH, and it only ever
tested the published release.

This wraps it, so you can validate a branch on real Windows hardware:

  1. authenticate to GitHub through `gh` (no token is ever written by this
     script or printed),
  2. fetch the branch into a scratch checkout, leaving your working tree alone,
  3. build the branch's launcher and package it with a real published backend,
  4. run test-windows-server.ps1 against THAT candidate,
  5. keep the evidence and print where it is.

Why this exists at all: no GitHub-hosted runner has a GPU, so every hosted job
proves the CPU path only. Real serving across real, mismatched cards is the one
thing CI cannot measure, and it is the thing ggrun is actually about.

.EXAMPLE
# Validate the current branch on the cards in this box.
.\scripts\windows-e2e.ps1 -Ref my-branch -ModelPath D:\models\GLM-5.3-Flash.gguf -MinWeightDevices 2

.EXAMPLE
# CPU-only smoke, fastest useful signal.
.\scripts\windows-e2e.ps1 -Backend cpu -MinWeightDevices 0

.EXAMPLE
# Agent-shaped workload, to compare against the Linux numbers under one run-id.
.\scripts\windows-e2e.ps1 -Ref my-branch -ModelPath D:\models\m.gguf -AgentLanes 3 -AgentRepeats 5 -PrefixReuse
#>
[CmdletBinding()]
param(
    [string]$Ref = 'main',
    [string]$Repo = 'raketenkater/ggrun',
    [ValidateSet('cpu', 'cuda')][string]$Backend = 'cuda',
    [string]$ModelPath = '',
    [string]$ModelRepo = '',
    [string]$Quant = 'Q4_K_M',
    [ValidateRange(128, 4194304)][int]$Context = 32768,
    [ValidateRange(0, 64)][int]$MinWeightDevices = 2,
    [ValidateRange(1, 86400)][int]$StartupTimeout = 2400,
    # Agent-shaped load. All default off: a bare serve is the cheap check, and
    # the agent lanes are what produce comparable numbers across machines.
    [ValidateRange(0, 32)][int]$AgentLanes = 0,
    [ValidateRange(1, 50)][int]$AgentRepeats = 3,
    [switch]$PrefixReuse,
    # Reuse a checkout you already have instead of fetching a scratch one. The
    # script still only ever reads it and builds in place.
    [string]$CheckoutDir = '',
    [string]$WorkRoot = $(Join-Path $env:TEMP 'ggrun-windows-e2e'),
    [switch]$KeepScratch
)

$ErrorActionPreference = 'Stop'
if ($env:OS -ne 'Windows_NT') { throw 'Run this script on native Windows.' }
if ($ModelPath -and $ModelRepo) { throw 'Choose ModelPath or ModelRepo, not both.' }
if ($Backend -eq 'cpu' -and $MinWeightDevices -ne 0) {
    throw 'CPU checks require -MinWeightDevices 0 (there is no device to place on).'
}
if ($Backend -eq 'cuda' -and $MinWeightDevices -lt 1) {
    throw 'A CUDA check must assert at least one device allocation; use -MinWeightDevices >= 1.'
}
# Keep this file pure ASCII. Windows PowerShell 5.1 reads a BOM-less script as the
# ANSI code page, and a smart quote or NBSP introduced by a non-ASCII glyph turns
# into a string delimiter that silently swallows the code between it and the next
# quote. install.ps1 lost five function definitions that way on 2026-09-22. The
# ci.yml guard covers install.ps1; the rule is worth following everywhere.

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required. $Hint"
    }
}

function Get-GhLogin {
    # `gh auth login` stores an OAuth token in the OS credential store. This
    # reads the login NAME only, for messages; the token itself is never
    # retrieved, written, or printed by this script.
    $status = gh auth status 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { return '' }
    $m = [regex]::Match($status, 'Logged in to \S+ account (\S+)')
    if ($m.Success) { return $m.Groups[1].Value }
    return 'unknown'
}

function Get-FreeVramMb {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return 0 }
    $total = 0
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits |
        ForEach-Object {
            $p = $_ -split ',\s*'
            $total += [int]$p[0] - [int]$p[1]
        }
    return $total
}

function Get-DeviceCount {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return 0 }
    return @(nvidia-smi --query-gpu=name --format=csv,noheader).Count
}

function Get-Checkout {
    if ($CheckoutDir) {
        if (-not (Test-Path (Join-Path $CheckoutDir 'install.ps1'))) {
            throw "$CheckoutDir does not look like a ggrun checkout."
        }
        return (Resolve-Path $CheckoutDir).Path
    }
    New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
    $dir = Join-Path $WorkRoot 'src'
    if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
    Write-Host "==> fetching $Repo@$Ref"
    # Full clone: a shallow one cannot check out an arbitrary branch by name.
    git clone --quiet "https://github.com/$Repo.git" $dir
    if ($LASTEXITCODE -ne 0) { throw "clone failed; is $Ref pushed?" }
    git -C $dir checkout --quiet $Ref
    if ($LASTEXITCODE -ne 0) { throw "checkout of $Ref failed" }
    $sha = (git -C $dir rev-parse HEAD).Trim()
    Write-Host "    $Ref = $sha"
    return $dir
}

function Get-PublishedBackend([string]$AssetName, [string]$Dest) {
    # The CI jobs pin a published release for the backend rather than building
    # llama.cpp, and so does this: a backend build is slow and is not what is
    # under test here.
    $zip = Join-Path $Dest $AssetName
    if (Test-Path $zip) { return (Get-ChildItem $Dest -Recurse -Filter 'llama-server.exe' | Select-Object -First 1).FullName }
    Write-Host "==> downloading published backend $AssetName"
    $rel = (gh api "repos/$Repo/releases/latest" | ConvertFrom-Json)
    $asset = $rel.assets | Where-Object { $_.name -eq $AssetName } | Select-Object -First 1
    if (-not $asset) { throw "release $($rel.tag_name) has no asset $AssetName" }
    Invoke-WebRequest -UseBasicParsing $asset.browser_download_url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath (Join-Path $Dest 'backend') -Force
    $server = Get-ChildItem (Join-Path $Dest 'backend') -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
    if (-not $server) { throw "$AssetName contained no llama-server.exe" }
    return $server.FullName
}

# ---------------------------------------------------------------- preflight
Write-Host '=== ggrun Windows end-to-end ==='
Require-Command gh 'Install from https://cli.github.com, then run: gh auth login'
Require-Command git 'Install Git for Windows.'
Require-Command python 'The serving checker is a Python script.'

$login = Get-GhLogin
if (-not $login) { throw 'gh is not authenticated. Run: gh auth login' }
Write-Host "authenticated as: $login"

if ($Backend -eq 'cuda') {
    Require-Command nvidia-smi 'CUDA checks need the NVIDIA driver.'
    $devices = Get-DeviceCount
    $free = Get-FreeVramMb
    Write-Host "devices: $devices    free VRAM: $free MiB"
    if ($devices -lt $MinWeightDevices) {
        throw "asked to assert $MinWeightDevices device(s) but nvidia-smi reports $devices."
    }
    if ($free -lt 3000) {
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
        throw "only $free MiB free across all cards; stop a loaded model first."
    }
}

if (-not (Get-Command go -ErrorAction SilentlyContinue) -and -not $CheckoutDir) {
    throw 'Building the branch launcher needs Go, or pass -CheckoutDir with a launcher already built.'
}

$evidence = Join-Path $WorkRoot ("evidence-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$stages = [ordered]@{ passed = $false; ref = $Ref; backend = $Backend }
$started = Get-Date
$sha = ''

try {
    # ------------------------------------------------------------- candidate
    $src = Get-Checkout
    $sha = (git -C $src rev-parse HEAD).Trim()
    $stages.sha = $sha

    Write-Host '==> building the branch launcher'
    Push-Location $src
    try {
        go build -C go -trimpath -o ggrun.exe ./cmd/ggrun
        if ($LASTEXITCODE -ne 0) { throw 'go build failed' }
    } finally { Pop-Location }

    $candidate = Join-Path $WorkRoot 'candidate'
    New-Item -ItemType Directory -Force -Path $candidate | Out-Null
    # Always the CPU asset, whatever $Backend says: there is no published
    # ggrun-windows-x86_64-cuda.zip. install.ps1 installs the CPU bundle
    # unconditionally and then, when -Backend cuda, adds a prebuilt llama.cpp
    # CUDA backend (Build-CudaBackend is only a fallback). The official
    # gpu-windows job packages the CPU asset for the same reason.
    $assetName = 'ggrun-windows-x86_64-cpu.zip'
    $serverExe = Get-PublishedBackend $assetName $WorkRoot
    Write-Host "==> packaging candidate (install backend: $Backend) with $([IO.Path]::GetFileName($serverExe))"
    & (Join-Path $src 'scripts\package-release.ps1') `
        -AssetName $assetName `
        -ServerBin $serverExe `
        -OutDir $candidate `
        -LlmServerBin (Join-Path $src 'go\ggrun.exe') | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'packaging failed' }

    # ---------------------------------------------------------------- e2e
    # test-windows-server.ps1 refuses an existing WorkDir so previous evidence
    # survives, so give each run its own.
    $check = Join-Path $WorkRoot ("server-check-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    # Not $args: that is a reserved automatic variable.
    $checkArgs = @{
        WorkDir          = $check
        ReleaseDir       = $candidate
        Backend          = $Backend
        Context          = $Context
        StartupTimeout   = $StartupTimeout
        MinWeightDevices = $MinWeightDevices
    }
    if ($ModelPath) { $checkArgs.ModelPath = $ModelPath }
    if ($ModelRepo) { $checkArgs.ModelRepo = $ModelRepo; $checkArgs.Quant = $Quant }

    Write-Host "==> running the install/serve/reinstall/relaunch check"
    & (Join-Path $src 'scripts\test-windows-server.ps1') @checkArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'the Windows end-to-end check failed' }

    # ----------------------------------------------------------------- data
    # The agent-shaped lane is what makes numbers comparable across machines:
    # same lanes, same repeats, same prefix-reuse setting, so a decode figure
    # from this box can be read against the Linux one.
    if ($AgentLanes -gt 0) {
        $smoke = Join-Path $check 'evidence\smoke-download\selected-model.json'
        $model = $ModelPath
        if (Test-Path $smoke) {
            $model = (Get-Content $smoke -Raw | ConvertFrom-Json).model
        }
        if (-not $model) { throw '-AgentLanes needs a model; pass -ModelPath.' }
        $dataDir = Join-Path $check 'evidence\agent-workload'
        New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
        Write-Host "==> agent-shaped workload: $AgentLanes lanes x $AgentRepeats repeats"
        $vargs = @(
            (Join-Path $src 'scripts\verify-installed-serving.py'),
            '--launcher', (Join-Path $check 'app\ggrun.cmd'),
            '--model', $model, '--output', $dataDir,
            '--ctx', "$Context",
            '--agent-lanes', "$AgentLanes",
            '--agent-repeats', "$AgentRepeats",
            '--min-weight-devices', "$MinWeightDevices",
            '--port', '18888'
        )
        if ($PrefixReuse) { $vargs += '--prefix-reuse' }
        & python @vargs | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'the agent workload failed' }
        $stages.agentWorkload = $dataDir
    }

    $stages.passed = $true
} catch {
    $stages.error = $_.Exception.Message
    throw
} finally {
    $stages.elapsedSeconds = [int]((Get-Date) - $started).TotalSeconds
    New-Item -ItemType Directory -Force -Path $evidence | Out-Null
    $stages | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $evidence 'run.json')
    if (Test-Path $WorkRoot) {
        # Copy the per-stage evidence out of the scratch tree, minus the
        # multi-GB weights, so the result survives cleanup.
        Get-ChildItem $WorkRoot -Directory -Filter 'server-check-*' | ForEach-Object {
            $dst = Join-Path $evidence $_.Name
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            Copy-Item (Join-Path $_.FullName 'evidence') $dst -Recurse -Force `
                -ErrorAction SilentlyContinue
        }
    }
    Write-Host ''
    Write-Host "Result:  $($stages.passed)"
    Write-Host "Ref:     $Ref ($sha)"
    Write-Host "Backend: $Backend"
    Write-Host "Evidence: $evidence"
    if (-not $KeepScratch -and -not $CheckoutDir) {
        Remove-Item (Join-Path $WorkRoot 'src') -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item (Join-Path $WorkRoot 'candidate') -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-Host 'Downloaded weights are retained locally by test-windows-server.ps1.'
}

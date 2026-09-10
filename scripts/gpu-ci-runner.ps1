<#
.SYNOPSIS
Run ggrun's Windows GPU end-to-end job on this machine.

.DESCRIPTION
No GitHub-hosted runner has a GPU, so every hosted job proves the CPU path only.
The install-e2e workflow carries a gpu-windows job for that gap; it needs a
runner on a Windows box with cards.

The runner comes up for one run and goes down again. ggrun is a public
repository, so a permanently-online runner on a personal machine is a standing
target. The workflow itself only accepts a manual dispatch from main.

.EXAMPLE
.\scripts\gpu-ci-runner.ps1 setup
.\scripts\gpu-ci-runner.ps1 run
.\scripts\gpu-ci-runner.ps1 status
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'run', 'status')]
    [string]$Command = 'status',
    [string]$Repo = $(if ($env:GGRUN_CI_REPO) { $env:GGRUN_CI_REPO } else { 'raketenkater/ggrun' }),
    [string]$RunnerDir = $(if ($env:GGRUN_RUNNER_DIR) { $env:GGRUN_RUNNER_DIR } else { Join-Path $env:USERPROFILE 'actions-runner' }),
    [string]$RunnerVersion = '2.337.0',
    [string]$RunnerName = "$env:COMPUTERNAME-gpu",
    # The job downloads a 563 MB Q4_0 model and serves it. Below this there is
    # no point starting: ggrun would refuse to fit or quietly place on the CPU,
    # and the job asserts a real device allocation.
    [int]$MinFreeVramMb = 3000
)

$ErrorActionPreference = 'Stop'
$Workflow = 'install-e2e.yml'

function Require-Command([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is required but not installed"
    }
}

# Windows PowerShell 5.1 does not escape double quotes embedded in arguments to
# native programs, so a --jq filter with a string literal reaches gh split into
# several arguments. It also turns redirected stderr into a terminating error
# under 'Stop'. Fetch the JSON and filter in PowerShell instead; returns $null
# on any API failure.
function Invoke-GhJson([string]$Path) {
    $ErrorActionPreference = 'Continue'
    $raw = gh api $Path 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    $json = ($raw -join "`n") | ConvertFrom-Json
    return $json
}

function Get-FreeVramMb {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { return 0 }
    $total = 0
    nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits |
        ForEach-Object {
            $parts = $_ -split ',\s*'
            $total += [int]$parts[0] - [int]$parts[1]
        }
    return $total
}

function Get-RunnerStatus {
    $resp = Invoke-GhJson "repos/$Repo/actions/runners"
    if (-not $resp) { return '' }
    $match = $resp.runners | Where-Object { $_.name -eq $RunnerName } | Select-Object -First 1
    if ($match) { return $match.status }
    return ''
}

function Invoke-Setup {
    Require-Command gh
    if (Test-Path (Join-Path $RunnerDir 'run.cmd')) {
        Write-Host "runner already configured at $RunnerDir"
        return
    }
    New-Item -ItemType Directory -Force -Path $RunnerDir | Out-Null
    Push-Location $RunnerDir
    try {
        $zip = "actions-runner-win-x64-$RunnerVersion.zip"
        Write-Host "==> downloading runner $RunnerVersion"
        Invoke-WebRequest -UseBasicParsing `
            "https://github.com/actions/runner/releases/download/v$RunnerVersion/$zip" -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath . -Force
        Remove-Item $zip
        Write-Host "==> registering as '$RunnerName' with label 'gpu'"
        # Minted per call and never written to disk or history.
        $token = gh api -X POST "repos/$Repo/actions/runners/registration-token" --jq .token
        if ($LASTEXITCODE -ne 0) { throw 'could not mint a registration token' }
        .\config.cmd --url "https://github.com/$Repo" --token $token `
            --labels gpu --name $RunnerName --unattended --replace
        Write-Host "==> configured. Nothing is running yet; use: .\scripts\gpu-ci-runner.ps1 run"
    } finally {
        Pop-Location
    }
}

# The gpu-windows job is gated on this as well as the runner label, so a
# repository with no Windows GPU machine skips the job instead of queueing one
# that can never be picked up.
function Enable-GpuRunnerVariable {
    $ErrorActionPreference = 'Continue'
    $current = Invoke-GhJson "repos/$Repo/actions/variables/GGRUN_GPU_RUNNER_WINDOWS"
    if ($current -and $current.value -eq 'true') { return }
    Write-Host '==> setting GGRUN_GPU_RUNNER_WINDOWS=true'
    gh api -X PATCH "repos/$Repo/actions/variables/GGRUN_GPU_RUNNER_WINDOWS" -f value=true 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        gh api -X POST "repos/$Repo/actions/variables" -f name=GGRUN_GPU_RUNNER_WINDOWS -f value=true | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'could not set GGRUN_GPU_RUNNER_WINDOWS' }
    }
}

function Invoke-Run {
    Require-Command gh
    Require-Command nvidia-smi
    if (-not (Test-Path (Join-Path $RunnerDir 'run.cmd'))) {
        throw "runner not configured; run: .\scripts\gpu-ci-runner.ps1 setup"
    }

    $free = Get-FreeVramMb
    Write-Host "==> free VRAM across all cards: $free MiB"
    if ($free -lt $MinFreeVramMb) {
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
        throw "only $free MiB free, need $MinFreeVramMb; stop a loaded model first"
    }

    Enable-GpuRunnerVariable

    Write-Host '==> starting runner'
    $runner = Start-Process (Join-Path $RunnerDir 'run.cmd') -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $RunnerDir 'run.log') `
        -RedirectStandardError (Join-Path $RunnerDir 'run.err')
    try {
        $online = $false
        foreach ($i in 1..40) {
            if ((Get-RunnerStatus) -eq 'online') { $online = $true; break }
            Start-Sleep 2
        }
        if (-not $online) {
            Get-Content (Join-Path $RunnerDir 'run.log') -Tail 20 -ErrorAction SilentlyContinue
            throw 'runner did not come online'
        }
        Write-Host '==> runner online'

        # Identify the dispatched run by being new, rather than by a timestamp.
        $before = gh api "repos/$Repo/actions/workflows/$Workflow/runs?per_page=1" --jq '.workflow_runs[0].id // 0'
        Write-Host "==> dispatching $Workflow on main"
        gh workflow run $Workflow --repo $Repo --ref main
        if ($LASTEXITCODE -ne 0) { throw 'dispatch failed' }

        $runId = ''
        foreach ($i in 1..30) {
            $candidate = gh api "repos/$Repo/actions/workflows/$Workflow/runs?per_page=1&event=workflow_dispatch" --jq '.workflow_runs[0].id // 0'
            if ($candidate -and $candidate -ne '0' -and $candidate -ne $before) { $runId = $candidate; break }
            Start-Sleep 2
        }
        if (-not $runId) { throw 'dispatched run did not appear' }
        Write-Host "==> run ${runId}: https://github.com/$Repo/actions/runs/$runId"

        while ($true) {
            $run = Invoke-GhJson "repos/$Repo/actions/runs/$runId"
            if ($run -and $run.status -eq 'completed') { break }
            $progress = Invoke-GhJson "repos/$Repo/actions/runs/$runId/jobs"
            if ($progress) {
                $progress.jobs | Where-Object { $_.name -eq 'gpu-windows' } |
                    ForEach-Object { Write-Host "    gpu-windows: $($_.status) $($_.conclusion)" }
            }
            Start-Sleep 20
        }

        Write-Host '==> finished'
        $final = Invoke-GhJson "repos/$Repo/actions/runs/$runId/jobs"
        $jobs = if ($final) { $final.jobs } else { @() }
        foreach ($job in $jobs) {
            $state = if ($job.conclusion) { $job.conclusion } else { $job.status }
            Write-Host "  $($job.name): $state"
        }
        $gpuJob = $jobs | Where-Object { $_.name -eq 'gpu-windows' } | Select-Object -First 1
        $result = if ($gpuJob) { [string]$gpuJob.conclusion } else { '' }
        switch ($result) {
            'success' { Write-Host '==> the Windows GPU job passed' }
            'skipped' { throw 'the gpu-windows job was SKIPPED: check GGRUN_GPU_RUNNER_WINDOWS and that the workflow is on main' }
            ''        { throw 'no gpu-windows job in this run; is the workflow on main?' }
            default   { throw "the gpu-windows job did not pass: $result" }
        }
    } finally {
        Write-Host '==> stopping runner'
        if ($runner -and -not $runner.HasExited) {
            Stop-Process -Id $runner.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

function Invoke-Status {
    Require-Command gh
    $configured = if (Test-Path (Join-Path $RunnerDir 'run.cmd')) { "yes ($RunnerDir)" } else { 'no' }
    Write-Host "configured locally: $configured"
    Write-Host 'registered runners:'
    $resp = Invoke-GhJson "repos/$Repo/actions/runners"
    if (-not $resp) { Write-Host '  (could not list runners; this needs admin on the repository)' }
    foreach ($r in $resp.runners) {
        $labels = ($r.labels | ForEach-Object { $_.name }) -join ','
        Write-Host "  $($r.name)  $($r.status)  labels=$labels"
    }
    $variable = Invoke-GhJson "repos/$Repo/actions/variables/GGRUN_GPU_RUNNER_WINDOWS"
    $enabled = if ($variable -and $variable.value) { $variable.value } else { '(unset)' }
    Write-Host "GGRUN_GPU_RUNNER_WINDOWS: $enabled"
    Write-Host "free VRAM: $(Get-FreeVramMb) MiB"
}

switch ($Command) {
    'setup'  { Invoke-Setup }
    'run'    { Invoke-Run }
    'status' { Invoke-Status }
}

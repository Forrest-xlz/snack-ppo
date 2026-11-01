param(
    [ValidateSet('A-ep2', 'B-ep2', 'C-ep2', 'A-ep4', 'B-ep4', 'C-ep4', 'A-ep8', 'B-ep8', 'C-ep8')]
    [string]$StartAt = 'A-ep2',
    [switch]$DryRun,
    [switch]$CheckRuntime,
    [switch]$AllowExistingCheckpoints
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$allExperiments = @('A-ep2', 'B-ep2', 'C-ep2', 'A-ep4', 'B-ep4', 'C-ep4', 'A-ep8', 'B-ep8', 'C-ep8')
$StartAt = $StartAt.Substring(0, 1).ToUpperInvariant() + $StartAt.Substring(1).ToLowerInvariant()
$startIndex = [Array]::IndexOf($allExperiments, $StartAt)
$experiments = $allExperiments[$startIndex..($allExperiments.Count - 1)]
$binary = Join-Path $repoRoot 'target/release/ppo-snake.exe'

foreach ($experiment in $experiments) {
    $config = Join-Path $repoRoot "config/experiments/epoch_scale/$experiment.yaml"
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "Missing experiment config: $config"
    }
    $checkpoint = Join-Path $repoRoot "checkpoints/E002_epoch_scale_$experiment/latest.ot"
    if (-not $DryRun -and -not $CheckRuntime -and -not $AllowExistingCheckpoints -and (Test-Path -LiteralPath $checkpoint)) {
        throw "Checkpoint already exists for $experiment at $checkpoint. Use -AllowExistingCheckpoints to rerun it, or -StartAt to begin later."
    }
}

if ($DryRun) {
    Write-Output 'cargo build --release'
    foreach ($experiment in $experiments) {
        Write-Output "target/release/ppo-snake.exe --config config/experiments/epoch_scale/$experiment.yaml"
    }
    return
}

$originalPath = $env:PATH
Push-Location -LiteralPath $repoRoot
try {
    & cargo build --release
    if ($LASTEXITCODE -ne 0) {
        throw "Release build failed with exit code $LASTEXITCODE"
    }

    if ($env:LIBTORCH) {
        $libtorchLib = Join-Path $env:LIBTORCH 'lib'
    }
    else {
        $bundled = @(Get-ChildItem -Path 'target/release/build/torch-sys-*/out/libtorch/libtorch/lib' -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'torch_cpu.dll') })
        if ($bundled.Count -ne 1) {
            throw 'Could not identify one bundled LibTorch DLL directory. Set LIBTORCH to the LibTorch root and rerun.'
        }
        $libtorchLib = $bundled[0].FullName
    }
    if (-not (Test-Path -LiteralPath (Join-Path $libtorchLib 'torch_cpu.dll') -PathType Leaf)) {
        throw "LibTorch runtime DLLs not found in $libtorchLib"
    }
    $env:PATH = "$libtorchLib;$originalPath"

    if ($CheckRuntime) {
        & $binary
        if ($LASTEXITCODE -ne 1) {
            throw "Runtime check failed with exit code $LASTEXITCODE"
        }
        Write-Host "LibTorch runtime loaded from $libtorchLib"
        return
    }

    foreach ($experiment in $experiments) {
        $config = "config/experiments/epoch_scale/$experiment.yaml"
        Write-Host "Starting epoch experiment $experiment with $config"
        & $binary --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $experiment failed with exit code $LASTEXITCODE"
        }
        Write-Host "Completed epoch experiment $experiment"
    }
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}

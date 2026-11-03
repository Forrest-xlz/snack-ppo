param(
    [ValidatePattern('^[A-Za-z]$')]
    [string]$StartAt = 'A',
    [switch]$DryRun,
    [switch]$CheckRuntime,
    [switch]$AllowExistingCheckpoints
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$configDirectory = Join-Path $repoRoot 'config/experiments/feature_architecture'
$allExperiments = @(Get-ChildItem -LiteralPath $configDirectory -Filter '*.yaml' -File |
    Where-Object { $_.BaseName -cmatch '^[A-Z]$' } | Sort-Object BaseName | ForEach-Object { $_.BaseName })
$StartAt = $StartAt.ToUpperInvariant()
$startIndex = [Array]::IndexOf($allExperiments, $StartAt)
if ($startIndex -lt 0) { throw "Unknown experiment '$StartAt'. Available: $($allExperiments -join ', ')" }
$experiments = @($allExperiments[$startIndex..($allExperiments.Count - 1)])
$binary = Join-Path $repoRoot 'target/release/ppo-snake.exe'

foreach ($experiment in $experiments) {
    $config = Join-Path $repoRoot "config/experiments/feature_architecture/$experiment.yaml"
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "Missing experiment config: $config"
    }
    $checkpoint = Join-Path $repoRoot "checkpoints/E004_feature_architecture_exp-$experiment/latest.ot"
    if (-not $DryRun -and -not $CheckRuntime -and -not $AllowExistingCheckpoints -and (Test-Path -LiteralPath $checkpoint)) {
        throw "Checkpoint already exists for $experiment at $checkpoint. Use -AllowExistingCheckpoints to rerun it, or -StartAt to begin later."
    }
}

if ($DryRun) {
    Write-Output 'cargo build --release'
    foreach ($experiment in $experiments) {
        Write-Output "target/release/ppo-snake.exe --config config/experiments/feature_architecture/$experiment.yaml"
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
        $config = "config/experiments/feature_architecture/$experiment.yaml"
        Write-Host "Starting feature/architecture experiment $experiment with $config"
        & $binary --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $experiment failed with exit code $LASTEXITCODE"
        }
        Write-Host "Completed feature/architecture experiment $experiment"
    }
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}

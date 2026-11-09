param(
    [ValidatePattern('^[A-Za-z]$')]
    [string]$StartAt = 'A',
    [switch]$Only,
    [switch]$DryRun,
    [switch]$CheckRuntime,
    [switch]$AllowExistingCheckpoints,
    [switch]$Benchmark,
    [ValidateSet('cpu', 'cuda', 'both')]
    [string]$BenchmarkDevice = 'both',
    [ValidateRange(0, 100)]
    [int]$WarmupRollouts = 1,
    [ValidateRange(1, 100)]
    [int]$BenchmarkRollouts = 2
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
if ($Only) { $experiments = @($StartAt) }
$binary = Join-Path $repoRoot 'target/release/ppo-snake.exe'

foreach ($experiment in $experiments) {
    $config = Join-Path $repoRoot "config/experiments/feature_architecture/$experiment.yaml"
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "Missing experiment config: $config"
    }
}

if ($DryRun) {
    if ($Benchmark) { Write-Output 'cargo build --release --bins --example benchmark_training' }
    else { Write-Output 'cargo build --release' }
    foreach ($experiment in $experiments) {
        if ($Benchmark) {
            $devices = if ($BenchmarkDevice -eq 'both') { @('cpu', 'cuda') } else { @($BenchmarkDevice) }
            foreach ($benchmarkTarget in $devices) {
                Write-Output "target/release/examples/benchmark_training.exe --config config/experiments/feature_architecture/$experiment.yaml --device $benchmarkTarget --warmup $WarmupRollouts --rollouts $BenchmarkRollouts --output target/benchmarks/E004-$experiment-$benchmarkTarget.json"
            }
        }
        else { Write-Output "target/release/ppo-snake.exe --config config/experiments/feature_architecture/$experiment.yaml" }
    }
    return
}

$originalPath = $env:PATH
Push-Location -LiteralPath $repoRoot
try {
    if ($Benchmark) { & cargo build --release --bins --example benchmark_training }
    else { & cargo build --release }
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

    if (-not $Benchmark -and -not $AllowExistingCheckpoints) {
        # Ask the same Rust config/parser/path sanitizer used by training;
        # letter B may point to a new versioned run without touching old B.
        foreach ($experiment in $experiments) {
            $config = "config/experiments/feature_architecture/$experiment.yaml"
            $configuredDirectory = & $binary --checkpoint-directory --config $config
            if ($LASTEXITCODE -ne 0) { throw "Could not inspect checkpoint directory for $experiment" }
            $runDirectory = [string]$configuredDirectory
            if (-not [System.IO.Path]::IsPathRooted($runDirectory)) { $runDirectory = Join-Path $repoRoot $runDirectory }
            $checkpoint = Join-Path $runDirectory 'latest.ot'
            if (Test-Path -LiteralPath $checkpoint) {
                throw "Checkpoint already exists for $experiment at $checkpoint. Use -AllowExistingCheckpoints for an intentional fresh rerun, or -StartAt to begin later."
            }
        }
    }

    foreach ($experiment in $experiments) {
        $config = "config/experiments/feature_architecture/$experiment.yaml"
        if ($Benchmark) {
            $devices = if ($BenchmarkDevice -eq 'both') { @('cpu', 'cuda') } else { @($BenchmarkDevice) }
            foreach ($benchmarkTarget in $devices) {
                Write-Host "Benchmarking $experiment on $benchmarkTarget (no W&B or formal checkpoints)"
                & (Join-Path $repoRoot 'target/release/examples/benchmark_training.exe') --config $config --device $benchmarkTarget --warmup $WarmupRollouts --rollouts $BenchmarkRollouts --output "target/benchmarks/E004-$experiment-$benchmarkTarget.json"
                if ($LASTEXITCODE -ne 0) { throw "Benchmark $experiment on $benchmarkTarget failed with exit code $LASTEXITCODE" }
            }
            continue
        }
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

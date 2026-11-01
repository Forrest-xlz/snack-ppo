param(
    [string]$StartAt = '',
    [ValidateSet('All', 'A', 'B', 'C')]
    [string]$SourceGroup = 'All',
    [ValidateSet('All', 'Early', 'Middle', 'Late')]
    [string]$Stage = 'All',
    [switch]$DryRun,
    [switch]$CheckRuntime,
    [switch]$AllowExistingCheckpoints
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$binary = Join-Path $repoRoot 'target/release/ppo-snake.exe'
$points = @(
    @{ Source = 'A'; Stage = 'Early'; Time = 100; Targets = @('B', 'C') },
    @{ Source = 'A'; Stage = 'Middle'; Time = 300; Targets = @('B', 'C') },
    @{ Source = 'A'; Stage = 'Late'; Time = 550; Targets = @('B', 'C') },
    @{ Source = 'B'; Stage = 'Early'; Time = 200; Targets = @('C') },
    @{ Source = 'B'; Stage = 'Middle'; Time = 500; Targets = @('C') },
    @{ Source = 'B'; Stage = 'Late'; Time = 850; Targets = @('C') },
    @{ Source = 'C'; Stage = 'Early'; Time = 150; Targets = @('B') },
    @{ Source = 'C'; Stage = 'Middle'; Time = 450; Targets = @('B') },
    @{ Source = 'C'; Stage = 'Late'; Time = 850; Targets = @('B') }
)
$experiments = @(
    foreach ($point in $points) {
        if (($SourceGroup -eq 'All' -or $SourceGroup -eq $point.Source) -and
            ($Stage -eq 'All' -or $Stage -eq $point.Stage)) {
            foreach ($target in $point.Targets) { "$($point.Source)-$($point.Time)M-to-$target" }
        }
    }
)
if ($StartAt) {
    $startIndex = -1
    for ($index = 0; $index -lt $experiments.Count; $index++) {
        if ($experiments[$index] -eq $StartAt) { $startIndex = $index; break }
    }
    if ($startIndex -lt 0) {
        throw "Unknown or filtered-out experiment '$StartAt'. Available: $($experiments -join ', ')"
    }
    $experiments = @($experiments[$startIndex..($experiments.Count - 1)])
}

# Check every selected input and output before starting any long training.
foreach ($experiment in $experiments) {
    $config = Join-Path $repoRoot "config/experiments/batch_rollout_switch/$experiment.yaml"
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) { throw "Missing experiment config: $config" }
    $yaml = [System.IO.File]::ReadAllText($config)
    $sourceMatch = [regex]::Match($yaml, '(?m)^  init_checkpoint: ([^\r\n]+)')
    if (-not $sourceMatch.Success) { throw "Missing weights-only initialization in $config" }
    $sourceValue = $sourceMatch.Groups[1].Value.Trim().Trim('"', "'")
    $source = if ([System.IO.Path]::IsPathRooted($sourceValue)) { $sourceValue } else { Join-Path $repoRoot $sourceValue }
    if (-not $CheckRuntime -and -not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing source checkpoint for ${experiment}: $source"
    }
    $nameMatch = [regex]::Match($yaml, '(?m)^  run_name: ([^\r\n]+)')
    if (-not $nameMatch.Success -or $nameMatch.Groups[1].Value.Trim().Trim('"', "'") -ne "E003 batch rollout switch $experiment") {
        throw "Unexpected run name in $config; checkpoint protection requires the E003 experiment name."
    }
    $directoryMatch = [regex]::Match($yaml, '(?m)^  checkpoint_dir: ([^\r\n]+)')
    if (-not $directoryMatch.Success -or $directoryMatch.Groups[1].Value.Trim().Trim('"', "'") -ne 'checkpoints') {
        throw "Unexpected checkpoint directory in $config; this sweep expects 'checkpoints'."
    }
    $outputDirectory = Join-Path $repoRoot "checkpoints/E003_batch_rollout_switch_$experiment"
    $existing = @(Get-ChildItem -LiteralPath $outputDirectory -Filter '*.ot' -File -ErrorAction SilentlyContinue)
    if (-not $DryRun -and -not $CheckRuntime -and -not $AllowExistingCheckpoints -and $existing.Count -gt 0) {
        throw "Checkpoints already exist for $experiment in $outputDirectory. Use -StartAt to begin later, or -AllowExistingCheckpoints for an intentional fresh rerun."
    }
}
if ($DryRun) {
    Write-Output 'cargo build --release'
    foreach ($experiment in $experiments) {
        Write-Output "target/release/ppo-snake.exe --config config/experiments/batch_rollout_switch/$experiment.yaml"
    }
    return
}

$originalPath = $env:PATH
Push-Location -LiteralPath $repoRoot
try {
    & cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "Release build failed with exit code $LASTEXITCODE" }
    if ($env:LIBTORCH) {
        $libtorchLib = Join-Path $env:LIBTORCH 'lib'
    }
    else {
        $bundled = @(Get-ChildItem -Path 'target/release/build/torch-sys-*/out/libtorch/libtorch/lib' -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'torch_cpu.dll') })
        if ($bundled.Count -ne 1) { throw 'Could not identify one bundled LibTorch DLL directory. Set LIBTORCH to the LibTorch root and rerun.' }
        $libtorchLib = $bundled[0].FullName
    }
    if (-not (Test-Path -LiteralPath (Join-Path $libtorchLib 'torch_cpu.dll') -PathType Leaf)) {
        throw "LibTorch runtime DLLs not found in $libtorchLib"
    }
    $env:PATH = "$libtorchLib;$originalPath"
    if ($CheckRuntime) {
        & $binary
        if ($LASTEXITCODE -ne 1) { throw "Runtime check failed with exit code $LASTEXITCODE" }
        Write-Host "LibTorch runtime loaded from $libtorchLib"
        return
    }
    foreach ($experiment in $experiments) {
        $config = "config/experiments/batch_rollout_switch/$experiment.yaml"
        Write-Host "Starting weights-only branch $experiment (Adam reset) with $config"
        & $binary --config $config
        if ($LASTEXITCODE -ne 0) { throw "Experiment $experiment failed with exit code $LASTEXITCODE" }
        Write-Host "Completed weights-only branch $experiment"
    }
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}

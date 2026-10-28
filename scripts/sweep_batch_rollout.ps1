param(
    [ValidateSet('A', 'B', 'C', 'D', 'E', 'F')]
    [string]$StartAt = 'A',
    [switch]$DryRun,
    [switch]$CheckRuntime,
    [switch]$AllowExistingCheckpoints
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$allGroups = @('A', 'B', 'C', 'D', 'E', 'F')
$startIndex = [Array]::IndexOf($allGroups, $StartAt)
$groups = $allGroups[$startIndex..($allGroups.Count - 1)]
$binary = Join-Path $repoRoot 'target/release/ppo-snake.exe'

foreach ($group in $groups) {
    $config = Join-Path $repoRoot "config/experiments/batch_rollout/$group.yaml"
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "Missing experiment config: $config"
    }
    $checkpoint = Join-Path $repoRoot "checkpoints/E001_batch_rollout_scale_exp-$group/latest.ot"
    if (-not $DryRun -and -not $CheckRuntime -and -not $AllowExistingCheckpoints -and (Test-Path -LiteralPath $checkpoint)) {
        throw "Checkpoint already exists for group $group at $checkpoint. Use -AllowExistingCheckpoints to rerun it, or -StartAt to begin later."
    }
}

if ($DryRun) {
    Write-Output 'cargo build --release'
    foreach ($group in $groups) {
        Write-Output "target/release/ppo-snake.exe --config config/experiments/batch_rollout/$group.yaml"
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

    foreach ($group in $groups) {
        $config = "config/experiments/batch_rollout/$group.yaml"
        Write-Host "Starting batch-rollout group $group with $config"
        & $binary --config $config
        if ($LASTEXITCODE -ne 0) {
            throw "Group $group failed with exit code $LASTEXITCODE"
        }
        Write-Host "Completed batch-rollout group $group"
    }
}
finally {
    $env:PATH = $originalPath
    Pop-Location
}

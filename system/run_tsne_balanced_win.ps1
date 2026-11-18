# ==============================================================================
# 1-CLICK WINDOWS GPU RUNNER: DENSE BALANCED t-SNE (3,000 SAMPLES, 300/CLASS)
# ==============================================================================
# Runs the 6 IEEE comparison algorithms (FedTGP, FedCOF, FedSA, FedORGP, FedGMKD, FedSCD)
# on CIFAR-10 (alpha = 0.5, seed = 2006, 20 clients, 100 rounds) with:
#   - FedSCD locked to verified 76.66% manuscript hyperparameters (c7tds897)
#   - --tsne_gap 100          : extract t-SNE only at the final round
#   - --tsne_max_samples 3000 : 300 unseen test images/class x 10 classes = 3,000 dense points
#   - --tsne_balanced         : balanced class sampling across all 20 clients' test sets
#   - --save_tsne_ckpt        : save Client 0 (.pt) checkpoint + 512-D features in .npz
#   - --wandb_group Cifar10_nc20_alpha0.5_tsne3k
#
# Usage (Windows PowerShell):
#   cd system
#   .\run_tsne_balanced_win.ps1
#   .\run_tsne_balanced_win.ps1 -DryRun
# ==============================================================================

param(
    [string]$Device = "cuda",
    [int]$DeviceId = 0,
    [string]$Dataset = "Cifar10",
    [string]$Model = "cnn",
    [int]$NumClients = 20,
    [int]$NumClasses = 10,
    [int]$GlobalRounds = 100,
    [int]$LocalEpochs = 1,
    [int]$BatchSize = 32,
    [double]$LocalLearningRate = 0.01,
    [double]$JoinRatio = 1.0,
    [double]$AlphaDirich = 0.5,
    [int]$Seed = 2006,
    [int]$TsneGap = 100,
    [int]$TsneMaxSamples = 3000,
    [string[]]$Algos = @("FedTGP", "FedCOF", "FedSA", "FedORGP", "FedGMKD", "FedSCD"),
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology",
    [string]$WandbGroup = "Cifar10_nc20_alpha0.5_tsne3k",
    [switch]$FairProjection,
    [switch]$ForceAll,
    [switch]$DryRun,
    [switch]$SkipPlot
)

Set-Location $PSScriptRoot

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$pyArgs = @(
    "-u", "run_tsne_balanced.py",
    "--device", $Device,
    "--device_id", "$DeviceId",
    "--dataset", $Dataset,
    "--model", $Model,
    "--num_clients", "$NumClients",
    "--num_classes", "$NumClasses",
    "--global_rounds", "$GlobalRounds",
    "--local_epochs", "$LocalEpochs",
    "--batch_size", "$BatchSize",
    "--lr", "$LocalLearningRate",
    "--join_ratio", "$JoinRatio",
    "--alpha", "$AlphaDirich",
    "--seed", "$Seed",
    "--tsne_gap", "$TsneGap",
    "--tsne_max_samples", "$TsneMaxSamples",
    "--wandb_project", $WandbProject,
    "--wandb_entity", $WandbEntity,
    "--wandb_group", $WandbGroup,
    "--algos"
) + $Algos

if ($FairProjection) {
    $pyArgs += "--tsne_fair"
}
if ($ForceAll) {
    $pyArgs += "--force_all"
}
if ($DryRun) {
    $pyArgs += "--dry_run"
}

Write-Host "Using Python: $PythonExe" -ForegroundColor Cyan
& $PythonExe @pyArgs
$ExitCode = $LASTEXITCODE

if ($ExitCode -ne 0) {
    Write-Host "`n[ERROR] Dense t-SNE extraction exited with code $ExitCode." -ForegroundColor Red
    exit $ExitCode
}

if ((-Not $SkipPlot) -and (-Not $DryRun)) {
    Write-Host "`n>>> Downloading dense 3k .npz files from WandB group '$WandbGroup' and rendering IEEE 2x3 t-SNE..." -ForegroundColor Green
    Push-Location ".."
    & $PythonExe "Figure_Plot\download_and_plot_tsne_ieee.py" `
        --entity $WandbEntity `
        --project $WandbProject `
        --dataset $Dataset `
        --alpha $AlphaDirich `
        --seed $Seed `
        --wandb_group $WandbGroup `
        --data_dir ".\Figure_Plot\data_tsne_3k" `
        --output_pdf ".\Figure_Plot\figures\ieee_tsne_cifar10_dir05.pdf"
    Pop-Location
}

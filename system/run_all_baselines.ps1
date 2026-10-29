# ==============================================================================
# SCRIPT BENCHMARK 8 THUAT TOAN PFL VOI THAM SO DAC THU CHO TUNG THUAT TOAN
# ==============================================================================
param(
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
    [int]$Seed = 42,
    [int]$EvalGap = 1,
    [string]$Device = "cuda",
    [int]$DeviceId = 0,
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology",
    [string]$WandbGroup = "baselines-cifar10"
)

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Danh sach 8 baseline can chay
$Algos = @(
    "FedTGP",
    "FedSA",
    "FedPLVM",
    "FedImpro",
    "FedCOF",
    "Fed3R",
    "FedORGP",
    "FedGMKD"
)

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " BAT DAU CHAY 8 THUAT TOAN BASELINE (SEED: $Seed | BATCH: $BatchSize | LR: $LocalLearningRate)" -ForegroundColor Cyan
Write-Host " Target Project : $WandbEntity / $WandbProject" -ForegroundColor Cyan
Write-Host " Device         : $Device (ID: $DeviceId)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

$CurrentIdx = 1
foreach ($algo in $Algos) {
    Write-Host "`n[$CurrentIdx/$($Algos.Count)] ===> DANG CHAY THUAT TOAN: $algo <===" -ForegroundColor Green
    $StartTime = Get-Date

    # --------------------------------------------------------------------------
    # KHAI BAO THAM SO DAC THU CUA TUNG THUAT TOAN
    # --------------------------------------------------------------------------
    $ExtraArgs = @()
    switch ($algo) {
        "FedTGP" {
            # FedTGP: Huấn luyện Task-Guided Generator tại server và margin threshold
            $ExtraArgs = @("-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1")
        }
        "FedSA" {
            # FedSA: Semantic Subspace Alignment (điều hòa căn chỉnh ngữ nghĩa)
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "FedPLVM" {
            # FedPLVM: Prototype Latent Variable Model (học biểu diễn phân phối tiềm ẩn)
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "FedImpro" {
            # FedImpro: Cải thiện biên phân loại và độ ổn định prototype
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "FedCOF" {
            # FedCOF: Collaborative Orthogonal / Contrastive Features
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "Fed3R" {
            # Fed3R: Robust Representation Regularization
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "FedORGP" {
            # FedORGP: Orthogonal Representation Guided Prototype
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
        "FedGMKD" {
            # FedGMKD: Generative / Mutual Knowledge Distillation
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1")
        }
    }

    # Tap hop toan bo tham so chung va tham so rieng
    $cmdArgs = @(
        "-u", "main.py",
        "-algo", $algo,
        "-data", $Dataset,
        "-m", $Model,
        "-nc", $NumClients,
        "-nb", $NumClasses,
        "-gr", $GlobalRounds,
        "-ls", $LocalEpochs,
        "-lbs", $BatchSize,
        "-lr", $LocalLearningRate,
        "-jr", $JoinRatio,
        "--noniid",
        "--alpha_dirich", $AlphaDirich,
        "-eg", $EvalGap,
        "--seed", $Seed,
        "-dev", $Device,
        "-did", $DeviceId,
        "-log",
        "--wandb_project", $WandbProject,
        "--wandb_entity", $WandbEntity,
        "--wandb_group", $WandbGroup
    ) + $ExtraArgs

    Write-Host "Lenh: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & $PythonExe @cmdArgs

    $EndTime = Get-Date
    $Elapsed = ($EndTime - $StartTime).TotalMinutes
    Write-Host "=> Hoan thanh $algo trong $([math]::Round($Elapsed, 2)) phut." -ForegroundColor Yellow
    $CurrentIdx++
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host " HOAN THANH TOAN BO 8 BASELINE!" -ForegroundColor Green
Write-Host " Kiem tra dashboard tai: https://forge.coreweave.com/wandb/$WandbEntity/$WandbProject" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan
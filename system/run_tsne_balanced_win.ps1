# ==============================================================================
# SCRIPT CHAY TU DONG TREN WINDOWS: TAI TAO DU LIEU t-SNE CAN BANG (2,000 MAU)
# ==============================================================================
# Chay 6 thuat toan trong Hinh 3 (FedTGP, FedCOF, FedSA, FedORGP, FedGMKD, FedSCD)
# tren CIFAR-10 (alpha = 0.5, seed = 2006, 20 clients, 100 rounds).
#
# Giu nguyen toan bo hyperparameter goc nhu run cu, chi thay doi phan t-SNE:
#   - --tsne_gap 100         : chi chay t-SNE o round cuoi (100)
#   - --tsne_max_samples 2000: 200 anh test/lop x 10 lop = 2,000 diem
#   - --tsne_balanced        : lay mau can bang tu tap test chua tung thay cua 20 clients
#   - --tsne_fair            : chuan hoa & chieu khong gian dong nhat 100% cho ca 6 thuat toan
#   - --save_tsne_ckpt       : luu checkpoint Client 0 (.pt) tai ./checkpoints_tsne/
#   - --wandb_group Cifar10_nc20_alpha0.5_tsne2k
#
# Cach chay tren PowerShell (Windows):
#   cd system
#   .\run_tsne_balanced_win.ps1
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
    [int]$TsneMaxSamples = 2000,
    [string[]]$Algos = @("FedTGP", "FedCOF", "FedSA", "FedORGP", "FedGMKD", "FedSCD"),
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology",
    [string]$WandbGroup = "Cifar10_nc20_alpha0.5_tsne2k",
    [switch]$SkipPlot
)

Set-Location $PSScriptRoot

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$LogsDir = "..\logs"
if (-Not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

$TotalRuns = $Algos.Count
Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " CHAY TU DONG 6 THUAT TOAN CHO t-SNE CAN BANG 2,000 DIEM (WINDOWS)" -ForegroundColor Cyan
Write-Host " Thiet lap chung : $Dataset | alpha=$AlphaDirich | nc=$NumClients | rounds=$GlobalRounds | seed=$Seed" -ForegroundColor Cyan
Write-Host " Cau hinh t-SNE  : gap=$TsneGap | max_samples=$TsneMaxSamples | balanced + fair + save_ckpt" -ForegroundColor Cyan
Write-Host " WandB Target    : $WandbEntity / $WandbProject (Group: $WandbGroup)" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

$CurrentStep = 1
$SuccessCount = 0
$FailCount = 0

foreach ($algo in $Algos) {
    Write-Host "`n[$CurrentStep/$TotalRuns] >>> DANG CHAY: $algo ($Dataset, alpha=$AlphaDirich, seed=$Seed) <<<" -ForegroundColor Green
    $StartTime = Get-Date

    # Hyperparameter rieng giu nguyen 100% theo cac run goc tren WandB
    $ExtraArgs = @()
    switch ($algo) {
        "FedTGP" {
            $ExtraArgs = @("-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1")
        }
        "FedCOF" {
            $ExtraArgs = @("-lam", "0.1", "--cof_gamma", "1.0")
        }
        "FedSA" {
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "-mcl", "0.01", "-cc", "1.0")
        }
        "FedORGP" {
            $ExtraArgs = @("-lam", "0.1", "-clam", "1.0", "-slam", "1.0", "-gam", "1.0", "--orgp_alpha", "0.5", "--orgp_server_epochs", "1", "--orgp_server_lr", "0.01")
        }
        "FedGMKD" {
            $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--gmkd_lam", "0.06", "--gmkd_gamma", "0.06", "--gmkd_temp", "0.6")
        }
        "FedSCD" {
            $ExtraArgs = @(
                "-lam", "0.1",
                "-nct", "0.1",
                "--phi_range", "0.1",
                "--proto_momentum", "0.1",
                "--bayes_shrink", "0.25",
                "--lamda_ins", "0.05",
                "--lamda_virt_global", "0.05",
                "--proto_warmup", "1",
                "--proto_rampup", "5"
            )
        }
    }

    $LogFile = "$LogsDir\tsne2k_${Dataset}_${algo}_alpha${AlphaDirich}_seed${Seed}.log"

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
        "-dev", $Device,
        "-did", $DeviceId,
        "-t", "1",
        "-sd", $Seed,
        "-log",
        "--wandb_project", $WandbProject,
        "--wandb_entity", $WandbEntity,
        "--wandb_group", $WandbGroup,
        "--tsne_gap", $TsneGap,
        "--tsne_max_samples", $TsneMaxSamples,
        "--tsne_balanced",
        "--tsne_fair",
        "--save_tsne_ckpt"
    ) + $ExtraArgs

    Write-Host "Lenh: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & $PythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $LogFile

    $ExitCode = $LASTEXITCODE
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime

    if ($ExitCode -eq 0) {
        $SuccessCount++
        Write-Host "[HOAN THANH $CurrentStep/$TotalRuns] $algo trong $([Math]::Round($Duration.TotalMinutes, 2)) phut." -ForegroundColor Green
    } else {
        $FailCount++
        Write-Host "[THAT BAI $CurrentStep/$TotalRuns] $algo ma loi $ExitCode! Xem log tai: $LogFile" -ForegroundColor Red
    }

    $CurrentStep++
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host " TONG KET: Thanh cong $SuccessCount / $TotalRuns | That bai: $FailCount" -ForegroundColor Yellow
Write-Host " Checkpoint Client 0 da luu tai: $PSScriptRoot\checkpoints_tsne\" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

if ((-Not $SkipPlot) -and ($FailCount -eq 0)) {
    Write-Host "`n>>> Dang tai du lieu moi tu WandB group '$WandbGroup' va ve lai hinh t-SNE IEEE..." -ForegroundColor Green
    Push-Location ".."
    & $PythonExe "Figure_Plot\download_and_plot_tsne_ieee.py" `
        --entity $WandbEntity `
        --project $WandbProject `
        --dataset $Dataset `
        --alpha $AlphaDirich `
        --seed $Seed `
        --wandb_group $WandbGroup `
        --data_dir ".\Figure_Plot\data_tsne_2k" `
        --output_pdf ".\Figure_Plot\figures\ieee_tsne_cifar10_dir05.pdf"
    Pop-Location
}

# ==============================================================================
# SCRIPT POWERSHELL CHAY CAC THI NGHIEM CON THIEU CHO BANG 4 (ICML 2026)
# Chay tren Windows voi GPU NVIDIA RTX 4050 (CUDA Device 0)
# Luon bat ghi log truc tuyen len WandB
# ==============================================================================

param(
    [string]$Device = "cuda",
    [int]$DeviceId = 0,
    [int]$GlobalRounds = 100,
    [int]$LocalEpochs = 1,
    [int]$BatchSize = 32,
    [double]$LocalLearningRate = 0.01,
    [int]$NumClients = 20,
    [double]$JoinRatio = 1.0,
    [int]$Times = 1,
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology"
)

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Tao thu muc logs neu chua co
$LogsDir = "..\logs"
if (-Not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Cac dataset can chay
$Datasets = @(
    @{ Name = "Cifar10";      Classes = 10;  Model = "cnn" },
    @{ Name = "Cifar100";     Classes = 100; Model = "cnn" },
    @{ Name = "Flowers102";   Classes = 102; Model = "cnn" },
    @{ Name = "TinyImagenet"; Classes = 200; Model = "cnn" }
)

$Alphas = @(0.1, 0.5, 1.0)
$Algos = @("FedTGP", "FedPLVM", "FedGMKD", "FedSA", "FedORGP", "FedCOF", "FedSCD")

# Danh sach cac thi nghiem da co ket qua trong Bang 4 (icml2026.tex)
$CompletedRuns = @{
    "Cifar10_FedTGP_0.1" = 88.83
    "Cifar10_FedTGP_0.5" = 68.64
    "Cifar10_FedTGP_1.0" = 61.21
    "Cifar10_FedSA_0.5"  = 69.92
    "Cifar10_FedORGP_0.5"= 70.46
    "Cifar10_FedCOF_0.5" = 70.61
    "Cifar10_FedSCD_0.1" = 90.71
    "Cifar10_FedSCD_0.5" = 75.15
    "Cifar10_FedSCD_1.0" = 69.60
}

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " HE THONG TU DONG CHAY BENCHMARK BANG 4 (7 THUAT TOAN x 4 DATASET x 3 ALPHA)" -ForegroundColor Cyan
Write-Host " Phien ban Windows GPU NVIDIA RTX 4050 (-dev $Device -did $DeviceId)" -ForegroundColor Cyan
Write-Host " Rounds: $GlobalRounds | Batch: $BatchSize | LR: $LocalLearningRate | Clients: $NumClients" -ForegroundColor Cyan
Write-Host " WandB: $WandbEntity / $WandbProject" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

$TotalPlanned = 84
$SkippedCount = $CompletedRuns.Count
$PendingCount = $TotalPlanned - $SkippedCount
$CurrentStep = 1

foreach ($ds in $Datasets) {
    $dsName = $ds.Name
    $nbClasses = $ds.Classes
    $model = $ds.Model

    foreach ($alpha in $Alphas) {
        foreach ($algo in $Algos) {
            $key = "${dsName}_${algo}_${alpha}"
            
            if ($CompletedRuns.ContainsKey($key)) {
                $acc = $CompletedRuns[$key]
                Write-Host "[BO QUA] $algo tren $dsName (alpha=$alpha) -> DA CHAY VOI ACC: $acc%" -ForegroundColor Yellow
                continue
            }

            Write-Host "`n[$CurrentStep/$PendingCount] ===> DANG CHAY: $algo | Dataset: $dsName | Alpha: $alpha <===" -ForegroundColor Green
            $StartTime = Get-Date

            # Tham so bo sung theo dac thu thuat toan
            $ExtraArgs = @()
            switch ($algo) {
                "FedTGP" {
                    $ExtraArgs = @("-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1")
                }
                "FedSA" {
                    $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "-mcl", "0.01", "-cc", "1.0")
                }
                "FedPLVM" {
                    $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--plvm_alpha", "0.25", "--plvm_lamb", "100.0", "--plvm_tau", "0.07")
                }
                "FedGMKD" {
                    $ExtraArgs = @("-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--gmkd_lam", "0.06", "--gmkd_gamma", "0.06", "--gmkd_temp", "0.6")
                }
                "FedORGP" {
                    $ExtraArgs = @("-clam", "1.0", "-slam", "1.0", "-gam", "1.0", "--orgp_alpha", "0.5", "--orgp_server_epochs", "1", "--orgp_server_lr", "0.01")
                }
                "FedCOF" {
                    $ExtraArgs = @("-lam", "0.1", "--cof_gamma", "1.0")
                }
                "FedSCD" {
                    $ExtraArgs = @(
                        "-lamda", "0.1",
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

            $WandbGroup = "${dsName}_nc${NumClients}_alpha${alpha}"
            $LogFile = "$LogsDir\${dsName}_${algo}_alpha${alpha}.log"

            $cmdArgs = @(
                "-u", "main.py",
                "-algo", $algo,
                "-data", $dsName,
                "-m", $model,
                "-nc", $NumClients,
                "-nb", $nbClasses,
                "-gr", $GlobalRounds,
                "-ls", $LocalEpochs,
                "-lbs", $BatchSize,
                "-lr", $LocalLearningRate,
                "-jr", $JoinRatio,
                "--noniid",
                "--alpha_dirich", $alpha,
                "-dev", $Device,
                "-did", $DeviceId,
                "-t", $Times,
                "-log",
                "--wandb_project", $WandbProject,
                "--wandb_entity", $WandbEntity,
                "--wandb_group", $WandbGroup
            ) + $ExtraArgs

            Write-Host "Lenh: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
            
            & $PythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $LogFile
            
            $Duration = (Get-Date) - $StartTime
            Write-Host "[HOAN THANH] $algo tren $dsName (alpha=$alpha) trong $([Math]::Round($Duration.TotalMinutes, 2)) phut." -ForegroundColor Green
            $CurrentStep++
        }
    }
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host " TAT CA CAC THI NGHIEM CON THIEU DA HOAN TAT!" -ForegroundColor Cyan
Write-Host " Kiem tra ket qua tren WandB Dashboard: https://wandb.ai/$WandbEntity/$WandbProject" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

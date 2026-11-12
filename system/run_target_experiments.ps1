# ==============================================================================
# HỆ THỐNG CHẠY THỰC NGHIỆM TARGET CHO BẢNG 4 (ICML/IEEE 2026)
# Chuyên biệt cho GPU NVIDIA RTX 4050 (-dev cuda -did 0) và Seed 2006
#
# PHẠM VI THỰC THI (TỔNG CỘNG 30 RUNS):
#   1. TOÀN BỘ FedSCD TRONG BẢNG:
#      - CIFAR-10      : alpha 0.1, 0.5, 1.0 (3 runs)
#      - CIFAR-100     : alpha 0.1, 0.5, 1.0 (3 runs)
#      - Flowers102    : alpha 0.1, 0.5, 1.0 (3 runs)
#      - TinyImagenet  : alpha 0.1, 0.5, 1.0 (3 runs)
#   2. TOÀN BỘ BASELINE TRÊN TẬP TINY-IMAGENET:
#      - TinyImagenet (alpha 0.1, 0.5, 1.0) cho:
#        FedTGP, FedPLVM, FedGMKD, FedSA, FedORGP, FedCOF (18 runs)
#
# ĐẶC ĐIỂM:
#   * Tự động lưu log chi tiết vào thư mục ../logs/
#   * Luôn kích hoạt WandB online (-log) và đồng bộ đồ thị t-SNE (.pdf, .npz)
#   * Tự động phát hiện lỗi và hiển thị thời gian từng run
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
    [int]$Seed = 2006,
    [int]$TsneGap = 20,
    [switch]$NoTsne,
    [switch]$SkipDoneCifar10SCD,   # Bỏ qua 3 run CIFAR-10 của FedSCD đã có sẵn trong bảng
    [switch]$SkipTinyAlpha01,      # Bỏ qua alpha=0.1 cho 6 baseline trên Tiny-ImageNet (còn đúng 24 runs)
    [switch]$OnlyFedSCD,           # Chỉ chạy 12 run của FedSCD
    [switch]$OnlyTiny,             # Chỉ chạy các run trên Tiny-ImageNet
    [string]$WandbProject = "FedSCD",
    [string]$WandbEntity = "gvquang25-hanoi-university-of-science-and-technology"
)

$PythonExe = "..\.venv\Scripts\python.exe"
if (-Not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# Tạo thư mục logs nếu chưa có
$LogsDir = "..\logs"
if (-Not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

# Danh sách 30 runs mục tiêu
$TargetTasks = @()

# ------------------------------------------------------------------------------
# 1. Toàn bộ FedSCD trên 4 dataset (CIFAR-10, CIFAR-100, Flowers102, TinyImagenet)
# ------------------------------------------------------------------------------
if (-Not $OnlyTiny) {
    $ScdDatasets = @(
        @{ Name = "Cifar10";      Classes = 10;  Model = "cnn" },
        @{ Name = "Cifar100";     Classes = 100; Model = "cnn" },
        @{ Name = "Flowers102";   Classes = 102; Model = "cnn" },
        @{ Name = "TinyImagenet"; Classes = 200; Model = "cnn" }
    )

    $Alphas = @(0.1, 0.5, 1.0)

    foreach ($ds in $ScdDatasets) {
        foreach ($a in $Alphas) {
            if ($SkipDoneCifar10SCD -and $ds.Name -eq "Cifar10") {
                continue
            }
            $TargetTasks += @{
                Dataset = $ds.Name
                Classes = $ds.Classes
                Model   = $ds.Model
                Alpha   = $a
                Algo    = "FedSCD"
                Group   = "Target_FedSCD"
            }
        }
    }
}

# ------------------------------------------------------------------------------
# 2. Toàn bộ các baseline trên Tiny-ImageNet (alpha 0.1, 0.5, 1.0)
# ------------------------------------------------------------------------------
if (-Not $OnlyFedSCD) {
    $TinyBaselines = @("FedTGP", "FedPLVM", "FedGMKD", "FedSA", "FedORGP", "FedCOF")
    $TinyAlphas = if ($SkipTinyAlpha01) { @(0.5, 1.0) } else { @(0.1, 0.5, 1.0) }

    foreach ($algo in $TinyBaselines) {
        foreach ($a in $TinyAlphas) {
            $TargetTasks += @{
                Dataset = "TinyImagenet"
                Classes = 200
                Model   = "cnn"
                Alpha   = $a
                Algo    = $algo
                Group   = "Target_TinyBaselines"
            }
        }
    }
}

$TotalRuns = $TargetTasks.Count

Write-Host "==========================================================================" -ForegroundColor Cyan
Write-Host " CHƯƠNG TRÌNH CHẠY TARGET BENCHMARK: FedSCD & TINY-IMAGENET BASELINES" -ForegroundColor Cyan
Write-Host " Phần cứng : Windows GPU NVIDIA RTX 4050 (-dev $Device -did $DeviceId)" -ForegroundColor Cyan
Write-Host " Thiết lập : Rounds: $GlobalRounds | Batch: $BatchSize | LR: $LocalLearningRate | Seed: $Seed" -ForegroundColor Cyan
Write-Host " WandB     : $WandbEntity / $WandbProject" -ForegroundColor Cyan
Write-Host " Tổng số   : $TotalRuns runs cần thực hiện" -ForegroundColor Yellow
Write-Host "==========================================================================" -ForegroundColor Cyan

$CurrentStep = 1
$SuccessCount = 0
$FailCount = 0

foreach ($task in $TargetTasks) {
    $dsName = $task.Dataset
    $nbClasses = $task.Classes
    $model = $task.Model
    $alpha = $task.Alpha
    $algo = $task.Algo

    Write-Host "`n[$CurrentStep/$TotalRuns] >>> TIẾN HÀNH CHẠY: $algo | Dataset: $dsName | Alpha: $alpha <<<" -ForegroundColor Green
    $StartTime = Get-Date

    # Tham số bổ sung chuẩn xác theo bài báo cho từng thuật toán
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
        "-sd", $Seed,
        "-log",
        "--wandb_project", $WandbProject,
        "--wandb_entity", $WandbEntity,
        "--wandb_group", $WandbGroup
    ) + $ExtraArgs

    if ($NoTsne) {
        $cmdArgs += "--no_tsne"
    } elseif ($TsneGap -gt 0) {
        $cmdArgs += @("--tsne_gap", $TsneGap)
    }

    Write-Host "Lệnh gọi: $PythonExe $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    
    # Thực thi và đồng thời in ra màn hình lẫn lưu file log
    & $PythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $LogFile

    $ExitCode = $LASTEXITCODE
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime

    if ($ExitCode -eq 0) {
        $SuccessCount++
        Write-Host "[HOÀN THÀNH $CurrentStep/$TotalRuns] $algo trên $dsName (alpha=$alpha) trong $([Math]::Round($Duration.TotalMinutes, 2)) phút." -ForegroundColor Green
    } else {
        $FailCount++
        Write-Host "[THẤT BẠI $CurrentStep/$TotalRuns] $algo trên $dsName (alpha=$alpha) văng mã lỗi $ExitCode! Xem chi tiết tại: $LogFile" -ForegroundColor Red
    }

    $CurrentStep++
}

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host " TỔNG KẾT TIẾN TRÌNH THỰC HIỆN TARGET BENCHMARK" -ForegroundColor Cyan
Write-Host " Thành công: $SuccessCount / $TotalRuns" -ForegroundColor Green
if ($FailCount -gt 0) {
    Write-Host " Thất bại  : $FailCount / $TotalRuns" -ForegroundColor Red
}
Write-Host " Kết quả và đồ thị t-SNE đã được đồng bộ lên WandB: $WandbEntity / $WandbProject" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan


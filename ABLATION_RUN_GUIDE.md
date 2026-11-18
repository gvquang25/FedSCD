# Hướng dẫn chạy Ablation & Sensitivity trên Windows (WandB)

Tài liệu tổng hợp các lệnh PowerShell trên Windows để tái tạo thực nghiệm **Table II (Component Ablation)** và **Table III (Sensitivity to Design Choices)** trên **CIFAR-10 ($\alpha = 0.5$)** và tự động đồng bộ kết quả lên **Weights & Biases (WandB)**.

---

## 1. Chuẩn bị môi trường (Windows PowerShell)

Mở PowerShell tại thư mục gốc của project, sau đó di chuyển vào `system/` và đăng nhập WandB (nếu máy chưa đăng nhập):

```powershell
cd system
..\.venv\Scripts\python.exe -m wandb login
```

> **Lưu ý:**
> - Nếu dùng `python` global hoặc conda thay cho `.venv`, thay `..\.venv\Scripts\python.exe` bằng `python`.
> - Có thể chỉnh lại `--wandb_project`, `--wandb_entity`, `--wandb_group` theo tài khoản WandB của bạn.
> - Thêm `--no_tsne` vào `$BASE` / `$BaseArgs` nếu muốn tắt vẽ t-SNE để chạy nhanh hơn.

---

## 2. Tham số nền dùng chung (CIFAR-10, $\alpha = 0.5$)

Cấu hình chuẩn khớp 100% với bản thảo `Overleaf/ieee2026.tex` (run tham chiếu `c7tds897`, đạt **76.66%** trên CIFAR-10, $\alpha = 0.5$):

```powershell
$BASE = "-u main.py -algo FedSCD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -sd 2006 -lam 0.3 -nct 0.1 --phi_range 0.1 --proto_momentum 0.6 --bayes_shrink 0.25 --lamda_ins 0.75 --lamda_virt_local 0.3 --lamda_virt_global 0.3 --proto_warmup 1 --proto_rampup 5 --mixup_alpha 0.2 --scd_aug 1 --proj_mode shared --no_tsne -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_ablation_alpha0.5"
```

---

## 3. Table II — Component Ablation trên CIFAR-10 ($\alpha = 0.5$)

| Cấu hình | Flag thay đổi | Lệnh PowerShell |
| :--- | :--- | :--- |
| **FedSCD (Full Framework)** | *(Mặc định)* | `..\.venv\Scripts\python.exe $BASE.Split(" ") -go "FedSCD_Full"` |
| **w/o Compactness ($\mathcal{L}_{\text{PL}}$)** | `--lamda_compact 0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --lamda_compact 0 -go "wo_Compactness"` |
| **w/o SupCon ($\lambda_{\text{ins}} = 0$)** | `--lamda_ins 0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --lamda_ins 0 -go "wo_SupCon"` |
| **w/o Virtual Features** | `--lamda_virt_global 0 --lamda_virt_local 0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --lamda_virt_global 0 --lamda_virt_local 0 -go "wo_VirtualFeatures"` |
| **w/o ProtoNCE ($\mathcal{L}_{\text{NCE}}$)** | `--lamda_nce 0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --lamda_nce 0 -go "wo_ProtoNCE"` |
| **Baseline (CE + Augmentation only)** | `-lp 0 --lamda_virt_global 0 --lamda_virt_local 0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") -lp 0 --lamda_virt_global 0 --lamda_virt_local 0 -go "Baseline_CE_Aug"` |

*(Lưu ý: ở cấu hình Baseline, `--scd_aug 1` vẫn được giữ mặc định cho Augmentation, còn `-lp 0` và `--lamda_virt_* 0` tắt toàn bộ prototype loss và virtual head loss.)*

---

## 4. Table III — Sensitivity to Design Choices trên CIFAR-10 ($\alpha = 0.5$)

*(Cấu hình **FedSCD (Default Configuration)** trùng với **FedSCD (Full Framework)** ở Table II.)*

| Cấu hình | Flag thay đổi | Lệnh PowerShell |
| :--- | :--- | :--- |
| **FedSCD (Default Configuration)** | *(Mặc định)* | `..\.venv\Scripts\python.exe $BASE.Split(" ") -go "FedSCD_Full"` |
| **Metric Projection: Class-specific (vs. Shared)** | `--proj_mode class` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --proj_mode class -go "ClassSpecific_Proj"` |
| **Temperature Modulation: Fixed ($\rho = 0$)** | `--fixed_temp` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --fixed_temp -go "Fixed_Temp"` |
| **Momentum Smoothing: No momentum ($\beta = 1.0$)** | `--proto_momentum 1.0` | `..\.venv\Scripts\python.exe $BASE.Split(" ") --proto_momentum 1.0 -go "No_Momentum"` |

---

## 5. Script PowerShell chạy gộp toàn bộ 9 runs liên tiếp

Chạy trong thư mục `system/`:

```powershell
$Python = if (Test-Path "..\.venv\Scripts\python.exe") { "..\.venv\Scripts\python.exe" } else { "python" }

$BaseArgs = @(
    "-u", "main.py",
    "-algo", "FedSCD",
    "-data", "Cifar10",
    "-m", "cnn",
    "-nc", "20",
    "-nb", "10",
    "-gr", "100",
    "-ls", "1",
    "-lbs", "32",
    "-lr", "0.01",
    "-jr", "1.0",
    "--noniid",
    "--alpha_dirich", "0.5",
    "-dev", "cuda",
    "-did", "0",
    "-t", "1",
    "-sd", "2006",
    "-lam", "0.3",
    "-nct", "0.1",
    "--phi_range", "0.1",
    "--proto_momentum", "0.6",
    "--bayes_shrink", "0.25",
    "--lamda_ins", "0.75",
    "--lamda_virt_local", "0.3",
    "--lamda_virt_global", "0.3",
    "--proto_warmup", "1",
    "--proto_rampup", "5",
    "--mixup_alpha", "0.2",
    "--scd_aug", "1",
    "--proj_mode", "shared",
    "--no_tsne",
    "-log",
    "--wandb_project", "FedSCD",
    "--wandb_entity", "gvquang25-hanoi-university-of-science-and-technology",
    "--wandb_group", "Cifar10_ablation_alpha0.5"
)

$Runs = @(
    @{ Name = "FedSCD_Full";        Extra = @() },
    @{ Name = "wo_Compactness";     Extra = @("--lamda_compact", "0") },
    @{ Name = "wo_SupCon";          Extra = @("--lamda_ins", "0") },
    @{ Name = "wo_VirtualFeatures"; Extra = @("--lamda_virt_global", "0", "--lamda_virt_local", "0") },
    @{ Name = "wo_ProtoNCE";        Extra = @("--lamda_nce", "0") },
    @{ Name = "Baseline_CE_Aug";    Extra = @("-lp", "0", "--lamda_virt_global", "0", "--lamda_virt_local", "0") },
    @{ Name = "ClassSpecific_Proj"; Extra = @("--proj_mode", "class") },
    @{ Name = "Fixed_Temp";         Extra = @("--fixed_temp") },
    @{ Name = "No_Momentum";        Extra = @("--proto_momentum", "1.0") }
)

foreach ($r in $Runs) {
    Write-Host "`n>>> Running: $($r.Name)" -ForegroundColor Cyan
    & $Python ($BaseArgs + $r.Extra + @("-go", $r.Name))
}
```

---

## 6. Chạy Hyperparameter Sensitivity (Figure 4 — Lưới 2×3, 18 Runs)

Khảo sát one-factor-at-a-time cho **6 siêu tham số** ($\lambda$, $\lambda_{\mathrm{ins}}$, $\lambda_{\mathrm{virt}}$, $\beta$, $\rho$, $\tau$) trên **CIFAR-10 ($\alpha = 0.5$, seed `2006`)**, giữ cố định các tham số còn lại theo run chuẩn `c7tds897` (`76.66%`) và đồng bộ lên nhóm WandB `Cifar10_hyperparam_alpha0.5`.

### 6.1. Chạy tự động 1-click trên Windows GPU (RTX 4050)

```powershell
cd system
.\run_hyperparam_sensitivity.ps1
```

*Các tùy chọn hữu ích:*
- Kiểm tra danh sách lệnh trước khi chạy: `.\run_hyperparam_sensitivity.ps1 -DryRun`
- Chỉ chạy riêng 1 siêu tham số (ví dụ `lam`, `ins`, `virt`, `beta`, `rho`, `tau`): `.\run_hyperparam_sensitivity.ps1 -OnlyParam lam`
- Chạy lại toàn bộ kể cả các run đã hoàn thành: `.\run_hyperparam_sensitivity.ps1 -ForceAll`

*(Script tự động bỏ qua các run đã có marker `.done` trong `logs/hyperparam_sensitivity/` hoặc đã ở trạng thái `finished` trên WandB, và tự động gọi `Figure_Plot\plot_hyperparam_sensitivity.py --fetch` sau khi xong.)*

### 6.2. Cập nhật lại biểu đồ Figure 4 từ WandB (khi chạy riêng bước vẽ)

```bash
python Figure_Plot/plot_hyperparam_sensitivity.py --fetch
```

Script sẽ tải đủ 18 run mới + run mặc định `c7tds897`, lưu CSV về `Figure_Plot/data_hyperparam_alpha05/` và xuất đồng thời `Figure_Plot/figures/hyperparam_sensitivity_6params.pdf` lẫn `Overleaf/figures/hyperparam_sensitivity_6params.pdf`.

# Hướng Dẫn Toàn Diện Chạy Benchmark 7 Thuật Toán (Bảng 4 - ICML 2026)

Tài liệu này chuẩn bị đầy đủ quy trình tải/chuẩn bị dữ liệu và lệnh chạy cho **7 thuật toán** trong **Bảng 4** của bài báo (`Overleaf/icml2026.tex`) trên **4 tập dataset** với **3 mức phân phối Dirichlet** ($\alpha \in \{0.1, 0.5, 1.0\}$), tối ưu hóa cho máy **Windows GPU NVIDIA RTX 4050**.

---

## 1. Ma Trận Tiến Độ Thí Nghiệm (Bảng 4 Trong Bài Báo)

Tổng số cấu hình: **7 thuật toán × 4 dataset × 3 mức $\alpha$ = 84 thí nghiệm**.
- **[ĐÃ CHẠY]**: 9 thí nghiệm đã có kết quả ghi nhận trong Bảng 4 / thư mục `results/`.
- **[CẦN CHẠY]**: 75 thí nghiệm còn trống (`--`).

| Thuật Toán | CIFAR-10 (0.1) | CIFAR-10 (0.5) | CIFAR-10 (1.0) | CIFAR-100 (0.1) | CIFAR-100 (0.5) | CIFAR-100 (1.0) | Flowers102 (0.1) | Flowers102 (0.5) | Flowers102 (1.0) | Tiny-ImageNet (0.1) | Tiny-ImageNet (0.5) | Tiny-ImageNet (1.0) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FedTGP** | **88.83%**<br>`[ĐÃ CHẠY]` | **68.64%**<br>`[ĐÃ CHẠY]` | **61.21%**<br>`[ĐÃ CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedPLVM** | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedGMKD** | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedSA** | `--`<br>`[CẦN CHẠY]` | **69.92%**<br>`[ĐÃ CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedORGP** | `--`<br>`[CẦN CHẠY]` | **70.46%**<br>`[ĐÃ CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedCOF** | `--`<br>`[CẦN CHẠY]` | **70.61%**<br>`[ĐÃ CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |
| **FedSCD (Ours)** | **90.71%**<br>`[ĐÃ CHẠY]` | **75.15%**<br>`[ĐÃ CHẠY]` | **69.60%**<br>`[ĐÃ CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` | `--`<br>`[CẦN CHẠY]` |

---

## 2. Bước 1: Clone & Chuẩn Bị 4 Tập Dataset (Chia 3 Dirichlet 0.1, 0.5, 1.0)

Script chuẩn hóa `scripts/prepare_all_datasets.py` tự động tải các bộ dữ liệu gốc và tạo phân chia Non-IID Dirichlet $K=20$ clients, 75% train / 25% test theo đúng cấu trúc thư mục mà hệ thống `system/utils/data_utils.py` yêu cầu:
- **Cifar10**: 10 lớp, ảnh $32 \times 32 \times 3$
- **Cifar100**: 100 lớp, ảnh $32 \times 32 \times 3$
- **Flowers102**: 102 lớp, resize về $64 \times 64 \times 3$
- **TinyImagenet**: 200 lớp, ảnh $64 \times 64 \times 3$ (tự động tải zip từ Stanford và giải nén)

### Lệnh chạy chuẩn bị toàn bộ 4 dataset:
Mở terminal (PowerShell hoặc CMD) tại thư mục gốc của dự án:

```powershell
# Chạy chuẩn bị tất cả 4 dataset với 3 mức Dirichlet: 0.1, 0.5, 1.0
python scripts/prepare_all_datasets.py --dataset all --alphas 0.1 0.5 1.0 --num_clients 20
```

> **Lưu ý kiểm tra trùng lặp:** Nếu phân chia nào đã có sẵn (ví dụ `dataset/Cifar10/20/False_True_0.5/`), script sẽ tự động nhận diện và bỏ qua `[SKIP]`, không tốn thời gian tải hoặc tạo lại.

---

## 3. Thiết Lập Môi Trường Windows GPU RTX 4050 & WandB

### Thông số huấn luyện chuẩn (theo phần Implementation Details của bài báo):
- **Device**: `-dev cuda -did 0` (NVIDIA GeForce RTX 4050 6GB VRAM)
- **Global Rounds**: `-gr 100` (100 vòng giao tiếp)
- **Clients**: `-nc 20` (20 clients)
- **Participation**: `-jr 1.0` (Full participation - 20 clients/vòng)
- **Batch Size**: `-lbs 32`
- **Learning Rate**: `-lr 0.01` (SGD không momentum/weight decay)
- **Backbone**: `-m cnn` (Feature dim $d=512$, Flatten dim $1600$ cho $32\times32$, $10816$ cho $64\times64$)
- **WandB Tracking**: Luôn bật `-log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology`

---

## 4. Cách 1: Chạy Tự Động 1-Click (Khuyên Dùng)

### Lựa chọn A: Dùng script PowerShell tự động bỏ qua các bài đã chạy
Tại thư mục `system/`:

```powershell
cd system
.\run_missing_experiments.ps1 -Device cuda -DeviceId 0 -BatchSize 32 -GlobalRounds 100 -LocalLearningRate 0.01
```

Script này sẽ:
1. Tự động kiểm tra danh sách 9 thí nghiệm đã chạy trong Bảng 4 và in `[BỎ QUA]`.
2. Tuần tự thực thi 75 thí nghiệm còn lại trên GPU RTX 4050.
3. Tự động lưu log vào `logs/<Dataset>_<Algo>_alpha<Alpha>.log` và đồng bộ live lên WandB.

### Lựa chọn B: Dùng Python Benchmark Matrix Runner
Tại thư mục `system/`:

```powershell
cd system
python run_benchmark_matrix.py --device cuda --device_id 0 --batch_size 32 --lr 0.01 --global_rounds 100
```
*(Nếu muốn chạy lại toàn bộ kể cả các bài đã chạy, thêm cờ `--force_all`)*.

---

## 5. Cách 2: Danh Sách Từng Lệnh Chạy Chi Tiết (Manual Copy-Paste)

Mỗi lệnh dưới đây đều được chuẩn hóa với đầy đủ tham số tối ưu riêng của từng thuật toán.

> **Chú thích:** Các lệnh được đánh dấu `# [ĐÃ CHẠY]` đã có kết quả. Nếu cần chạy lại, bạn chỉ cần bỏ dấu `#`.

### 5.1. CIFAR-10 (10 Classes, -data Cifar10 -nb 10 -m cnn)

#### Mức Dirichlet $\alpha = 0.1$
```powershell
# [ĐÃ CHẠY - 88.83%] FedTGP
# python -u main.py -algo FedTGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1

# [CẦN CHẠY] FedPLVM
python -u main.py -algo FedPLVM -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07

# [CẦN CHẠY] FedGMKD
python -u main.py -algo FedGMKD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6

# [CẦN CHẠY] FedSA
python -u main.py -algo FedSA -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0

# [CẦN CHẠY] FedORGP
python -u main.py -algo FedORGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01

# [CẦN CHẠY] FedCOF
python -u main.py -algo FedCOF -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lam 0.1 --cof_gamma 1.0

# [ĐÃ CHẠY - 90.71%] FedSCD (Ours)
# python -u main.py -algo FedSCD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.1 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 0.5$
```powershell
# [ĐÃ CHẠY - 68.64%] FedTGP
# python -u main.py -algo FedTGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1

# [CẦN CHẠY] FedPLVM
python -u main.py -algo FedPLVM -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07

# [CẦN CHẠY] FedGMKD
python -u main.py -algo FedGMKD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6

# [ĐÃ CHẠY - 69.92%] FedSA
# python -u main.py -algo FedSA -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0

# [ĐÃ CHẠY - 70.46%] FedORGP
# python -u main.py -algo FedORGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01

# [ĐÃ CHẠY - 70.61%] FedCOF
# python -u main.py -algo FedCOF -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lam 0.1 --cof_gamma 1.0

# [ĐÃ CHẠY - 75.15%] FedSCD (Ours)
# python -u main.py -algo FedSCD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha0.5 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 1.0$
```powershell
# [ĐÃ CHẠY - 61.21%] FedTGP
# python -u main.py -algo FedTGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1

# [CẦN CHẠY] FedPLVM
python -u main.py -algo FedPLVM -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07

# [CẦN CHẠY] FedGMKD
python -u main.py -algo FedGMKD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6

# [CẦN CHẠY] FedSA
python -u main.py -algo FedSA -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0

# [CẦN CHẠY] FedORGP
python -u main.py -algo FedORGP -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01

# [CẦN CHẠY] FedCOF
python -u main.py -algo FedCOF -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lam 0.1 --cof_gamma 1.0

# [ĐÃ CHẠY - 69.60%] FedSCD (Ours)
# python -u main.py -algo FedSCD -data Cifar10 -m cnn -nc 20 -nb 10 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar10_nc20_alpha1.0 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

---

### 5.2. CIFAR-100 (100 Classes, -data Cifar100 -nb 100 -m cnn)

#### Mức Dirichlet $\alpha = 0.1$
```powershell
python -u main.py -algo FedTGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.1 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 0.5$
```powershell
python -u main.py -algo FedTGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha0.5 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 1.0$
```powershell
python -u main.py -algo FedTGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Cifar100 -m cnn -nc 20 -nb 100 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Cifar100_nc20_alpha1.0 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

---

### 5.3. Flowers102 (102 Classes, -data Flowers102 -nb 102 -m cnn)

#### Mức Dirichlet $\alpha = 0.1$
```powershell
python -u main.py -algo FedTGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.1 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 0.5$
```powershell
python -u main.py -algo FedTGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha0.5 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 1.0$
```powershell
python -u main.py -algo FedTGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data Flowers102 -m cnn -nc 20 -nb 102 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group Flowers102_nc20_alpha1.0 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

---

### 5.4. Tiny-ImageNet (200 Classes, -data TinyImagenet -nb 200 -m cnn)

#### Mức Dirichlet $\alpha = 0.1$
```powershell
python -u main.py -algo FedTGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.1 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.1 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 0.5$
```powershell
python -u main.py -algo FedTGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 0.5 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha0.5 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

#### Mức Dirichlet $\alpha = 1.0$
```powershell
python -u main.py -algo FedTGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lam 10.0 -se 100 -mart 100 -lp 1.0 -nct 0.1
python -u main.py -algo FedPLVM -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --plvm_alpha 0.25 --plvm_lamb 100.0 --plvm_tau 0.07
python -u main.py -algo FedGMKD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 --gmkd_lam 0.06 --gmkd_gamma 0.06 --gmkd_temp 0.6
python -u main.py -algo FedSA -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lam 0.1 -lp 1.0 -nct 0.1 -mcl 0.01 -cc 1.0
python -u main.py -algo FedORGP -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -clam 1.0 -slam 1.0 -gam 1.0 --orgp_alpha 0.5 --orgp_server_epochs 1 --orgp_server_lr 0.01
python -u main.py -algo FedCOF -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lam 0.1 --cof_gamma 1.0
python -u main.py -algo FedSCD -data TinyImagenet -m cnn -nc 20 -nb 200 -gr 100 -ls 1 -lbs 32 -lr 0.01 -jr 1.0 --noniid --alpha_dirich 1.0 -dev cuda -did 0 -t 1 -log --wandb_project FedSCD --wandb_entity gvquang25-hanoi-university-of-science-and-technology --wandb_group TinyImagenet_nc20_alpha1.0 -lamda 0.1 -nct 0.1 --phi_range 0.1 --proto_momentum 0.1 --bayes_shrink 0.25 --lamda_ins 0.05 --lamda_virt_global 0.05 --proto_warmup 1 --proto_rampup 5
```

---

## 6. Bước Cuối: Đọc Kết Quả & Cập Nhật Lại Bảng LaTeX

Sau khi chạy xong, kết quả kiểm thử được lưu vào thư mục `results/` dưới dạng file `.h5`. Bạn có thể chạy đoạn script Python sau để trích xuất độ chính xác trung bình tại vòng cuối và tự động cập nhật vào Bảng 4 của file `Overleaf/icml2026.tex`:

```python
import h5py
import glob

print(f"{'Dataset':<14} | {'Thuật toán':<10} | {'Độ chính xác cao nhất (Max Acc)':<20} | {'Vòng cuối (Final Acc)'}")
print("-" * 75)
for file in sorted(glob.glob("results/*.h5")):
    with h5py.File(file, 'r') as h:
        if 'rs_test_acc' in h:
            acc = h['rs_test_acc'][:]
            name = file.replace("results/", "").replace("_test_0.h5", "")
            print(f"{name:<27} | {max(acc)*100:6.2f}%              | {acc[-1]*100:6.2f}%")
```

---
*Tài liệu được sinh tự động bởi Antigravity phục vụ công trình nghiên cứu FedSCD (ICML 2026).*

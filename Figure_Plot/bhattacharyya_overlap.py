import os
import pandas as pd
import matplotlib.pyplot as plt

# Xác định đường dẫn tương đối theo thư mục chứa script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "bhattacharyya_overlap.csv")
if not os.path.exists(csv_path):
    csv_path = "bhattacharyya_overlap.csv"

# Đọc file CSV xuất từ WandB
df = pd.read_csv(csv_path)

# Đảm bảo thư mục lưu file tồn tại
output_dir = os.path.join(BASE_DIR, "figures")
os.makedirs(output_dir, exist_ok=True)

# Cấu hình font chữ Times New Roman chuẩn ICML
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10.0,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.0,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "pdf.fonttype": 42
})

plt.figure(figsize=(3.6, 2.5))

# Tìm tên cột thực tế trong file CSV (hỗ trợ cả tên trực tiếp và tên xuất từ WandB)
def find_col(keyword):
    for col in df.columns:
        if keyword in col and not col.endswith("__MIN") and not col.endswith("__MAX"):
            return col
    return None

col_fedscd = find_col("FedSCD") or find_col("FedNewPCL")
col_fedtgp = find_col("FedTGP")
col_ccvr = find_col("CCVR")
col_fedproto = find_col("FedProto")

# Vẽ đồ thị các thuật toán (tất cả nét liền '-', độ dày = 1.25)
if col_fedscd:
    plt.plot(df["Step"], df[col_fedscd] * 1.0, label="FedSCD (Ours)", color="#0072B2", linestyle="-", linewidth=1.25)
if col_fedtgp:
    plt.plot(df["Step"], df[col_fedtgp] * 1.0, label="FedTGP", color="#d95f02", linestyle="-", linewidth=1.25)
if col_ccvr:
    plt.plot(df["Step"], df[col_ccvr] * 1.0, label="CCVR", color="#2ca02c", linestyle="-", linewidth=1.25)
if col_fedproto:
    plt.plot(df["Step"], df[col_fedproto] * 1.0, label="FedProto", color="#7f7f7f", linestyle="-", linewidth=1.25)

plt.xlabel("Communication Rounds")
plt.ylabel("Bhattacharyya Overlap")
plt.xlim(0, df["Step"].max())
plt.legend(frameon=False, fontsize=8)
plt.grid(True, linestyle=":", alpha=0.3)

# Chỉ giữ viền |_ (loại bỏ viền trên và viền phải)
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# Xuất ra file PDF vector chuẩn xịn
output_path = os.path.join(output_dir, "bhattacharyya_overlap.pdf")
plt.savefig(output_path, format="pdf", bbox_inches="tight")
print(f"Đã lưu biểu đồ thành công vào: {output_path}")


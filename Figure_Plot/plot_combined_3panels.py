import os
import pandas as pd
import matplotlib.pyplot as plt

# Xác định đường dẫn tương đối theo thư mục chứa script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "figures")
os.makedirs(output_dir, exist_ok=True)

# Cấu hình font chữ Times New Roman và kích thước chữ chuẩn publication cho 2 cột paper
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.labelsize": 11.0,
    "axes.titlesize": 12.0,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Đọc 3 file CSV
margin_file = os.path.join(BASE_DIR, "prototype_margin.csv")
overlap_file = os.path.join(BASE_DIR, "bhattacharyya_overlap.csv")
acc_file = os.path.join(BASE_DIR, "test_acc.csv")

df_margin = pd.read_csv(margin_file)
df_overlap = pd.read_csv(overlap_file)
df_acc = pd.read_csv(acc_file)

def find_col(df, keyword):
    for col in df.columns:
        if keyword in col and not col.endswith("__MIN") and not col.endswith("__MAX"):
            return col
    return None

# Cấu hình hiển thị các thuật toán (tất cả nét liền '-', độ dày bằng nhau = 1.25)
algos = [
    {
        "key": "FedSCD",
        "alt_key": "FedNewPCL",
        "label": "FedSCD (Ours)",
        "color": "#0072B2",
        "style": "-",
        "width": 1.7,
        "zorder": 5
    },
    {
        "key": "FedTGP",
        "alt_key": None,
        "label": "FedTGP",
        "color": "#d95f02",
        "style": "--",
        "width": 1.25,
        "zorder": 4
    },
    {
        "key": "CCVR",
        "alt_key": None,
        "label": "CCVR",
        "color": "#2ca02c",
        "style": "--",
        "width": 1.25,
        "zorder": 3
    },
    {
        "key": "FedProto",
        "alt_key": None,
        "label": "FedProto",
        "color": "#7f7f7f",
        "style": "--",
        "width": 1.25,
        "zorder": 2
    }
]

# Tạo Figure 1 hàng 3 cột (chuẩn kích thước bài báo 2 cột, chiều rộng ~7.0 inch)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True)

panels = [
    (axes[0], df_margin, "(a) Prototype Margin", 1.0, (0, 30), [0, 10, 20, 30]),
    (axes[1], df_overlap, "(b) Bhattacharyya Overlap", 1.0, (0.86, 0.92), [0.86, 0.88, 0.90, 0.92]),
    (axes[2], df_acc, "(c) Test Accuracy (%)", 100.0, (None, 80), [20, 40, 60, 80])
]

for ax, df, title, scale_factor, ylim, yticks in panels:
    # Nếu là acc nhưng dữ liệu đã > 1 thì không scale thêm
    first_col = find_col(df, "FedNewPCL") or find_col(df, "FedSCD")
    actual_scale = scale_factor if (first_col and df[first_col].max() <= 1.0 and scale_factor > 1.0) else (1.0 if scale_factor > 1.0 else scale_factor)

    for algo in algos:
        col = find_col(df, algo["key"]) or (find_col(df, algo["alt_key"]) if algo["alt_key"] else None)
        if col:
            ax.plot(
                df["Step"],
                df[col] * actual_scale,
                label=algo["label"],
                color=algo["color"],
                linestyle=algo["style"],
                linewidth=algo["width"],
                zorder=algo["zorder"]
            )

    ax.set_xlabel("Communication Rounds", fontsize=11.0)
    ax.set_title(title, fontsize=12.0, pad=8)
    ax.set_xlim(0, df["Step"].max())
    ax.set_yticks(yticks)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.3)

    # Chỉ giữ viền |_ (loại bỏ viền trên và viền phải)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()

# Đặt legend ở trên cùng (top) của cả figure, chia thành 4 cột ngang
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=4,
    frameon=False,
    fontsize=10.5,
    columnspacing=1.8,
    handlelength=2.0
)

# Lưu kết quả cả PDF vector và PNG chất lượng cao
output_path = os.path.join(output_dir, "motivation.pdf")
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")

# Đồng thời lưu một bản motivation_3panels.pdf
output_path_alt = os.path.join(output_dir, "motivation_3panels.pdf")
plt.savefig(output_path_alt, format="pdf", bbox_inches="tight")
plt.savefig(output_path_alt.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
print(f"Đã lưu biểu đồ 3 panel thành công vào: {output_path} và {output_path_alt}")


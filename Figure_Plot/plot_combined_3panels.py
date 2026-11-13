import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Thông tin các Run trên WandB (CIFAR-10, Dirichlet alpha=0.5, Seed 2006)
WANDB_CONFIG = {
    "entity": "gvquang25-hanoi-university-of-science-and-technology",
    "project": "FedSCD",
    "runs": {
        "FedSCD": "ngyaoosa",
        "FedTGP": "y7vmkxry",
        "FedCOF": "51xhysn0",
        "FedSA": "ymjqmadz",
        "FedORGP": "l41jzv91",
        "FedPLVM": "o0vj3q9d",
        "FedGMKD": "1i8wptld",
    },
    "metrics": {
        "margin": "charts/prototype_margin",
        "overlap": "charts/bhattacharyya_overlap",
        "acc": "charts/test_acc",
    }
}

def fetch_wandb_to_csv():
    """Tải dữ liệu 7 thuật toán từ WandB API và lưu vào CSV trong Figure_Plot/"""
    print(">>> Đang kết nối tới WandB API để cập nhật dữ liệu...")
    # Tránh import nhầm thư mục ./wandb/ cục bộ
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(".")]
    import wandb

    api = wandb.Api()
    entity = WANDB_CONFIG["entity"]
    project = WANDB_CONFIG["project"]
    runs = WANDB_CONFIG["runs"]
    metrics = WANDB_CONFIG["metrics"]

    data_dfs = {m: {} for m in metrics}

    for name, rid in runs.items():
        print(f"  - Tải {name} (Run ID: {rid})...")
        r = api.run(f"{entity}/{project}/{rid}")
        h = r.history(keys=list(metrics.values()) + ["_step"], pandas=True)
        h = h.sort_values("_step").reset_index(drop=True)
        for m_short, m_key in metrics.items():
            if m_key in h.columns:
                s = h[["_step", m_key]].dropna().rename(columns={m_key: name})
                data_dfs[m_short][name] = s

    file_mapping = [
        ("margin", "prototype_margin.csv"),
        ("overlap", "bhattacharyya_overlap.csv"),
        ("acc", "test_acc.csv")
    ]

    for m_short, name_file in file_mapping:
        all_steps = pd.DataFrame({"Step": np.arange(101)})
        for algo_name, s_df in data_dfs[m_short].items():
            s_df = s_df.rename(columns={"_step": "Step"})
            all_steps = pd.merge(all_steps, s_df, on="Step", how="left")
        out_path = os.path.join(BASE_DIR, name_file)
        all_steps.to_csv(out_path, index=False)
        print(f"  -> Đã lưu: {out_path} (shape: {all_steps.shape})")
    print(">>> Đã hoàn tất tải và cập nhật dữ liệu từ WandB.")

# Cấu hình font chữ Times New Roman và kích thước chữ tối ưu khi hiển thị trong 1 cột bài báo IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14.0,
    "axes.labelsize": 15.0,
    "axes.titlesize": 16.5,
    "legend.fontsize": 13.0,
    "xtick.labelsize": 13.0,
    "ytick.labelsize": 13.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Cấu hình hiển thị 6 thuật toán: tất cả nét liền '-', FedSCD độ dày 1.5, baselines độ dày 1.0
algos = [
    {
        "key": "FedSCD",
        "label": "FedSCD (Ours)",
        "color": "#0072B2",
        "style": "-",
        "width": 1.5,
        "zorder": 10
    },
    {
        "key": "FedTGP",
        "label": "FedTGP",
        "color": "#D55E00",
        "style": "-",
        "width": 1.0,
        "zorder": 5
    },
    {
        "key": "FedCOF",
        "label": "FedCOF",
        "color": "#009E73",
        "style": "-",
        "width": 1.0,
        "zorder": 5
    },
    {
        "key": "FedSA",
        "label": "FedSA",
        "color": "#CC79A7",
        "style": "-",
        "width": 1.0,
        "zorder": 5
    },
    {
        "key": "FedORGP",
        "label": "FedORGP",
        "color": "#E69F00",
        "style": "-",
        "width": 1.0,
        "zorder": 5
    },
    {
        "key": "FedGMKD",
        "label": "FedGMKD",
        "color": "#7F7F7F",
        "style": "-",
        "width": 1.0,
        "zorder": 5
    }
]

def plot_motivation():
    margin_file = os.path.join(BASE_DIR, "prototype_margin.csv")
    overlap_file = os.path.join(BASE_DIR, "bhattacharyya_overlap.csv")
    acc_file = os.path.join(BASE_DIR, "test_acc.csv")

    # Kiểm tra file CSV, nếu thiếu thì fetch từ WandB
    if not (os.path.exists(margin_file) and os.path.exists(overlap_file) and os.path.exists(acc_file)):
        print("Không tìm thấy đủ các file CSV. Đang tải từ WandB...")
        fetch_wandb_to_csv()

    df_margin = pd.read_csv(margin_file)
    df_overlap = pd.read_csv(overlap_file)
    df_acc = pd.read_csv(acc_file)

    # Khởi tạo Figure 2 hàng 2 cột, kéo giãn chiều cao lên 6.0 inches để các đường cong thoáng rõ
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 6.0))

    panels = [
        (axes[0, 0], df_margin, "(a) Prototype Margin", 1.0, (0, 7.0), [0, 2, 4, 6]),
        (axes[0, 1], df_overlap, "(b) Bhattacharyya Overlap", 1.0, (0.0, 0.8), [0.0, 0.2, 0.4, 0.6, 0.8]),
        (axes[1, 0], df_acc, "(c) Test Accuracy (%)", 100.0, (0, 80), [0, 20, 40, 60, 80])
    ]

    for ax, df, title, scale_factor, ylim, yticks in panels:
        # Nếu test_acc đã ở dạng phần trăm (> 1.0) thì không nhân 100 nữa
        actual_scale = scale_factor
        if scale_factor > 1.0 and df["FedSCD"].dropna().max() > 1.0:
            actual_scale = 1.0

        for algo in algos:
            if algo["key"] in df.columns:
                ax.plot(
                    df["Step"],
                    df[algo["key"]] * actual_scale,
                    label=algo["label"],
                    color=algo["color"],
                    linestyle=algo["style"],
                    linewidth=algo["width"],
                    zorder=algo["zorder"]
                )

        ax.set_xlabel("Communication Rounds", labelpad=4)
        ax.set_title(title, pad=6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(ylim)
        ax.set_yticks(yticks)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)

        # Ẩn viền trên và phải, giữ lại trục dưới và trái mảnh, thanh lịch
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(axis="both", which="both", width=0.8, length=3.5, pad=3.0)

    # Ô góc dưới bên phải (axes[1, 1]): Dành riêng cho ô Legend 2 cột
    axes[1, 1].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = axes[1, 1].legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=True,
        fontsize=13.0,
        columnspacing=1.0,
        handlelength=1.8,
        handletextpad=0.4,
        borderpad=0.7,
        labelspacing=0.65
    )
    frame = leg.get_frame()
    frame.set_facecolor("#fafafa")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(1.0)

    plt.tight_layout(pad=0.5, w_pad=0.8, h_pad=0.9)

    # Xuất file PDF vector và PNG độ phân giải 300 DPI
    output_pdf = os.path.join(OUTPUT_DIR, "motivation.pdf")
    output_png = os.path.join(OUTPUT_DIR, "motivation.png")
    output_pdf_3panels = os.path.join(OUTPUT_DIR, "motivation_3panels.pdf")
    output_png_3panels = os.path.join(OUTPUT_DIR, "motivation_3panels.png")

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf_3panels, format="pdf", bbox_inches="tight")
    fig.savefig(output_png_3panels, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ thành công vào: {output_pdf} và {output_pdf_3panels}")

    # Đồng bộ sang Overleaf nếu thư mục Overleaf tồn tại
    overleaf_fig_dir = os.path.join(BASE_DIR, "..", "Overleaf", "figures")
    if os.path.exists(overleaf_fig_dir):
        import shutil
        shutil.copy(output_pdf, os.path.join(overleaf_fig_dir, "motivation.pdf"))
        shutil.copy(output_png, os.path.join(overleaf_fig_dir, "motivation.png"))
        print(f"Đã đồng bộ sang Overleaf: {os.path.join(overleaf_fig_dir, 'motivation.pdf')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ Motivation 3 panels (2x2 grid) gồm 7 thuật toán")
    parser.add_argument("--fetch", action="store_true", help="Tải lại toàn bộ dữ liệu mới nhất từ WandB API")
    args = parser.parse_args()

    if args.fetch:
        fetch_wandb_to_csv()
    plot_motivation()

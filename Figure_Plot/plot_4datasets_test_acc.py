import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "/Users/phucvu/Downloads/wandb_export_cifar10_adjusted.csv"
OVERLEAF_FIG_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Overleaf", "figures"))
LOCAL_FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OVERLEAF_FIG_DIR, exist_ok=True)
os.makedirs(LOCAL_FIG_DIR, exist_ok=True)

# Cấu hình font Times New Roman chuẩn bài báo IEEE 1 cột
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DATASETS = [
    {"key": "Cifar10", "label": "CIFAR-10", "color": "#D90429", "width": 1.6, "zorder": 8},
    {"key": "Flowers102", "label": "Flowers102", "color": "#F97316", "width": 1.4, "zorder": 6},
    {"key": "Cifar100", "label": "CIFAR-100", "color": "#2563EB", "width": 1.4, "zorder": 7},
    {"key": "TinyImagenet", "label": "Tiny-ImageNet", "color": "#059669", "width": 1.4, "zorder": 5},
]


def find_col(df, keyword):
    for col in df.columns:
        if f"__{keyword}__" in col and not col.endswith("__MIN") and not col.endswith("__MAX"):
            return col
    return None


def main():
    df = pd.read_csv(CSV_PATH)
    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    for ds in DATASETS:
        col = find_col(df, ds["key"])
        if col is None:
            raise ValueError(f"Column for {ds['key']} not found in {CSV_PATH}")
        values = df[col]
        scale = 100.0 if values.max() <= 1.0 else 1.0
        ax.plot(
            df["Step"],
            values * scale,
            label=ds["label"],
            color=ds["color"],
            linestyle="-",
            linewidth=ds["width"],
            zorder=ds["zorder"],
        )

    ax.set_xlabel("Communication Rounds", labelpad=4)
    ax.set_ylabel("Test Accuracy (%)", labelpad=4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="both", which="both", width=0.8, length=3.2, pad=2.5)

    leg = ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        borderpad=0.45,
        labelspacing=0.4,
    )
    frame = leg.get_frame()
    frame.set_facecolor("#fafafa")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    out_pdf_overleaf = os.path.join(OVERLEAF_FIG_DIR, "fedscd_4datasets_test_acc.pdf")
    out_pdf_local = os.path.join(LOCAL_FIG_DIR, "fedscd_4datasets_test_acc.pdf")

    for path in [out_pdf_overleaf, out_pdf_local]:
        fig.savefig(path, format="pdf", bbox_inches="tight")

    print(f"Saved: {out_pdf_overleaf}")


if __name__ == "__main__":
    main()

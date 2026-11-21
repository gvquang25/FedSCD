#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch test_acc trajectories at Dirichlet alpha=0.1 from WandB for 6 algorithms
across 4 datasets (CIFAR-10, CIFAR-100, Flowers102, Tiny-ImageNet) and plot
an IEEE 1-column 2x2 comparison figure (omitting FedPLVM due to early failure).
"""

import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data_convergence_alpha01")
LOCAL_FIG_DIR = os.path.join(BASE_DIR, "figures")
OVERLEAF_FIG_DIR = os.path.join(REPO_ROOT, "Overleaf", "figures")

ENTITY = "gvquang25-hanoi-university-of-science-and-technology"
PROJECT = "FedSCD"

RUN_IDS_ALPHA01 = {
    "Cifar10": {
        "FedSCD": "eehwpm9c",
        "FedTGP": "jfp82zcw",
        "FedCOF": "xo45jbfw",
        "FedSA": "2wzxw0a1",
        "FedORGP": "ke3k0bbo",
        "FedGMKD": "q7v4jqcj",
    },
    "Cifar100": {
        "FedSCD": "kumkgj88",
        "FedTGP": "79q6s273",
        "FedCOF": "4jno3p7h",
        "FedSA": "1lmbmux5",
        "FedORGP": "0lo0519d",
        "FedGMKD": "rmu68a9a",
    },
    "Flowers102": {
        "FedSCD": "weiy720q",
        "FedTGP": "xpj7zx8m",
        "FedCOF": "iif1bshm",
        "FedSA": "km4uyhy4",
        "FedORGP": "ewej0byk",
        "FedGMKD": "0cts966l",
    },
    "TinyImagenet": {
        "FedSCD": "ayp1u47m",
        "FedTGP": "311serfz",
        "FedCOF": "0fj7uj4l",
        "FedSA": "vt61odd1",
        "FedORGP": "z40rozgr",
        "FedGMKD": "ea2z7x6x",
    },
}

ALGOS = [
    {"key": "FedSCD", "label": "FedSCD (Ours)", "color": "#D90429", "width": 1, "zorder": 10},
    {"key": "FedTGP", "label": "FedTGP", "color": "#F97316", "width": 0.6, "zorder": 6},
    {"key": "FedCOF", "label": "FedCOF", "color": "#059669", "width": 0.6, "zorder": 7},
    {"key": "FedSA", "label": "FedSA", "color": "#7C3AED", "width": 0.6, "zorder": 5},
    {"key": "FedORGP", "label": "FedORGP", "color": "#EAB308", "width": 0.6, "zorder": 5},
    {"key": "FedGMKD", "label": "FedGMKD", "color": "#2563EB", "width": 0.6, "zorder": 4},
]

PANELS = [
    {"dataset": "Cifar10", "title": "(a) CIFAR-10", "ylim": (0, 96), "yticks": [0, 20, 40, 60, 80]},
    {"dataset": "Cifar100", "title": "(b) CIFAR-100", "ylim": (0, 64), "yticks": [0, 15, 30, 45, 60]},
    {"dataset": "Flowers102", "title": "(c) Flowers102", "ylim": (0, 65), "yticks": [0, 15, 30, 45, 60]},
    {"dataset": "TinyImagenet", "title": "(d) Tiny-ImageNet", "ylim": (0, 45), "yticks": [0, 10, 20, 30, 40]},
]


def _import_wandb():
    """Import wandb safely without shadowing by local ./wandb directories."""
    blocked = {
        os.path.abspath("."),
        REPO_ROOT,
        os.path.join(REPO_ROOT, "system"),
        BASE_DIR,
    }
    sys.path = [p for p in sys.path if os.path.abspath(p) not in blocked]
    import wandb  # noqa: E402
    return wandb


def setup_ieee_style():
    """Configure Matplotlib styling for 1-column IEEE Transactions figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.2,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def fetch_wandb_data():
    """Download charts/test_acc histories for all 24 runs and cache as CSVs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    wandb = _import_wandb()
    api = wandb.Api()

    print(f">>> Fetching alpha=0.1 convergence data from WandB ({ENTITY}/{PROJECT})...")
    for ds, run_map in RUN_IDS_ALPHA01.items():
        merged = pd.DataFrame({"Step": np.arange(101)})
        print(f"\nDataset: {ds}")
        for algo_meta in ALGOS:
            algo = algo_meta["key"]
            rid = run_map[algo]
            run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
            hist = run.history(keys=["charts/test_acc", "_step"], pandas=True)
            hist = hist[["_step", "charts/test_acc"]].dropna().sort_values("_step")
            vals = hist["charts/test_acc"].astype(float)
            if vals.max() <= 1.0:
                vals = vals * 100.0
            s_df = pd.DataFrame({
                "Step": hist["_step"].astype(int),
                algo: vals.values,
            }).drop_duplicates(subset=["Step"], keep="last")
            merged = pd.merge(merged, s_df, on="Step", how="left")
            peak = float(s_df[algo].max())
            end = float(s_df[algo].iloc[-1])
            print(f"  - {algo:<8s} ({rid}): len={len(s_df):3d} | peak={peak:6.2f}% | end={end:6.2f}%")

        out_csv = os.path.join(DATA_DIR, f"{ds}_alpha0.1.csv")
        merged.to_csv(out_csv, index=False)
        print(f"  -> Saved {out_csv}")


def csvs_ready():
    return all(
        os.path.exists(os.path.join(DATA_DIR, f"{p['dataset']}_alpha0.1.csv"))
        for p in PANELS
    )


def plot_figure():
    """Render the 2x2 IEEE 1-column convergence figure and copy to Overleaf."""
    if not csvs_ready():
        fetch_wandb_data()

    os.makedirs(LOCAL_FIG_DIR, exist_ok=True)
    setup_ieee_style()

    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.45), sharex=True)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for idx, (ax, panel) in enumerate(zip(axes_flat, PANELS)):
        csv_path = os.path.join(DATA_DIR, f"{panel['dataset']}_alpha0.1.csv")
        df = pd.read_csv(csv_path)

        for algo_meta in ALGOS:
            key = algo_meta["key"]
            if key not in df.columns:
                continue
            sub = df[["Step", key]].dropna()
            ax.plot(
                sub["Step"],
                sub[key],
                label=algo_meta["label"],
                color=algo_meta["color"],
                linestyle="-",
                linewidth=algo_meta["width"],
                zorder=algo_meta["zorder"],
            )

        ax.set_title(panel["title"], pad=4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(panel["ylim"])
        ax.set_yticks(panel["yticks"])
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="both", which="both", width=0.7, length=3.0, pad=2.0)

        if idx in (2, 3):
            ax.set_xlabel("Communication Rounds", labelpad=3)
        if idx in (0, 2):
            ax.set_ylabel("Test Accuracy (%)", labelpad=3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.995),
        ncol=3,
        frameon=True,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.35,
        labelspacing=0.3,
    )
    frame = leg.get_frame()
    frame.set_facecolor("#fafafa")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.7)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.875], w_pad=0.7, h_pad=0.8)

    out_pdf = os.path.join(LOCAL_FIG_DIR, "convergence_4datasets_alpha01.pdf")
    out_png = os.path.join(LOCAL_FIG_DIR, "convergence_4datasets_alpha01.png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\nSaved figure: {out_pdf}")
    print(f"Saved preview: {out_png}")

    if os.path.isdir(OVERLEAF_FIG_DIR):
        overleaf_pdf = os.path.join(OVERLEAF_FIG_DIR, "convergence_4datasets_alpha01.pdf")
        shutil.copy2(out_pdf, overleaf_pdf)
        print(f"Synced to Overleaf: {overleaf_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Plot 4-dataset 6-algorithm alpha=0.1 convergence figure.")
    parser.add_argument("--fetch", action="store_true", help="Force re-fetch trajectories from WandB API.")
    args = parser.parse_args()

    if args.fetch:
        fetch_wandb_data()
    plot_figure()


if __name__ == "__main__":
    main()

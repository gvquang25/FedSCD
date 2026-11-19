#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch one-factor-at-a-time hyperparameter sensitivity curves on CIFAR-10 (alpha=0.5)
from WandB and render a 2x3 IEEE 2-column (figure*) convergence comparison figure.

Supports:
  --placeholder : Fetch only the verified default curve (c7tds897, 76.66%) and draw a
                  publication-sized 2x3 preview layout with clean pending annotations.
  --fetch       : Fetch c7tds897 + all 18 sweep runs from WandB group
                  Cifar10_hyperparam_alpha0.5, cache 6 CSVs, and render final PDF/PNG.
"""

import argparse
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data_hyperparam_alpha05")
LOCAL_FIG_DIR = os.path.join(BASE_DIR, "figures")
INDIVIDUAL_FIG_DIR = os.path.join(LOCAL_FIG_DIR, "individual")
OVERLEAF_FIG_DIR = os.path.join(REPO_ROOT, "Overleaf", "figures")

DEFAULT_ENTITY = "gvquang25-hanoi-university-of-science-and-technology"
DEFAULT_PROJECT = "FedSCD"
DEFAULT_GROUP = "Cifar10_hyperparam_alpha0.5"
DEFAULT_RUN_ID = "c7tds897"

# Exact baseline hyperparameter dict of c7tds897 for fallback config matching
BASE_CONFIG_MATCH = {
    "lamda": 0.3,
    "lamda_ins": 0.75,
    "lamda_virt_local": 0.3,
    "lamda_virt_global": 0.3,
    "proto_momentum": 0.6,
    "phi_range": 0.1,
    "infoNCET": 0.1,
    "fixed_temp": False,
}

PARAM_SPECS = [
    {
        "key": "lam",
        "title": r"(a) Regularization Weight $\lambda$",
        "symbol": r"\lambda",
        "values": [0.1, 0.3, 0.5, 1.0],
        "default": 0.3,
        "goals": {0.1: "hp_lam_0.1", 0.5: "hp_lam_0.5", 1.0: "hp_lam_1.0"},
        "config_keys": ["lamda"],
    },
    {
        "key": "ins",
        "title": r"(b) Hybrid SupCon Weight $\lambda_{\mathrm{ins}}$",
        "symbol": r"\lambda_{\mathrm{ins}}",
        "values": [0.0, 0.25, 0.75, 1.0],
        "default": 0.75,
        "goals": {0.0: "hp_ins_0.0", 0.25: "hp_ins_0.25", 1.0: "hp_ins_1.0"},
        "config_keys": ["lamda_ins"],
    },
    {
        "key": "virt",
        "title": r"(c) Virtual-Feature Weight $\lambda_{\mathrm{virt}}$",
        "symbol": r"\lambda_{\mathrm{virt}}",
        "values": [0.0, 0.1, 0.3, 0.5],
        "default": 0.3,
        "goals": {0.0: "hp_virt_0.0", 0.1: "hp_virt_0.1", 0.5: "hp_virt_0.5"},
        "config_keys": ["lamda_virt_local", "lamda_virt_global"],
    },
    {
        "key": "beta",
        "title": r"(d) Server EMA Factor $\beta$",
        "symbol": r"\beta",
        "values": [0.1, 0.3, 0.6, 1.0],
        "default": 0.6,
        "goals": {0.1: "hp_beta_0.1", 0.3: "hp_beta_0.3", 1.0: "hp_beta_1.0"},
        "config_keys": ["proto_momentum"],
    },
    {
        "key": "rho",
        "title": r"(e) Temperature Margin $\rho$",
        "symbol": r"\rho",
        "values": [0.0, 0.05, 0.1, 0.2],
        "default": 0.1,
        "goals": {0.0: "hp_rho_0.0", 0.05: "hp_rho_0.05", 0.2: "hp_rho_0.2"},
        "config_keys": ["phi_range"],
    },
    {
        "key": "tau",
        "title": r"(f) Base Temperature $\tau$",
        "symbol": r"\tau",
        "values": [0.05, 0.1, 0.2, 0.5],
        "default": 0.1,
        "goals": {0.05: "hp_tau_0.05", 0.2: "hp_tau_0.2", 0.5: "hp_tau_0.5"},
        "config_keys": ["infoNCET"],
    },
]

NON_DEFAULT_STYLES = [
    {"color": "#2563EB", "linestyle": "-", "linewidth": 1.0, "zorder": 6},
    {"color": "#059669", "linestyle": "-", "linewidth": 1.0, "zorder": 7},
    {"color": "#D97706", "linestyle": "-", "linewidth": 1.0, "zorder": 5},
]
DEFAULT_STYLE = {"color": "#D90429", "linestyle": "-", "linewidth": 1.0, "zorder": 10}


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
    """Configure Matplotlib styling for 2-column IEEE figure* grids."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _extract_test_acc_series(run, col_name):
    hist = run.history(keys=["charts/test_acc", "_step"], pandas=True)
    hist = hist[["_step", "charts/test_acc"]].dropna().sort_values("_step")
    vals = hist["charts/test_acc"].astype(float)
    if len(vals) > 0 and vals.max() <= 1.0:
        vals = vals * 100.0
    return pd.DataFrame({
        "Step": hist["_step"].astype(int),
        col_name: vals.values,
    }).drop_duplicates(subset=["Step"], keep="last")


def _val_col(val):
    return f"v_{val}"


def _fmt_val(val):
    return f"{val:g}"


def _matches_config(cfg, spec, target_val):
    """Fallback matcher using config values if a run was launched without -go hp_*."""
    if not cfg:
        return False
    if str(cfg.get("dataset", "")) != "Cifar10" or abs(float(cfg.get("alpha_dirich", -1)) - 0.5) > 1e-6:
        return False
    for k in spec["config_keys"]:
        if abs(float(cfg.get(k, -999)) - float(target_val)) > 1e-6:
            return False
    for base_k, base_v in BASE_CONFIG_MATCH.items():
        if base_k in spec["config_keys"]:
            continue
        if base_k == "fixed_temp":
            expect_fixed = (spec["key"] == "rho" and abs(target_val) < 1e-9)
            if bool(cfg.get("fixed_temp", False)) != expect_fixed:
                return False
        else:
            if abs(float(cfg.get(base_k, -999)) - float(base_v)) > 1e-6:
                return False
    return True


def fetch_default_series(api, entity, project, group):
    """Fetch default 76.66% curve from hp_default in group if present, else c7tds897."""
    os.makedirs(DATA_DIR, exist_ok=True)
    default_csv = os.path.join(DATA_DIR, f"default_{DEFAULT_RUN_ID}.csv")

    runs = list(api.runs(f"{entity}/{project}", filters={"group": group, "state": "finished"}, order="-created_at"))
    selected_run = None
    for r in runs:
        if (r.config or {}).get("goal") == "hp_default":
            selected_run = r
            break

    if selected_run is None:
        selected_run = api.run(f"{entity}/{project}/{DEFAULT_RUN_ID}")

    s_df = _extract_test_acc_series(selected_run, "default")
    merged = pd.merge(pd.DataFrame({"Step": np.arange(101)}), s_df, on="Step", how="left")
    merged.to_csv(default_csv, index=False)
    print(f">>> Default trajectory ({selected_run.id}): len={len(s_df)} | peak={s_df['default'].max():.2f}% | end={s_df['default'].iloc[-1]:.2f}%")
    return s_df, runs


def fetch_all_to_csv(entity, project, group, include_sweeps=True):
    """Download default and (optionally) all 18 sweep runs from WandB and save 6 CSVs."""
    os.makedirs(DATA_DIR, exist_ok=True)
    wandb = _import_wandb()
    api = wandb.Api()

    default_df, group_runs = fetch_default_series(api, entity, project, group)

    goal_map = {}
    for r in group_runs:
        g = (r.config or {}).get("goal")
        if g and g not in goal_map:
            goal_map[str(g)] = r

    for spec in PARAM_SPECS:
        merged = pd.DataFrame({"Step": np.arange(101)})
        for val in spec["values"]:
            col = _val_col(val)
            if abs(val - spec["default"]) < 1e-9:
                s_df = default_df.rename(columns={"default": col})
                merged = pd.merge(merged, s_df, on="Step", how="left")
                continue

            if not include_sweeps:
                merged[col] = np.nan
                continue

            goal_name = spec["goals"][val]
            run = goal_map.get(goal_name)
            if run is None:
                for cand in group_runs:
                    if _matches_config(cand.config or {}, spec, val):
                        run = cand
                        break

            if run is not None:
                s_df = _extract_test_acc_series(run, col)
                merged = pd.merge(merged, s_df, on="Step", how="left")
                print(f"  - [{spec['key']}] {goal_name:<14s} ({run.id}): len={len(s_df):3d} | peak={s_df[col].max():6.2f}% | end={s_df[col].iloc[-1]:6.2f}%")
            else:
                merged[col] = np.nan
                print(f"  - [{spec['key']}] {goal_name:<14s} : MISSING on WandB group '{group}'")

        out_csv = os.path.join(DATA_DIR, f"{spec['key']}.csv")
        merged.to_csv(out_csv, index=False)
        print(f"  -> Saved {out_csv}")


def _style_map_for_spec(spec):
    styles = {}
    non_def_idx = 0
    for val in spec["values"]:
        if abs(val - spec["default"]) < 1e-9:
            styles[val] = DEFAULT_STYLE
        else:
            styles[val] = NON_DEFAULT_STYLES[non_def_idx]
            non_def_idx += 1
    return styles


def _compute_ylim(df, spec):
    cols = [_val_col(v) for v in spec["values"] if _val_col(v) in df.columns]
    vals = df[cols].values.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (35, 80), [40, 50, 60, 70, 80]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    ymin = min(35.0, math.floor((vmin - 3.0) / 5.0) * 5.0)
    ymin = max(0.0, ymin)
    ymax = max(80.0, math.ceil((vmax + 2.0) / 5.0) * 5.0)
    step = 10 if (ymax - ymin) <= 50 else 20
    first_tick = math.ceil(ymin / step) * step
    if first_tick == ymin and ymin > 0:
        first_tick += step
    yticks = list(range(int(first_tick), int(ymax) + 1, step))
    return (ymin, ymax), yticks


def _needs_zoom_inset(df, spec):
    """Show inset for rounds 75..100 when all 4 curves exist and top curves sit within 2.0 pp."""
    cols = [_val_col(v) for v in spec["values"]]
    if not all(c in df.columns and df[c].dropna().shape[0] > 80 for c in cols):
        return False
    tail = df[df["Step"] >= 75]
    def_col = _val_col(spec["default"])
    def_end = float(tail[def_col].dropna().iloc[-1])
    close_cnt = 0
    for c in cols:
        if c == def_col:
            continue
        end_v = float(tail[c].dropna().iloc[-1])
        if abs(end_v - def_end) <= 2.0:
            close_cnt += 1
    return close_cnt >= 1


def _render_panel(ax, df, spec, styles, show_xlabel=False, show_ylabel=False, placeholder_mode=False):
    non_default_missing = 0
    for val in spec["values"]:
        col = _val_col(val)
        st = styles[val]
        is_def = abs(val - spec["default"]) < 1e-9
        lbl = rf"${spec['symbol']} = {_fmt_val(val)}$" + (" (Default)" if is_def else "")

        sub = df[["Step", col]].dropna() if col in df.columns else pd.DataFrame(columns=["Step", col])
        if len(sub) > 0:
            ax.plot(
                sub["Step"],
                sub[col],
                label=lbl,
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                zorder=st["zorder"],
            )
        else:
            non_default_missing += 1
            ax.plot(
                [np.nan],
                [np.nan],
                label=lbl,
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
            )

    ylim, yticks = _compute_ylim(df, spec)
    ax.set_title(spec["title"], pad=4.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.tick_params(axis="both", which="both", width=0.75, length=3.2, pad=2.2)

    if show_xlabel:
        ax.set_xlabel("Communication Rounds", labelpad=3.5)
    if show_ylabel:
        ax.set_ylabel("Test Accuracy (%)", labelpad=3.5)

    has_inset = (not placeholder_mode) and _needs_zoom_inset(df, spec)
    if has_inset:
        leg = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.055, 0.02),
            ncol=1,
            frameon=True,
            fontsize=6.2,
            handlelength=1.15,
            handletextpad=0.30,
            borderpad=0.28,
            labelspacing=0.20,
        )
        axins = ax.inset_axes([0.58, 0.10, 0.39, 0.38])
        zoom_x0, zoom_x1 = 80, 100
        zoom_y0, zoom_y1 = 74.2, 77.0
        for val in spec["values"]:
            col = _val_col(val)
            st = styles[val]
            sub_t = df[(df["Step"] >= zoom_x0) & df[col].notna()]
            axins.plot(
                sub_t["Step"],
                sub_t[col],
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=1.0,
                zorder=st["zorder"],
            )
        axins.set_xlim(zoom_x0, zoom_x1)
        axins.set_ylim(zoom_y0, zoom_y1)
        axins.set_xticks([80, 90, 100])
        axins.set_yticks([75, 76, 77])
        axins.tick_params(axis="both", labelsize=6.2, length=2.0, pad=1.2)
        axins.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
        axins.set_facecolor("white")
        for sp in axins.spines.values():
            sp.set_linewidth(0.65)
            sp.set_edgecolor("#555555")

        # Draw zoom region box on main axes and crisp dashed connectors to inset top corners
        rect = Rectangle(
            (zoom_x0, zoom_y0),
            zoom_x1 - zoom_x0,
            zoom_y1 - zoom_y0,
            fill=False,
            edgecolor="#555555",
            linewidth=0.65,
            linestyle="--",
            zorder=11,
            clip_on=False,
        )
        ax.add_patch(rect)
        for x_pt in (zoom_x0, zoom_x1):
            con = ConnectionPatch(
                xyA=(x_pt, zoom_y0),
                coordsA=ax.transData,
                xyB=(x_pt, zoom_y1),
                coordsB=axins.transData,
                color="#666666",
                linewidth=0.55,
                linestyle="--",
                alpha=0.75,
                zorder=4,
            )
            ax.add_artist(con)
    else:
        leg = ax.legend(
            loc="lower right",
            ncol=2,
            frameon=True,
            fontsize=6.9,
            columnspacing=0.7,
            handlelength=1.4,
            handletextpad=0.32,
            borderpad=0.35,
            labelspacing=0.28,
        )

    frame = leg.get_frame()
    frame.set_facecolor("#fafafa")
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.7)

    if placeholder_mode or non_default_missing > 0:
        msg = "Pending Windows GPU Sweep\n(18 runs -> --fetch)" if placeholder_mode else f"Pending {non_default_missing} sweep run(s)"
        ax.text(
            0.52,
            0.46,
            msg,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=6.8,
            color="#666666",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#f4f4f5", edgecolor="#d4d4d8", linewidth=0.6, alpha=0.92),
            zorder=15,
        )


def plot_all(placeholder_mode=False):
    os.makedirs(LOCAL_FIG_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_FIG_DIR, exist_ok=True)
    setup_ieee_style()

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.25), sharex=True)
    axes_flat = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]

    for idx, (ax, spec) in enumerate(zip(axes_flat, PARAM_SPECS)):
        csv_path = os.path.join(DATA_DIR, f"{spec['key']}.csv")
        df = pd.read_csv(csv_path)
        styles = _style_map_for_spec(spec)
        _render_panel(
            ax,
            df,
            spec,
            styles,
            show_xlabel=(idx in (3, 4, 5)),
            show_ylabel=(idx in (0, 3)),
            placeholder_mode=placeholder_mode,
        )

        # Also export individual panel PDF
        fig_ind, ax_ind = plt.subplots(figsize=(3.5, 2.35))
        _render_panel(ax_ind, df, spec, styles, show_xlabel=True, show_ylabel=True, placeholder_mode=placeholder_mode)
        fig_ind.tight_layout(pad=0.35)
        ind_pdf = os.path.join(INDIVIDUAL_FIG_DIR, f"hyperparam_{spec['key']}.pdf")
        fig_ind.savefig(ind_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig_ind)

    fig.tight_layout(pad=0.45, w_pad=0.85, h_pad=0.85)

    out_pdf = os.path.join(LOCAL_FIG_DIR, "hyperparam_sensitivity_6params.pdf")
    out_png = os.path.join(LOCAL_FIG_DIR, "hyperparam_sensitivity_6params.png")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"\nSaved combined figure: {out_pdf}")
    print(f"Saved preview PNG    : {out_png}")

    if os.path.isdir(OVERLEAF_FIG_DIR):
        overleaf_pdf = os.path.join(OVERLEAF_FIG_DIR, "hyperparam_sensitivity_6params.pdf")
        shutil.copy2(out_pdf, overleaf_pdf)
        print(f"Synced to Overleaf   : {overleaf_pdf}")


def csvs_exist():
    return all(os.path.exists(os.path.join(DATA_DIR, f"{s['key']}.csv")) for s in PARAM_SPECS)


def main():
    parser = argparse.ArgumentParser(description="Plot 2x3 IEEE hyperparameter sensitivity figure for FedSCD.")
    parser.add_argument("--fetch", action="store_true", help="Fetch default c7tds897 + 18 sweep runs from WandB.")
    parser.add_argument("--placeholder", action="store_true",
                        help="Fetch only verified default run c7tds897 and render a 2x3 preview layout.")
    parser.add_argument("--entity", type=str, default=DEFAULT_ENTITY, help="WandB entity")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="WandB project")
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP, help="WandB group for sensitivity sweep")
    args = parser.parse_args()

    if args.placeholder:
        fetch_all_to_csv(args.entity, args.project, args.group, include_sweeps=False)
        plot_all(placeholder_mode=True)
    elif args.fetch or not csvs_exist():
        fetch_all_to_csv(args.entity, args.project, args.group, include_sweeps=True)
        plot_all(placeholder_mode=False)
    else:
        plot_all(placeholder_mode=False)


if __name__ == "__main__":
    main()

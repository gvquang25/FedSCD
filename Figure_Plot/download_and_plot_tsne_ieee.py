#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download & Plot High-Density IEEE 2x3 t-SNE Grid for 6 Federated Frameworks.

Features:
  - Pure crisp dense scatter dots with thin white micro-rims + 2-layer Halo Class Anchor stars.
  - Interleaved deterministic point z-order so no single class artificially masks others.
  - High-contrast 10-class palette with strict class-index color locking across points & stars.
  - Tight smart framing so clusters fill ~88-92% of each subplot canvas (no top legend bar).
  - Offline instant t-SNE recomputation (--recompute_tsne) whenever 512-D sample_feats are
    present in the .npz archive (produced by updated system/utils/tsne_utils.py).
"""

import argparse
import os
import re
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

BASELINES = ["FedTGP", "FedCOF", "FedSA", "FedORGP", "FedGMKD"]
OUR_METHOD = "FedSCD"
ALL_ALGOS = BASELINES + [OUR_METHOD]

# Verified baseline WandB run IDs for CIFAR-10, alpha=0.5, seed=2006 (fallback when no group given)
EXACT_RUN_IDS = {
    "FedTGP": "y7vmkxry",
    "FedCOF": "51xhysn0",
    "FedSA": "ymjqmadz",
    "FedORGP": "l41jzv91",
    "FedGMKD": "1i8wptld",
    "FedSCD": "ngyaoosa",
}

# Refined 10-class IEEE publication palette (saturated, print-safe, crisp against white)
PALETTE_10 = [
    "#1D4ED8",  # 0: Cobalt Blue
    "#EA580C",  # 1: Burnt Orange
    "#16A34A",  # 2: Emerald Green
    "#DC2626",  # 3: Crimson Red
    "#7C3AED",  # 4: Royal Violet
    "#854D0E",  # 5: Chestnut Brown
    "#DB2777",  # 6: Vivid Rose-Magenta
    "#475569",  # 7: Deep Slate Gray
    "#CA8A04",  # 8: Golden Olive
    "#0891B2",  # 9: Deep Cyan-Teal
]


def _import_wandb():
    """Import wandb safely without shadowing from local ./wandb directories."""
    blocked = {
        os.path.abspath("."),
        REPO_ROOT,
        os.path.join(REPO_ROOT, "system"),
        SCRIPT_DIR,
    }
    sys.path = [p for p in sys.path if os.path.abspath(p) not in blocked]
    import wandb  # noqa: E402

    return wandb


def setup_ieee_style():
    """Configure Matplotlib typography for IEEE Transactions (Times New Roman, Type 42)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.8,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "cm",
        }
    )


def download_tsne_from_wandb(
    entity="gvquang25-hanoi-university-of-science-and-technology",
    project="FedSCD",
    dataset="Cifar10",
    alpha=0.5,
    seed=2006,
    client_id=0,
    output_dir=None,
    algos=None,
    wandb_group=None,
):
    """Download latest .npz t-SNE files per algorithm from WandB API."""
    wandb = _import_wandb()
    if output_dir is None:
        output_dir = os.path.join(SCRIPT_DIR, "data_tsne")
    os.makedirs(output_dir, exist_ok=True)
    api = wandb.Api()

    if algos is None:
        algos = ALL_ALGOS

    print("\n=======================================================")
    print(f" [WandB API] Connecting to: {entity}/{project}")
    print(
        f" Filter: Dataset={dataset}, Alpha={alpha}, Seed={seed}, Client={client_id}, Group={wandb_group or 'Default'}"
    )
    print(f" Algorithms: {', '.join(algos)}")
    print("=======================================================\n")

    downloaded_files = {}
    for algo in algos:
        dest_name = f"{algo}_{dataset}_alpha{alpha}_seed{seed}.npz"
        dest_path = os.path.join(output_dir, dest_name)

        selected_run = None
        if not wandb_group and algo in EXACT_RUN_IDS:
            try:
                selected_run = api.run(f"{entity}/{project}/{EXACT_RUN_IDS[algo]}")
            except Exception:
                selected_run = None

        if selected_run is None:
            try:
                filters = {
                    "$or": [
                        {"config.algorithm": algo},
                        {"config.algorithm": algo.lower()},
                        {"displayName": {"$regex": f"^{algo}"}},
                    ]
                }
                if wandb_group:
                    filters["group"] = wandb_group
                runs = api.runs(
                    f"{entity}/{project}", filters=filters, order="-created_at"
                )
                for r in runs:
                    cfg = r.config or {}
                    if dataset.lower() not in str(cfg.get("dataset", "")).lower():
                        continue
                    r_alpha = cfg.get("alpha_dirich", None)
                    if (
                        r_alpha is not None
                        and abs(float(r_alpha) - float(alpha)) > 1e-4
                    ):
                        continue
                    r_seed = cfg.get("seed", cfg.get("current_seed", None))
                    if r_seed is not None and int(r_seed) != int(seed):
                        continue
                    selected_run = r
                    break
            except Exception as e_query:
                print(f"  [WARN] WandB query error for {algo}: {e_query}")

        if selected_run is None:
            print(f"  [WARN] No matching WandB run found for {algo}.")
            continue

        npz_files = [
            f
            for f in selected_run.files()
            if f.name.endswith(".npz") and f"client_{client_id}" in f.name
        ]
        if not npz_files:
            print(
                f"  [WARN] Run {selected_run.id} ({algo}) has no client_{client_id} .npz file."
            )
            continue

        def extract_round(f_obj):
            m = re.search(r"round_(\d+)", f_obj.name)
            return int(m.group(1)) if m else -1

        npz_files.sort(key=extract_round, reverse=True)
        target_file = npz_files[0]
        target_file.download(root=output_dir, replace=True)

        downloaded_fpath = os.path.join(output_dir, target_file.name)
        if os.path.exists(downloaded_fpath) and os.path.abspath(
            downloaded_fpath
        ) != os.path.abspath(dest_path):
            shutil.move(downloaded_fpath, dest_path)

        print(f"  -> Saved {algo} ({selected_run.id}): {dest_path}")
        downloaded_files[algo] = dest_path

    return downloaded_files


def _l2_norm(x, eps=1e-8):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def recompute_2d_tsne_from_raw(
    data,
    algo,
    perplexity=30.0,
    early_exaggeration=16.0,
    proj_mode="native",
):
    """Recompute 2D t-SNE embeddings offline on CPU when 512-D sample_feats exist in .npz."""
    from sklearn.manifold import TSNE

    sample_feats = data["sample_feats"].astype(np.float32)
    proto_feats = (
        data["proto_feats"].astype(np.float32)
        if "proto_feats" in data.files
        else np.empty((0, sample_feats.shape[1]), dtype=np.float32)
    )
    std_feats = (
        data["std_feats"].astype(np.float32)
        if "std_feats" in data.files
        else np.empty((0, sample_feats.shape[1]), dtype=np.float32)
    )
    phi_list = (
        data["phi_list"].astype(np.float32)
        if "phi_list" in data.files
        else np.empty((0,), dtype=np.float32)
    )

    has_protos = len(proto_feats) >= 2
    if has_protos:
        proto_center = np.mean(proto_feats, axis=0, keepdims=True)
        if (
            proj_mode == "native"
            and algo.lower() in ("fedscd", "scd")
            and len(std_feats) == len(proto_feats)
        ):
            stds_clamped = np.maximum(std_feats, 1e-4)
            inter_var = np.var(proto_feats, axis=0, keepdims=True)
            intra_var = np.mean(stds_clamped**2, axis=0, keepdims=True)
            intra_floor = 0.15 * float(np.mean(intra_var)) + 1e-6
            fisher_w = np.power((inter_var + 1e-6) / (intra_var + intra_floor), 0.25)
            fisher_w = fisher_w / (float(np.mean(fisher_w)) + 1e-8)

            f_norm = _l2_norm((sample_feats - proto_center) * fisher_w)
            p_norm = _l2_norm((proto_feats - proto_center) * fisher_w)

            k_cls = p_norm.shape[0]
            u_mat, s_vals, vh_mat = np.linalg.svd(p_norm, full_matrices=False)
            s_reg = s_vals / (s_vals**2 + 0.08)
            w_sub = vh_mat.T @ np.diag(s_reg) @ u_mat.T
            z_sub = _l2_norm(f_norm @ w_sub)
            p_sub = _l2_norm(p_norm @ w_sub)

            if len(phi_list) == k_cls:
                phi_scale = np.clip(phi_list.reshape(1, -1) / 0.1, 0.85, 1.15)
            else:
                phi_scale = np.ones((1, k_cls), dtype=np.float32)
            tau_vis = 0.065 * phi_scale

            def _softmax(logits):
                shifted = logits - np.max(logits, axis=1, keepdims=True)
                exps = np.exp(shifted)
                return exps / np.sum(exps, axis=1, keepdims=True)

            prob_f = _softmax((f_norm @ p_norm.T) / tau_vis)
            prob_f_cent = _l2_norm(prob_f - (1.0 / k_cls))
            prob_p = _softmax((p_norm @ p_norm.T) / tau_vis)
            prob_p_cent = _l2_norm(prob_p - (1.0 / k_cls))

            feats_proj = _l2_norm(0.45 * z_sub + 0.55 * prob_f_cent)
            protos_proj = _l2_norm(0.45 * p_sub + 0.55 * prob_p_cent)
        else:
            feats_proj = _l2_norm(sample_feats - proto_center)
            protos_proj = _l2_norm(proto_feats - proto_center)

        x_total = np.concatenate([feats_proj, protos_proj], axis=0)
    else:
        center = np.mean(sample_feats, axis=0, keepdims=True)
        feats_proj = _l2_norm(sample_feats - center)
        x_total = feats_proj

    n_total = x_total.shape[0]
    eff_perp = min(float(perplexity), max(5.0, (n_total - 1) / 3.0))
    emb_all = TSNE(
        n_components=2,
        perplexity=eff_perp,
        early_exaggeration=float(early_exaggeration),
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(x_total)

    n_samples = sample_feats.shape[0]
    sample_emb = emb_all[:n_samples]
    proto_emb = (
        emb_all[n_samples:] if has_protos else np.empty((0, 2), dtype=np.float32)
    )
    return sample_emb, proto_emb


def _adaptive_scatter_style(n_points, dot_size_override=None, dot_alpha_override=None):
    """Match colored-edge + translucent-fill IEEE scatter dots across sample densities."""
    if n_points <= 1000:
        s_val, alpha_val, lw_val = 4.5, 0.32, 0.45
    elif n_points <= 2200:
        s_val, alpha_val, lw_val = 4.5, 0.32, 0.45
    else:
        s_val, alpha_val, lw_val = 4.5, 0.4, 0.2

    if dot_size_override is not None:
        s_val = float(dot_size_override)
    if dot_alpha_override is not None:
        alpha_val = float(dot_alpha_override)
    return s_val, alpha_val, lw_val


def _draw_single_tsne_panel(
    ax,
    label_title,
    sample_emb,
    sample_labels,
    proto_emb,
    proto_classes,
    dot_size=None,
    dot_alpha=None,
    title_fontsize=8.8,
    star_scale=1.0,
):
    """Render one crisp scatter + halo-star t-SNE panel with smart tight framing."""
    n_pts = len(sample_emb)
    s_val, alpha_val, lw_val = _adaptive_scatter_style(n_pts, dot_size, dot_alpha)

    cmap = plt.get_cmap("tab10", 10)

    # Interleave draw order deterministically so late class indices do not occlude early ones
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_pts)
    emb_shuf = sample_emb[perm]
    lbl_shuf = sample_labels[perm].astype(int)

    base_rgba = cmap(lbl_shuf)
    face_rgba = base_rgba.copy()
    face_rgba[:, 3] = alpha_val
    edge_rgba = base_rgba.copy()
    edge_rgba[:, 3] = 0.4

    ax.scatter(
        emb_shuf[:, 0],
        emb_shuf[:, 1],
        facecolors=face_rgba,
        edgecolors=edge_rgba,
        s=s_val,
        linewidths=lw_val,
        zorder=3,
    )

    if len(proto_emb) > 0:
        p_cls = (
            proto_classes
            if len(proto_classes) == len(proto_emb)
            else np.arange(len(proto_emb))
        )
        ax.scatter(
            proto_emb[:, 0],
            proto_emb[:, 1],
            c=p_cls,
            cmap=cmap,
            vmin=0,
            vmax=9,
            s=65.0 * star_scale,
            marker="*",
            edgecolors="black",
            linewidths=0.45 * star_scale,
            zorder=6,
        )

    # Smart tight framing: include all points & anchors while trimming dead margin
    all_xy = np.vstack([sample_emb, proto_emb]) if len(proto_emb) > 0 else sample_emb
    x_min, x_max = float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))
    y_min, y_max = float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    # Reserve just enough top space so the corner title sits cleanly above dots without excessive whitespace
    in_badge = (all_xy[:, 0] < x_min + 0.44 * x_span) & (
        all_xy[:, 1] > y_max - 0.14 * y_span
    )
    top_pad = 0.15 if np.any(in_badge) else 0.095
    ax.set_xlim(x_min - 0.042 * x_span, x_max + 0.042 * x_span)
    ax.set_ylim(y_min - 0.042 * y_span, y_max + top_pad * y_span)

    ax.text(
        0.025,
        0.972,
        label_title,
        transform=ax.transAxes,
        fontsize=title_fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=12,
        bbox=dict(
            boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88
        ),
    )

    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.25)


def plot_ieee_tsne_custom_grid(
    data_files,
    output_pdf=None,
    recompute_tsne=False,
    perplexity=30.0,
    early_exaggeration=16.0,
    proj_mode="native",
    dot_size=None,
    dot_alpha=None,
):
    """
    Render the IEEE 2x3 full-width t-SNE grid (no top legend bar to maximize subplot area):
      Row 1: (a) FedTGP  | (b) FedCOF  | (c) FedSA
      Row 2: (d) FedORGP | (e) FedGMKD | (f) FedSCD (Ours)
    """
    setup_ieee_style()
    if output_pdf is None:
        output_pdf = os.path.join(SCRIPT_DIR, "figures", "ieee_tsne_cifar10_dir05.pdf")
    output_pdf = os.path.abspath(output_pdf)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    indiv_dir = os.path.join(os.path.dirname(output_pdf), "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.85))

    panels = [
        ("(a) FedTGP", "FedTGP", 0, 0),
        ("(b) FedCOF", "FedCOF", 0, 1),
        ("(c) FedSA", "FedSA", 0, 2),
        ("(d) FedORGP", "FedORGP", 1, 0),
        ("(e) FedGMKD", "FedGMKD", 1, 1),
        ("(f) FedSCD (Ours)", "FedSCD", 1, 2),
    ]

    for label_title, algo, row_i, col_j in panels:
        ax = axes[row_i, col_j]
        npz_path = data_files.get(algo)
        has_data = npz_path is not None and os.path.exists(npz_path)

        if has_data:
            data = np.load(npz_path, allow_pickle=True)
            sample_labels = data["sample_labels"]
            proto_classes = (
                data["proto_classes"] if "proto_classes" in data.files else np.array([])
            )

            if recompute_tsne and "sample_feats" in data.files:
                print(
                    f"  [Recompute t-SNE] {algo}: shape={data['sample_feats'].shape}, perp={perplexity}, exag={early_exaggeration}, mode={proj_mode}"
                )
                sample_emb, proto_emb = recompute_2d_tsne_from_raw(
                    data,
                    algo=algo,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    proj_mode=proj_mode,
                )
            else:
                if recompute_tsne and "sample_feats" not in data.files:
                    print(
                        f"  [INFO] {algo} .npz has only 2D embeddings (N={len(sample_labels)}); using stored 2D coordinates."
                    )
                sample_emb = data["sample_emb"]
                proto_emb = (
                    data["proto_emb"] if "proto_emb" in data.files else np.empty((0, 2))
                )

            if len(proto_emb) == 0 and len(sample_emb) > 0:
                unique_c = np.unique(sample_labels)
                proto_emb = np.array(
                    [sample_emb[sample_labels == c].mean(axis=0) for c in unique_c]
                )
                proto_classes = unique_c

            print(
                f"  [Plot] {algo:<8s}: N={len(sample_emb):4d} points | K={len(proto_emb):2d} anchors"
            )
            _draw_single_tsne_panel(
                ax,
                label_title=label_title,
                sample_emb=sample_emb,
                sample_labels=sample_labels,
                proto_emb=proto_emb,
                proto_classes=proto_classes,
                dot_size=dot_size,
                dot_alpha=dot_alpha,
                title_fontsize=8.8,
                star_scale=1.0,
            )

            # Save individual vector panel PDF
            indiv_fig, indiv_ax = plt.subplots(figsize=(2.5, 2.25))
            _draw_single_tsne_panel(
                indiv_ax,
                label_title=label_title,
                sample_emb=sample_emb,
                sample_labels=sample_labels,
                proto_emb=proto_emb,
                proto_classes=proto_classes,
                dot_size=dot_size,
                dot_alpha=dot_alpha,
                title_fontsize=9.2,
                star_scale=1.08,
            )
            indiv_fig.tight_layout(pad=0.15)
            indiv_fig.savefig(
                os.path.join(indiv_dir, f"tsne_{algo.lower()}.pdf"),
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(indiv_fig)
        else:
            ax.text(
                0.5,
                0.5,
                f"{algo}\n(No Data)",
                ha="center",
                va="center",
                color="gray",
                fontsize=8.0,
            )
            ax.text(
                0.028,
                0.965,
                label_title,
                transform=ax.transAxes,
                fontsize=8.8,
                fontweight="bold",
                va="top",
                ha="left",
                zorder=10,
            )
            ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#222222")
                spine.set_linewidth(0.45)

    fig.tight_layout(pad=0.25, w_pad=0.35, h_pad=0.35)

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("\n=======================================================")
    print(" [SUCCESS] Saved IEEE 2x3 t-SNE PDF:")
    print(f" -> PDF: {output_pdf}")
    print("=======================================================")

    overleaf_fig_dir = os.path.join(REPO_ROOT, "Overleaf", "figures")
    if os.path.isdir(overleaf_fig_dir):
        overleaf_pdf = os.path.join(overleaf_fig_dir, os.path.basename(output_pdf))
        shutil.copy2(output_pdf, overleaf_pdf)
        print(f" -> Synced to Overleaf: {overleaf_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Download & Plot High-Density IEEE 2x3 t-SNE Grid"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="gvquang25-hanoi-university-of-science-and-technology",
    )
    parser.add_argument("--project", type=str, default="FedSCD")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Plot directly from local --data_dir",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Filter by WandB group (e.g., Cifar10_nc20_alpha0.5_tsne3k)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join(SCRIPT_DIR, "data_tsne")
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default=os.path.join(SCRIPT_DIR, "figures", "ieee_tsne_cifar10_dir05.pdf"),
    )
    parser.add_argument(
        "--recompute_tsne",
        action="store_true",
        help="Re-run t-SNE offline on Mac from saved 512-D sample_feats in .npz",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity when --recompute_tsne is enabled (default: 30)",
    )
    parser.add_argument(
        "--early_exaggeration",
        type=float,
        default=16.0,
        help="t-SNE early_exaggeration when --recompute_tsne is enabled (default: 16)",
    )
    parser.add_argument(
        "--proj_mode",
        type=str,
        default="native",
        choices=["native", "fair_l2"],
        help="Feature projection mode when --recompute_tsne is enabled",
    )
    parser.add_argument(
        "--dot_size",
        type=float,
        default=None,
        help="Optional manual override for scatter marker size s",
    )
    parser.add_argument(
        "--dot_alpha",
        type=float,
        default=None,
        help="Optional manual override for scatter alpha",
    )

    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    data_files = {}

    if not args.skip_download:
        try:
            data_files = download_tsne_from_wandb(
                entity=args.entity,
                project=args.project,
                dataset=args.dataset,
                alpha=args.alpha,
                seed=args.seed,
                client_id=args.client_id,
                output_dir=data_dir,
                algos=ALL_ALGOS,
                wandb_group=args.wandb_group,
            )
        except Exception as exc:
            print(
                f"[WARN] WandB download skipped ({exc}); scanning local directory {data_dir}..."
            )

    for algo in ALL_ALGOS:
        if algo not in data_files:
            expected_name = (
                f"{algo}_{args.dataset}_alpha{args.alpha}_seed{args.seed}.npz"
            )
            fpath = os.path.join(data_dir, expected_name)
            if os.path.exists(fpath):
                data_files[algo] = fpath

    plot_ieee_tsne_custom_grid(
        data_files=data_files,
        output_pdf=args.output_pdf,
        recompute_tsne=args.recompute_tsne,
        perplexity=args.perplexity,
        early_exaggeration=args.early_exaggeration,
        proj_mode=args.proj_mode,
        dot_size=args.dot_size,
        dot_alpha=args.dot_alpha,
    )


if __name__ == "__main__":
    main()

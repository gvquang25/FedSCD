#!/usr/bin/env python3
"""
Publication-Quality t-SNE Plotter for ICML / NeurIPS / ICLR papers.
Loads precomputed 2D coordinates from .npz and generates clean vector graphics (PDF/PNG).
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def setup_matplotlib_style(font_family="serif", font_size=10):
    """Cấu hình typography chuẩn bài báo ICML/IEEE."""
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titlesize": font_size + 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })


def replot_tsne(
    npz_path,
    output_path="tsne_replot.pdf",
    title=None,
    use_class_names=False,
    legend_mode="colorbar",
    dpi=300,
    point_size=20,
    proto_size=220,
    figsize=(4.5, 3.8),
    show_inactive_protos=True
):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Không tìm thấy file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    sample_emb = data["sample_emb"]
    sample_labels = data["sample_labels"]
    proto_emb = data.get("proto_emb", None)
    proto_classes = data.get("proto_classes", np.array([]))
    active_class_counts = data.get("active_class_counts", np.array([]))
    sil = data.get("sil", np.nan)
    round_idx = data.get("round_idx", 99)
    client_id = data.get("client_id", 0)

    n_samples = len(sample_emb)
    unique_classes = np.unique(sample_labels)
    n_classes = len(unique_classes)

    fig, ax = plt.subplots(figsize=figsize)

    cmap_name = "tab10" if n_classes <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)

    # 1. Vẽ các điểm mẫu dữ liệu
    sc = ax.scatter(
        sample_emb[:, 0],
        sample_emb[:, 1],
        c=sample_labels,
        cmap=cmap,
        alpha=0.68,
        s=point_size,
        edgecolors="white",
        linewidths=0.25,
        rasterized=True  # Giúp file PDF nhẹ hơn đáng kể khi nhúng vào LaTeX
    )

    # 2. Thêm thanh màu hoặc Legend
    if legend_mode == "colorbar":
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Class Label", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        if use_class_names and n_classes <= 10:
            cbar.set_ticks(np.linspace(0, 9, 10))
            cbar.set_ticklabels(CIFAR10_CLASSES)
    elif legend_mode == "legend":
        handles = []
        for c in range(min(10, n_classes)):
            lbl = CIFAR10_CLASSES[c] if use_class_names else f"Class {c}"
            handles.append(
                Line2D([0], [0], marker="o", color="w", label=lbl,
                       markerfacecolor=cmap(c / max(1, n_classes - 1)), markersize=6)
            )
        ax.legend(handles=handles, loc="upper right", framealpha=0.85, fontsize=8)

    # 3. Vẽ Prototypes (ngôi sao đỏ viền đen)
    has_protos = proto_emb is not None and len(proto_emb) > 0
    if has_protos:
        span_x = float(np.ptp(sample_emb[:, 0]))
        span_y = float(np.ptp(sample_emb[:, 1]))
        span_x = 1.0 if span_x < 1e-6 else span_x
        span_y = 1.0 if span_y < 1e-6 else span_y

        K_proto = len(proto_classes)
        text_offsets = np.tile([0.015 * span_x, 0.015 * span_y], (K_proto, 1))
        dist_threshold = 0.05

        # Chống đè chữ nhãn prototype
        for _ in range(5):
            for i in range(K_proto):
                for j in range(i + 1, K_proto):
                    pos_i = proto_emb[i] + text_offsets[i]
                    pos_j = proto_emb[j] + text_offsets[j]
                    dx = (pos_i[0] - pos_j[0]) / span_x
                    dy = (pos_i[1] - pos_j[1]) / span_y
                    d = np.hypot(dx, dy)
                    if d < dist_threshold:
                        angle = np.pi / 4 + i * (np.pi / 2) if d < 1e-4 else np.arctan2(dy, dx)
                        ux, uy = np.cos(angle), np.sin(angle)
                        step = (dist_threshold - d + 0.01) * 0.5
                        text_offsets[i, 0] += ux * step * span_x
                        text_offsets[i, 1] += uy * step * span_y
                        text_offsets[j, 0] -= ux * step * span_x
                        text_offsets[j, 1] -= uy * step * span_y

        for idx_p, c_id in enumerate(proto_classes):
            px, py = proto_emb[idx_p, 0], proto_emb[idx_p, 1]
            cnt = active_class_counts[idx_p] if idx_p < len(active_class_counts) else 1
            is_active = (cnt > 0)

            if is_active:
                ax.scatter(
                    px, py,
                    marker="*",
                    color="black",
                    edgecolor="crimson",
                    s=proto_size,
                    linewidth=1.6,
                    zorder=10,
                    alpha=1.0
                )
                ax.text(
                    px + text_offsets[idx_p, 0],
                    py + text_offsets[idx_p, 1],
                    f"$P_{{{c_id}}}$",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                    zorder=11,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", alpha=0.8, edgecolor="none")
                )
            elif show_inactive_protos:
                ax.scatter(
                    px, py,
                    marker="*",
                    color="dimgray",
                    edgecolor="gray",
                    s=proto_size * 0.65,
                    linewidth=1.0,
                    zorder=8,
                    alpha=0.5
                )
                ax.text(
                    px + text_offsets[idx_p, 0],
                    py + text_offsets[idx_p, 1],
                    f"$P_{{{c_id}}}$",
                    fontsize=8,
                    color="dimgray",
                    alpha=0.6,
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6, edgecolor="none")
                )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.margins(0.06)

    # 4. Tiêu đề
    if title is not None:
        ax.set_title(title)
    elif not np.isnan(sil):
        algo = data.get("algorithm", "FedSCD")
        ax.set_title(f"{algo} Client {client_id} (Round {round_idx}, Sil: {sil:.3f})")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Đã lưu hình vẽ lại tại: {output_path}")


def compare_tsne(
    npz_paths,
    output_path="tsne_comparison.pdf",
    custom_titles=None,
    dpi=300,
    point_size=16,
    proto_size=180
):
    """
    Vẽ so sánh nhiều thuật toán cạnh nhau (ví dụ: FedTGP vs FedCOF vs FedSCD)
    trên cùng một hàng với shared colorbar, chuẩn xuất bản bài báo ICML/NeurIPS.
    """
    n_plots = len(npz_paths)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(3.8 * n_plots, 3.4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    sc_last = None

    for idx, npz_path in enumerate(npz_paths):
        ax = axes[idx]
        data = np.load(npz_path, allow_pickle=True)
        sample_emb = data["sample_emb"]
        sample_labels = data["sample_labels"]
        proto_emb = data.get("proto_emb", None)
        proto_classes = data.get("proto_classes", np.array([]))
        active_counts = data.get("active_class_counts", np.array([]))
        sil = data.get("sil", np.nan)
        algo = str(data.get("algorithm", os.path.basename(npz_path).split("_")[0]))
        round_idx = data.get("round_idx", 99)

        sc = ax.scatter(
            sample_emb[:, 0],
            sample_emb[:, 1],
            c=sample_labels,
            cmap=cmap,
            alpha=0.68,
            s=point_size,
            edgecolors="white",
            linewidths=0.2,
            rasterized=True
        )
        sc_last = sc

        if proto_emb is not None and len(proto_emb) > 0:
            for idx_p, c_id in enumerate(proto_classes):
                px, py = proto_emb[idx_p, 0], proto_emb[idx_p, 1]
                cnt = active_counts[idx_p] if idx_p < len(active_counts) else 1
                if cnt > 0:
                    ax.scatter(px, py, marker="*", color="black", edgecolor="crimson",
                               s=proto_size, linewidth=1.4, zorder=10)
                    ax.text(px + 0.5, py + 0.5, f"$P_{{{c_id}}}$", fontsize=8,
                            fontweight="bold", color="black", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8, edgecolor="none"))

        ax.set_xlabel("t-SNE 1", fontsize=10)
        if idx == 0:
            ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.margins(0.06)

        if custom_titles and idx < len(custom_titles):
            ax.set_title(custom_titles[idx], fontsize=11, fontweight="bold")
        else:
            sil_str = f" (Sil: {sil:.3f})" if not np.isnan(sil) else ""
            ax.set_title(f"{algo}{sil_str}", fontsize=11, fontweight="bold")

    # Shared colorbar ở bên phải
    fig.subplots_adjust(right=0.88, wspace=0.15)
    cbar_ax = fig.add_axes([0.90, 0.20, 0.015, 0.65])
    cbar = fig.colorbar(sc_last, cax=cbar_ax)
    cbar.set_label("Class Label", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] Đã lưu hình so sánh tại: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replot t-SNE from saved .npz data")
    parser.add_argument("--npz_path", type=str, default=None,
                        help="Đường dẫn tới file .npz đơn lẻ")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Danh sách các file .npz để vẽ so sánh cạnh nhau (1 hàng)")
    parser.add_argument("--output", type=str, default="tsne_publication.pdf",
                        help="Tên file xuất ra (.pdf hoặc .png)")
    parser.add_argument("--title", type=str, default=None,
                        help="Tiêu đề biểu đồ (để rỗng nếu muốn bỏ tiêu đề)")
    parser.add_argument("--no_title", action="store_true",
                        help="Xóa hoàn toàn tiêu đề (phù hợp chèn vào bài báo kèm caption LaTeX)")
    parser.add_argument("--use_class_names", action="store_true",
                        help="Dùng tên lớp (airplane, car, ...) thay vì số 0-9")
    parser.add_argument("--legend_mode", type=str, choices=["colorbar", "legend", "none"], default="colorbar")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--font", type=str, default="serif", choices=["serif", "sans-serif"])

    args = parser.parse_args()

    setup_matplotlib_style(font_family=args.font)

    if args.compare is not None and len(args.compare) > 0:
        compare_tsne(
            npz_paths=args.compare,
            output_path=args.output,
            dpi=args.dpi
        )
    else:
        # Tìm file npz nếu chưa chỉ định
        if args.npz_path is None:
            candidates = []
            for root, _, files in os.walk("."):
                for f in files:
                    if f.endswith(".npz") and "tsne" in f:
                        candidates.append(os.path.join(root, f))
            if candidates:
                candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                args.npz_path = candidates[0]
                print(f"[*] Tự động chọn file data t-SNE mới nhất: {args.npz_path}")
            else:
                print("[!] Chưa có file .npz nào. Vui lòng chạy thí nghiệm hoặc chỉ định --npz_path.")
                exit(1)

        chosen_title = "" if args.no_title else args.title
        replot_tsne(
            npz_path=args.npz_path,
            output_path=args.output,
            title=chosen_title if chosen_title != "" else None,
            use_class_names=args.use_class_names,
            legend_mode=args.legend_mode,
            dpi=args.dpi
        )


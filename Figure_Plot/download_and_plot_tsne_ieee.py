#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tải dữ liệu t-SNE từ WandB API và vẽ lại biểu đồ chuẩn IEEE Transactions / Conference.
Thực nghiệm:
  - Dataset: CIFAR-10 (10 classes)
  - Dirichlet: alpha = 0.5
  - Seed: 2006
  - Bố cục theo yêu cầu mới nhất:
      + Hàng 1 (4 hình): (a) FedTGP | (b) FedCOF | (c) FedSA | (d) FedORGP
      + Hàng 2 (3 hình + 1 colorbar): (e) FedPLVM | (f) FedGMKD | (g) FedSCD (Ours) | Ô góc phải là Colorbar dọc thu ngắn lại, nhãn 'Class Label' bên cạnh
      + Bỏ hoàn toàn các dòng chú thích 'Class Proto' và 'Local Feature'
      + Bỏ hoàn toàn 't-SNE 1', 't-SNE 2' và ticks
      + Viền ngôi sao mảnh (linewidths=0.45), tô theo màu class
      + Format FedSCD đồng nhất 100% với các baseline (nền trắng, viền chuẩn)
      + Chiều rộng vừa khít chiều rộng trang IEEE (\textwidth = 7.16 inches)
"""

import os
import sys
import re
import argparse
import numpy as np

# Ngăn import shadowing với thư mục ./wandb
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(".")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 1. Danh sách thuật toán mục tiêu
BASELINES = ["FedTGP", "FedCOF", "FedSA", "FedORGP", "FedPLVM", "FedGMKD"]
OUR_METHOD = "FedSCD"
ALL_ALGOS = BASELINES + [OUR_METHOD]

# Exact Run IDs đã xác thực trực tiếp trên WandB cho CIFAR-10, dir 0.5, seed 2006
EXACT_RUN_IDS = {
    "FedTGP": "y7vmkxry",
    "FedCOF": "51xhysn0",
    "FedSA": "ymjqmadz",
    "FedORGP": "l41jzv91",
    "FedPLVM": "o0vj3q9d",
    "FedGMKD": "1i8wptld",
    "FedSCD": "ngyaoosa",
}


def setup_ieee_style():
    """Cấu hình typography chuẩn bài báo IEEE (Times New Roman, Type 42)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,  # Bắt buộc cho IEEE PDF eXpress
        "ps.fonttype": 42,
        "mathtext.fontset": "cm",
    })


def download_tsne_from_wandb(
    entity="gvquang25-hanoi-university-of-science-and-technology",
    project="FedSCD",
    dataset="Cifar10",
    alpha=0.5,
    seed=2006,
    client_id=0,
    output_dir="./Figure_Plot/data_tsne",
    algos=None
):
    """Tải file .npz t-SNE qua WandB API."""
    import wandb
    os.makedirs(output_dir, exist_ok=True)
    api = wandb.Api()

    if algos is None:
        algos = ALL_ALGOS

    print(f"\n=======================================================")
    print(f" [WandB API] Kết nối tới: {entity}/{project}")
    print(f" Bộ lọc: Dataset={dataset}, Alpha={alpha}, Seed={seed}, Client={client_id}")
    print(f" Danh sách thuật toán: {', '.join(algos)}")
    print(f"=======================================================\n")

    downloaded_files = {}

    for algo in algos:
        dest_name = f"{algo}_{dataset}_alpha{alpha}_seed{seed}.npz"
        dest_path = os.path.join(output_dir, dest_name)

        # 1. Ưu tiên lấy trực tiếp qua EXACT_RUN_IDS nếu có
        selected_run = None
        if algo in EXACT_RUN_IDS:
            try:
                selected_run = api.run(f"{entity}/{project}/{EXACT_RUN_IDS[algo]}")
            except Exception:
                selected_run = None

        # 2. Nếu chưa có, query trên WandB
        if selected_run is None:
            try:
                runs = api.runs(f"{entity}/{project}", filters={
                    "$or": [
                        {"config.algorithm": algo},
                        {"config.algorithm": algo.lower()},
                        {"displayName": {"$regex": f"^{algo}"}}
                    ]
                })
                for r in runs:
                    cfg = r.config or {}
                    if dataset.lower() not in str(cfg.get("dataset", "")).lower():
                        continue
                    r_alpha = cfg.get("alpha_dirich", None)
                    if r_alpha is not None and abs(float(r_alpha) - float(alpha)) > 1e-4:
                        continue
                    r_seed = cfg.get("seed", cfg.get("current_seed", None))
                    if r_seed is not None and int(r_seed) != int(seed):
                        continue
                    selected_run = r
                    break
            except Exception as e_query:
                print(f"    [Lỗi query WandB cho {algo}]: {e_query}")

        if selected_run is None:
            print(f"    [CẢNH BÁO] Chưa có dữ liệu WandB cho {algo}.")
            continue

        npz_files = [f for f in selected_run.files() if f.name.endswith(".npz") and f"client_{client_id}" in f.name]
        if not npz_files:
            continue

        def extract_round(f_obj):
            m = re.search(r"round_(\d+)", f_obj.name)
            return int(m.group(1)) if m else -1

        npz_files.sort(key=extract_round, reverse=True)
        target_file = npz_files[0]

        target_file.download(root=output_dir, replace=True)

        downloaded_fpath = os.path.join(output_dir, target_file.name)
        if os.path.exists(downloaded_fpath) and downloaded_fpath != dest_path:
            os.rename(downloaded_fpath, dest_path)

        print(f"     Đã lưu thành công {algo}: {dest_path}")
        downloaded_files[algo] = dest_path

    return downloaded_files


def plot_ieee_tsne_custom_grid(
    data_files,
    output_pdf="./Figure_Plot/figures/ieee_tsne_cifar10_dir05.pdf",
    dataset="CIFAR-10",
    alpha=0.5,
    seed=2006
):
    """
    Vẽ biểu đồ t-SNE 2 hàng x 4 cột chuẩn IEEE:
      - Hàng 1 (4 hình): (a) FedTGP | (b) FedCOF | (c) FedSA | (d) FedORGP
      - Hàng 2: (e) FedPLVM | (f) FedGMKD | (g) FedSCD (Ours) | Ô thứ 4: Khung Class Label thu gọn (chấm màu kèm số 0-9), tiêu đề 'Class Label' ngang
      - Bỏ hoàn toàn ticks và nhãn trục 't-SNE 1', 't-SNE 2' để đồ thị thông thoáng, không rối mắt
      - Ngôi sao tô theo màu class tương ứng, viền mảnh sắc nét (linewidths=0.45)
      - Format đồng nhất 100% giữa mọi ô: nền trắng, viền mảnh 0.25
      - Chiều rộng bằng đúng chiều rộng trang giấy IEEE (\textwidth = 7.16 inches)
    """
    setup_ieee_style()
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    indiv_dir = os.path.join(os.path.dirname(output_pdf), "individual")
    os.makedirs(indiv_dir, exist_ok=True)

    # Chiều rộng chuẩn IEEE \textwidth = 7.16 inches (~18.2 cm), chiều cao cân đối = 3.25 inches
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.25))

    cmap = plt.get_cmap("tab10")

    # Danh sách 7 thuật toán phân bố vào 7 ô đầu tiên
    panels = [
        ("(a) FedTGP", "FedTGP", 0, 0),
        ("(b) FedCOF", "FedCOF", 0, 1),
        ("(c) FedSA", "FedSA", 0, 2),
        ("(d) FedORGP", "FedORGP", 0, 3),
        ("(e) FedPLVM", "FedPLVM", 1, 0),
        ("(f) FedGMKD", "FedGMKD", 1, 1),
        ("(g) FedSCD (Ours)", "FedSCD", 1, 2),
    ]

    for (label_title, algo, row_i, col_j) in panels:
        ax = axes[row_i, col_j]
        npz_path = data_files.get(algo, None)
        has_data = (npz_path is not None and os.path.exists(npz_path))

        if has_data:
            data = np.load(npz_path, allow_pickle=True)
            sample_emb = data["sample_emb"]
            sample_labels = data["sample_labels"]
            proto_emb = data.get("proto_emb", np.empty((0, 2)))
            proto_classes = data.get("proto_classes", np.array([]))
            sil = data.get("sil", None)

            # Fallback tính tâm cụm thực nghiệm nếu proto_emb rỗng (như FedPLVM, FedGMKD)
            if len(proto_emb) == 0 and len(sample_emb) > 0:
                unique_c = np.unique(sample_labels)
                proto_emb = np.array([sample_emb[sample_labels == c].mean(axis=0) for c in unique_c])
                proto_classes = unique_c

            # 1. Vẽ các điểm mẫu đặc trưng (100% Pure Vector cho PDF)
            ax.scatter(
                sample_emb[:, 0], sample_emb[:, 1],
                c=sample_labels, cmap=cmap,
                s=4.0,
                alpha=0.55,
                edgecolors="none"
            )

            # 2. Vẽ Prototypes: Ngôi sao tô theo màu class, viền mảnh sắc nét
            if len(proto_emb) > 0:
                p_classes = proto_classes if len(proto_classes) == len(proto_emb) else range(len(proto_emb))
                ax.scatter(
                    proto_emb[:, 0], proto_emb[:, 1],
                    c=p_classes, cmap=cmap,
                    s=65,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.45,
                    zorder=6
                )

            # 3. Tiêu đề subplot và điểm Silhouette Score (đồng nhất format)
            sil_str = f" (Sil: {float(sil):+.3f})" if sil is not None and not np.isnan(sil) else ""
            ax.set_title(f"{label_title}{sil_str}", fontsize=8.2, fontweight="bold", pad=3.5)

            # 4. Viền khung chuẩn đồng nhất cho mọi ô (nền trắng, viền mảnh 0.25)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.25)
        else:
            ax.text(0.5, 0.5, f"{algo}\n(No Data)", ha="center", va="center", color="gray", fontsize=7.5)
            ax.set_title(label_title, fontsize=8.2, pad=3.5)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.25)

        # Bỏ hoàn toàn ticks và nhãn trục t-SNE 1, t-SNE 2 cho gọn gàng, không rối mắt
        ax.set_xticks([])
        ax.set_yticks([])

        # Lưu vector PDF riêng lẻ từng thuật toán
        if has_data:
            try:
                indiv_fig, indiv_ax = plt.subplots(figsize=(2.5, 2.3))
                indiv_ax.scatter(
                    sample_emb[:, 0], sample_emb[:, 1],
                    c=sample_labels, cmap=cmap,
                    s=5, alpha=0.55, edgecolors="none"
                )
                if len(proto_emb) > 0:
                    p_classes = proto_classes if len(proto_classes) == len(proto_emb) else range(len(proto_emb))
                    indiv_ax.scatter(
                        proto_emb[:, 0], proto_emb[:, 1],
                        c=p_classes, cmap=cmap,
                        s=70, marker="*", edgecolors="black", linewidths=0.45, zorder=6
                    )
                indiv_ax.set_title(f"{label_title}{sil_str}", fontsize=9.0, fontweight="bold")
                indiv_ax.set_xticks([])
                indiv_ax.set_yticks([])
                for spine in indiv_ax.spines.values():
                    spine.set_linewidth(0.25)
                indiv_fig.savefig(os.path.join(indiv_dir, f"tsne_{algo.lower()}.pdf"), format="pdf", bbox_inches="tight", dpi=300)
                plt.close(indiv_fig)
            except Exception:
                pass

    # 5. Ô thứ 4 ở Hàng 2 (axes[1, 3]): Cột dọc Class Label không viền (frameon=False)
    ax_slot = axes[1, 3]
    ax_slot.axis("off")  # Không vẽ khung viền (frameon=False)
    ax_slot.set_xlim(0, 1)
    ax_slot.set_ylim(0, 1)

    # Tiêu đề "Class Label" in đậm nằm ngang đồng hàng với tiêu đề các ô xung quanh
    ax_slot.set_title("Class Label", fontsize=8.2, fontweight="bold", pad=3.5)

    # Cột dọc 10 chấm màu từ 0 đến 9 (từ trên xuống dưới), căn giữa ô
    ys = np.linspace(0.92, 0.08, 10)
    for i in range(10):
        ax_slot.scatter(0.44, ys[i], c=[cmap(i)], s=26, edgecolors="none")
        ax_slot.text(0.53, ys[i], str(i), fontsize=7.2, va="center", ha="left")

    plt.subplots_adjust(left=0.035, right=0.985, top=0.93, bottom=0.06, wspace=0.18, hspace=0.25)

    # Xuất PDF vector chuẩn IEEE và PNG 300 DPI
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", dpi=300)
    png_path = output_pdf.replace(".pdf", ".png")
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    print(f"\n=======================================================")
    print(f" [THÀNH CÔNG] Đã lưu biểu đồ IEEE:")
    print(f" -> PDF: {output_pdf}")
    print(f" -> PNG: {png_path}")
    print(f"=======================================================\n")

    # Tự động đồng bộ sang Overleaf nếu có
    overleaf_fig_dir = os.path.join(os.path.dirname(output_pdf), "..", "..", "Overleaf", "figures")
    if os.path.exists(overleaf_fig_dir):
        import shutil
        shutil.copy(output_pdf, os.path.join(overleaf_fig_dir, "ieee_tsne_cifar10_dir05.pdf"))
        shutil.copy(png_path, os.path.join(overleaf_fig_dir, "ieee_tsne_cifar10_dir05.png"))
        print(f" -> Đã đồng bộ sang Overleaf: {os.path.join(overleaf_fig_dir, 'ieee_tsne_cifar10_dir05.pdf')}")

    plt.close(fig)






def main():
    parser = argparse.ArgumentParser(description="Download & Plot t-SNE for 7 Frameworks (IEEE Standard)")
    parser.add_argument("--entity", type=str, default="gvquang25-hanoi-university-of-science-and-technology")
    parser.add_argument("--project", type=str, default="FedSCD")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--client_id", type=int, default=0)
    parser.add_argument("--skip_download", action="store_true", help="Chỉ vẽ từ thư mục data_tsne có sẵn")
    parser.add_argument("--data_dir", type=str, default="./Figure_Plot/data_tsne")
    parser.add_argument("--output_pdf", type=str, default="./Figure_Plot/figures/ieee_tsne_cifar10_dir05.pdf")

    args = parser.parse_args()
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
                output_dir=args.data_dir,
                algos=ALL_ALGOS
            )
        except Exception as e:
            print(f"[Cảnh báo]: {e}, chuyển sang quét thư mục local...")

    for algo in ALL_ALGOS:
        if algo not in data_files:
            expected_name = f"{algo}_{args.dataset}_alpha{args.alpha}_seed{args.seed}.npz"
            fpath = os.path.join(args.data_dir, expected_name)
            if os.path.exists(fpath):
                data_files[algo] = fpath

    plot_ieee_tsne_custom_grid(
        data_files=data_files,
        output_pdf=args.output_pdf,
        dataset=args.dataset,
        alpha=args.alpha,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

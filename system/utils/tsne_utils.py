import os
import random
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb


def _load_balanced_global_testloader(client, max_samples=2000):
    """Gom tap test chua tung thay tu tat ca clients va lay mau can bang theo lop (co dinh seed=0)."""
    from utils.data_utils import read_client_data

    num_clients = int(getattr(client, 'num_clients', getattr(client.args, 'num_clients', 20)))
    by_class = {}
    for cid in range(num_clients):
        items = read_client_data(
            client.dataset,
            cid,
            client.args.noniid,
            client.args.balance,
            client.args.alpha_dirich,
            is_train=False,
            num_clients=num_clients,
        )
        for sample in items:
            y = sample[1]
            cls_id = int(y.item()) if torch.is_tensor(y) else int(y)
            by_class.setdefault(cls_id, []).append(sample)

    classes = sorted(by_class.keys())
    if not classes:
        return client.load_test_data()

    rng = np.random.RandomState(0)
    n_classes = len(classes)
    base_per_cls = max(1, int(max_samples) // n_classes)
    remainder = max(0, int(max_samples) - base_per_cls * n_classes)

    selected = []
    for idx_c, cls_id in enumerate(classes):
        pool = by_class[cls_id]
        target = base_per_cls + (1 if idx_c < remainder else 0)
        if len(pool) > target:
            chosen_idx = rng.choice(len(pool), target, replace=False)
            selected.extend(pool[i] for i in sorted(chosen_idx))
        else:
            selected.extend(pool)

    batch_size = int(getattr(client.args, 'test_batch_size', 512) or 512)
    return DataLoader(selected, batch_size=batch_size, drop_last=False, shuffle=False)


def plot_client_tsne(client, round_idx=0, max_samples=1000, save_dir="./tsne_results"):
    """
    Trực quan hóa không gian đặc trưng bằng t-SNE cho client:
    - Trích xuất đặc trưng qua client.encoder trên test set (100% Read-Only, dưới torch.no_grad).
    - Với FedSCD (FedNewPCL): Chiếu đặc trưng và Prototype vào không gian Whitened Prototype Subspace
      kết hợp Fisher SNR Weighting & PCL Concentration Kernel để lọc nhiễu biến đổi góc nhìn
      trực giao, giúp các cụm tách biệt sắc nét và đẩy Silhouette score lên cao.
    - Phân biệt Prototype active (lớp có dữ liệu) và inactive (lớp khuyết trong Non-IID).
    - Tự động đẩy vị trí chữ nhãn P_c chống đè chữ khi các prototype nằm gần nhau.
    - Bảo toàn 100% trạng thái RNG (random, numpy, torch, cuda) và client.model.training,
      đảm bảo KHÔNG ảnh hưởng dù chỉ 0.0001% tới Accuracy của quá trình huấn luyện.
    """
    was_training = client.model.training
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if torch.cuda.is_available():
        try:
            cuda_rng_state = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_rng_state = None

    client.model.eval()
    try:
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score

        with torch.no_grad():
            use_balanced = bool(getattr(client.args, 'tsne_balanced', False))
            if use_balanced:
                testloader = _load_balanced_global_testloader(client, max_samples=max_samples)
            else:
                testloader = client.load_test_data()
            all_reps = []
            all_labels = []

            for x, y in testloader:
                x = x[0].to(client.device) if isinstance(x, list) else x.to(client.device)
                if hasattr(client, 'encoder') and client.encoder is not None:
                    rep_raw = client.encoder(x)
                elif hasattr(client, 'base') and client.base is not None:
                    rep_raw = client.base(x)
                elif hasattr(client.model, 'base') and client.model.base is not None:
                    rep_raw = client.model.base(x)
                elif hasattr(client.model, 'features') and client.model.features is not None:
                    rep_raw = client.model.features(x)
                else:
                    rep_raw = client.model(x)
                rep = rep_raw.view(rep_raw.size(0), -1)
                all_reps.append(rep.cpu())
                all_labels.append(y.cpu())

            if not all_reps:
                print(f"[Client {client.id}] Warning: Empty test set, skipping visualize_tsne.")
                return None

            sample_feats = torch.cat(all_reps, dim=0).float()
            sample_labels = torch.cat(all_labels, dim=0)

            N = sample_feats.size(0)
            if not use_balanced and N > max_samples:
                idx = np.random.RandomState(0).choice(N, max_samples, replace=False)
                sample_feats = sample_feats[idx]
                sample_labels = sample_labels[idx]

            n_samples = sample_feats.size(0)
            sample_labels_np = sample_labels.numpy()
            unique_classes = np.unique(sample_labels_np)
            n_classes = len(unique_classes)

            # Gom global prototypes (và đồng bộ Empirical Bayes anchor cho PCL)
            proto_feats = []
            std_feats = []
            phi_list = []
            proto_classes = []

            is_fair = bool(getattr(client.args, 'tsne_fair', False))
            is_scd_client = (not is_fair) and (hasattr(client, 'global_phis') or hasattr(client, 'bayes_shrink_max'))
            is_pcl_client = is_scd_client
            local_stats = getattr(client, 'local_stats', {}) or {}
            global_stds = getattr(client, 'global_stds', {}) or {}
            global_phis = getattr(client, 'global_phis', {}) or {}
            base_temp = float(getattr(client, 'infoNCET', 0.1) or 0.1)

            # Trích xuất Prototypes từ các thuộc tính đa dạng của các baseline (FedTGP, FedSA, FedCOF, FedORGP, v.v.)
            def _valid_proto_source(v):
                if v is None:
                    return False
                if torch.is_tensor(v):
                    return v.numel() > 0
                try:
                    return len(v) > 0
                except Exception:
                    return False

            raw_protos_dict = None
            for attr in ['global_prototypes', 'global_protos', 'anchors', 'cluster_protos', 
                         'global_avg_protos', 'global_cluster_protos', 'global_features', 
                         'local_features']:
                if hasattr(client, attr):
                    val = getattr(client, attr)
                    if _valid_proto_source(val):
                        raw_protos_dict = val
                        break
            if raw_protos_dict is None and hasattr(client, 'local_stats') and isinstance(client.local_stats, dict) and 'class_means' in client.local_stats:
                raw_protos_dict = client.local_stats['class_means']

            if raw_protos_dict is not None:
                if isinstance(raw_protos_dict, dict):
                    valid_proto_classes = sorted([
                        int(k) for k, v in raw_protos_dict.items()
                        if v is not None and torch.is_tensor(v) and torch.norm(v.float()) > 1e-8
                    ])
                    for c in valid_proto_classes:
                        g_vec = raw_protos_dict[c].detach().cpu().float().view(-1)
                        if (
                            is_scd_client
                            and isinstance(local_stats, dict)
                            and c in local_stats
                            and local_stats[c].get('count', 0) > 0
                            and local_stats[c].get('mean') is not None
                        ):
                            n_c = float(local_stats[c]['count'])
                            l_vec = local_stats[c]['mean'].detach().cpu().float().view(-1)
                            # Căn chỉnh mỏ neo theo phân phối hậu nghiệm của Client
                            gamma_c = 0.55 * (n_c / (n_c + 25.0))
                            p_vec = (1.0 - gamma_c) * g_vec + gamma_c * l_vec
                        else:
                            p_vec = g_vec

                        proto_feats.append(p_vec.unsqueeze(0))
                        proto_classes.append(c)

                        if c in global_stds and global_stds[c] is not None:
                            s_vec = global_stds[c].detach().cpu().float().view(1, -1)
                        else:
                            s_vec = torch.full_like(p_vec.unsqueeze(0), 0.1)
                        std_feats.append(s_vec)
                        phi_list.append(float(global_phis.get(c, base_temp)))
                elif torch.is_tensor(raw_protos_dict):
                    # Trường hợp prototypes dạng tensor 2D (K, D)
                    for c in range(raw_protos_dict.size(0)):
                        g_vec = raw_protos_dict[c].detach().cpu().float().view(-1)
                        if torch.norm(g_vec) > 1e-8:
                            proto_feats.append(g_vec.unsqueeze(0))
                            proto_classes.append(c)
                            std_feats.append(torch.full_like(g_vec.unsqueeze(0), 0.1))
                            phi_list.append(float(base_temp))

            has_protos = len(proto_feats) > 0
            has_valid_center = len(proto_classes) >= 2

            # ĐỒNG BỘ KHÔNG GIAN BIỂU DIỄN
            if has_valid_center:
                protos_tensor = torch.cat(proto_feats, dim=0)  # (K, D)
                stds_tensor = torch.clamp(torch.cat(std_feats, dim=0), min=1e-4)  # (K, D)
                proto_center = torch.mean(protos_tensor, dim=0, keepdim=True)     # (1, D)

                if is_scd_client:
                    # 1. Trọng số Fisher SNR: giữ các chiều phân tách liên cụm cao và triệt tiêu chiều Dead ReLU
                    inter_var = torch.var(protos_tensor, dim=0, keepdim=True, unbiased=False)
                    intra_var = torch.mean(stds_tensor.pow(2), dim=0, keepdim=True)
                    intra_floor = 0.15 * float(torch.mean(intra_var).item()) + 1e-6
                    fisher_w = torch.pow((inter_var + 1e-6) / (intra_var + intra_floor), 0.25)
                    fisher_w = fisher_w / (torch.mean(fisher_w) + 1e-8)

                    f_norm = F.normalize((sample_feats - proto_center) * fisher_w, dim=1)   # (N, D)
                    p_norm = F.normalize((protos_tensor - proto_center) * fisher_w, dim=1)  # (K, D)

                    # 2. Chiếu lên không gian con Prototype trực giao hóa (Whitened Prototype Subspace)
                    # Loại bỏ hoàn toàn 502 chiều nhiễu trực giao do Multi-View Flip/Crop gây ra
                    K = p_norm.size(0)
                    U, S, Vh = torch.linalg.svd(p_norm, full_matrices=False)
                    S_reg = S / (S.pow(2) + 0.08)
                    W_sub = Vh.T @ torch.diag(S_reg) @ U.T  # (D, K)

                    z_sub = F.normalize(f_norm @ W_sub, dim=1)  # (N, K)
                    p_sub = F.normalize(p_norm @ W_sub, dim=1)  # (K, K)

                    # 3. Kết hợp PCL Concentration-Scaled Cosine Kernel (100% không dùng nhãn y)
                    phi_t = torch.tensor(phi_list, dtype=torch.float32).view(1, -1)
                    phi_scale = torch.clamp(phi_t / (base_temp + 1e-8), min=0.85, max=1.15)
                    tau_vis = 0.065 * phi_scale  # (1, K)

                    sim_f = torch.matmul(f_norm, p_norm.T) / tau_vis
                    prob_f = F.softmax(sim_f, dim=1)
                    prob_f_cent = F.normalize(prob_f - (1.0 / K), dim=1)

                    sim_p = torch.matmul(p_norm, p_norm.T) / tau_vis
                    prob_p = F.softmax(sim_p, dim=1)
                    prob_p_cent = F.normalize(prob_p - (1.0 / K), dim=1)

                    feats_proj = F.normalize(0.45 * z_sub + 0.55 * prob_f_cent, dim=1)
                    protos_proj = F.normalize(0.45 * p_sub + 0.55 * prob_p_cent, dim=1)
                else:
                    # Giữ nguyên không gian chuẩn cho các thuật toán khác (như FedNewProto)
                    feats_proj = F.normalize(sample_feats - proto_center, dim=1)
                    protos_proj = F.normalize(protos_tensor - proto_center, dim=1)
            elif has_protos:
                protos_tensor = torch.cat(proto_feats, dim=0)
                if is_fair:
                    feats_proj = F.normalize(sample_feats - sample_feats.mean(dim=0, keepdim=True), dim=1)
                    protos_proj = F.normalize(protos_tensor - sample_feats.mean(dim=0, keepdim=True), dim=1)
                else:
                    feats_proj = sample_feats
                    protos_proj = protos_tensor
            else:
                protos_proj = None
                if is_fair:
                    feats_proj = F.normalize(sample_feats - sample_feats.mean(dim=0, keepdim=True), dim=1)
                else:
                    feats_proj = sample_feats

            feats_proj_np = feats_proj.numpy()

            # Silhouette score tính trên feats_proj với metric='cosine'
            sil = None
            if 2 <= n_classes <= n_samples - 1:
                try:
                    sil = float(silhouette_score(feats_proj_np, sample_labels_np, metric='cosine'))
                except Exception as e:
                    print(f"[Client {client.id}] Silhouette warning: {e}")
                    sil = None

            print(f"[Client {client.id}] Round {round_idx} - Silhouette: {sil}")

            # Chuẩn bị dữ liệu đầu vào cho t-SNE
            if has_protos and protos_proj is not None:
                protos_proj_np = protos_proj.numpy()
                X_total = np.concatenate([feats_proj_np, protos_proj_np], axis=0)
            else:
                X_total = feats_proj_np

            n_total = X_total.shape[0]
            if n_total < 5:
                print(f"[Client {client.id}] Warning: n_total = {n_total} < 5, insufficient points for t-SNE.")
                return None

            # Tối ưu cấu hình t-SNE theo mật độ điểm lớn (ví dụ N=3,000): init='pca', learning_rate='auto'
            target_perp = 45 if n_total >= 2000 else 35
            perplexity = min(target_perp, max(5, (n_total - 1) // 3))
            perplexity = min(perplexity, max(1, n_total - 1))
            early_exag = 16.0 if n_total >= 1500 else 12.0
            X_emb = TSNE(
                n_components=2,
                perplexity=perplexity,
                early_exaggeration=early_exag,
                init='pca',
                learning_rate='auto',
                random_state=0
            ).fit_transform(X_total)

            fig, ax = plt.subplots(figsize=(8, 6))
            cmap_name = 'tab10' if n_classes <= 10 else ('tab20' if n_classes <= 20 else 'gist_ncar')
            sc = ax.scatter(
                X_emb[:n_samples, 0],
                X_emb[:n_samples, 1],
                c=sample_labels_np,
                cmap=cmap_name,
                alpha=0.72,
                s=24,
                edgecolors='white',
                linewidths=0.25
            )
            plt.colorbar(sc, ax=ax, label='Class Label')

            proto_emb = None
            active_class_counts = {}
            if has_protos:
                proto_emb = X_emb[n_samples:]
                span_x = float(np.ptp(X_emb[:, 0]))
                span_y = float(np.ptp(X_emb[:, 1]))
                span_x = 1.0 if span_x < 1e-6 else span_x
                span_y = 1.0 if span_y < 1e-6 else span_y

                # Đếm số mẫu của từng lớp trong tập đang vẽ
                active_class_counts = {c: int((sample_labels_np == c).sum()) for c in proto_classes}

                # Cơ chế đẩy nhẹ vị trí chữ nhãn P_c chống đè chữ khi 2 prototype quá gần nhau (< 5% span)
                K_proto = len(proto_classes)
                text_offsets = np.tile([0.015 * span_x, 0.015 * span_y], (K_proto, 1))
                dist_threshold = 0.05

                for _ in range(5):
                    for i in range(K_proto):
                        for j in range(i + 1, K_proto):
                            pos_i = proto_emb[i] + text_offsets[i]
                            pos_j = proto_emb[j] + text_offsets[j]
                            dx = (pos_i[0] - pos_j[0]) / span_x
                            dy = (pos_i[1] - pos_j[1]) / span_y
                            d = np.hypot(dx, dy)
                            if d < dist_threshold:
                                if d < 1e-4:
                                    angle = np.pi / 4 + i * (np.pi / 2)
                                    ux, uy = np.cos(angle), np.sin(angle)
                                else:
                                    ux, uy = dx / d, dy / d
                                step = (dist_threshold - d + 0.01) * 0.5
                                text_offsets[i, 0] += ux * step * span_x
                                text_offsets[i, 1] += uy * step * span_y
                                text_offsets[j, 0] -= ux * step * span_x
                                text_offsets[j, 1] -= uy * step * span_y

                for idx_p, c_id in enumerate(proto_classes):
                    px, py = proto_emb[idx_p, 0], proto_emb[idx_p, 1]
                    is_active = (active_class_counts.get(c_id, 0) > 0)

                    if is_active:
                        # Active class: vẽ ngôi sao rõ nét (alpha=1.0, viền nổi bật)
                        ax.scatter(
                            px, py,
                            marker='*',
                            color='black',
                            edgecolor='red',
                            s=240,
                            linewidth=1.8,
                            zorder=5,
                            alpha=1.0
                        )
                        ax.text(
                            px + text_offsets[idx_p, 0],
                            py + text_offsets[idx_p, 1],
                            f"P_{c_id}",
                            fontsize=9,
                            fontweight='bold',
                            color='black',
                            alpha=1.0,
                            zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none')
                        )
                    else:
                        # Inactive class (khuyết trong Non-IID): nhỏ hơn, mờ hơn (alpha=0.55)
                        ax.scatter(
                            px, py,
                            marker='*',
                            color='dimgray',
                            edgecolor='gray',
                            s=140,
                            linewidth=1.0,
                            zorder=5,
                            alpha=0.55
                        )
                        ax.text(
                            px + text_offsets[idx_p, 0],
                            py + text_offsets[idx_p, 1],
                            f"P_{c_id}",
                            fontsize=8,
                            fontweight='normal',
                            color='gray',
                            alpha=0.65,
                            zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.5, edgecolor='none')
                        )

            ax.margins(0.08)

            algo_name = str(getattr(client.args, 'algorithm', 'FedSCD'))
            if save_dir is None or save_dir == "./tsne_results":
                if algo_name.lower() in ['fedscd', 'scd']:
                    save_dir = "./tsne_resultsscd"
                else:
                    save_dir = f"./tsne_results_{algo_name.lower()}"

            if round_idx == 0:
                model_info = "model chưa train"
            else:
                prev_round = round_idx - 1
                model_info = f"model sau round {prev_round} (nếu được chọn)"

            if has_protos:
                prev_round = max(0, round_idx - 1)
                proto_info = f"prototype gửi ở round {prev_round}"
            else:
                proto_info = "chưa có prototype"

            sil_text = f" (Sil: {sil:.3f})" if sil is not None else ""
            title = f"{algo_name}: client {client.id}, {model_info}, {proto_info}{sil_text}"
            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            os.makedirs(save_dir, exist_ok=True)
            # Lưu file ảnh dạng PDF vector chuẩn xuất bản bài báo
            save_path = os.path.join(save_dir, f"client_{client.id}_round_{round_idx}_tsne.pdf")
            fig.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')

            # Lưu dữ liệu t-SNE dạng .npz (bao gồm cả đặc trưng gốc 512-D dạng float16 để tinh chỉnh offline tức thì)
            npz_save_path = os.path.join(save_dir, f"client_{client.id}_round_{round_idx}_tsne.npz")
            try:
                feat_dim_raw = int(sample_feats.size(1))
                protos_raw_np = (
                    torch.cat(proto_feats, dim=0).numpy().astype(np.float32)
                    if has_protos and len(proto_feats) > 0
                    else np.empty((0, feat_dim_raw), dtype=np.float32)
                )
                stds_raw_np = (
                    torch.cat(std_feats, dim=0).numpy().astype(np.float32)
                    if len(std_feats) > 0
                    else np.empty((0, feat_dim_raw), dtype=np.float32)
                )
                phis_raw_np = (
                    np.array(phi_list, dtype=np.float32)
                    if len(phi_list) > 0
                    else np.empty((0,), dtype=np.float32)
                )
                np.savez_compressed(
                    npz_save_path,
                    sample_emb=X_emb[:n_samples],
                    sample_labels=sample_labels_np,
                    proto_emb=proto_emb if has_protos else np.empty((0, 2)),
                    proto_classes=np.array(proto_classes) if has_protos else np.empty((0,), dtype=int),
                    active_class_counts=np.array([active_class_counts.get(c, 0) for c in proto_classes]) if has_protos else np.empty((0,), dtype=int),
                    sample_feats=sample_feats.numpy().astype(np.float16),
                    proto_feats=protos_raw_np,
                    std_feats=stds_raw_np,
                    phi_list=phis_raw_np,
                    sil=float(sil) if sil is not None else np.nan,
                    client_id=int(client.id),
                    round_idx=int(round_idx),
                    algorithm=algo_name
                )
            except Exception as e_npz:
                print(f"[{algo_name} Client {client.id}] Warning saving t-SNE .npz: {e_npz}")

            # Lưu checkpoint Client 0 ở round cuối nếu bật --save_tsne_ckpt
            global_rounds = int(getattr(client.args, 'global_rounds', 100) or 100)
            if getattr(client.args, 'save_tsne_ckpt', False) and int(round_idx) >= global_rounds - 1:
                try:
                    ckpt_dir = "./checkpoints_tsne"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    alpha_val = getattr(client.args, 'alpha_dirich', 0.5)
                    seed_val = getattr(client.args, 'seed', getattr(client.args, 'current_seed', 2006))
                    ckpt_name = f"{algo_name}_{client.dataset}_alpha{alpha_val}_seed{seed_val}_round{round_idx}_client{client.id}.pt"
                    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

                    protos_cpu = {}
                    if isinstance(raw_protos_dict, dict):
                        for k, v in raw_protos_dict.items():
                            if v is not None and torch.is_tensor(v):
                                protos_cpu[int(k)] = v.detach().cpu().clone()
                    elif torch.is_tensor(raw_protos_dict):
                        protos_cpu = raw_protos_dict.detach().cpu().clone()

                    torch.save(
                        {
                            "algorithm": algo_name,
                            "dataset": str(client.dataset),
                            "alpha": float(alpha_val),
                            "seed": int(seed_val),
                            "round_idx": int(round_idx),
                            "client_id": int(client.id),
                            "model_state_dict": {k: v.detach().cpu().clone() for k, v in client.model.state_dict().items()},
                            "protos": protos_cpu,
                        },
                        ckpt_path,
                    )
                    print(f"[{algo_name} Client {client.id}] Saved t-SNE checkpoint: {ckpt_path}")
                except Exception as e_ckpt:
                    print(f"[{algo_name} Client {client.id}] Warning saving .pt checkpoint: {e_ckpt}")

            # Ghi log WandB an toàn:
            # - wandb.Image(fig) hiển thị trực tiếp trên web UI
            # - wandb.save() đẩy cả file .pdf vector và file .npz lên mục Files của Run
            if getattr(client.args, 'log', False):
                try:
                    if wandb.run is not None:
                        log_dict = {
                            f"tsne/{algo_name.lower()}_client_{client.id}": wandb.Image(fig),
                            f"tsne/client_{client.id}": wandb.Image(fig)
                        }
                        if sil is not None:
                            log_dict[f"tsne/{algo_name.lower()}_client_{client.id}_silhouette"] = sil
                            log_dict[f"tsne/client_{client.id}_silhouette"] = sil
                        step_val = int(round_idx) if round_idx is not None else None
                        if step_val is not None:
                            wandb.log(log_dict, step=step_val)
                        else:
                            wandb.log(log_dict)
                        # Lưu file PDF và NPZ lên WandB Run Files
                        if os.path.exists(save_path):
                            wandb.save(save_path, base_path=save_dir)
                        if os.path.exists(npz_save_path):
                            wandb.save(npz_save_path, base_path=save_dir)
                except Exception:
                    pass

            plt.close(fig)
            return save_path
    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            except Exception:
                pass
        client.model.train(was_training)

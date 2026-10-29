import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def shrink_cov(cov, gamma=1.0, device=None):
    """
    Applies shrinkage regularization to a covariance matrix:
    cov_shrunk = cov + gamma * I
    """
    target_device = device if device is not None else cov.device
    dim = cov.shape[0]
    iden = torch.eye(dim, device=target_device, dtype=cov.dtype)
    return cov.to(target_device) + (gamma * iden)


def solve_fedcof(client_stats_list, num_classes, feature_dim, gamma=1.0, prev_weights=None, device="cpu"):
    """
    Computes optimal linear classifier weights via FedCOF (Federated Learning with Covariances for Free):
    - Estimates intra-class covariance matrices delta_c across clients from client-level class means.
    - Estimates the global Gram matrix est_gram = sum((N_c - 1) * Sigma_c) + N * (mu_g * mu_g^T) + gamma * I.
    - Solves the ridge regression problem W_c = (est_gram)^(-1) * (sum_k N_{k, c} * mu_{k, c}).
    - Normalizes the weight vectors: W_c = W_c / ||W_c||_2.

    Args:
        client_stats_list: List of dicts, each with 'class_means' {c: Tensor(D)} and 'sample_per_mean' {c: int}
        num_classes: int
        feature_dim: int
        gamma: float (shrinkage coefficient, default 1.0)
        prev_weights: Tensor of shape (num_classes, feature_dim), fallback for missing classes
        device: torch device / string

    Returns:
        W: Tensor of shape (num_classes, feature_dim), normalized
    """
    class_mean_sums = {c: torch.zeros(feature_dim, device=device, dtype=torch.float32) for c in range(num_classes)}
    class_sample_counts = {c: 0 for c in range(num_classes)}
    class_mean_clients = {c: [] for c in range(num_classes)}
    class_counts = {c: [] for c in range(num_classes)}

    for user_data in client_stats_list:
        if not user_data:
            continue
        c_means = user_data.get("class_means", {})
        s_counts = user_data.get("sample_per_mean", {})

        for class_id, class_mean in c_means.items():
            cnt = s_counts.get(class_id, 0)
            if cnt > 0 and class_mean is not None:
                m_tensor = class_mean.to(device).float()
                class_mean_sums[class_id] += m_tensor * float(cnt)
                class_sample_counts[class_id] += int(cnt)
                class_mean_clients[class_id].append(m_tensor)
                class_counts[class_id].append(float(cnt))

    class_means = {}
    for c in range(num_classes):
        if class_sample_counts[c] > 0:
            class_means[c] = class_mean_sums[c] / class_sample_counts[c]
        else:
            class_means[c] = torch.zeros(feature_dim, device=device, dtype=torch.float32)

    covs = {}
    all_count = 0
    g_means = torch.zeros(feature_dim, device=device, dtype=torch.float32)

    for c in range(num_classes):
        if class_sample_counts[c] > 0 and len(class_mean_clients[c]) > 0:
            client_means_c = torch.stack(class_mean_clients[c], dim=0)  # (K_c, D)
            counts_c = torch.tensor(class_counts[c], device=device, dtype=torch.float32)
            all_count += class_sample_counts[c]
            g_means += class_means[c] * float(class_sample_counts[c])

            mu_x = client_means_c - class_means[c]  # (K_c, D)
            mu_x *= torch.sqrt(counts_c).unsqueeze(1)
            inside = torch.mm(mu_x.T, mu_x)  # (D, D)

            K_c = len(class_mean_clients[c])
            delta = inside / (K_c - 1) if K_c > 1 else inside
            covs[c] = shrink_cov(delta, gamma=gamma, device=device)

    if all_count > 0:
        g_means = g_means / all_count
        t1 = float(all_count) * torch.mm(g_means.unsqueeze(1), g_means.unsqueeze(0))
    else:
        t1 = torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32)

    est_gram = torch.zeros((feature_dim, feature_dim), device=device, dtype=torch.float32)
    for c in range(num_classes):
        if class_sample_counts[c] > 0 and c in covs:
            weight_factor = max(0, class_sample_counts[c] - 1)
            est_gram += float(weight_factor) * covs[c]

    est_gram = est_gram + t1 + (gamma * torch.eye(feature_dim, device=device, dtype=torch.float32))

    inv_G = torch.linalg.pinv(est_gram).float()

    W_list = []
    for c in range(num_classes):
        if class_sample_counts[c] > 0:
            w_c = torch.matmul(inv_G, class_mean_sums[c].unsqueeze(1)).squeeze(1)
        else:
            if prev_weights is not None and prev_weights.shape[0] > c:
                w_c = prev_weights[c].to(device).float()
            else:
                w_c = torch.randn(feature_dim, device=device, dtype=torch.float32)
        W_list.append(w_c)

    W = torch.stack(W_list, dim=0)
    W = F.normalize(W, p=2, dim=1)

    return W


def solve_fedncm(client_stats_list, num_classes, feature_dim, prev_weights=None, device="cpu"):
    """
    Baseline FedNCM solver (Nearest Class Mean):
    Weights are normalized global class average prototypes:
        W_c = normalize(sum_k N_{k, c} * mu_{k, c} / N_c)
    """
    class_mean_sums = {c: torch.zeros(feature_dim, device=device, dtype=torch.float32) for c in range(num_classes)}
    class_sample_counts = {c: 0 for c in range(num_classes)}

    for user_data in client_stats_list:
        if not user_data:
            continue
        c_means = user_data.get("class_means", {})
        s_counts = user_data.get("sample_per_mean", {})

        for class_id, class_mean in c_means.items():
            cnt = s_counts.get(class_id, 0)
            if cnt > 0 and class_mean is not None:
                class_mean_sums[class_id] += class_mean.to(device).float() * float(cnt)
                class_sample_counts[class_id] += int(cnt)

    W_list = []
    for c in range(num_classes):
        if class_sample_counts[c] > 0:
            w_c = class_mean_sums[c] / float(class_sample_counts[c])
        else:
            if prev_weights is not None and prev_weights.shape[0] > c:
                w_c = prev_weights[c].to(device).float()
            else:
                w_c = torch.randn(feature_dim, device=device, dtype=torch.float32)
        W_list.append(w_c)

    W = torch.stack(W_list, dim=0)
    W = F.normalize(W, p=2, dim=1)
    return W

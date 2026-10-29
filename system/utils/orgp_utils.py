import copy
import itertools
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def agg_func(protos):
    """
    Returns the average prototype per label across samples collected on a client.
    Input:
        protos: dict {label: list_of_tensors}
    Output:
        aggregated: dict {label: mean_tensor}
    """
    aggregated = {}
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            stacked = torch.stack(proto_list)
            aggregated[label] = torch.mean(stacked, dim=0).detach()
        elif len(proto_list) == 1:
            aggregated[label] = proto_list[0].detach()
    return aggregated


def proto_cluster(protos_list):
    """
    Clusters and averages class prototypes across all participating clients.
    Input:
        protos_list: list of dicts [{label: tensor, ...}, ...]
    Output:
        result: dict {label: mean_tensor}
    """
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        if protos is None:
            continue
        for k, v in protos.items():
            if isinstance(v, torch.Tensor):
                proto_clusters[k].append(v.detach().cpu())

    result = {}
    for k, v_list in proto_clusters.items():
        if len(v_list) > 0:
            result[k] = torch.mean(torch.stack(v_list), dim=0).detach()

    return result


def filter_outliers_multi_metric(features, alpha=0.5, min_keep_ratio=0.5):
    """
    Adaptive multi-metric feature outlier filtering for FedORGP / FedOrth.
    Combines Euclidean distance and cosine similarity using dynamic coefficient
    of variation (CV) weighting, followed by IQR-based outlier rejection.

    Args:
        features: list of 1D feature tensors for a single class.
        alpha: base weight for combining distance and cosine similarity scores (0-1).
        min_keep_ratio: minimum fraction of features to retain (guards against over-filtering).

    Returns:
        filtered: list of retained 1D feature tensors.
    """
    if len(features) <= 2:
        return features

    # Stack features onto CPU for robust metric calculations
    feats = torch.stack(features).cpu()
    n_samples = feats.shape[0]

    # Mean prototype center
    center = torch.mean(feats, dim=0)

    # Euclidean distance from center
    dists = torch.norm(feats - center, dim=1)

    # Cosine similarity to center
    feats_norm = F.normalize(feats, p=2, dim=1)
    center_norm = F.normalize(center.unsqueeze(0), p=2, dim=1)
    cos_sims = torch.matmul(feats_norm, center_norm.t()).squeeze(1)

    # Compute variation coefficients (CV = std / mean)
    dist_std = torch.clamp(torch.std(dists), min=1e-8)
    cos_std = torch.clamp(torch.std(cos_sims), min=1e-8)
    dist_mean = torch.clamp(torch.mean(dists), min=1e-8)
    cos_mean = torch.clamp(torch.abs(torch.mean(cos_sims)), min=1e-8)

    dist_cv = dist_std / dist_mean
    cos_cv = cos_std / cos_mean
    cv_ratio = dist_cv / cos_cv

    # Dynamic smooth adjustment factor using tanh
    adj_factor = 1.0 - 0.3 * torch.tanh(0.5 * (cv_ratio - 1.0)).item()
    adj_factor = max(0.4, min(1.6, adj_factor))

    # Apply adaptive alpha clamped within [0.1, 0.9]
    adaptive_alpha = max(0.1, min(0.9, alpha * adj_factor))

    # Normalized scores in [0, 1]
    max_d = torch.max(dists)
    min_d = torch.min(dists)
    dist_scores = 1.0 - (dists - min_d) / torch.clamp(max_d - min_d, min=1e-8)

    max_c = torch.max(cos_sims)
    min_c = torch.min(cos_sims)
    cos_scores = (cos_sims - min_c) / torch.clamp(max_c - min_c, min=1e-8)

    combined_scores = adaptive_alpha * dist_scores + (1.0 - adaptive_alpha) * cos_scores

    # IQR-based outlier thresholding
    q1 = torch.quantile(combined_scores, 0.25)
    q3 = torch.quantile(combined_scores, 0.75)
    iqr = q3 - q1

    low_threshold = q1 - 1.5 * iqr
    high_threshold = q3 + 1.5 * iqr

    keep_idx = (combined_scores >= low_threshold) & (combined_scores <= high_threshold)
    n_keep = torch.sum(keep_idx).item()
    min_keep = max(1, int(n_samples * min_keep_ratio))

    if n_keep < min_keep:
        _, top_indices = torch.topk(combined_scores, min_keep)
        keep_idx = torch.zeros(n_samples, dtype=torch.bool)
        keep_idx[top_indices] = True

    filtered = [features[i] for i in range(len(features)) if keep_idx[i]]
    if len(filtered) == 0:
        best_idx = torch.argmax(combined_scores).item()
        filtered = [features[best_idx]]

    return filtered


def _prepare_protos_tensor(protos, num_classes, feature_dim, device):
    """
    Helper to convert dictionary or tensor prototypes into a normalized (num_classes, feature_dim) tensor.
    Guarantees consistent class ordering 0 .. num_classes-1.
    """
    if isinstance(protos, dict):
        proto_list = []
        for c in range(num_classes):
            if c in protos and isinstance(protos[c], torch.Tensor):
                p = protos[c].to(device)
                if p.dim() > 1:
                    p = p.squeeze()
                proto_list.append(p)
            else:
                proto_list.append(torch.zeros(feature_dim, device=device))
        protos_tensor = torch.stack(proto_list)
    elif isinstance(protos, torch.Tensor):
        protos_tensor = protos.to(device)
    else:
        raise TypeError(f"Unsupported protos type: {type(protos)}")

    return protos_tensor


def intra_orth_loss(rep, labels, protos, adaptive=False):
    """
    Calculates intra-class alignment loss between representations and their corresponding class prototypes.
    Optionally applies dynamic adaptive weights based on repulsion from other class prototypes.

    Args:
        rep (Tensor): Feature representation tensor of shape (batch_size, feature_dim).
        labels (Tensor): Target class labels of shape (batch_size,).
        protos (dict or Tensor): Prototypes keyed by class or stacked as (num_classes, feature_dim).
        adaptive (bool): Whether to weight intra-loss adaptively based on other class similarities.

    Returns:
        Tensor: Scalar intra-class alignment loss.
    """
    if rep.dim() > 2:
        rep = rep.view(rep.size(0), -1)

    device = rep.device
    feature_dim = rep.shape[1]
    batch_size = rep.shape[0]

    if batch_size == 0:
        return torch.tensor(0.0, device=device)

    labels = labels.to(device=device, dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    if isinstance(protos, dict):
        num_classes = max(num_classes, max(protos.keys()) + 1 if len(protos) > 0 else num_classes)
    elif isinstance(protos, torch.Tensor):
        num_classes = max(num_classes, protos.shape[0])

    protos_tensor = _prepare_protos_tensor(protos, num_classes, feature_dim, device)

    rep_norm = F.normalize(rep, p=2, dim=1)
    protos_norm = F.normalize(protos_tensor, p=2, dim=1)

    # Sample's corresponding prototype
    proto_of_sample = protos_norm[labels]

    # Cosine similarity with target prototype: want similarity -> 1, so minimize (1 - cos)
    similarity_intra = torch.sum(rep_norm * proto_of_sample, dim=1)
    loss_intra = 1.0 - similarity_intra

    if adaptive and num_classes > 1:
        # Measure similarity against all prototypes
        similarity_all = torch.abs(torch.matmul(rep_norm, protos_norm.t()))
        mask = torch.ones_like(similarity_all)
        mask.scatter_(1, labels.view(-1, 1), 0.0)

        other_class_sim_matrix = similarity_all * mask
        non_zero_counts = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        row_sums = torch.sum(other_class_sim_matrix, dim=1)
        other_class_sim = torch.div(row_sums, non_zero_counts)

        # Dynamic temperature / sigmoid weighting
        adaptive_weights = 1.0 + torch.sigmoid((other_class_sim - 0.3) / 0.5)
        weighted_loss = loss_intra * adaptive_weights
        return torch.mean(weighted_loss)
    else:
        return torch.mean(loss_intra)


def inter_orth_loss(rep, labels, protos, adaptive=True):
    """
    Calculates inter-class orthogonality loss forcing representations and prototypes
    of different classes to be mutually orthogonal (absolute cosine similarity -> 0).

    Args:
        rep (Tensor): Feature representation tensor of shape (batch_size, feature_dim).
        labels (Tensor): Target class labels of shape (batch_size,).
        protos (dict or Tensor): Prototypes keyed by class or stacked as (num_classes, feature_dim).
        adaptive (bool): Whether to scale inter-loss with non-linear penalty.

    Returns:
        Tensor: Scalar inter-class orthogonality loss.
    """
    if rep.dim() > 2:
        rep = rep.view(rep.size(0), -1)

    device = rep.device
    feature_dim = rep.shape[1]
    batch_size = rep.shape[0]

    if batch_size == 0:
        return torch.tensor(0.0, device=device)

    labels = labels.to(device=device, dtype=torch.long)
    num_classes = int(labels.max().item()) + 1
    if isinstance(protos, dict):
        num_classes = max(num_classes, max(protos.keys()) + 1 if len(protos) > 0 else num_classes)
    elif isinstance(protos, torch.Tensor):
        num_classes = max(num_classes, protos.shape[0])

    if num_classes <= 1:
        return torch.tensor(0.0, device=device)

    protos_tensor = _prepare_protos_tensor(protos, num_classes, feature_dim, device)

    rep_norm = F.normalize(rep, p=2, dim=1)
    protos_norm = F.normalize(protos_tensor, p=2, dim=1)

    # Absolute cosine similarity between representations and all prototypes
    similarity_all = torch.abs(torch.matmul(rep_norm, protos_norm.t()))

    # Mask out the ground truth class to isolate other classes
    mask = torch.ones_like(similarity_all)
    mask.scatter_(1, labels.view(-1, 1), 0.0)

    if adaptive:
        adaptive_weights = 1.0 + torch.sigmoid((similarity_all - 0.3) / 0.5)
        weighted_similarity = similarity_all * adaptive_weights * mask
        denom = torch.clamp(torch.sum(mask), min=1.0)
        return torch.sum(weighted_similarity) / denom
    else:
        similarity_other = similarity_all * mask
        denom = torch.clamp(torch.sum(mask), min=1.0)
        return torch.sum(similarity_other) / denom


class Trainable_Orthogonal_Prototypes(nn.Module):
    """
    Trainable Global Prototypes Generator for FedORGP / FedOrth.
    Projects discrete class identifiers through an embedding and MLP into
    the continuous feature representation space where mutual orthogonality is enforced.
    """
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [
            nn.Sequential(
                nn.Linear(feature_dim, server_hidden_dim),
                nn.ReLU()
            )
        ]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        if not isinstance(class_id, torch.Tensor):
            class_id = torch.tensor(class_id, dtype=torch.long, device=self.device)
        else:
            class_id = class_id.to(device=self.device, dtype=torch.long)

        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out

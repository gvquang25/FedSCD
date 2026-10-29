import copy
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_scaled_softmax(logits, temperature=0.6):
    """
    Computes temperature-scaled softmax probabilities:
    Q = softmax(Z / temp, dim=-1)
    """
    temp = max(1e-4, float(temperature))
    return F.softmax(logits / temp, dim=-1)


def temperature_scaled_log_softmax(logits, temperature=0.6):
    """
    Computes temperature-scaled log-softmax for numerical stability in KL divergence:
    log_Q = log_softmax(Z / temp, dim=-1)
    """
    temp = max(1e-4, float(temperature))
    return F.log_softmax(logits / temp, dim=-1)


def kd_loss(student_logits, teacher_probs, temperature=0.6):
    """
    Computes Knowledge Distillation KL-divergence loss with tau^2 gradient scaling:
    L_KD = tau^2 * KL( log_softmax(Z_s / tau) || P_t )

    Args:
        student_logits (Tensor): shape (B, C) or (C, C).
        teacher_probs (Tensor): shape matching student, probability distribution over classes.
        temperature (float): distillation temperature tau.

    Returns:
        Tensor: scalar loss.
    """
    temp = max(1e-4, float(temperature))
    log_student = temperature_scaled_log_softmax(student_logits, temp)

    # Ensure teacher_probs is detached and clamped for numerical stability
    teacher_probs = teacher_probs.detach().clamp(min=1e-8)
    # Normalize teacher probs along last dim if needed
    teacher_probs = teacher_probs / torch.clamp(teacher_probs.sum(dim=-1, keepdim=True), min=1e-8)

    kl = F.kl_div(log_student, teacher_probs, reduction='batchmean')
    return kl * (temp ** 2)


def feature_alignment_loss(local_features, target_features):
    """
    Computes Mean Squared Error (MSE) feature alignment loss between local sample
    representations and target global class prototypes:
    L_feat = MSE(F_local, F_target)

    Args:
        local_features (Tensor): shape (B, D).
        target_features (Tensor): shape (B, D).

    Returns:
        Tensor: scalar MSE loss.
    """
    if local_features.dim() > 2:
        local_features = local_features.view(local_features.size(0), -1)
    if target_features.dim() > 2:
        target_features = target_features.view(target_features.size(0), -1)

    return F.mse_loss(local_features, target_features)


def compute_client_diversity_weights(class_counts_list, nks_list, num_classes, param_a=0.2, param_b=0.2):
    """
    Computes client aggregation weights (p_k) based on sample volume and
    Dirichlet distribution skew relative to uniform class distribution:
    D_kc = n_kc / (n_k + eps)
    d_k = || D_k - 1/C ||_2
    w_k = max(0, n_k - a * d_k + b)
    p_k = w_k / sum(w)

    Args:
        class_counts_list (list of dict): class sample counts for each selected client.
        nks_list (list of int): total sample counts for each selected client.
        num_classes (int): number of dataset classes C.
        param_a (float): penalty factor for Non-IID skew.
        param_b (float): positive offset.

    Returns:
        pks (list of float): normalized aggregation weights summing to 1.0.
    """
    num_clients = len(class_counts_list)
    if num_clients == 0:
        return []

    target_uniform = 1.0 / max(1, num_classes)
    raw_weights = []

    for idx in range(num_clients):
        nk = nks_list[idx]
        class_dict = class_counts_list[idx]

        # Compute L2 distance from uniform distribution
        dist_sq = 0.0
        for c in range(num_classes):
            c_cnt = class_dict.get(c, 0)
            p_c = c_cnt / max(1.0, float(nk))
            dist_sq += (p_c - target_uniform) ** 2
        dk = math.sqrt(dist_sq)

        # Skew-adjusted weight with ReLU activation
        w = max(0.0, float(nk) - param_a * dk + param_b)
        raw_weights.append(w)

    total_w = sum(raw_weights)
    if total_w > 1e-8:
        pks = [w / total_w for w in raw_weights]
    else:
        pks = [1.0 / num_clients] * num_clients

    return pks


def build_knowledge_tensors(global_features_dict, global_soft_dict, num_classes, feature_dim, device):
    """
    Converts dictionary representations of global features and soft predictions
    into unified, ordered PyTorch tensors:
    - tensor_features: shape (num_classes, feature_dim)
    - tensor_soft_preds: shape (num_classes, num_classes)
    """
    feat_list = []
    soft_list = []

    for c in range(num_classes):
        # Features
        if c in global_features_dict and isinstance(global_features_dict[c], torch.Tensor):
            feat = global_features_dict[c].to(device)
            if feat.dim() > 1:
                feat = feat.view(-1)
            feat_list.append(feat)
        else:
            feat_list.append(torch.zeros(feature_dim, device=device))

        # Soft predictions
        if c in global_soft_dict and isinstance(global_soft_dict[c], torch.Tensor):
            soft = global_soft_dict[c].to(device)
            if soft.dim() > 1:
                soft = soft.view(-1)
            soft_list.append(soft)
        else:
            # Default to uniform probability
            uniform = torch.full((num_classes,), 1.0 / max(1, num_classes), device=device)
            soft_list.append(uniform)

    tensor_features = torch.stack(feat_list)
    tensor_soft_preds = torch.stack(soft_list)

    return tensor_features, tensor_soft_preds


def aggregate_knowledge_initial(local_features_list, local_soft_list, num_classes, feature_dim, device):
    """
    Round 0 global knowledge aggregation:
    Computes unweighted average of local feature prototypes and soft predictions per class.
    """
    global_features = {}
    global_soft_prediction = {}

    for c in range(num_classes):
        feats_c = []
        softs_c = []
        for client_idx in range(len(local_features_list)):
            if c in local_features_list[client_idx]:
                f = local_features_list[client_idx][c].detach().cpu()
                if f.dim() > 1:
                    f = f.view(-1)
                feats_c.append(f)

            if c in local_soft_list[client_idx]:
                s = local_soft_list[client_idx][c].detach().cpu()
                if s.dim() > 1:
                    s = s.view(-1)
                softs_c.append(s)

        if len(feats_c) > 0:
            global_features[c] = torch.mean(torch.stack(feats_c), dim=0)
        else:
            global_features[c] = torch.zeros(feature_dim)

        if len(softs_c) > 0:
            global_soft_prediction[c] = torch.mean(torch.stack(softs_c), dim=0)
        else:
            global_soft_prediction[c] = torch.full((num_classes,), 1.0 / max(1, num_classes))

    return global_features, global_soft_prediction


def aggregate_knowledge_adaptive(
    local_features_list,
    local_soft_list,
    prev_global_features,
    prev_global_soft,
    nks_list,
    num_classes,
    feature_dim,
    param_a=0.4,
    param_b=0.2,
    device='cpu'
):
    """
    Round > 0 adaptive knowledge aggregation for FedGMKD:
    Computes class-specific client aggregation weights based on KL divergence / L2 distance
    from the previous round's global representations, then computes weighted average.
    """
    num_clients = len(local_features_list)
    if num_clients == 0:
        return prev_global_features, prev_global_soft

    global_features = {}
    global_soft_prediction = {}

    for c in range(num_classes):
        # Target representation from previous round
        prev_f = prev_global_features.get(c, torch.zeros(feature_dim)).to(device)
        prev_s = prev_global_soft.get(c, torch.full((num_classes,), 1.0 / max(1, num_classes))).to(device)

        client_indices_with_c = []
        divergences = []

        for idx in range(num_clients):
            if c in local_features_list[idx] and c in local_soft_list[idx]:
                client_indices_with_c.append(idx)
                f_k = local_features_list[idx][c].to(device)
                s_k = local_soft_list[idx][c].to(device)

                # Compute KL divergence between normalized soft predictions
                log_s_k = F.log_softmax(s_k.view(1, -1), dim=-1)
                p_prev_s = F.softmax(prev_s.view(1, -1), dim=-1)
                div_s = F.kl_div(log_s_k, p_prev_s, reduction='batchmean').item()

                # Optional L2 feature distance
                div_f = torch.norm(f_k - prev_f, p=2).item() if prev_f.norm() > 0 else 0.0

                combined_div = max(0.0, div_s + 0.1 * div_f)
                divergences.append(combined_div)

        if len(client_indices_with_c) == 0:
            # Fallback to previous round representation
            global_features[c] = prev_f.detach().cpu()
            global_soft_prediction[c] = prev_s.detach().cpu()
            continue

        # Compute adaptive weights per class
        raw_weights = []
        for i, client_idx in enumerate(client_indices_with_c):
            nk = nks_list[client_idx]
            div = divergences[i]
            w = max(0.0, float(nk) - param_a * div + param_b)
            raw_weights.append(w)

        total_w = sum(raw_weights)
        if total_w > 1e-8:
            norm_weights = [w / total_w for w in raw_weights]
        else:
            norm_weights = [1.0 / len(raw_weights)] * len(raw_weights)

        # Weighted sum of features and soft predictions
        agg_f = torch.zeros(feature_dim, device=device)
        agg_s = torch.zeros(num_classes, device=device)

        for i, client_idx in enumerate(client_indices_with_c):
            w = norm_weights[i]
            agg_f += w * local_features_list[client_idx][c].to(device).view(-1)
            agg_s += w * local_soft_list[client_idx][c].to(device).view(-1)

        global_features[c] = agg_f.detach().cpu()
        global_soft_prediction[c] = agg_s.detach().cpu()

    return global_features, global_soft_prediction

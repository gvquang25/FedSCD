import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianSynthesisLabel(nn.Module):
    """
    Gaussian Feature Synthesis Module for FedImpro (DecoupleGradVariance).
    Maintains class-wise mean prototypes and standard deviations to sample
    synthetic feature representations for classifier head regularization.
    """

    def __init__(self, num_classes, feature_dim, noise_std=0.1, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.noise_std = noise_std
        self.device = device

        self.feature_means = torch.zeros(num_classes, feature_dim, device=device)
        self.feature_stds = torch.full((num_classes, feature_dim), noise_std, device=device)

    def set_params(self, means, stds, device=None):
        if device is not None:
            self.device = device
        if isinstance(means, torch.Tensor):
            self.feature_means = means.to(self.device).float()
        elif isinstance(means, np.ndarray):
            self.feature_means = torch.from_numpy(means).to(self.device).float()

        if isinstance(stds, torch.Tensor):
            self.feature_stds = stds.to(self.device).float()
        elif isinstance(stds, np.ndarray):
            self.feature_stds = torch.from_numpy(stds).to(self.device).float()

        self.feature_stds = torch.clamp(self.feature_stds, min=1e-6, max=max(self.noise_std, 1.0))

    def sample(self, repeat_times=1, device=None):
        """
        Samples synthetic feature vectors from N(feature_means, feature_stds^2).
        Returns:
            fake_features: Tensor of shape (num_classes * repeat_times, feature_dim)
            fake_labels: Tensor of shape (num_classes * repeat_times,)
        """
        target_device = device if device is not None else self.device
        repeat_times = max(1, int(repeat_times))

        means = self.feature_means.repeat(repeat_times, 1).to(target_device)
        stds = self.feature_stds.repeat(repeat_times, 1).to(target_device)

        fake_features = torch.normal(mean=means, std=stds)
        fake_labels = torch.arange(self.num_classes, device=target_device).repeat(repeat_times)

        return fake_features, fake_labels


def aggregate_feature_statistics(local_stats_list, num_classes, max_noise_std=0.1, prev_stats=None):
    """
    Server-side aggregation of local feature statistics (Means and Variances)
    using the Law of Total Variance (Pooled Variance):
        mean_c = sum(w_k * mean_{k, c})
        var_c  = sum(w_k * var_{k, c}) + sum(w_k * (mean_{k, c} - mean_c)^2)
        std_c  = sqrt(var_c)
    """
    if not local_stats_list:
        return prev_stats

    # Determine feature dimension from first available non-empty client stat
    feature_dim = None
    for stat in local_stats_list:
        if stat and "means" in stat and len(stat["means"]) > 0:
            for k, m in stat["means"].items():
                feature_dim = m.shape[-1]
                break
        if feature_dim is not None:
            break

    if feature_dim is None:
        return prev_stats

    global_means = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
    global_stds = torch.full((num_classes, feature_dim), max_noise_std, dtype=torch.float32)

    if prev_stats is not None:
        if "means" in prev_stats and isinstance(prev_stats["means"], torch.Tensor):
            global_means = prev_stats["means"].clone().cpu().float()
        if "stds" in prev_stats and isinstance(prev_stats["stds"], torch.Tensor):
            global_stds = prev_stats["stds"].clone().cpu().float()

    for c in range(num_classes):
        client_means = []
        client_stds = []
        client_counts = []

        for stat in local_stats_list:
            if not stat or "counts" not in stat:
                continue
            cnt = stat["counts"].get(c, 0)
            if cnt > 0 and c in stat["means"] and c in stat["stds"]:
                client_means.append(stat["means"][c].cpu().float())
                client_stds.append(stat["stds"][c].cpu().float())
                client_counts.append(cnt)

        total_cnt = sum(client_counts)
        if total_cnt > 0:
            # Weighted mean
            weights = [cnt / total_cnt for cnt in client_counts]
            pooled_mean = torch.zeros(feature_dim, dtype=torch.float32)
            for w, m in zip(weights, client_means):
                pooled_mean += w * m

            # Law of total variance: Intra-client variance + Inter-client variance
            pooled_var = torch.zeros(feature_dim, dtype=torch.float32)
            for w, s, m in zip(weights, client_stds, client_means):
                intra_var = s ** 2
                inter_var = (m - pooled_mean) ** 2
                pooled_var += w * (intra_var + inter_var)

            pooled_std = torch.sqrt(torch.clamp(pooled_var, min=1e-8))
            if max_noise_std > 0:
                pooled_std = torch.clamp(pooled_std, max=max_noise_std)

            global_means[c] = pooled_mean
            global_stds[c] = pooled_std
        else:
            # Missing class in this round: preserve previous stats or default
            if prev_stats is not None and "means" in prev_stats:
                global_means[c] = prev_stats["means"][c].cpu().float()
                global_stds[c] = prev_stats["stds"][c].cpu().float()

    return {
        "means": global_means,
        "stds": global_stds
    }


def cov_non_diag_norm(x):
    """
    Covariance non-diagonal norm loss for feature representation regularization:
    penalizes cross-channel / off-diagonal correlation among features.
    """
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    b, d = x.size()
    if b <= 1:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    x_norm = x - x.mean(dim=0, keepdim=True)
    cov = torch.mm(x_norm.t(), x_norm) / (b - 1)
    eye = torch.eye(d, device=x.device)
    non_diag_cov = cov * (1.0 - eye)
    return (non_diag_cov ** 2).mean()


class Distance_loss(nn.Module):
    """
    Feature distance / alignment loss (Cosine or L2 norm).
    """

    def __init__(self, distance="cosine"):
        super().__init__()
        self.distance = distance

    def forward(self, feat_real, feat_fake):
        if feat_real.dim() > 2:
            feat_real = feat_real.view(feat_real.size(0), -1)
        if feat_fake.dim() > 2:
            feat_fake = feat_fake.view(feat_fake.size(0), -1)

        min_len = min(feat_real.size(0), feat_fake.size(0))
        if min_len == 0:
            return torch.tensor(0.0, device=feat_real.device, requires_grad=True)

        f1 = feat_real[:min_len]
        f2 = feat_fake[:min_len]

        if self.distance == "cosine":
            cos_sim = F.cosine_similarity(f1, f2, dim=-1)
            return (1.0 - cos_sim).mean()
        elif self.distance == "L2_norm":
            return (f1 - f2).norm(p=2, dim=-1).mean()
        else:
            return (f1 - f2).norm(p=2, dim=-1).mean()

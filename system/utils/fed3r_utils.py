import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def calc_ridge_params(A, b):
    """
    Solves the linear ridge regression system A @ W = b.
    Separates the weight matrix and bias vector:
        W = W_full[:-1, :]
        bias = W_full[-1, :]
    """
    try:
        W_full = torch.linalg.solve(A, b)
    except Exception:
        W_full = torch.matmul(torch.linalg.pinv(A), b)

    bias = W_full[-1, :]
    W = W_full[:-1, :]
    return W, bias


def normalize_W(W, bias=None, eps=1e-6):
    """
    Normalizes weight columns and scales the corresponding bias:
        W_c = W_c / ||W_c||_2
        bias_c = bias_c / ||W_c||_2
    """
    norm = torch.norm(W, dim=0, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    W_norm = W / norm

    if bias is not None:
        bias_norm = bias / norm[0, :]
    else:
        bias_norm = None

    return W_norm, bias_norm


def fed3r_step(A, b, dataloader, model_base, num_classes, device, use_bias=True):
    """
    Computes and accumulates local Gram matrix Ak and cross-product bk:
        Ak = sum_i z_i @ z_i^T
        bk = sum_i z_i @ y_i^T
    """
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if isinstance(data, list):
                data = data[0].to(device)
            else:
                data = data.to(device)
            target = target.to(device)

            rep = model_base(data)
            if rep.dim() > 2:
                rep = rep.view(rep.size(0), -1)

            if use_bias:
                bias_ones = torch.ones((rep.shape[0], 1), dtype=torch.float32, device=device)
                feats = torch.cat((rep, bias_ones), dim=1)
            else:
                feats = rep

            one_hot = F.one_hot(target, num_classes=num_classes).float()

            Ak = feats.T @ feats
            bk = feats.T @ one_hot

            A += Ak
            b += bk

    return A, b


def solve_fed3r(A_list, b_list, feature_dim, num_classes, lamb=0.01, use_bias=True, normalize=True, device="cpu"):
    """
    Server-side aggregation and closed-form ridge solution for Fed3R:
    - Sums client Gram matrices: A = sum_k A_k + lambda * I
    - Sums client cross-product matrices: b = sum_k b_k
    - Solves: W, bias = calc_ridge_params(A, b)
    - Normalizes W and bias.

    Returns:
        W_final: Tensor of shape (num_classes, feature_dim)
        bias_final: Tensor of shape (num_classes,) or None
    """
    target_dim = (feature_dim + 1) if use_bias else feature_dim

    A_total = torch.zeros((target_dim, target_dim), dtype=torch.float32, device=device)
    b_total = torch.zeros((target_dim, num_classes), dtype=torch.float32, device=device)

    for Ak, bk in zip(A_list, b_list):
        if Ak is not None and bk is not None:
            A_total += Ak.to(device).float()
            b_total += bk.to(device).float()

    # Ridge regularization: add lambda * I
    A_reg = A_total + (lamb * torch.eye(target_dim, dtype=torch.float32, device=device))

    if use_bias:
        W, bias = calc_ridge_params(A_reg, b_total)
        if normalize:
            W, bias = normalize_W(W, bias)
        # W has shape (feature_dim, num_classes), transpose to (num_classes, feature_dim)
        W_final = W.T
        bias_final = bias
    else:
        try:
            W = torch.linalg.solve(A_reg, b_total)
        except Exception:
            W = torch.matmul(torch.linalg.pinv(A_reg), b_total)
        if normalize:
            norm = torch.norm(W, dim=0, keepdim=True).clamp(min=1e-6)
            W = W / norm
        W_final = W.T
        bias_final = None

    return W_final, bias_final

import copy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_semantic_projector(feat_channel):
    """
    Builds the projector MLP used to map random class centers to initial semantic anchors.
    Uses pure PyTorch (Linear -> BatchNorm1d -> ReLU -> Linear -> BatchNorm1d -> ReLU).
    Does not require external dependencies like mmcv.
    """
    return nn.Sequential(
        nn.Linear(feat_channel, feat_channel),
        nn.BatchNorm1d(feat_channel),
        nn.ReLU(inplace=True),
        nn.Linear(feat_channel, feat_channel),
        nn.BatchNorm1d(feat_channel),
        nn.ReLU(inplace=True)
    )


def proto_aggregation(local_protos_list):
    """
    Aggregates prototypes across all clients by averaging per class label.
    Input:
        local_protos_list: list of dicts [{class_id: tensor, ...}, ...]
    Output:
        agg_protos: dict {class_id: tensor}
    """
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        if local_protos is None:
            continue
        for label, proto in local_protos.items():
            if isinstance(proto, torch.Tensor):
                agg_protos_label[label].append(proto.detach().cpu())

    agg_protos = {}
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            agg_protos[label] = torch.stack(proto_list, dim=0).mean(dim=0).detach()
        elif len(proto_list) == 1:
            agg_protos[label] = proto_list[0].detach()

    return agg_protos


def proto2anchors(global_protos, class_numbers, feat_channel, device='cpu'):
    """
    Converts a prototype dictionary into a dense anchor tensor of shape (class_numbers, feat_channel).
    """
    anchors = torch.zeros(class_numbers, feat_channel, device=device)
    for label, proto in global_protos.items():
        if isinstance(proto, torch.Tensor) and int(label) < class_numbers:
            anchors[int(label)] = proto.to(device)
    return anchors


def cal_class_margins(anchors):
    """
    Calculates pairwise distance matrix and class margin statistics (min, avg, max)
    between anchor vectors.
    """
    if anchors is None or anchors.size(0) < 2:
        return 0.0, 0.0, 0.0, None

    dist_matrix = torch.cdist(anchors, anchors, p=2)
    dist_matrix.fill_diagonal_(float('inf'))

    min_margins = torch.min(dist_matrix, dim=1).values
    min_margin = torch.min(min_margins).item()
    max_margin = torch.max(min_margins).item()
    avg_margin = torch.mean(min_margins).item()

    return min_margin, avg_margin, max_margin, dist_matrix

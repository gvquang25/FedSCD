import copy
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
        protos: dict {label: mean_tensor}
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
    Clusters and averages prototypes per label across all uploaded clients.
    Input:
        protos_list: list of dicts [{label: tensor, ...}, ...]
    Output:
        proto_clusters: dict {label: mean_tensor}
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


class Trainable_Global_Prototypes(nn.Module):
    """
    Trainable Global Prototypes (TGP) module for FedTGP:
    Embeds class IDs and projects them through a middle MLP layer into the shared prototype feature space.
    """
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
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

import copy
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client
from utils.orgp_utils import (
    agg_func,
    filter_outliers_multi_metric,
    intra_orth_loss
)


class clientORGP(Client):
    """
    Client for FedORGP (Federated Orthogonal Global Prototypes):
    - Inherits from standardized Client base class.
    - Decoupled base (feature representation extractor) and classification head.
    - Trains locally with Cross-Entropy Loss + Intra-Class Prototype Alignment Loss against Global Prototypes.
    - Applies adaptive multi-metric outlier filtering (distance + cosine similarity) before prototype aggregation.
    - Extracts and uploads class prototypes to Server after local training.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()

        # Hyperparameters for FedORGP
        raw_clamda = getattr(args, 'c_lamda', getattr(args, 'orgp_c_lamda', 1.0))
        self.c_lamda = float(raw_clamda) if raw_clamda is not None else 1.0

        raw_alpha = getattr(args, 'orgp_alpha', getattr(args, 'alpha', 0.5))
        self.alpha = float(raw_alpha) if raw_alpha is not None else 0.5

        self.ablation_client = bool(getattr(args, 'ablation_client', False))

        self.global_protos = None
        self.protos = None
        self.local_stats = {}

        # Split model into base (representation extractor) and head (classifier)
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            self.base = self.model.base
            self.head = self.model.head
        elif hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            self.base = self.model.features
            self.head = self.model.classifier
        else:
            self.base = self.model
            self.head = nn.Identity()

    def set_protos(self, global_protos):
        """
        Receives global orthogonal prototypes dispatched from the server.
        """
        self.global_protos = global_protos

    def train(self):
        """
        Local client training:
        1. Optimizes model using Cross-Entropy Loss + Intra-Class Prototype Loss.
        2. Collects representations and extracts class prototypes with adaptive outlier filtering.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                output = self.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None and len(self.global_protos) > 0:
                    loss_intra = intra_orth_loss(rep, y, self.global_protos, adaptive=False)
                    loss = loss + self.c_lamda * loss_intra

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Collect local prototypes after training epochs
        self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        """
        Collects feature representations from local training data.
        Applies adaptive multi-metric feature outlier filtering (if not ablation_client),
        and computes class prototypes.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        protos_raw = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                for idx, yy in enumerate(y):
                    y_c = yy.item()
                    protos_raw[y_c].append(rep[idx].detach().cpu())

        # Adaptive outlier filtering
        if not self.ablation_client:
            for label in list(protos_raw.keys()):
                protos_raw[label] = filter_outliers_multi_metric(
                    protos_raw[label],
                    alpha=self.alpha
                )

        # Average filtered prototypes per class
        self.protos = agg_func(protos_raw)

        # Store local statistics for logging and inspection
        local_stats = {}
        for k, proto in self.protos.items():
            local_stats[k] = {
                'mean': proto,
                'count': len(protos_raw[k])
            }
        self.local_stats = local_stats

        return self.protos

    def test_metrics(self):
        """
        Evaluates local model accuracy and AUC on test dataset using classification head.
        """
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                output = self.head(rep)
                preds = torch.argmax(output, dim=1)
                test_acc += (torch.sum(preds == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        auc = 0.0
        if len(y_prob) > 0 and test_num > 0:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            try:
                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            except Exception:
                auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        """
        Evaluates local loss across training dataset.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0.0

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                output = self.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None and len(self.global_protos) > 0:
                    loss_intra = intra_orth_loss(rep, y, self.global_protos, adaptive=False)
                    loss = loss + self.c_lamda * loss_intra

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

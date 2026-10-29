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
from utils.tgp_utils import agg_func


class clientTGP(Client):
    """
    Client for FedTGP (Trainable Global Prototypes):
    - Inherits from standard Client base class.
    - Decoupled base (feature representation extractor) and classification head.
    - Trains locally with Cross-Entropy Loss + Prototype MSE Loss against Global Prototypes.
    - Collects and uploads class prototypes to Server after local training.
    - Evaluates accuracy using distance-to-prototype nearest centroid classification.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        raw_lamda = getattr(args, 'lamda', 10.0)
        self.lamda = float(raw_lamda) if raw_lamda is not None else 10.0

        self.global_protos = None
        self.protos = None
        self.local_stats = {}

        # Split model into base and head if available
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
        Receives global prototypes dispatched from the server.
        """
        self.global_protos = global_protos

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        # Cache global prototypes to device if present
        gp_device = None
        if self.global_protos is not None and len(self.global_protos) > 0:
            gp_device = {
                k: v.to(self.device)
                for k, v in self.global_protos.items()
                if isinstance(v, torch.Tensor)
            }

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.base(x_in)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                output = self.head(rep)
                loss = self.loss(output, y)

                if gp_device is not None:
                    proto_new = rep.detach().clone()
                    for idx, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in gp_device:
                            proto_new[idx, :] = gp_device[y_c]
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.collect_protos()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        """
        Extracts feature representations from local training data and computes
        the average representation (prototype) for each class, as well as local_stats.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.base(x_in)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                for idx, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[idx, :].detach().cpu())

        self.protos = agg_func(protos)

        # Populate local_stats for seamless compatibility with motivation metrics
        local_stats = {}
        for k, v_list in protos.items():
            if len(v_list) > 0:
                stacked = torch.stack(v_list)
                mean_c = stacked.mean(dim=0).cpu()
                std_c = stacked.std(dim=0, unbiased=False).cpu() if len(v_list) > 1 else torch.full_like(mean_c, 1e-6)
                local_stats[k] = {'mean': mean_c, 'std': std_c, 'count': len(v_list)}
        self.local_stats = local_stats

        return self.protos

    def test_metrics(self):
        """
        Evaluates local model accuracy and AUC.
        If global prototypes are available, uses nearest prototype MSE distance classification (FedTGP metric).
        Otherwise falls back to the classification head.
        """
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        if self.global_protos is not None and len(self.global_protos) > 0:
            gp_device = {
                k: v.to(self.device)
                for k, v in self.global_protos.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                for x, y in testloader:
                    if isinstance(x, list):
                        x_in = x[0].to(self.device)
                    else:
                        x_in = x.to(self.device)
                    y = y.to(self.device)

                    rep = self.base(x_in)
                    if rep.dim() > 2:
                        rep = rep.view(rep.size(0), -1)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes, device=self.device)
                    for j, pro in gp_device.items():
                        if j < self.num_classes:
                            output[:, j] = torch.mean((rep - pro.unsqueeze(0)) ** 2, dim=1)

                    preds = torch.argmin(output, dim=1)
                    test_acc += (torch.sum(preds == y)).item()
                    test_num += y.shape[0]

                    logits = -output
                    y_prob.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                    nc = self.num_classes
                    if self.num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if self.num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            auc = 0.0
            if len(y_prob) > 0:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                try:
                    auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
                except Exception:
                    auc = 0.0

            return test_acc, test_num, auc
        else:
            with torch.no_grad():
                for x, y in testloader:
                    if isinstance(x, list):
                        x_in = x[0].to(self.device)
                    else:
                        x_in = x.to(self.device)
                    y = y.to(self.device)

                    rep = self.base(x_in)
                    if rep.dim() > 2:
                        rep = rep.view(rep.size(0), -1)

                    output = self.head(rep)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
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
            if len(y_prob) > 0:
                y_prob = np.concatenate(y_prob, axis=0)
                y_true = np.concatenate(y_true, axis=0)
                try:
                    auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
                except Exception:
                    auc = 0.0

            return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0

        gp_device = None
        if self.global_protos is not None and len(self.global_protos) > 0:
            gp_device = {
                k: v.to(self.device)
                for k, v in self.global_protos.items()
                if isinstance(v, torch.Tensor)
            }

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x_in)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                output = self.head(rep)
                loss = self.loss(output, y)

                if gp_device is not None:
                    proto_new = rep.detach().clone()
                    for idx, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in gp_device:
                            proto_new[idx, :] = gp_device[y_c]
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

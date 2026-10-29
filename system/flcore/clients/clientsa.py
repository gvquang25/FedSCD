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


class clientSA(Client):
    """
    Client for FedSA (Federated Semantic Anchors):
    - Inherits from standard Client base class.
    - Utilizes global Semantic Anchors provided by the Server.
    - Performs 3-fold anchor-guided local learning:
        1. Anchor-based Classifier Calibration (CC): Aligns classifier head on global anchors.
        2. Anchor-based Regularization (MSE): Pulls sample representations to corresponding class anchors.
        3. Margin-enhanced Contrastive Learning (MCL): Maximizes inter-class margins dynamically.
    - Vectorized prototype collection and memory-efficient evaluation.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.protos = None
        self.semantic_anchors = None
        self.local_gap = 0.0
        self.global_gap = 0.0
        self.global_protos = None
        self.local_stats = {}

        raw_lamda = getattr(args, 'lamda', 10.0)
        self.lamda = float(raw_lamda) if raw_lamda is not None else 10.0
        self.mcl = float(getattr(args, 'margin_contrastive', 1.0))
        self.cc = float(getattr(args, 'classifier_calibration', 1.0))

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

    def set_semantic_anchors(self, semantic_anchors):
        """
        Receives global semantic anchors dispatched from the Server.
        """
        if semantic_anchors is not None:
            self.semantic_anchors = semantic_anchors.detach().to(self.device)

    def set_protos(self, global_protos):
        """
        Receives aggregated global prototypes from the Server.
        """
        self.global_protos = global_protos

    def cal_local_gap(self):
        """
        Calculates the average minimum inter-class margin among local prototypes.
        """
        if self.protos is not None and len(self.protos) > 1:
            all_anchors = [p.to(self.device) for p in self.protos.values() if isinstance(p, torch.Tensor)]
            if len(all_anchors) > 1:
                anchors_tensor = torch.stack(all_anchors)
                dist_matrix = torch.cdist(anchors_tensor, anchors_tensor, p=2)
                dist_matrix.fill_diagonal_(float('inf'))
                min_margins = torch.min(dist_matrix, dim=1).values
                self.local_gap = torch.mean(min_margins).item()
            else:
                self.local_gap = 0.0
        else:
            self.local_gap = 0.0

    def cal_global_gap(self):
        """
        Calculates the average minimum inter-class margin among global semantic anchors.
        """
        if self.semantic_anchors is not None and self.semantic_anchors.size(0) > 1:
            global_dist_matrix = torch.cdist(self.semantic_anchors, self.semantic_anchors, p=2)
            global_dist_matrix.fill_diagonal_(float('inf'))
            global_min_margins = torch.min(global_dist_matrix, dim=1).values
            self.global_gap = torch.mean(global_min_margins).item()
        else:
            self.global_gap = 0.0

    def cal_anchors_loss(self, feats, y):
        """
        Computes the composite anchor-based auxiliary losses:
        1. Classifier Calibration: Head forward on global semantic anchors.
        2. Representation Regularization: MSE alignment with target anchor.
        3. Margin-enhanced Contrastive Loss: Distance-based cross-entropy with adaptive gap.
        """
        if self.semantic_anchors is None:
            return 0.0

        loss = 0.0

        # 1. Anchor-based Classifier Calibration (CC)
        if self.cc > 0.0:
            sa_lbl = torch.arange(self.num_classes, device=self.device)
            sa_logits = self.head(self.semantic_anchors)
            loss_c = self.loss(sa_logits, sa_lbl)
            loss = loss + loss_c * self.cc

        # 2. Anchor-based Regularization (MSE)
        if self.lamda > 0.0:
            target_anchors = self.semantic_anchors[y]
            loss_reg = self.loss_mse(feats, target_anchors)
            loss = loss + loss_reg * self.lamda

        # 3. Anchor-based Margin-enhanced Contrastive Learning (MCL)
        if self.mcl > 0.0:
            loss_mcl = self.cal_mcl_loss(feats, y)
            loss = loss + loss_mcl * self.mcl

        return loss

    def cal_mcl_loss(self, feats, y):
        """
        Calculates the margin-enhanced contrastive distance loss against semantic anchors.
        """
        features_square = torch.sum(torch.pow(feats, 2), dim=1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.semantic_anchors, 2), dim=1, keepdim=True)
        features_into_centers = torch.matmul(feats, self.semantic_anchors.T)
        dist = features_square - 2 * features_into_centers + centers_square.T
        dist = torch.clamp(dist, min=1e-8)
        dist = torch.sqrt(dist)

        one_hot = F.one_hot(y, self.num_classes).to(self.device)
        gap = max(self.local_gap, self.global_gap)
        dist = dist + one_hot * gap
        loss = self.loss(-dist, y)

        return loss

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        self.cal_local_gap()
        self.cal_global_gap()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        protos = defaultdict(list)
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

                # Composite FedSA anchor loss
                anchors_loss = self.cal_anchors_loss(rep, y)
                loss = loss + anchors_loss

                # Collect batch representations on last epoch
                if step == max_local_epochs - 1:
                    for idx, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[idx, :].detach().cpu())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Update local prototypes & local stats
        self.protos = agg_func(protos)

        local_stats = {}
        for k, v_list in protos.items():
            if len(v_list) > 0:
                stacked = torch.stack(v_list)
                mean_c = stacked.mean(dim=0).cpu()
                std_c = stacked.std(dim=0, unbiased=False).cpu() if len(v_list) > 1 else torch.full_like(mean_c, 1e-6)
                local_stats[k] = {'mean': mean_c, 'std': std_c, 'count': len(v_list)}
        self.local_stats = local_stats

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        """
        Evaluates local model accuracy and AUC.
        If global prototypes or semantic anchors are available, uses nearest prototype MSE distance classification.
        Otherwise falls back to the classification head.
        """
        testloader = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        # Determine reference prototypes: prefer global_protos if available, else semantic_anchors
        ref_protos = None
        if self.global_protos is not None and len(self.global_protos) > 0:
            ref_protos = {k: v.to(self.device) for k, v in self.global_protos.items() if isinstance(v, torch.Tensor)}
        elif self.semantic_anchors is not None and self.semantic_anchors.size(0) == self.num_classes:
            ref_protos = {c: self.semantic_anchors[c] for c in range(self.num_classes)}

        if ref_protos is not None and len(ref_protos) > 0:
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
                    for j, pro in ref_protos.items():
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

                anchors_loss = self.cal_anchors_loss(rep, y)
                loss = loss + anchors_loss

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

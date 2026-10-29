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
from utils.gmkd_utils import (
    kd_loss,
    feature_alignment_loss,
    temperature_scaled_softmax,
    build_knowledge_tensors
)


class clientGMKD(Client):
    """
    Client for FedGMKD (Federated Group Mutual Knowledge Distillation):
    - Inherits from standardized Client base class.
    - Decoupled base (feature representation extractor) and classification head.
    - Local training optimizes classification loss + group knowledge distillation loss (KD)
      + feature prototype alignment loss against global representative knowledge.
    - Generates local representative knowledge (feature prototypes and soft predictions)
      and uploads them to the server for adaptive group knowledge aggregation.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        # FedGMKD Hyperparameters
        raw_lam = getattr(args, 'gmkd_lam', getattr(args, 'lam_kd', 0.06))
        self.lam = float(raw_lam) if raw_lam is not None else 0.06

        raw_gamma = getattr(args, 'gmkd_gamma', getattr(args, 'gamma_feat', 0.06))
        self.gamma = float(raw_gamma) if raw_gamma is not None else 0.06

        raw_temp = getattr(args, 'gmkd_temp', getattr(args, 'temperature', 0.6))
        self.temp = float(raw_temp) if raw_temp is not None else 0.6

        raw_std = getattr(args, 'gmkd_std', 0.0)
        self.std = float(raw_std) if raw_std is not None else 0.0

        # Global knowledge dispatched from server
        self.global_features = None
        self.global_soft_prediction = None
        self.tensor_global_features = None
        self.tensor_global_soft_prediction = None

        # Local knowledge generated on client
        self.local_features = {}
        self.local_soft_prediction = {}
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

        # Cache class counts for fast Dirichlet diversity weight calculation
        self._cached_nk = None
        self._cached_class_dict = None

    def get_class_counts(self):
        """
        Returns the total number of local training samples (nk) and sample count per class.
        Cached across rounds since local data partition does not change.
        """
        if self._cached_nk is not None and self._cached_class_dict is not None:
            return self._cached_nk, self._cached_class_dict

        trainloader = self.load_train_data()
        class_dict = {c: 0 for c in range(self.num_classes)}
        nk = 0

        for x, y in trainloader:
            for yy in y:
                y_val = yy.item()
                if y_val in class_dict:
                    class_dict[y_val] += 1
                nk += 1

        self._cached_nk = nk
        self._cached_class_dict = class_dict
        return nk, class_dict

    def set_knowledge(self, global_features, global_soft_prediction):
        """
        Receives global knowledge representations from the server and prepares
        vectorized tensors on client device.
        """
        self.global_features = global_features
        self.global_soft_prediction = global_soft_prediction

        if global_features is not None and global_soft_prediction is not None:
            # Determine feature dimension from the first available feature tensor
            sample_feat = next(iter(global_features.values())) if len(global_features) > 0 else None
            feature_dim = sample_feat.shape[-1] if sample_feat is not None else 512

            tensor_f, tensor_s = build_knowledge_tensors(
                global_features,
                global_soft_prediction,
                self.num_classes,
                feature_dim,
                self.device
            )
            self.tensor_global_features = tensor_f
            self.tensor_global_soft_prediction = tensor_s
        else:
            self.tensor_global_features = None
            self.tensor_global_soft_prediction = None

    def train(self):
        """
        Local client training:
        1. Trains model using CE + KD + Feature Alignment losses.
        2. Generates local representative knowledge for server aggregation.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        has_global_knowledge = (
            self.tensor_global_features is not None and
            self.tensor_global_soft_prediction is not None and
            len(self.global_features) > 0
        )

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
                loss1 = self.loss(output, y)

                if has_global_knowledge:
                    # Loss 2: Group Knowledge Distillation on Global Feature Prototypes
                    # Pass global features through local classifier head
                    z_help = self.head(self.tensor_global_features)
                    loss2 = kd_loss(z_help, self.tensor_global_soft_prediction, temperature=self.temp)

                    # Loss 3: Feature alignment with global class prototypes
                    target_features = self.tensor_global_features[y]
                    loss3 = feature_alignment_loss(rep, target_features)

                    loss = loss1 + self.lam * loss2 + self.gamma * loss3
                else:
                    loss = loss1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Extract local representative knowledge after training
        self.generate_knowledge()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def generate_knowledge(self):
        """
        Computes local representative feature prototypes and soft predictions
        per class on local training data.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        class_features = defaultdict(list)
        class_soft_preds = defaultdict(list)

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

                logits = self.head(rep)
                soft_prob = temperature_scaled_softmax(logits, temperature=self.temp)

                for idx, yy in enumerate(y):
                    c = yy.item()
                    class_features[c].append(rep[idx].detach().cpu())
                    class_soft_preds[c].append(soft_prob[idx].detach().cpu())

        local_features = {}
        local_soft_prediction = {}
        local_stats = {}

        for c, f_list in class_features.items():
            stacked_f = torch.stack(f_list)
            mean_f = torch.mean(stacked_f, dim=0)
            if self.std > 0:
                mean_f = mean_f + self.std * torch.randn_like(mean_f)
            local_features[c] = mean_f

            stacked_s = torch.stack(class_soft_preds[c])
            mean_s = torch.mean(stacked_s, dim=0)
            local_soft_prediction[c] = mean_s

            local_stats[c] = {
                'mean': mean_f,
                'soft_pred': mean_s,
                'count': len(f_list)
            }

        self.local_features = local_features
        self.local_soft_prediction = local_soft_prediction
        self.local_stats = local_stats

        return self.local_features, self.local_soft_prediction

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

        has_global_knowledge = (
            self.tensor_global_features is not None and
            self.tensor_global_soft_prediction is not None and
            len(self.global_features) > 0
        )

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
                loss1 = self.loss(output, y)

                if has_global_knowledge:
                    z_help = self.head(self.tensor_global_features)
                    loss2 = kd_loss(z_help, self.tensor_global_soft_prediction, temperature=self.temp)
                    target_features = self.tensor_global_features[y]
                    loss3 = feature_alignment_loss(rep, target_features)
                    loss = loss1 + self.lam * loss2 + self.gamma * loss3
                else:
                    loss = loss1

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

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
from utils.plvm_utils import cluster_protos_finch, average_protos


class clientPLVM(Client):
    """
    Client for FedPLVM (Personalized Latent Variable Model for Federated Learning):
    - Decoupled representation extractor (base) and classification head (head).
    - Local training loss: CrossEntropy + lamb * (InfoNCE_alpha + correction).
    - Extracts feature representations and clusters them into local cluster prototypes using FINCH.
    - Sends model weights and local cluster prototypes to Server.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        # FedPLVM Hyperparameters
        raw_alpha = getattr(args, 'plvm_alpha', None)
        self.alpha = float(raw_alpha) if raw_alpha is not None else float(getattr(args, 'alpha', 0.25))

        raw_lamb = getattr(args, 'plvm_lamb', getattr(args, 'lamb', None))
        self.lamb = float(raw_lamb) if raw_lamb is not None else 100.0

        raw_tau = getattr(args, 'plvm_tau', None)
        self.tau = float(raw_tau) if raw_tau is not None else float(getattr(args, 'tau', 0.07))

        self.criterion_CE = nn.CrossEntropyLoss()
        self.criterion_MSE = nn.MSELoss()

        self.global_cluster_protos = {}
        self.global_avg_protos = {}
        self.local_cluster_protos = {}
        self.local_protos_dict = {}

        # Split base and head if available
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            self.base = self.model.base
            self.head = self.model.head
        elif hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            self.base = self.model.features
            self.head = self.model.classifier
        else:
            self.base = self.model
            self.head = nn.Identity()

    def set_protos(self, global_cluster_protos, global_avg_protos):
        """
        Receives global cluster prototypes and global average prototypes from the server.
        """
        self.global_cluster_protos = global_cluster_protos if global_cluster_protos is not None else {}
        self.global_avg_protos = global_avg_protos if global_avg_protos is not None else {}

    def criterion_InfoNCE_alpha(self, features, labels, cluster_protos_tensor, cluster_labels_tensor):
        """
        Computes alpha-sparsity InfoNCE contrastive loss against global cluster prototypes.
        """
        if cluster_protos_tensor is None or cluster_protos_tensor.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Cosine similarity between feature embeddings and cluster prototypes
        feat_norm = F.normalize(features, p=2, dim=1)
        prot_norm = F.normalize(cluster_protos_tensor, p=2, dim=1)
        logits = torch.matmul(feat_norm, prot_norm.t())  # (B, M)

        # Mask of positive prototypes for each sample's label
        mask = (labels.unsqueeze(1) == cluster_labels_tensor.unsqueeze(0))  # (B, M)
        valid_samples = mask.sum(dim=1) > 0
        if not valid_samples.any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Alpha-sparsity transformation
        logits = torch.clamp(logits, min=1e-7).pow(self.alpha) / self.tau

        exp_logits = torch.exp(logits)
        sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)
        pos_logits = exp_logits * mask.float()
        sum_pos_logits = pos_logits.sum(dim=1, keepdim=True)

        valid_sum_pos = torch.clamp(sum_pos_logits[valid_samples], min=1e-12)
        valid_sum_exp = torch.clamp(sum_exp_logits[valid_samples], min=1e-12)

        loss = -torch.log(valid_sum_pos / valid_sum_exp)
        return loss.mean()

    def criterion_correction(self, features, labels, global_avg_protos_device):
        """
        Computes cosine similarity correction loss against class mean prototypes:
        L_corr = 2 * MSE(cos(z_i, p_bar_{y_i}), 1.0)
        """
        if not global_avg_protos_device:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        valid_indices = []
        target_protos = []
        for i in range(labels.size(0)):
            lbl = labels[i].item()
            if lbl in global_avg_protos_device:
                valid_indices.append(i)
                target_protos.append(global_avg_protos_device[lbl])

        if not valid_indices:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        valid_indices = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        target_protos = torch.stack(target_protos, dim=0)

        selected_features = features[valid_indices]
        cos_sim = F.cosine_similarity(selected_features, target_protos, dim=1)
        target = torch.ones_like(cos_sim)

        loss = 2.0 * self.criterion_MSE(cos_sim, target)
        return loss

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        # Prepare cached prototype tensors for device
        cluster_protos_tensor = None
        cluster_labels_tensor = None
        global_avg_protos_device = {}

        if self.global_cluster_protos and len(self.global_cluster_protos) > 0:
            p_list = []
            l_list = []
            for lbl, p_centers in self.global_cluster_protos.items():
                for center in p_centers:
                    p_list.append(torch.tensor(center, dtype=torch.float32))
                    l_list.append(lbl)
            if p_list:
                cluster_protos_tensor = torch.stack(p_list, dim=0).to(self.device)
                cluster_labels_tensor = torch.tensor(l_list, dtype=torch.long, device=self.device)

        if self.global_avg_protos and len(self.global_avg_protos) > 0:
            for lbl, mean_p in self.global_avg_protos.items():
                if isinstance(mean_p, torch.Tensor):
                    global_avg_protos_device[lbl] = mean_p.to(self.device)
                else:
                    global_avg_protos_device[lbl] = torch.tensor(mean_p, dtype=torch.float32, device=self.device)

        has_global_protos = (cluster_protos_tensor is not None and len(global_avg_protos_device) > 0)

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

                probs = self.head(rep)
                loss_ce = self.criterion_CE(probs, y)

                if has_global_protos:
                    loss_infonce = self.criterion_InfoNCE_alpha(rep, y, cluster_protos_tensor, cluster_labels_tensor)
                    loss_corr = self.criterion_correction(rep, y, global_avg_protos_device)
                    loss = loss_ce + self.lamb * (loss_infonce + loss_corr)
                else:
                    loss = loss_ce

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Extract local representations and cluster with FINCH
        self.collect_protos()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos(self):
        """
        Extracts representations from local data and computes cluster prototypes via FINCH.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        protos_dict = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x_in)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                rep_cpu = rep.detach().cpu().numpy()
                y_cpu = y.detach().cpu().numpy()

                for j in range(len(y_cpu)):
                    lbl = int(y_cpu[j])
                    protos_dict[lbl].append(rep_cpu[j])

        self.local_protos_dict = protos_dict
        # Hierarchical agglomerative clustering using FINCH
        local_cluster_protos, num_cls = cluster_protos_finch(protos_dict)
        self.local_cluster_protos = local_cluster_protos

    def test_metrics(self):
        """
        Evaluates test accuracy and test AUC on local test dataset.
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
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                rep = self.base(x_in)
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
        if len(y_prob) > 0:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            try:
                auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
            except Exception:
                auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        """
        Computes training loss and total training samples.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0.0

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
                loss = self.criterion_CE(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

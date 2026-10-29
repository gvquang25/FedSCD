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
from utils.impro_utils import GaussianSynthesisLabel, Distance_loss


class clientImpro(Client):
    """
    Client for FedImpro (Decoupling Gradient Variance via Feature Synthesis):
    - Decoupled representation backbone (self.base) and classification head (self.head).
    - Local training uses real data features + Gaussian synthetic features of all classes
      to decouple and regularize gradient variance on missing/minority classes.
    - Collects local representation means, standard deviations, and class counts to upload to Server.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        # FedImpro Hyperparameters
        self.impro_weight = float(getattr(args, 'impro_weight', getattr(args, 'fed_split_loss_weight', 1.0)))
        self.impro_noise_std = float(getattr(args, 'impro_noise_std', getattr(args, 'fed_split_noise_std', 0.1)))
        self.impro_sched = getattr(args, 'impro_sched', getattr(args, 'fed_split_feature_weight_sched', 'gradual2half'))
        self.impro_max_rounds = int(getattr(args, 'impro_max_rounds', getattr(args, 'fed_split_feature_weight_max_epochs', 500)))
        self.impro_constrain_feat = getattr(args, 'impro_constrain_feat', False)
        self.impro_distance = getattr(args, 'impro_distance', 'cosine')

        self.distance_loss = Distance_loss(self.impro_distance) if self.impro_constrain_feat else None

        self.global_stats = None
        self.current_round = 0
        self.local_stats = {}
        self.synthesis_model = None

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

    def set_stats(self, global_stats, current_round=0):
        """
        Receives global feature statistics (pooled means and stds) from the server.
        """
        self.global_stats = global_stats
        self.current_round = current_round

    def train(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        # Compute synthetic feature loss weight according to schedule
        progress = min(1.0, float(self.current_round) / max(1, self.impro_max_rounds))
        if self.impro_sched == "gradual2half" or self.impro_sched == "gradual":
            fake_weight = self.impro_weight * progress
        else:
            fake_weight = self.impro_weight

        # Prepare synthesis module if global feature statistics are available
        has_synthesis = False
        if self.global_stats is not None and "means" in self.global_stats and "stds" in self.global_stats:
            means = self.global_stats["means"]
            stds = self.global_stats["stds"]
            feat_dim = means.shape[-1]
            if self.synthesis_model is None or self.synthesis_model.feature_dim != feat_dim:
                self.synthesis_model = GaussianSynthesisLabel(
                    num_classes=self.num_classes,
                    feature_dim=feat_dim,
                    noise_std=self.impro_noise_std,
                    device=self.device
                )
            self.synthesis_model.set_params(means, stds, device=self.device)
            has_synthesis = True

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

                real_bs = rep.size(0)

                if has_synthesis and fake_weight > 0:
                    # Sample synthetic features for all classes from global Gaussian distribution
                    repeat_times = max(1, real_bs // self.num_classes)
                    fake_features, fake_labels = self.synthesis_model.sample(
                        repeat_times=repeat_times,
                        device=self.device
                    )

                    all_features = torch.cat([rep, fake_features], dim=0)
                    all_output = self.head(all_features)

                    output_real = all_output[:real_bs]
                    output_fake = all_output[real_bs:]

                    loss_real = self.loss(output_real, y)
                    loss_fake = self.loss(output_fake, fake_labels)

                    if self.impro_sched == "gradual":
                        loss = loss_real * (1.0 - fake_weight) + loss_fake * fake_weight
                    else:
                        loss = loss_real + fake_weight * loss_fake

                    if self.impro_constrain_feat and self.distance_loss is not None:
                        loss += 0.1 * self.distance_loss(rep, fake_features)
                else:
                    output_real = self.head(rep)
                    loss = self.loss(output_real, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Extract local feature statistics (means, standard deviations, sample counts)
        self.collect_local_stats()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_local_stats(self):
        """
        Computes exact local class-wise feature representations, means, and standard deviations.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        features_per_class = defaultdict(list)
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

                rep_cpu = rep.detach().cpu()
                y_cpu = y.detach().cpu()

                for j in range(len(y_cpu)):
                    lbl = int(y_cpu[j].item())
                    features_per_class[lbl].append(rep_cpu[j])

        local_means = {}
        local_stds = {}
        local_counts = {}

        for c, feat_list in features_per_class.items():
            if len(feat_list) > 0:
                feat_mat = torch.stack(feat_list, dim=0).float()
                local_means[c] = feat_mat.mean(dim=0)
                if feat_mat.size(0) > 1:
                    local_stds[c] = feat_mat.std(dim=0)
                else:
                    local_stds[c] = torch.full_like(local_means[c], self.impro_noise_std)
                local_counts[c] = feat_mat.size(0)

        self.local_stats = {
            "means": local_means,
            "stds": local_stds,
            "counts": local_counts
        }

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
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

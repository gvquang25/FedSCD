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
from utils.fed3r_utils import fed3r_step


class client3R(Client):
    """
    Client for Fed3R (Federated Representation Learning with Ridge Regression Classifier):
    - Decoupled representation extractor (self.base) and classification head (self.head).
    - Local training optimizes representation backbone with standard CrossEntropy loss.
    - Post-training extraction accumulates the local Gram matrix (A_k) and cross-product matrix (b_k)
      for exact closed-form global ridge regression calibration at the server.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.fed3r_lambda = float(getattr(args, 'fed3r_lambda', getattr(args, 'lamb', 0.01)))

        self.A_k = None
        self.b_k = None

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

        self.use_bias = (hasattr(self.head, 'bias') and self.head.bias is not None)

    def train(self):
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
                    x_in = x[0].to(self.device)
                else:
                    x_in = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x_in)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Compute and collect local ridge regression matrices Ak and bk
        self.collect_ridge_matrices()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_ridge_matrices(self):
        """
        Extracts representations from local training data and computes
        the local Gram matrix Ak = Z.T @ Z and projection bk = Z.T @ Y.
        """
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        feature_dim = self.head.weight.shape[1] if hasattr(self.head, 'weight') else 64
        target_dim = (feature_dim + 1) if self.use_bias else feature_dim

        A = torch.zeros((target_dim, target_dim), dtype=torch.float32, device=self.device)
        b = torch.zeros((target_dim, self.num_classes), dtype=torch.float32, device=self.device)

        A, b = fed3r_step(
            A, b,
            dataloader=trainloader,
            model_base=self.base,
            num_classes=self.num_classes,
            device=self.device,
            use_bias=self.use_bias
        )

        self.A_k = A.detach().cpu()
        self.b_k = b.detach().cpu()

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

                output = self.model(x_in)

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

                output = self.model(x_in)
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

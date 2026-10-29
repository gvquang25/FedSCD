import copy
import itertools
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clientorgp import clientORGP
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.orgp_utils import (
    Trainable_Orthogonal_Prototypes,
    proto_cluster,
    intra_orth_loss,
    inter_orth_loss
)


class FedORGP(Server):
    """
    Server for FedORGP (Federated Orthogonal Global Prototypes, also known as FedOrth):
    - Maintains a generator (Trainable_Orthogonal_Prototypes) that outputs orthogonal global class prototypes.
    - Gathers class prototypes from participating clients.
    - Optimizes generator using intra-class alignment and inter-class orthogonality losses.
    - Dispatches updated orthogonal global prototypes to all clients.
    - Evaluates personalized/heterogeneous client models cleanly within the benchmark framework.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientORGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedORGP server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size

        # Resolve feature dimension from model head if decoupled, fallback to feature_dim argument
        if hasattr(args, 'head') and hasattr(args.head, 'in_features'):
            self.feature_dim = args.head.in_features
        elif hasattr(args, 'model') and hasattr(args.model, 'head') and hasattr(args.model.head, 'in_features'):
            self.feature_dim = args.model.head.in_features
        else:
            self.feature_dim = getattr(args, 'feature_dim', 512)

        self.server_hidden_dim = self.feature_dim

        # FedORGP Loss & Optimization Hyperparameters
        raw_clamda = getattr(args, 'c_lamda', getattr(args, 'orgp_c_lamda', 1.0))
        self.c_lamda = float(raw_clamda) if raw_clamda is not None else 1.0

        raw_slamda = getattr(args, 's_lamda', getattr(args, 'orgp_s_lamda', 1.0))
        self.s_lamda = float(raw_slamda) if raw_slamda is not None else 1.0

        raw_gamma = getattr(args, 'gamma', getattr(args, 'orgp_gamma', 1.0))
        self.gamma = float(raw_gamma) if raw_gamma is not None else 1.0

        raw_alpha = getattr(args, 'orgp_alpha', getattr(args, 'alpha', 0.5))
        self.alpha = float(raw_alpha) if raw_alpha is not None else 0.5

        self.ablation_server = bool(getattr(args, 'ablation_server', False))
        self.ablation_client = bool(getattr(args, 'ablation_client', False))

        # Server prototype generator epochs & learning rate
        self.server_epochs = getattr(args, 'orgp_server_epochs', 1)
        self.server_learning_rate = getattr(args, 'orgp_server_lr', 0.01)

        # Initialize Trainable Orthogonal Prototypes generator
        self.PROTO = Trainable_Orthogonal_Prototypes(
            self.num_classes,
            self.server_hidden_dim,
            self.feature_dim,
            self.device
        ).to(self.device)

        self.Gen_opt = torch.optim.SGD(self.PROTO.parameters(), lr=self.server_learning_rate)

        self.uploaded_protos = []
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = torch.tensor(0.0, device=self.device)
        self.max_gap = torch.tensor(1.0, device=self.device)

        # Initialize global prototypes and distribute to clients
        self.PROTO.eval()
        with torch.no_grad():
            self.global_protos = {
                c: self.PROTO(torch.tensor(c, device=self.device)).detach().cpu()
                for c in range(self.num_classes)
            }
        self.send_protos()

    def train(self):
        """
        Federated training loop for FedORGP:
        1. Evaluates models at specified eval_gap intervals.
        2. Selected clients train locally with CE + Intra-Orth prototype loss.
        3. Server receives client prototypes.
        4. Server updates prototype generator via Intra-Orth & Inter-Orth losses.
        5. Server broadcasts updated orthogonal prototypes to clients.
        """
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Evaluate personalized / prototype models
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models")
                self.evaluate()

            # Client local training
            for client in self.selected_clients:
                client.train()

            # Server receives uploaded class prototypes
            self.receive_protos()

            # Server updates orthogonal prototype generator
            self.update_Gen()

            # Broadcast updated global orthogonal prototypes
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, round(self.Budget[-1], 2), 's')

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        if self.global_rounds % self.eval_gap != 0:
            self.current_round = self.global_rounds
            print(f"\n-------------Final Evaluation at Round: {self.global_rounds}-------------")
            self.evaluate()

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / max(len(self.Budget[1:]), 1))

        self.save_results()

    def send_protos(self):
        """
        Dispatches updated orthogonal global prototypes to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        """
        Collects class prototypes from participating clients, builds uploaded proto buffer,
        and computes prototype separation and cosine orthogonality metrics.
        """
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = getattr(client, 'protos', None)
            if protos is not None:
                for k, proto in protos.items():
                    if isinstance(proto, torch.Tensor):
                        self.uploaded_protos.append((proto.detach().cpu(), k))
                uploaded_protos_per_client.append(protos)

        # Compute prototype metrics across uploaded client prototypes
        self.calculate_prototype_metrics(uploaded_protos_per_client)

    def calculate_prototype_metrics(self, uploaded_protos_per_client):
        """
        Computes class-wise minimum Euclidean distance and pairwise cosine similarities.
        """
        if len(uploaded_protos_per_client) == 0:
            return

        avg_protos = proto_cluster(uploaded_protos_per_client)
        if len(avg_protos) < 2:
            return

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        cosine_sim_values = []
        avg_keys = list(avg_protos.keys())

        for idx1 in range(len(avg_keys)):
            k1 = avg_keys[idx1]
            p1 = avg_protos[k1].to(self.device)
            for idx2 in range(idx1 + 1, len(avg_keys)):
                k2 = avg_keys[idx2]
                p2 = avg_protos[k2].to(self.device)

                euc_dist = torch.norm(p1 - p2, p=2)
                self.gap[k1] = torch.min(self.gap[k1], euc_dist)
                self.gap[k2] = torch.min(self.gap[k2], euc_dist)

                cos_sim = F.cosine_similarity(p1.unsqueeze(0), p2.unsqueeze(0)).item()
                cosine_sim_values.append(cos_sim)

        valid_gaps = [self.gap[k] for k in avg_keys if self.gap[k] < 1e8]
        if len(valid_gaps) > 0:
            self.min_gap = torch.min(torch.stack(valid_gaps))
            for i in range(len(self.gap)):
                if self.gap[i] > 1e8:
                    self.gap[i] = self.min_gap
            self.max_gap = torch.max(self.gap)

        if len(cosine_sim_values) > 0:
            avg_cos = sum(cosine_sim_values) / len(cosine_sim_values)
            min_cos = min(cosine_sim_values)
            max_cos = max(cosine_sim_values)
            print(f"Prototype Orthogonality Metrics -> Avg Cosine Sim: {avg_cos:.4f}, Min: {min_cos:.4f}, Max: {max_cos:.4f}")

    def update_Gen(self):
        """
        Updates the Trainable Orthogonal Prototypes generator using client-uploaded prototypes.
        Optimizes intra-class alignment and inter-class orthogonality.
        """
        if len(self.uploaded_protos) == 0:
            return

        self.PROTO.to(self.device)
        self.PROTO.train()

        proto_loader = DataLoader(
            self.uploaded_protos,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

        last_loss = 0.0
        adaptive = not self.ablation_server

        for e in range(self.server_epochs):
            for proto, y in proto_loader:
                proto = proto.to(self.device)
                y = torch.as_tensor(y, dtype=torch.int64, device=self.device)

                # Generate global prototypes from generator for all classes
                proto_gen = {}
                for class_id in range(self.num_classes):
                    proto_gen[class_id] = self.PROTO(torch.tensor(class_id, device=self.device))

                loss_intra = intra_orth_loss(proto, y, proto_gen, adaptive=adaptive)
                loss_inter = inter_orth_loss(proto, y, proto_gen, adaptive=adaptive)

                loss = self.s_lamda * loss_intra + self.gamma * loss_inter

                self.Gen_opt.zero_grad()
                loss.backward()
                self.Gen_opt.step()
                last_loss = loss.item()

        print(f"FedORGP Server Generator Loss: {last_loss:.4f}")

        # Clear uploaded prototypes to avoid memory leaks
        self.uploaded_protos = []

        # Generate updated global orthogonal prototypes
        self.PROTO.eval()
        with torch.no_grad():
            self.global_protos = {
                c: self.PROTO(torch.tensor(c, device=self.device)).detach().cpu()
                for c in range(self.num_classes)
            }

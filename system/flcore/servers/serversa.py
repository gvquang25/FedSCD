import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientsa import clientSA
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.sa_utils import (
    build_semantic_projector,
    proto_aggregation,
    proto2anchors,
    cal_class_margins
)


class FedSA(Server):
    """
    Server for FedSA (Federated Semantic Anchors):
    - Maintains and updates global Semantic Anchors via Momentum EMA aggregation.
    - Dispatches Semantic Anchors to clients to guide feature representation alignment
      and calibrate local classification heads.
    - Collects and aggregates class prototypes from participating clients.
    - Integrates cleanly with serverbase evaluation and logging framework.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientSA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedSA server and clients.")

        self.Budget = []
        self.class_numbers = args.num_classes
        self.feat_channel = getattr(args, 'feature_dim', 512)

        raw_lamda = getattr(args, 'lamda', 10.0)
        self.lamda = float(raw_lamda) if raw_lamda is not None else 10.0
        self.mcl = float(getattr(args, 'margin_contrastive', 1.0))
        self.cc = float(getattr(args, 'classifier_calibration', 1.0))

        # Build pure-PyTorch projector to generate initial semantic anchors
        projector = build_semantic_projector(self.feat_channel)
        projector.eval()

        with torch.no_grad():
            random_center = torch.randn(self.class_numbers, self.feat_channel)
            self.semantic_anchors = projector(random_center).detach().to(self.device)

        self.global_protos = {}

    def train(self):
        print_experiment_config(self.args, self.times)

        # Initial distribution of semantic anchors
        self.send_semantic_anchors()
        min_m, avg_m, max_m, _ = cal_class_margins(self.semantic_anchors)
        print(f"Initial Semantic Anchors Margin -> Min: {min_m:.4f}, Avg: {avg_m:.4f}, Max: {max_m:.4f}")

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Evaluate personalized / prototype models
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models")
                self.evaluate()

            # Client local training with Semantic Anchors
            for client in self.selected_clients:
                client.train()

            # Server receives uploaded class prototypes
            self.receive_protos()

            # Update semantic anchors via EMA
            self.update_semantic_anchors()

            # Dispatch updated semantic anchors & global prototypes
            self.send_semantic_anchors()
            self.send_protos()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

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

    def send_semantic_anchors(self):
        """
        Dispatches semantic anchors to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            client.set_semantic_anchors(self.semantic_anchors)

    def send_protos(self):
        """
        Dispatches aggregated global prototypes to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        """
        Gathers class prototypes uploaded by participating clients and aggregates them.
        """
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        uploaded_protos = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = getattr(client, 'protos', None)
            if protos is not None:
                uploaded_protos.append(protos)

        self.global_protos = proto_aggregation(uploaded_protos)

    def update_semantic_anchors(self):
        """
        Updates global semantic anchors using a momentum moving average (EMA):
        A_c = 0.9999 * A_c + 0.0001 * P_c
        """
        if self.global_protos is None or len(self.global_protos) == 0:
            return

        with torch.no_grad():
            for label, proto in self.global_protos.items():
                c_idx = int(label)
                if c_idx < self.class_numbers and isinstance(proto, torch.Tensor):
                    p = proto.to(self.device)
                    self.semantic_anchors[c_idx] = (
                        0.9999 * self.semantic_anchors[c_idx] + 0.0001 * p
                    ).detach()

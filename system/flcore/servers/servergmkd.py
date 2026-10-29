import copy
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientgmkd import clientGMKD
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.gmkd_utils import (
    compute_client_diversity_weights,
    aggregate_knowledge_initial,
    aggregate_knowledge_adaptive
)


class FedGMKD(Server):
    """
    Server for FedGMKD (Federated Group Mutual Knowledge Distillation):
    - Coordinates federated training with group mutual knowledge distillation.
    - Aggregates model parameters using Non-IID Dirichlet skew diversity weights (p_k).
    - Aggregates global representative feature prototypes and soft predictions across clients.
    - Uses adaptive divergence weighting (p_kc) across communication rounds.
    - Dispatches updated global model and global representative knowledge to clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientGMKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedGMKD server and clients.")

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

        # FedGMKD Hyperparameters
        raw_lam = getattr(args, 'gmkd_lam', getattr(args, 'lam_kd', 0.06))
        self.lam = float(raw_lam) if raw_lam is not None else 0.06

        raw_gamma = getattr(args, 'gmkd_gamma', getattr(args, 'gamma_feat', 0.06))
        self.gamma = float(raw_gamma) if raw_gamma is not None else 0.06

        raw_temp = getattr(args, 'gmkd_temp', getattr(args, 'temperature', 0.6))
        self.temp = float(raw_temp) if raw_temp is not None else 0.6

        self.std = float(getattr(args, 'gmkd_std', 0.0))
        self.param_a = float(getattr(args, 'gmkd_param_a', 0.2))
        self.param_b = float(getattr(args, 'gmkd_param_b', 0.2))
        self.param_a_cls = float(getattr(args, 'gmkd_param_a_cls', 0.4))
        self.param_b_cls = float(getattr(args, 'gmkd_param_b_cls', 0.2))

        # Global knowledge stores
        self.global_features = {}
        self.global_soft_prediction = {}

    def train(self):
        """
        Federated training loop for FedGMKD:
        1. Evaluates model at specified eval_gap intervals.
        2. Selected clients train locally with CE + Group KD + Feature Alignment.
        3. Server receives local models, sample counts, and knowledge representations.
        4. Server computes diversity aggregation weights and aggregates global model.
        5. Server aggregates global knowledge representations (features & soft predictions).
        6. Server broadcasts global model and global knowledge to clients.
        """
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Evaluate personalized / global models
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models")
                self.evaluate()

            # Client local training
            for client in self.selected_clients:
                client.train()

            # Server receives models, diversity metrics, and knowledge
            self.receive_models_and_knowledge()

            # Server aggregates model parameters using p_k
            self.aggregate_parameters()

            # Server broadcasts updated global model and global knowledge
            self.send_models()
            self.send_knowledge()

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

    def send_knowledge(self):
        """
        Dispatches updated global features and soft predictions to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            client.set_knowledge(self.global_features, self.global_soft_prediction)

    def receive_models_and_knowledge(self):
        """
        Receives model weights and representative knowledge from selected clients.
        Computes Dirichlet skew diversity weights (p_k) for model aggregation,
        and aggregates global features and soft predictions.
        """
        assert len(self.selected_clients) > 0

        self.uploaded_ids = []
        self.uploaded_models = []
        local_features_list = []
        local_soft_list = []
        class_counts_list = []
        nks_list = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

            local_features_list.append(client.local_features)
            local_soft_list.append(client.local_soft_prediction)

            nk, class_dict = client.get_class_counts()
            nks_list.append(nk)
            class_counts_list.append(class_dict)

        # 1. Compute Dirichlet diversity weights p_k for model parameter aggregation
        self.uploaded_weights = compute_client_diversity_weights(
            class_counts_list,
            nks_list,
            self.num_classes,
            param_a=self.param_a,
            param_b=self.param_b
        )

        # 2. Aggregate global representative knowledge (features & soft predictions)
        if self.current_round == 0 or len(self.global_features) == 0:
            self.global_features, self.global_soft_prediction = aggregate_knowledge_initial(
                local_features_list,
                local_soft_list,
                self.num_classes,
                self.feature_dim,
                self.device
            )
        else:
            self.global_features, self.global_soft_prediction = aggregate_knowledge_adaptive(
                local_features_list,
                local_soft_list,
                self.global_features,
                self.global_soft_prediction,
                nks_list,
                self.num_classes,
                self.feature_dim,
                param_a=self.param_a_cls,
                param_b=self.param_b_cls,
                device=self.device
            )

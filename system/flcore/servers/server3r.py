import copy
import time
from collections import defaultdict
import numpy as np
import torch

from flcore.clients.client3r import client3R
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.fed3r_utils import solve_fed3r

try:
    from utils.wandb_utils import log_server_summary
except ImportError:
    def log_server_summary(server):
        pass


class Fed3R(Server):
    """
    Server for Fed3R (Federated Representation Learning with Ridge Regression Classifier):
    - Aggregates model backbone parameters via sample-weighted FedAvg.
    - Gathers local Gram matrices (A_k) and cross-product projections (b_k) from clients.
    - Solves the closed-form global ridge regression problem to compute optimal linear classifier parameters.
    - Normalizes the classifier weights and updates the global head.
    - Broadcasts the updated model to all clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(client3R)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating Fed3R server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        # Fed3R Hyperparameters
        self.fed3r_lambda = float(getattr(args, 'fed3r_lambda', getattr(args, 'lamb', 0.01)))
        self.fed3r_norm = bool(getattr(args, 'fed3r_norm', True))

        self.uploaded_A = []
        self.uploaded_b = []

    def train(self):
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Evaluate global/personalized models
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate models")
                self.evaluate()

            if i == self.global_rounds:
                break

            # Broadcast global model to clients
            self.send_models()

            # Clients local training
            for client in self.selected_clients:
                client.train()

            # Receive models and local ridge regression matrices
            self.receive_models()
            self.receive_matrices()

            # Aggregate backbone parameters via FedAvg
            self.aggregate_parameters()

            # Calibrate classifier head via Fed3R closed-form ridge solution
            self.calibrate_classifier()

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
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        elif len(self.Budget) == 1:
            print(self.Budget[0])

        log_server_summary(self)
        self.save_results()

    def receive_matrices(self):
        """
        Collects local Gram matrices (A_k) and cross-product projections (b_k) from participating clients.
        """
        assert len(self.selected_clients) > 0
        self.uploaded_A = []
        self.uploaded_b = []
        for client in self.selected_clients:
            if hasattr(client, 'A_k') and client.A_k is not None:
                self.uploaded_A.append(copy.deepcopy(client.A_k))
            if hasattr(client, 'b_k') and client.b_k is not None:
                self.uploaded_b.append(copy.deepcopy(client.b_k))

    def calibrate_classifier(self):
        """
        Solves for the Fed3R optimal ridge classifier weights and updates
        the global model's classification head.
        """
        if not self.uploaded_A or not self.uploaded_b:
            return

        head = getattr(self.global_model, 'head', None)
        if head is None:
            return

        if not hasattr(head, 'weight'):
            return

        feature_dim = head.weight.shape[1]
        use_bias = (hasattr(head, 'bias') and head.bias is not None)

        W, bias = solve_fed3r(
            A_list=self.uploaded_A,
            b_list=self.uploaded_b,
            feature_dim=feature_dim,
            num_classes=self.num_classes,
            lamb=self.fed3r_lambda,
            use_bias=use_bias,
            normalize=self.fed3r_norm,
            device=self.device
        )

        head.weight.data.copy_(W)
        if use_bias and bias is not None and head.bias is not None:
            head.bias.data.copy_(bias)

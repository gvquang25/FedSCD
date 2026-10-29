import copy
import time
from collections import defaultdict
import numpy as np
import torch

from flcore.clients.clientcof import clientCOF
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.cof_utils import solve_fedcof

try:
    from utils.wandb_utils import log_server_summary
except ImportError:
    def log_server_summary(server):
        pass


class FedCOF(Server):
    """
    Server for FedCOF (Federated Learning with Covariances for Free):
    - Aggregates model backbone parameters via sample-weighted FedAvg.
    - Collects local class means and sample counts from participating clients (O(D) communication).
    - Estimates intra-class covariances and the global Gram matrix via closed-form shrinkage.
    - Solves the ridge regression problem to calibrate the global classification head weights.
    - Broadcasts the updated model to all clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientCOF)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedCOF server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        # FedCOF Hyperparameters
        self.cof_gamma = float(getattr(args, 'cof_gamma', getattr(args, 'shrink', 1.0)))

        self.uploaded_stats = []

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

            # Receive models and local representation statistics
            self.receive_models()
            self.receive_stats()

            # Aggregate backbone and base parameters (FedAvg)
            self.aggregate_parameters()

            # Calibrate classifier head via FedCOF closed-form ridge solution
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

    def receive_stats(self):
        """
        Collects local class means and sample counts from participating clients.
        """
        assert len(self.selected_clients) > 0
        self.uploaded_stats = []
        for client in self.selected_clients:
            if hasattr(client, 'local_stats') and client.local_stats:
                self.uploaded_stats.append(copy.deepcopy(client.local_stats))

    def calibrate_classifier(self):
        """
        Solves for the FedCOF optimal ridge classifier weights and updates
        the global model's classification head.
        """
        if not self.uploaded_stats:
            return

        head = getattr(self.global_model, 'head', None)
        if head is None:
            return

        if not hasattr(head, 'weight'):
            return

        feature_dim = head.weight.shape[1]
        prev_weights = head.weight.data.clone().to(self.device)

        W = solve_fedcof(
            client_stats_list=self.uploaded_stats,
            num_classes=self.num_classes,
            feature_dim=feature_dim,
            gamma=self.cof_gamma,
            prev_weights=prev_weights,
            device=self.device
        )

        head.weight.data.copy_(W)
        if head.bias is not None:
            head.bias.data.zero_()

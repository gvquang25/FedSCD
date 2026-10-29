import copy
import time
from collections import defaultdict
import numpy as np
import torch

from flcore.clients.clientimpro import clientImpro
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.impro_utils import aggregate_feature_statistics

try:
    from utils.wandb_utils import log_server_summary
except ImportError:
    def log_server_summary(server):
        pass


class FedImpro(Server):
    """
    Server for FedImpro (Decoupling Gradient Variance in Federated Learning via Feature Synthesis):
    - Coordinates global model aggregation (FedAvg).
    - Collects local representation statistics (means, standard deviations, class counts) from clients.
    - Aggregates feature statistics using the Law of Total Variance (Pooled Variance).
    - Broadcasts global model parameters and pooled feature statistics to clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientImpro)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedImpro server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        # FedImpro Hyperparameters
        self.impro_noise_std = float(getattr(args, 'impro_noise_std', getattr(args, 'fed_split_noise_std', 0.1)))
        self.impro_weight = float(getattr(args, 'impro_weight', getattr(args, 'fed_split_loss_weight', 1.0)))
        self.impro_sched = getattr(args, 'impro_sched', getattr(args, 'fed_split_feature_weight_sched', 'gradual2half'))
        self.impro_max_rounds = int(getattr(args, 'impro_max_rounds', getattr(args, 'fed_split_feature_weight_max_epochs', 500)))

        self.global_stats = None
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

            # Broadcast global model and global feature statistics to clients
            self.send_models()
            self.send_stats()

            # Client local training with real data + synthetic Gaussian features
            for client in self.selected_clients:
                client.train()

            # Receive models and local feature statistics
            self.receive_models()
            self.receive_stats()

            # Aggregate model weights (FedAvg)
            self.aggregate_parameters()

            # Aggregate feature statistics (Pooled Variance)
            self.aggregate_stats()

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

    def send_stats(self):
        """
        Dispatches global feature statistics and current round index to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_stats(self.global_stats, self.current_round)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_stats(self):
        """
        Collects local feature statistics from participating clients.
        """
        assert len(self.selected_clients) > 0
        self.uploaded_stats = []
        for client in self.selected_clients:
            if hasattr(client, 'local_stats') and client.local_stats:
                self.uploaded_stats.append(copy.deepcopy(client.local_stats))

    def aggregate_stats(self):
        """
        Aggregates local feature statistics using the Law of Total Variance
        to update global feature means and pooled standard deviations.
        """
        if not self.uploaded_stats:
            return

        self.global_stats = aggregate_feature_statistics(
            local_stats_list=self.uploaded_stats,
            num_classes=self.num_classes,
            max_noise_std=self.impro_noise_std,
            prev_stats=self.global_stats
        )

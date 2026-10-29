import copy
import time
from collections import defaultdict
import numpy as np
import torch

from flcore.clients.clientplvm import clientPLVM
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.plvm_utils import (
    local_cluster_collect,
    cluster_protos_finch,
    proto_aggregation_cluster,
)

try:
    from utils.wandb_utils import log_server_summary
except ImportError:
    def log_server_summary(server):
        pass


class FedPLVM(Server):
    """
    Server for FedPLVM (Personalized Latent Variable Model for Federated Learning):
    - Coordinates global model aggregation (FedAvg).
    - Collects local cluster prototypes from participating clients.
    - Hierarchically clusters local cluster prototypes into global cluster prototypes using FINCH.
    - Averages cluster prototypes per class to maintain global average prototypes.
    - Dispatches updated global model weights and prototypes to clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientPLVM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedPLVM server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        # FedPLVM Hyperparameters
        raw_alpha = getattr(args, 'plvm_alpha', None)
        self.alpha = float(raw_alpha) if raw_alpha is not None else float(getattr(args, 'alpha', 0.25))

        raw_lamb = getattr(args, 'plvm_lamb', getattr(args, 'lamb', None))
        self.lamb = float(raw_lamb) if raw_lamb is not None else 100.0

        raw_tau = getattr(args, 'plvm_tau', None)
        self.tau = float(raw_tau) if raw_tau is not None else float(getattr(args, 'tau', 0.07))

        # Server-side prototype storage
        self.global_cluster_protos = {}
        self.global_avg_protos = {}
        self.uploaded_cluster_protos = {}

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

            # Broadcast global model and global prototypes to clients
            self.send_models()
            self.send_protos()

            # Clients local training
            for client in self.selected_clients:
                client.train()

            # Receive models and local cluster prototypes
            self.receive_models()
            self.receive_protos()

            # Aggregate model weights (FedAvg)
            self.aggregate_parameters()

            # Aggregate prototypes via FINCH hierarchical clustering
            self.aggregate_protos()

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

    def send_protos(self):
        """
        Dispatches global cluster prototypes and global average prototypes to clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_cluster_protos, self.global_avg_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        """
        Collects local cluster prototypes from selected clients.
        """
        assert len(self.selected_clients) > 0
        self.uploaded_cluster_protos = {}
        for client in self.selected_clients:
            if hasattr(client, 'local_cluster_protos') and client.local_cluster_protos:
                self.uploaded_cluster_protos[client.id] = copy.deepcopy(client.local_cluster_protos)

    def aggregate_protos(self):
        """
        Performs global hierarchical clustering of local cluster prototypes using FINCH,
        and computes class-wise average prototypes.
        """
        if not self.uploaded_cluster_protos:
            return

        # 1. Collect all local cluster prototypes across clients
        global_collected_protos = local_cluster_collect(self.uploaded_cluster_protos)

        # 2. Cluster collected prototypes using FINCH per class
        new_cluster_protos, avg_num_cls = cluster_protos_finch(global_collected_protos)

        # Update global cluster prototypes, preserving previously observed classes
        for lbl, p_list in new_cluster_protos.items():
            self.global_cluster_protos[lbl] = p_list

        # 3. Average cluster prototypes to form global average prototypes
        new_avg_protos = proto_aggregation_cluster(self.global_cluster_protos)
        self.global_avg_protos.update(new_avg_protos)

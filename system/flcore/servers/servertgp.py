import copy
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.tgp_utils import Trainable_Global_Prototypes, proto_cluster


class FedTGP(Server):
    """
    Server for FedTGP (Trainable Global Prototypes):
    - Inherits from standard Server base class.
    - Maintains a generator (Trainable_Global_Prototypes) on the server.
    - Gathers class prototypes from participating clients.
    - Computes class-wise pairwise minimum distances and adaptive margin.
    - Trains TGP generator using contrastive/margin distance loss.
    - Dispatches updated global prototypes to all clients.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedTGP server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes

        # Learning rate and training parameters for TGP model on server
        self.server_learning_rate = getattr(args, 'tgp_server_lr', getattr(args, 'local_learning_rate', 0.01))
        self.batch_size = args.batch_size

        # Resolve server_epochs: prioritize --tgp_server_epochs, fallback to -se/--server_epochs
        tgp_se = getattr(args, 'tgp_server_epochs', None)
        generic_se = getattr(args, 'server_epochs', None)
        if tgp_se is not None:
            self.server_epochs = tgp_se
        elif generic_se is not None:
            self.server_epochs = generic_se
        else:
            self.server_epochs = 100

        self.margin_threthold = getattr(args, 'margin_threthold', 100.0)
        self.feature_dim = getattr(args, 'feature_dim', 512)
        self.server_hidden_dim = self.feature_dim

        # Initialize Trainable Global Prototypes generator
        self.TGP = Trainable_Global_Prototypes(
            self.num_classes,
            self.server_hidden_dim,
            self.feature_dim,
            self.device
        ).to(self.device)

        self.TGP_opt = torch.optim.SGD(self.TGP.parameters(), lr=self.server_learning_rate)

        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap = torch.tensor(0.0, device=self.device)
        self.max_gap = torch.tensor(self.margin_threthold, device=self.device)

        self.uploaded_protos = []
        self.global_protos = None

    def train(self):
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

            # Server receives uploaded prototypes
            self.receive_protos()

            # Server optimizes TGP and generates global prototypes
            self.update_TGP()

            # Server broadcasts updated global prototypes to clients
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

    def send_protos(self):
        """
        Dispatches updated global prototypes to all clients.
        """
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        """
        Collects class prototypes from participating clients, aggregates them
        per class, and computes the class-wise minimum distance and adaptive margin.
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

        # Calculate class-wise minimum distance
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(uploaded_protos_per_client)

        has_pairs = False
        avg_keys = list(avg_protos.keys())
        for idx1 in range(len(avg_keys)):
            k1 = avg_keys[idx1]
            for idx2 in range(idx1 + 1, len(avg_keys)):
                k2 = avg_keys[idx2]
                has_pairs = True
                p1 = avg_protos[k1].to(self.device)
                p2 = avg_protos[k2].to(self.device)
                dis = torch.norm(p1 - p2, p=2)
                self.gap[k1] = torch.min(self.gap[k1], dis)
                self.gap[k2] = torch.min(self.gap[k2], dis)

        if has_pairs:
            valid_gaps = [self.gap[k] for k in avg_keys if self.gap[k] < 1e8]
            if len(valid_gaps) > 0:
                self.min_gap = torch.min(torch.stack(valid_gaps))
                for i in range(len(self.gap)):
                    if self.gap[i] > torch.tensor(1e8, device=self.device):
                        self.gap[i] = self.min_gap
                self.max_gap = torch.max(self.gap)
            else:
                self.min_gap = torch.tensor(0.0, device=self.device)
                self.max_gap = torch.tensor(self.margin_threthold, device=self.device)
        else:
            self.min_gap = torch.tensor(0.0, device=self.device)
            self.max_gap = torch.tensor(self.margin_threthold, device=self.device)

        print('class-wise minimum distance:', [round(g.item(), 4) for g in self.gap])
        print('min_gap:', round(self.min_gap.item(), 4))
        print('max_gap:', round(self.max_gap.item(), 4))

    def update_TGP(self):
        """
        Updates the Trainable Global Prototypes model using client-uploaded prototypes
        and adaptive margin contrastive loss.
        """
        if len(self.uploaded_protos) == 0:
            return

        self.TGP.to(self.device)
        self.TGP.train()

        proto_loader = DataLoader(
            self.uploaded_protos,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True
        )

        all_classes = torch.arange(self.num_classes, device=self.device)
        margin = min(self.max_gap.item(), self.margin_threthold)

        last_loss = 0.0
        for e in range(self.server_epochs):
            for proto, y in proto_loader:
                proto = proto.to(self.device)
                y = torch.as_tensor(y, dtype=torch.int64, device=self.device)

                proto_gen = self.TGP(all_classes)

                features_square = torch.sum(torch.pow(proto, 2), dim=1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), dim=1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.clamp(dist, min=1e-8)
                dist = torch.sqrt(dist)

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)

                self.TGP_opt.zero_grad()
                loss.backward()
                self.TGP_opt.step()
                last_loss = loss.item()

        print(f'Server loss: {last_loss:.4f}')
        self.uploaded_protos = []

        # Generate global prototypes for the next round
        self.TGP.eval()
        global_protos = defaultdict(list)
        with torch.no_grad():
            gen_protos = self.TGP(all_classes).detach().cpu()
            for class_id in range(self.num_classes):
                global_protos[class_id] = gen_protos[class_id].clone()
        self.global_protos = global_protos

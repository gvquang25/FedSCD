#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import random

# Active Server Algorithms
from flcore.servers.serverscd import FedSCD
from flcore.servers.servertgp import FedTGP
from flcore.servers.serversa import FedSA
from flcore.servers.serverplvm import FedPLVM
from flcore.servers.serverimpro import FedImpro
from flcore.servers.servercof import FedCOF
from flcore.servers.server3r import Fed3R
from flcore.servers.serverorgp import FedORGP
from flcore.servers.servergmkd import FedGMKD

# Models
from flcore.trainmodel.models import *
from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *

from utils.result_utils import average_data, print_experiment_config
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Hyper-params for Text tasks
vocab_size = 98635
max_len = 200
emb_dim = 32


def split_base_head(model):
    """Tách head phân loại ra khỏi backbone để dùng trong BaseHeadSplit nếu chưa được tách."""
    if hasattr(model, 'fc') and not isinstance(model.fc, nn.Identity):
        head = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
        return BaseHeadSplit(model, head)
    elif not hasattr(model, 'base'):
        head = nn.Identity()
        return BaseHeadSplit(model, head)
    return model


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_str = model_str

    base_seed = args.seed if args.seed is not None else (int(time.time() * 1000) % 1000000)

    for i in range(args.prev, args.times):
        current_seed = base_seed + i
        set_seed(current_seed)
        args.current_seed = current_seed

        print(f"\n============= Running time: {i}th (Seed: {current_seed}) =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "mlr":  # convex
            if "mnist" in args.dataset:
                args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        elif model_str == "cnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "cnn20":  # non-convex
            if "mnist" in args.dataset:
                args.model = FedAvgCNN20(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN20(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "omniglot" in args.dataset:
                args.model = FedAvgCNN20(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN().to(args.device)
            elif "imagenet" in args.dataset:
                args.model = FedAvgCNN20_Tiny(in_features=3, num_classes=args.num_classes).to(args.device)
            else:
                args.model = FedAvgCNN20(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "custom_resnet":
            if "mnist" in args.dataset:
                args.model = CustomResNet(in_channels=1, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = CustomResNet(in_channels=3, num_classes=args.num_classes).to(args.device)
            elif "imagenet" in args.dataset:
                args.model = CustomResNet(in_channels=3, num_classes=args.num_classes).to(args.device)

        elif model_str == "dnn":  # non-convex
            if "mnist" in args.dataset:
                args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

        elif model_str == "resnet":
            in_ch = 1 if ("mnist" in args.dataset.lower() or "emnist" in args.dataset.lower()) else 3
            args.model = resnet18(num_classes=args.num_classes, in_channels=in_ch).to(args.device)

        elif model_str == "alexnet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "googlenet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "mobilenet_v2":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

        elif model_str == "lstm":
            args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "bilstm":
            args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim, output_size=args.num_classes, 
                        num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                        embedding_length=emb_dim).to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=2, d_hid=emb_dim, nlayers=2, 
                            num_classes=args.num_classes).to(args.device)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP().to(args.device)

        elif model_str == "harcnn":
            if args.dataset == 'har':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)
            elif args.dataset == 'pamap':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(args.device)

        else:
            raise NotImplementedError(f"Model '{model_str}' not supported.")

        # Select algorithm
        algo_name = args.algorithm.lower()
        if algo_name in ["fedscd", "scd"]:
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSCD(args, i)

        elif algo_name in ["fedtgp", "tgp"]:
            args.model = split_base_head(args.model)
            server = FedTGP(args, i)

        elif algo_name in ["fedsa", "sa"]:
            args.model = split_base_head(args.model)
            server = FedSA(args, i)

        elif algo_name in ["fedplvm", "plvm"]:
            args.model = split_base_head(args.model)
            server = FedPLVM(args, i)

        elif algo_name in ["fedimpro", "impro"]:
            args.model = split_base_head(args.model)
            server = FedImpro(args, i)

        elif algo_name in ["fedcof", "cof"]:
            args.model = split_base_head(args.model)
            server = FedCOF(args, i)

        elif algo_name in ["fed3r", "3r"]:
            args.model = split_base_head(args.model)
            server = Fed3R(args, i)

        elif algo_name in ["fedorgp", "fedorth", "orgp", "orth"]:
            args.model = split_base_head(args.model)
            server = FedORGP(args, i)

        elif algo_name in ["fedgmkd", "gmkd"]:
            args.model = split_base_head(args.model)
            server = FedGMKD(args, i)

        else:
            raise NotImplementedError(f"Algorithm '{args.algorithm}' is not supported. Available: FedSCD, FedTGP, FedSA, FedPLVM, FedImpro, FedCOF, Fed3R, FedORGP, FedGMKD")

        print_experiment_config(args, i)
        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average across runs
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")
    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedSCD & Baselines Benchmark Runner")
    # General arguments
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=int, default=0)
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="cnn")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-tbs', '--test_batch_size', type=int, default=512, help="Test batch size")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedSCD",
                        help="Algorithm: FedSCD, FedTGP, FedSA, FedPLVM, FedImpro, FedCOF, Fed3R, FedORGP, FedGMKD")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=2,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)
    parser.add_argument('-fceal', "--force_evaluate", action='store_true')
    parser.add_argument('-log', "--log", action='store_true')
    parser.add_argument('--wandb_project', type=str, default='PFLA',
                        help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="WandB entity/username/team name")
    parser.add_argument('--wandb_group', type=str, default=None,
                        help="WandB group name")

    # Data partitioning
    parser.add_argument("--noniid", action='store_true')
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--alpha_dirich', default=0.1, type=float)

    # Practical simulation settings
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    # Regularization weight
    parser.add_argument('-lam', "--lamda", type=float, default=0.1,
                        help="Regularization weight / prototype regularization weight")

    # FedSCD Hyperparameters
    parser.add_argument('-nct', "--infoNCET", type=float, default=0.1, 
                        help="Temperature parameter for InfoNCE loss")
    parser.add_argument('-lp', '--lamda_proto', type=float, default=1.0, 
                        help="Weight coefficient for prototype loss; set to 0.0 to disable")
    parser.add_argument('--proto_warmup', type=int, default=1, 
                        help="Number of warmup rounds before applying prototype loss")
    parser.add_argument('--proto_rampup', type=int, default=5, 
                        help="Ramp-up rounds for prototype loss")
    parser.add_argument('--encoder_lr', type=float, default=None, 
                        help="Separate learning rate for the encoder/backbone")
    parser.add_argument('--tsne_gap', type=int, default=None, 
                        help="Interval for t-SNE visualization")
    parser.add_argument('--proto_momentum', type=float, default=0.1,
                        help="Momentum EMA coefficient for prototype update")
    parser.add_argument('--lamda_ins', type=float, default=0.05,
                        help="Weight for intra-batch instance contrastive loss in FedSCD")
    parser.add_argument('--lamda_nce', type=float, default=1.0,
                        help="Scaling factor for the ProtoNCE term")
    parser.add_argument('--proj_mode', type=str, default='shared', choices=['class', 'shared'],
                        help="Projection hypersphere mode: class-specific or shared")
    parser.add_argument('--phi_range', type=float, default=0.1,
                        help="Dynamic temperature clipping range")
    parser.add_argument('--lamda_virt_local', type=float, default=0.0,
                        help="Local virtual feature regularization weight")
    parser.add_argument('--lamda_virt_global', type=float, default=0.05,
                        help="Global virtual feature regularization weight")
    parser.add_argument('--scd_aug', '--pcl_aug', dest='scd_aug', type=int, default=1,
                        help="Enable GPU contrastive view augmentation (1: True, 0: False)")
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help="Feature manifold mixup alpha for classifier head")
    parser.add_argument('--bayes_shrink', type=float, default=0.25,
                        help="Max Empirical Bayes shrinkage weight for local prototype")
    parser.add_argument('--lamda_compact', type=float, default=1,
                        help="Weight for compactness loss L_PL")
    parser.add_argument('--fixed_temp', action='store_true', default=False,
                        help="Disable dynamic temperature and fix temperature to base tau")

    # FedTGP Hyperparameters
    parser.add_argument('--tgp_server_epochs', '-se', '--server_epochs', type=int, default=100,
                        help="Number of server epochs for FedTGP prototype model")
    parser.add_argument('-mart', '--margin_threthold', type=float, default=100.0,
                        help="Margin threshold for FedTGP adaptive margin")
    parser.add_argument('--tgp_server_lr', type=float, default=0.01,
                        help="Server learning rate for FedTGP")
    parser.add_argument('-fd', '--feature_dim', type=int, default=512,
                        help="Feature dimension for FedTGP prototypes")

    # FedSA Hyperparameters
    parser.add_argument('-mcl', "--margin_contrastive", type=float, default=1.0,
                        help="Weight for margin-enhanced contrastive learning (FedSA)")
    parser.add_argument('-cc', "--classifier_calibration", type=float, default=1.0,
                        help="Weight for anchor-based classifier calibration (FedSA)")

    # FedPLVM Hyperparameters
    parser.add_argument('--plvm_alpha', type=float, default=0.25,
                        help="alpha-sparsity parameter for FedPLVM")
    parser.add_argument('--plvm_lamb', '--lamb', type=float, default=100.0,
                        help="Weight parameter for prototype loss terms in FedPLVM")
    parser.add_argument('--plvm_tau', type=float, default=0.07,
                        help="Temperature parameter for alpha-sparsity loss in FedPLVM")

    # FedImpro Hyperparameters
    parser.add_argument('--impro_noise_std', type=float, default=0.1,
                        help="Noise std for FedImpro Gaussian feature synthesis")
    parser.add_argument('--impro_weight', '--impro_loss_weight', type=float, default=1.0,
                        help="Weight for synthetic feature loss in FedImpro")
    parser.add_argument('--impro_sched', type=str, default="gradual2half",
                        choices=["gradual2half", "default", "gradual"],
                        help="Weight schedule for synthetic features in FedImpro")
    parser.add_argument('--impro_max_rounds', type=int, default=500,
                        help="Max rounds for synthetic feature weight schedule in FedImpro")
    parser.add_argument('--impro_constrain_feat', action='store_true', default=False,
                        help="Enable feature distance constraint in FedImpro")

    # FedCOF Hyperparameters
    parser.add_argument('--cof_gamma', '--shrink', type=float, default=1.0,
                        help="Covariance shrinkage parameter for FedCOF")

    # Fed3R Hyperparameters
    parser.add_argument('--fed3r_lambda', '--lambda_', type=float, default=0.01,
                        help="Ridge regression regularization parameter for Fed3R")
    parser.add_argument('--fed3r_norm', action='store_true', default=True,
                        help="Normalize Fed3R classifier weights")

    # FedORGP / FedOrth Hyperparameters
    parser.add_argument('-clam', '--c_lamda', '--orgp_c_lamda', type=float, default=1.0,
                        help="Client prototype alignment weight for FedORGP")
    parser.add_argument('-slam', '--s_lamda', '--orgp_s_lamda', type=float, default=1.0,
                        help="Server intra-class loss weight for FedORGP")
    parser.add_argument('-gam', '--gamma', '--orgp_gamma', type=float, default=1.0,
                        help="Server inter-class orthogonal loss weight for FedORGP")
    parser.add_argument('--orgp_alpha', type=float, default=0.5,
                        help="Adaptive filtering weight alpha for FedORGP")
    parser.add_argument('--orgp_server_epochs', type=int, default=1,
                        help="Server training epochs for FedORGP prototype generator")
    parser.add_argument('--orgp_server_lr', type=float, default=0.01,
                        help="Server learning rate for FedORGP prototype generator")
    parser.add_argument('-abs', '--ablation_server', action='store_true', default=False,
                        help="Disable adaptive weights in server loss for FedORGP")
    parser.add_argument('-abc', '--ablation_client', action='store_true', default=False,
                        help="Disable adaptive outlier filtering in client for FedORGP")

    # FedGMKD Hyperparameters
    parser.add_argument('--gmkd_lam', '--lam_kd', type=float, default=0.06,
                        help="Weight for Group Knowledge Distillation loss in FedGMKD")
    parser.add_argument('--gmkd_gamma', '--gamma_feat', type=float, default=0.06,
                        help="Weight for feature prototype alignment loss in FedGMKD")
    parser.add_argument('--gmkd_temp', '--temperature', type=float, default=0.6,
                        help="Temperature for soft prediction in FedGMKD")
    parser.add_argument('--gmkd_std', type=float, default=0.0,
                        help="Std of Gaussian noise for prototype features in FedGMKD")
    parser.add_argument('--gmkd_param_a', type=float, default=0.2,
                        help="Dirichlet skew penalty param_a for client model weighting in FedGMKD")
    parser.add_argument('--gmkd_param_b', type=float, default=0.2,
                        help="Offset param_b for client model weighting in FedGMKD")
    parser.add_argument('--gmkd_param_a_cls', type=float, default=0.4,
                        help="Divergence penalty param_a for class-wise knowledge aggregation in FedGMKD")
    parser.add_argument('--gmkd_param_b_cls', type=float, default=0.2,
                        help="Offset param_b for class-wise knowledge aggregation in FedGMKD")

    # Seed & Data Loader
    parser.add_argument('-sd', "--seed", type=int, default=None,
                        help="Random seed for model initialization and training")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle training data in client local DataLoader")

    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        args.device = torch.device("cuda", index=args.device_id)
    else:
        args.device = torch.device("cpu")

    run(args)

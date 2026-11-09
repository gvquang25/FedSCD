#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Benchmark Matrix Runner for FedSCD & 6 Baselines (Table 4, ICML 2026).

Algorithms (7):
  FedTGP, FedPLVM, FedGMKD, FedSA, FedORGP, FedCOF, FedSCD

Datasets (4):
  Cifar10 (10 classes)
  Cifar100 (100 classes)
  Flowers102 (102 classes)
  TinyImagenet (200 classes)

Dirichlet Alphas (3):
  0.1, 0.5, 1.0
Total cells in matrix: 7 * 4 * 3 = 84 runs.

Features:
  - Automatically identifies completed runs according to Table 4 / results folder.
  - Supports running on Windows with NVIDIA GPU (RTX 4050, CUDA device 0).
  - WandB logging enabled for all runs.
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# Completed runs already recorded in Table 4 of icml2026.tex
COMPLETED_RUNS = {
    # (dataset, algo, alpha)
    ("Cifar10", "FedTGP", 0.1),
    ("Cifar10", "FedTGP", 0.5),
    ("Cifar10", "FedTGP", 1.0),
    ("Cifar10", "FedSA", 0.5),
    ("Cifar10", "FedORGP", 0.5),
    ("Cifar10", "FedCOF", 0.5),
    ("Cifar10", "FedSCD", 0.1),
    ("Cifar10", "FedSCD", 0.5),
    ("Cifar10", "FedSCD", 1.0),
}

DATASET_CONFIG = {
    "Cifar10": {"num_classes": 10, "model": "cnn"},
    "Cifar100": {"num_classes": 100, "model": "cnn"},
    "Flowers102": {"num_classes": 102, "model": "cnn"},
    "TinyImagenet": {"num_classes": 200, "model": "cnn"},
}

ALL_ALGOS = ["FedTGP", "FedPLVM", "FedGMKD", "FedSA", "FedORGP", "FedCOF", "FedSCD"]
ALL_ALPHAS = [0.1, 0.5, 1.0]


def get_algo_extra_args(algo):
    """Return tuned hyperparameters matching the paper benchmark."""
    algo_lower = algo.lower()
    if algo_lower in ["fedtgp", "tgp"]:
        return ["-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1"]
    elif algo_lower in ["fedsa", "sa"]:
        return ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "-mcl", "0.01", "-cc", "1.0"]
    elif algo_lower in ["fedplvm", "plvm"]:
        return ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--plvm_alpha", "0.25", "--plvm_lamb", "100.0", "--plvm_tau", "0.07"]
    elif algo_lower in ["fedgmkd", "gmkd"]:
        return ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--gmkd_lam", "0.06", "--gmkd_gamma", "0.06", "--gmkd_temp", "0.6"]
    elif algo_lower in ["fedorgp", "orgp"]:
        return ["-clam", "1.0", "-slam", "1.0", "-gam", "1.0", "--orgp_alpha", "0.5", "--orgp_server_epochs", "1", "--orgp_server_lr", "0.01"]
    elif algo_lower in ["fedcof", "cof"]:
        return ["-lam", "0.1", "--cof_gamma", "1.0"]
    elif algo_lower in ["fedscd", "scd"]:
        return [
            "-lamda", "0.1",
            "-nct", "0.1",
            "--phi_range", "0.1",
            "--proto_momentum", "0.1",
            "--bayes_shrink", "0.25",
            "--lamda_ins", "0.05",
            "--lamda_virt_global", "0.05",
            "--proto_warmup", "1",
            "--proto_rampup", "5"
        ]
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="FedSCD Benchmark Matrix Runner (Table 4, ICML 2026)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIG.keys()),
                        choices=list(DATASET_CONFIG.keys()), help="Datasets to execute")
    parser.add_argument("--algos", nargs="+", default=ALL_ALGOS,
                        choices=ALL_ALGOS, help="Algorithms to execute")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALL_ALPHAS,
                        help="Dirichlet alpha levels (e.g. 0.1 0.5 1.0)")
    parser.add_argument("--global_rounds", "-gr", type=int, default=100, help="Global communication rounds")
    parser.add_argument("--local_epochs", "-ls", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", "-lbs", type=int, default=32, help="Local batch size (default: 32 as per paper)")
    parser.add_argument("--lr", "-lr", type=float, default=0.01, help="Learning rate (default: 0.01 as per paper)")
    parser.add_argument("--num_clients", "-nc", type=int, default=20, help="Total number of clients")
    parser.add_argument("--join_ratio", "-jr", type=float, default=1.0, help="Client participation ratio per round")
    parser.add_argument("--device", "-dev", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Compute device")
    parser.add_argument("--device_id", "-did", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--times", "-t", type=int, default=1, help="Independent runs/seeds (default: 1)")
    parser.add_argument("--wandb_project", type=str, default="FedSCD", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default="gvquang25-hanoi-university-of-science-and-technology", help="WandB entity")
    parser.add_argument("--skip_completed", action="store_true", default=True,
                        help="Skip runs already present in Table 4 (default: True)")
    parser.add_argument("--force_all", action="store_true", default=False,
                        help="Force running all configurations including completed ones")
    return parser.parse_args()


def main():
    args = parse_args()
    skip_completed = args.skip_completed and not args.force_all

    print("=" * 85)
    print("FEDERATED LEARNING BENCHMARK MATRIX RUNNER (TABLE 4, ICML 2026)")
    print(f"Device               : {args.device} (ID: {args.device_id})")
    print(f"Global Rounds        : {args.global_rounds} | Local Epochs: {args.local_epochs}")
    print(f"Batch Size           : {args.batch_size} | Learning Rate: {args.lr}")
    print(f"Clients              : {args.num_clients} (Join Ratio: {args.join_ratio})")
    print(f"WandB Project        : {args.wandb_entity} / {args.wandb_project}")
    print(f"Skip Completed Runs  : {skip_completed}")
    print("=" * 85)

    # Build execution matrix
    tasks = []
    for ds in args.datasets:
        for alpha in args.alphas:
            for algo in args.algos:
                is_done = (ds, algo, float(alpha)) in COMPLETED_RUNS
                tasks.append({
                    "dataset": ds,
                    "algo": algo,
                    "alpha": float(alpha),
                    "is_completed": is_done
                })

    pending_tasks = [t for t in tasks if not (skip_completed and t["is_completed"])]
    skipped_tasks = [t for t in tasks if skip_completed and t["is_completed"]]

    print(f"Total tasks in matrix : {len(tasks)}")
    print(f"Completed (Skipped)   : {len(skipped_tasks)}")
    print(f"Pending to Execute    : {len(pending_tasks)}")
    print("=" * 85)

    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    summary_records = []
    total_start = time.time()

    for idx, task in enumerate(pending_tasks, 1):
        ds = task["dataset"]
        algo = task["algo"]
        alpha = task["alpha"]
        nb_classes = DATASET_CONFIG[ds]["num_classes"]
        model = DATASET_CONFIG[ds]["model"]

        wandb_group = f"{ds}_nc{args.num_clients}_alpha{alpha}"

        print(f"\n[{idx}/{len(pending_tasks)}] >>> RUNNING: {algo} on {ds} (alpha={alpha}, classes={nb_classes}) <<<")

        cmd = [
            sys.executable, "-u", "main.py",
            "-algo", algo,
            "-data", ds,
            "-m", model,
            "-nc", str(args.num_clients),
            "-nb", str(nb_classes),
            "-gr", str(args.global_rounds),
            "-ls", str(args.local_epochs),
            "-lbs", str(args.batch_size),
            "-lr", str(args.lr),
            "-jr", str(args.join_ratio),
            "--noniid",
            "--alpha_dirich", str(alpha),
            "-dev", args.device,
            "-did", str(args.device_id),
            "-t", str(args.times),
            "-log",
            "--wandb_project", args.wandb_project,
            "--wandb_entity", args.wandb_entity,
            "--wandb_group", wandb_group
        ]

        extra_args = get_algo_extra_args(algo)
        cmd.extend(extra_args)

        log_file = os.path.join(logs_dir, f"{ds}_{algo}_alpha{alpha}.log")
        print(f"Command:\n{' '.join(cmd)}")
        print(f"Log path: {log_file}")

        task_start = time.time()
        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
                for line in process.stdout:
                    sys.stdout.write(line)
                    lf.write(line)
                process.wait()

            if process.returncode == 0:
                elapsed = time.time() - task_start
                print(f"[SUCCESS] {algo} on {ds} (alpha={alpha}) finished in {elapsed / 60:.2f} mins.")
                summary_records.append((ds, algo, alpha, "SUCCESS", elapsed))
            else:
                elapsed = time.time() - task_start
                print(f"[FAIL] {algo} on {ds} (alpha={alpha}) failed with returncode {process.returncode}.")
                summary_records.append((ds, algo, alpha, f"FAILED (code {process.returncode})", elapsed))

        except Exception as e:
            elapsed = time.time() - task_start
            print(f"[ERROR] Exception running {algo} on {ds} (alpha={alpha}): {e}")
            summary_records.append((ds, algo, alpha, f"ERROR: {e}", elapsed))

        except KeyboardInterrupt:
            print("\nExecution interrupted by user (Ctrl+C). Terminating current process...")
            break

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 85)
    print("BENCHMARK EXECUTION SUMMARY")
    print(f"Total Elapsed Time : {total_elapsed / 60:.2f} mins")
    print("-" * 85)
    for ds, algo, alpha, status, duration in summary_records:
        print(f"  * {ds:<14s} | {algo:<10s} | alpha={alpha:<3.1f} : {status:<18s} ({duration / 60:6.2f} mins)")
    print("=" * 85)
    print(f"WandB Dashboard: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    print("=" * 85)


if __name__ == "__main__":
    main()

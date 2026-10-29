#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sequential Benchmark Runner for 8 Baseline PFL Algorithms + WandB Logging.

Algorithms:
  1. FedTGP
  2. FedSA
  3. FedPLVM
  4. FedImpro
  5. FedCOF
  6. Fed3R
  7. FedORGP
  8. FedGMKD
(Optional: FedSCD)
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# Configure UTF-8 for console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DEFAULT_BASELINES = [
    "FedTGP",
    "FedSA",
    "FedPLVM",
    "FedImpro",
    "FedCOF",
    "Fed3R",
    "FedORGP",
    "FedGMKD",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run 8 PFL baseline algorithms sequentially and log to WandB")
    
    # Dataset and model
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10", help="Dataset name (e.g. Cifar10, mnist)")
    parser.add_argument('-m', "--model", type=str, default="cnn", help="Backbone model (e.g. cnn, cnn20, custom_resnet)")
    parser.add_argument('-nc', "--num_clients", type=int, default=20, help="Number of clients")
    parser.add_argument('-nb', "--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument('-gr', "--global_rounds", type=int, default=200, help="Global communication rounds")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10, help="Local batch size")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('--noniid', action='store_true', default=True, help="Enable Non-IID Dirichlet partition")
    parser.add_argument('--alpha_dirich', type=float, default=0.1, help="Dirichlet alpha parameter")
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")
    parser.add_argument('-did', "--device_id", type=int, default=0, help="CUDA device index")
    
    # WandB
    parser.add_argument('--wandb_project', type=str, default="FedSCD-Benchmark", help="WandB project name")
    parser.add_argument('--wandb_entity', type=str, default="gvquang25-hanoi-university-of-science-and-technology", help="WandB entity/username")
    parser.add_argument('--wandb_group', type=str, default=None, help="WandB group name for grouping runs")
    
    # Algorithm choices
    parser.add_argument('--algos', nargs='+', default=DEFAULT_BASELINES, 
                        help="List of algorithms to run (default: 8 baselines)")
    parser.add_argument('--include_scd', action='store_true', help="Include FedSCD in the execution list")
    parser.add_argument('--times', type=int, default=1, help="Number of repetitions per algorithm")

    return parser.parse_args()


def main():
    args = parse_args()
    
    algos_to_run = list(args.algos)
    if args.include_scd and "FedSCD" not in algos_to_run:
        algos_to_run.insert(0, "FedSCD")
        
    wandb_group = args.wandb_group or f"{args.dataset}_nc{args.num_clients}_alpha{args.alpha_dirich}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("STARTING SEQUENTIAL PFL BENCHMARK & WANDB LOGGING")
    print(f"Dataset          : {args.dataset} (Classes: {args.num_classes}, Clients: {args.num_clients})")
    print(f"Model            : {args.model}")
    print(f"Rounds           : {args.global_rounds} (Local Epochs: {args.local_epochs})")
    print(f"WandB Project    : {args.wandb_project}")
    print(f"WandB Entity     : {args.wandb_entity}")
    print(f"WandB Group      : {wandb_group}")
    print(f"Algorithms ({len(algos_to_run)}) : {', '.join(algos_to_run)}")
    print("=" * 80)

    summary_results = []
    total_start = time.time()

    for idx, algo in enumerate(algos_to_run, 1):
        print(f"\n[{idx}/{len(algos_to_run)}] Running algorithm: {algo} ...")
        cmd = [
            sys.executable, "-u", "main.py",
            "-algo", algo,
            "-data", str(args.dataset),
            "-m", str(args.model),
            "-nc", str(args.num_clients),
            "-nb", str(args.num_classes),
            "-gr", str(args.global_rounds),
            "-ls", str(args.local_epochs),
            "-lbs", str(args.batch_size),
            "-lr", str(args.local_learning_rate),
            "-dev", str(args.device),
            "-did", str(args.device_id),
            "-t", str(args.times),
            "-log",
            "--wandb_project", str(args.wandb_project),
            "--wandb_group", str(wandb_group),
        ]
        
        if args.wandb_entity:
            cmd.extend(["--wandb_entity", str(args.wandb_entity)])
        if args.noniid:
            cmd.append("--noniid")
        if args.alpha_dirich is not None:
            cmd.extend(["--alpha_dirich", str(args.alpha_dirich)])

        # Specific parameters for FedTGP
        if algo.lower() in ["fedtgp", "tgp"]:
            cmd.extend(["-lam", "10", "-se", "100", "-mart", "100"])

        algo_start = time.time()
        print(f"Command:\n{' '.join(cmd)}\n")
        
        try:
            res = subprocess.run(cmd, check=True)
            elapsed = time.time() - algo_start
            print(f"[DONE] {algo} completed in {elapsed / 60:.2f} mins.")
            summary_results.append((algo, "SUCCESS", elapsed))
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - algo_start
            print(f"[FAIL] {algo} failed with exit code {e.returncode} after {elapsed / 60:.2f} mins.")
            summary_results.append((algo, f"FAILED (code {e.returncode})", elapsed))
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user (Ctrl+C)!")
            break

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print(f"Total Time: {total_elapsed / 60:.2f} mins")
    print("-" * 80)
    for algo, status, duration in summary_results:
        print(f"  * {algo:12s} : {status:20s} ({duration / 60:6.2f} mins)")
    print("=" * 80)
    print(f"WandB Dashboard URL: https://wandb.ai/{args.wandb_entity}/{args.wandb_project}")
    print("=" * 80)


if __name__ == "__main__":
    main()

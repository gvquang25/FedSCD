#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chạy thực nghiệm Target Benchmark (Table 4, ICML/IEEE 2026).
Tự động chạy:
  1. Toàn bộ FedSCD trong bảng (4 datasets x 3 alphas = 12 runs)
  2. Toàn bộ Baseline trên Tiny-ImageNet (6 baselines x 3 alphas = 18 runs)
Tổng cộng: 30 runs.
"""

import os
import sys
import time
import argparse
import subprocess

ALGO_CONFIG = {
    "FedTGP": ["-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1"],
    "FedSA": ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "-mcl", "0.01", "-cc", "1.0"],
    "FedPLVM": ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--plvm_alpha", "0.25", "--plvm_lamb", "100.0", "--plvm_tau", "0.07"],
    "FedGMKD": ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "--gmkd_lam", "0.06", "--gmkd_gamma", "0.06", "--gmkd_temp", "0.6"],
    "FedORGP": ["-clam", "1.0", "-slam", "1.0", "-gam", "1.0", "--orgp_alpha", "0.5", "--orgp_server_epochs", "1", "--orgp_server_lr", "0.01"],
    "FedCOF": ["-lam", "0.1", "--cof_gamma", "1.0"],
    "FedSCD": [
        "-lam", "0.1",
        "-nct", "0.1",
        "--phi_range", "0.1",
        "--proto_momentum", "0.1",
        "--bayes_shrink", "0.25",
        "--lamda_ins", "0.05",
        "--lamda_virt_global", "0.05",
        "--proto_warmup", "1",
        "--proto_rampup", "5"
    ]
}

DATASET_INFO = {
    "Cifar10": {"num_classes": 10, "model": "cnn"},
    "Cifar100": {"num_classes": 100, "model": "cnn"},
    "Flowers102": {"num_classes": 102, "model": "cnn"},
    "TinyImagenet": {"num_classes": 200, "model": "cnn"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Target Benchmark Runner for FedSCD & Tiny-ImageNet Baselines")
    parser.add_argument("--device", "-dev", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--device_id", "-did", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--global_rounds", "-gr", type=int, default=100, help="Global communication rounds (default: 100)")
    parser.add_argument("--batch_size", "-lbs", type=int, default=32, help="Local batch size (default: 32)")
    parser.add_argument("--lr", "-lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--num_clients", "-nc", type=int, default=20, help="Total clients (default: 20)")
    parser.add_argument("--seed", "-sd", type=int, default=2006, help="Random seed (default: 2006)")
    parser.add_argument("--tsne_gap", type=int, default=20, help="Interval for t-SNE (default: 20)")
    parser.add_argument("--no_tsne", action="store_true", default=False, help="Disable t-SNE")
    parser.add_argument("--skip_done_cifar10", action="store_true", default=False,
                        help="Skip 3 CIFAR-10 FedSCD runs already present in Table")
    parser.add_argument("--skip_tiny_alpha01", action="store_true", default=False,
                        help="Skip alpha=0.1 for baselines on Tiny-ImageNet (remaining 24 runs)")
    parser.add_argument("--only_fedscd", action="store_true", default=False, help="Run only the 12 FedSCD runs")
    parser.add_argument("--only_tiny", action="store_true", default=False, help="Run only the 21 Tiny-ImageNet runs")
    parser.add_argument("--wandb_project", type=str, default="FedSCD", help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default="gvquang25-hanoi-university-of-science-and-technology", help="WandB entity")
    return parser.parse_args()


def build_tasks(args):
    tasks = []
    alphas = [0.1, 0.5, 1.0]

    # 1. Toàn bộ FedSCD trên 4 dataset
    if not args.only_tiny:
        for ds in ["Cifar10", "Cifar100", "Flowers102", "TinyImagenet"]:
            if ds == "Cifar10" and args.skip_done_cifar10:
                continue
            for a in alphas:
                tasks.append({
                    "dataset": ds,
                    "alpha": a,
                    "algo": "FedSCD",
                    "group": "Target_FedSCD"
                })

    # 2. Toàn bộ baseline trên Tiny-ImageNet
    if not args.only_fedscd:
        tiny_alphas = [0.5, 1.0] if args.skip_tiny_alpha01 else [0.1, 0.5, 1.0]
        for algo in ["FedTGP", "FedPLVM", "FedGMKD", "FedSA", "FedORGP", "FedCOF"]:
            for a in tiny_alphas:
                tasks.append({
                    "dataset": "TinyImagenet",
                    "alpha": a,
                    "algo": algo,
                    "group": "Target_TinyBaselines"
                })

    return tasks


def main():
    args = parse_args()
    tasks = build_tasks(args)

    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(logs_dir, exist_ok=True)

    print("=" * 85)
    print("TARGET BENCHMARK RUNNER: FedSCD & TINY-IMAGENET BASELINES")
    print(f"Device      : {args.device}:{args.device_id}")
    print(f"Rounds      : {args.global_rounds} | Batch: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}")
    print(f"Total runs  : {len(tasks)}")
    print("=" * 85)

    summary_records = []

    for idx, t in enumerate(tasks, start=1):
        ds = t["dataset"]
        alpha = t["alpha"]
        algo = t["algo"]
        info = DATASET_INFO[ds]

        wandb_group = f"{ds}_nc{args.num_clients}_alpha{alpha}"
        log_file = os.path.join(logs_dir, f"{ds}_{algo}_alpha{alpha}.log")

        cmd = [
            sys.executable, "-u", "main.py",
            "-algo", algo,
            "-data", ds,
            "-m", info["model"],
            "-nc", str(args.num_clients),
            "-nb", str(info["num_classes"]),
            "-gr", str(args.global_rounds),
            "-ls", "1",
            "-lbs", str(args.batch_size),
            "-lr", str(args.lr),
            "-jr", "1.0",
            "--noniid",
            "--alpha_dirich", str(alpha),
            "-dev", args.device,
            "-did", str(args.device_id),
            "-t", "1",
            "-sd", str(args.seed),
            "-log",
            "--wandb_project", args.wandb_project,
            "--wandb_entity", args.wandb_entity,
            "--wandb_group", wandb_group
        ]

        if args.no_tsne:
            cmd.append("--no_tsne")
        elif args.tsne_gap > 0:
            cmd.extend(["--tsne_gap", str(args.tsne_gap)])

        extra_args = ALGO_CONFIG.get(algo, [])
        cmd.extend(extra_args)

        print(f"\n[{idx}/{len(tasks)}] >>> RUNNING: {algo} on {ds} (alpha={alpha}) <<<")
        print(f"Log: {log_file}")
        start_time = time.time()

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                process = subprocess.Popen(
                    cmd,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                for line in process.stdout:
                    sys.stdout.write(line)
                    f_log.write(line)
                process.wait()

            elapsed = time.time() - start_time
            if process.returncode == 0:
                print(f"[SUCCESS] {algo} on {ds} (alpha={alpha}) in {elapsed / 60:.2f} mins.")
                summary_records.append((ds, algo, alpha, "SUCCESS", elapsed))
            else:
                print(f"[FAILED] {algo} on {ds} (alpha={alpha}) code {process.returncode}.")
                summary_records.append((ds, algo, alpha, f"FAIL({process.returncode})", elapsed))
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[ERROR] Exception running {algo} on {ds} (alpha={alpha}): {e}")
            summary_records.append((ds, algo, alpha, f"ERROR({e})", elapsed))

    print("\n" + "=" * 85)
    print("FINAL SUMMARY REPORT")
    print("=" * 85)
    for ds, algo, alpha, status, duration in summary_records:
        print(f"  * {ds:<14s} | {algo:<10s} | alpha={alpha:<3.1f} : {status:<15s} ({duration / 60:6.2f} mins)")
    print("=" * 85)


if __name__ == "__main__":
    main()


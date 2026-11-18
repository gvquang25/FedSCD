#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated runner to generate dense balanced t-SNE embeddings (default: 3,000 samples =
300 samples/class x 10 classes) for the 6 IEEE comparison algorithms on CIFAR-10 (alpha=0.5).

Synchronizes FedSCD with the verified 76.66% manuscript configuration (c7tds897) and
saves both 2D t-SNE coordinates and 512-D embeddings (.npz) + Client 0 checkpoints (.pt).
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs", "tsne_balanced_3k")

DEFAULT_ALGOS = ["FedTGP", "FedCOF", "FedSA", "FedORGP", "FedGMKD", "FedSCD"]

ALGO_EXTRA_ARGS = {
    "FedTGP": ["-lam", "10.0", "-se", "100", "-mart", "100", "-lp", "1.0", "-nct", "0.1"],
    "FedCOF": ["-lam", "0.1", "--cof_gamma", "1.0"],
    "FedSA": ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1", "-mcl", "0.01", "-cc", "1.0"],
    "FedORGP": ["-lam", "0.1", "-clam", "1.0", "-slam", "1.0", "-gam", "1.0",
                "--orgp_alpha", "0.5", "--orgp_server_epochs", "1", "--orgp_server_lr", "0.01"],
    "FedGMKD": ["-lam", "0.1", "-lp", "1.0", "-nct", "0.1",
                "--gmkd_lam", "0.06", "--gmkd_gamma", "0.06", "--gmkd_temp", "0.6"],
    # Verified 76.66% manuscript configuration on CIFAR-10 (alpha=0.5, seed=2006) — DO NOT pass -lp!
    "FedSCD": [
        "-lam", "0.3",
        "-nct", "0.1",
        "--phi_range", "0.1",
        "--proto_momentum", "0.6",
        "--bayes_shrink", "0.25",
        "--lamda_ins", "0.75",
        "--lamda_virt_local", "0.3",
        "--lamda_virt_global", "0.3",
        "--proto_warmup", "1",
        "--proto_rampup", "5",
        "--mixup_alpha", "0.2",
        "--scd_aug", "1",
        "--proj_mode", "shared",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 6 algorithms to extract dense balanced t-SNE embeddings (3,000 samples) on CIFAR-10."
    )
    parser.add_argument("--device", "-dev", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device_id", "-did", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--global_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--join_ratio", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--tsne_gap", type=int, default=100)
    parser.add_argument("--tsne_max_samples", type=int, default=3000,
                        help="Total balanced test points across all classes (default: 3000 = 300/class)")
    parser.add_argument("--tsne_fair", action="store_true", default=False,
                        help="Use unweighted centered-L2 projection during online t-SNE (512-D features are always saved regardless)")
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS, choices=DEFAULT_ALGOS)
    parser.add_argument("--wandb_project", type=str, default="FedSCD")
    parser.add_argument("--wandb_entity", type=str,
                        default="gvquang25-hanoi-university-of-science-and-technology")
    parser.add_argument("--wandb_group", type=str, default="Cifar10_nc20_alpha0.5_tsne3k")
    parser.add_argument("--force_all", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def build_command(args, algo):
    goal = f"tsne3k_{algo}_alpha{args.alpha}"
    cmd = [
        sys.executable, "-u", "main.py",
        "-algo", algo,
        "-data", args.dataset,
        "-m", args.model,
        "-nc", str(args.num_clients),
        "-nb", str(args.num_classes),
        "-gr", str(args.global_rounds),
        "-ls", str(args.local_epochs),
        "-lbs", str(args.batch_size),
        "-lr", str(args.lr),
        "-jr", str(args.join_ratio),
        "--noniid",
        "--alpha_dirich", str(args.alpha),
        "-dev", args.device,
        "-did", str(args.device_id),
        "-t", "1",
        "-sd", str(args.seed),
        "-go", goal,
        "-log",
        "--wandb_project", args.wandb_project,
        "--wandb_entity", args.wandb_entity,
        "--wandb_group", args.wandb_group,
        "--tsne_gap", str(args.tsne_gap),
        "--tsne_max_samples", str(args.tsne_max_samples),
        "--tsne_balanced",
        "--save_tsne_ckpt",
    ]
    if args.tsne_fair:
        cmd.append("--tsne_fair")
    cmd.extend(ALGO_EXTRA_ARGS[algo])
    return goal, cmd


def main():
    args = parse_args()
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=" * 88)
    print(f"DENSE BALANCED t-SNE RUNNER ({args.tsne_max_samples} SAMPLES | {args.dataset} | alpha={args.alpha})")
    print(f"Device       : {args.device}:{args.device_id}")
    print(f"WandB Target : {args.wandb_entity}/{args.wandb_project} (group: {args.wandb_group})")
    print(f"Algorithms   : {', '.join(args.algos)}")
    print("=" * 88)

    records = []
    for idx, algo in enumerate(args.algos, start=1):
        goal, cmd = build_command(args, algo)
        done_marker = os.path.join(LOGS_DIR, f"{goal}.done")
        log_file = os.path.join(LOGS_DIR, f"{goal}.log")

        if args.dry_run:
            print(f"[{idx}/{len(args.algos)}] {algo} ({goal}):")
            print("  " + " ".join(cmd))
            records.append((algo, "DRY_RUN", 0.0))
            continue

        if (not args.force_all) and os.path.exists(done_marker):
            print(f"[{idx}/{len(args.algos)}] [SKIP] {algo} already completed ({done_marker}).")
            records.append((algo, "SKIPPED", 0.0))
            continue

        print(f"\n[{idx}/{len(args.algos)}] >>> RUNNING: {algo} ({args.tsne_max_samples} balanced samples) <<<")
        print(f"Log: {log_file}")
        t0 = time.time()

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                proc = subprocess.Popen(
                    cmd,
                    cwd=SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    sys.stdout.write(line)
                    f_log.write(line)
                proc.wait()

            elapsed = time.time() - t0
            if proc.returncode == 0:
                with open(done_marker, "w", encoding="utf-8") as f_done:
                    f_done.write(f"algo={algo}\ngoal={goal}\nelapsed_sec={elapsed:.2f}\n")
                print(f"[SUCCESS] {algo} finished in {elapsed / 60:.2f} mins.")
                records.append((algo, "SUCCESS", elapsed))
            else:
                print(f"[FAILED] {algo} exited with code {proc.returncode}.")
                records.append((algo, f"FAIL({proc.returncode})", elapsed))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[ERROR] {algo} raised exception: {exc}")
            records.append((algo, f"ERROR({exc})", elapsed))

    print("\n" + "=" * 88)
    print("DENSE t-SNE RUN SUMMARY")
    print("=" * 88)
    for algo, status, elapsed in records:
        print(f"  * {algo:<10s} : {status:<15s} ({elapsed / 60:6.2f} mins)")
    print("=" * 88)

    failures = [r for r in records if r[1].startswith("FAIL") or r[1].startswith("ERROR")]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

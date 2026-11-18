#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-factor-at-a-time hyperparameter sensitivity runner for FedSCD on CIFAR-10 (alpha=0.5).
Holds all parameters fixed at the verified 76.66% reference configuration (WandB run c7tds897)
and sweeps 6 parameters across 4 values each (18 non-default runs + optional default run).

CRITICAL: Never pass -lp / --lamda_proto (clientscd.py treats -lp != 1.0 as overriding -lam).
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs", "hyperparam_sensitivity")

# Exact reference configuration matching c7tds897 (76.66% on CIFAR-10, alpha=0.5, seed=2006)
BASE_CONFIG = {
    "-algo": "FedSCD",
    "-data": "Cifar10",
    "-m": "cnn",
    "-nc": "20",
    "-nb": "10",
    "-gr": "100",
    "-ls": "1",
    "-lbs": "32",
    "-lr": "0.01",
    "-jr": "1.0",
    "--alpha_dirich": "0.5",
    "-t": "1",
    "-sd": "2006",
    "-lam": "0.3",
    "-nct": "0.1",
    "--phi_range": "0.1",
    "--proto_momentum": "0.6",
    "--bayes_shrink": "0.25",
    "--lamda_ins": "0.75",
    "--lamda_virt_local": "0.3",
    "--lamda_virt_global": "0.3",
    "--proto_warmup": "1",
    "--proto_rampup": "5",
    "--mixup_alpha": "0.2",
    "--scd_aug": "1",
    "--proj_mode": "shared",
}

BASE_BOOL_FLAGS = ["--noniid", "--no_tsne", "-log"]

SWEEP_DEFINITIONS = [
    ("lam", "hp_lam_0.1", {"-lam": "0.1"}, []),
    ("lam", "hp_lam_0.5", {"-lam": "0.5"}, []),
    ("lam", "hp_lam_1.0", {"-lam": "1.0"}, []),
    ("ins", "hp_ins_0.0", {"--lamda_ins": "0.0"}, []),
    ("ins", "hp_ins_0.25", {"--lamda_ins": "0.25"}, []),
    ("ins", "hp_ins_1.0", {"--lamda_ins": "1.0"}, []),
    ("virt", "hp_virt_0.0", {"--lamda_virt_local": "0.0", "--lamda_virt_global": "0.0"}, []),
    ("virt", "hp_virt_0.1", {"--lamda_virt_local": "0.1", "--lamda_virt_global": "0.1"}, []),
    ("virt", "hp_virt_0.5", {"--lamda_virt_local": "0.5", "--lamda_virt_global": "0.5"}, []),
    ("beta", "hp_beta_0.1", {"--proto_momentum": "0.1"}, []),
    ("beta", "hp_beta_0.3", {"--proto_momentum": "0.3"}, []),
    ("beta", "hp_beta_1.0", {"--proto_momentum": "1.0"}, []),
    ("rho", "hp_rho_0.0", {"--phi_range": "0.0"}, ["--fixed_temp"]),
    ("rho", "hp_rho_0.05", {"--phi_range": "0.05"}, []),
    ("rho", "hp_rho_0.2", {"--phi_range": "0.2"}, []),
    ("tau", "hp_tau_0.05", {"-nct": "0.05"}, []),
    ("tau", "hp_tau_0.2", {"-nct": "0.2"}, []),
    ("tau", "hp_tau_0.5", {"-nct": "0.5"}, []),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 1-factor-at-a-time FedSCD hyperparameter sensitivity sweep on CIFAR-10 (alpha=0.5)."
    )
    parser.add_argument("--device", "-dev", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Compute device passed to main.py (default: cuda)")
    parser.add_argument("--device_id", "-did", type=int, default=0,
                        help="CUDA device ID (default: 0)")
    parser.add_argument("--wandb_project", type=str, default="FedSCD",
                        help="WandB project name (default: FedSCD)")
    parser.add_argument("--wandb_entity", type=str,
                        default="gvquang25-hanoi-university-of-science-and-technology",
                        help="WandB entity name")
    parser.add_argument("--wandb_group", type=str, default="Cifar10_hyperparam_alpha0.5",
                        help="WandB group name (default: Cifar10_hyperparam_alpha0.5)")
    parser.add_argument("--only_param", type=str, default=None,
                        choices=["lam", "ins", "virt", "beta", "rho", "tau"],
                        help="Optional: run only the 3 sweeps for one parameter")
    parser.add_argument("--include_default", action="store_true", default=False,
                        help="Also run the default configuration as hp_default")
    parser.add_argument("--skip_finished", action="store_true", default=True,
                        help="Skip runs already marked done locally or finished on WandB (default: True)")
    parser.add_argument("--force_all", action="store_true", default=False,
                        help="Re-run all selected experiments even if already finished")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print commands without executing main.py")
    return parser.parse_args()


def fetch_wandb_finished_goals(entity, project, group):
    """Query WandB safely to detect already-finished goals in wandb_group."""
    finished_goals = set()
    try:
        saved_path = list(sys.path)
        blocked = {os.path.abspath("."), REPO_ROOT, SCRIPT_DIR}
        sys.path = [p for p in sys.path if os.path.abspath(p) not in blocked]
        import wandb  # noqa: E402
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"group": group, "state": "finished"})
        for r in runs:
            goal = (r.config or {}).get("goal")
            if goal:
                finished_goals.add(str(goal))
        sys.path = saved_path
    except Exception as exc:
        print(f"[WARN] Could not query WandB for finished runs ({exc}); relying on local .done markers only.")
    return finished_goals


def build_tasks(args):
    tasks = []
    if args.include_default:
        tasks.append(("default", "hp_default", {}, []))
    for param, goal, overrides, extra_flags in SWEEP_DEFINITIONS:
        if args.only_param is None or param == args.only_param:
            tasks.append((param, goal, overrides, extra_flags))
    return tasks


def build_command(args, goal, overrides, extra_flags):
    cfg = dict(BASE_CONFIG)
    cfg.update(overrides)

    cmd = [sys.executable, "-u", "main.py"]
    for flag, val in cfg.items():
        cmd.extend([flag, str(val)])

    cmd.extend(BASE_BOOL_FLAGS)
    for flag in extra_flags:
        if flag not in cmd:
            cmd.append(flag)

    cmd.extend([
        "-dev", args.device,
        "-did", str(args.device_id),
        "-go", goal,
        "--wandb_project", args.wandb_project,
        "--wandb_entity", args.wandb_entity,
        "--wandb_group", args.wandb_group,
    ])
    return cmd


def main():
    args = parse_args()
    skip_finished = False if args.force_all else args.skip_finished
    tasks = build_tasks(args)

    os.makedirs(LOGS_DIR, exist_ok=True)
    wandb_finished = set()
    if skip_finished and not args.dry_run:
        wandb_finished = fetch_wandb_finished_goals(args.wandb_entity, args.wandb_project, args.wandb_group)

    print("=" * 88)
    print("FED-SCD HYPERPARAMETER SENSITIVITY RUNNER (CIFAR-10, alpha=0.5, seed=2006)")
    print(f"Device       : {args.device}:{args.device_id}")
    print(f"WandB Target : {args.wandb_entity}/{args.wandb_project} (group: {args.wandb_group})")
    print(f"Total tasks  : {len(tasks)} | Skip finished: {skip_finished} | Dry run: {args.dry_run}")
    print("=" * 88)

    records = []
    for idx, (param, goal, overrides, extra_flags) in enumerate(tasks, start=1):
        done_marker = os.path.join(LOGS_DIR, f"{goal}.done")
        log_file = os.path.join(LOGS_DIR, f"{goal}.log")
        cmd = build_command(args, goal, overrides, extra_flags)

        if args.dry_run:
            print(f"[{idx:02d}/{len(tasks):02d}] ({param}) {goal}:")
            print("  " + " ".join(cmd))
            records.append((goal, "DRY_RUN", 0.0))
            continue

        if skip_finished and (os.path.exists(done_marker) or goal in wandb_finished):
            reason = "local .done" if os.path.exists(done_marker) else "WandB finished"
            print(f"[{idx:02d}/{len(tasks):02d}] [SKIP] {goal} already finished ({reason}).")
            records.append((goal, "SKIPPED", 0.0))
            continue

        print(f"\n[{idx:02d}/{len(tasks):02d}] >>> RUNNING: {goal} (param={param}, overrides={overrides}, extra={extra_flags}) <<<")
        print(f"Log: {log_file}")
        t0 = time.time()

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                process = subprocess.Popen(
                    cmd,
                    cwd=SCRIPT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in process.stdout:
                    sys.stdout.write(line)
                    f_log.write(line)
                process.wait()

            elapsed = time.time() - t0
            if process.returncode == 0:
                with open(done_marker, "w", encoding="utf-8") as f_done:
                    f_done.write(f"goal={goal}\nreturncode=0\nelapsed_sec={elapsed:.2f}\n")
                print(f"[SUCCESS] {goal} finished in {elapsed / 60:.2f} mins.")
                records.append((goal, "SUCCESS", elapsed))
            else:
                print(f"[FAILED] {goal} exited with code {process.returncode}.")
                records.append((goal, f"FAIL({process.returncode})", elapsed))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[ERROR] {goal} raised exception: {exc}")
            records.append((goal, f"ERROR({exc})", elapsed))

    print("\n" + "=" * 88)
    print("HYPERPARAMETER SENSITIVITY SUMMARY")
    print("=" * 88)
    for goal, status, elapsed in records:
        print(f"  * {goal:<16s} : {status:<15s} ({elapsed / 60:6.2f} mins)")
    print("=" * 88)

    failures = [r for r in records if r[1].startswith("FAIL") or r[1].startswith("ERROR")]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()

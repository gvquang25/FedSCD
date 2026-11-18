"""Sequential FedSCD LR/lambda tuning on train-only validation, alpha=0.5.

Run from any directory: python system/tune_fedscd.py
Each dataset gets three LR trials, then two lambda trials at its winning LR.
Test files are never read by trials. Existing training code is unchanged.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
from datetime import datetime

SYSTEM = Path(__file__).resolve().parent
ROOT = SYSTEM.parent
DATASETS = {"Cifar10": 10, "Cifar100": 100, "Flowers102": 102, "TinyImageNet": 200}


def split_indices(labels, fraction, seed):
    """Deterministic per-class holdout; singleton classes stay in training."""
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(groups):
        indices = groups[label][:]
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        val.extend(indices[:count])
        train.extend(indices[count:])
    if not val:
        raise ValueError("Client has no validation samples after stratified split")
    return sorted(train), sorted(val)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def worker(spec_path):
    # Install the train-only reader before importing main/client/server modules.
    import numpy as np
    from utils import data_utils

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    original_read = data_utils.read_data
    split_cache = {}

    def validation_read(dataset, idx, niid, balance, alpha, is_train=True, num_clients=None):
        if spec.get("data_root"):
            path = Path(spec["data_root"]) / dataset / str(num_clients) / f"{balance}_{niid}_{alpha}" / "train" / f"{idx}.npz"
            with np.load(path, allow_pickle=True) as archive:
                raw = archive["data"].tolist()
        else:
            raw = original_read(dataset, idx, niid, balance, alpha, True, num_clients)
        key = (dataset, idx, niid, balance, alpha, num_clients)
        if key not in split_cache:
            split_cache[key] = split_indices(raw["y"], spec["val_fraction"], spec["split_seed"] + idx)
        train, val = split_cache[key]
        indices = train if is_train else val
        return {"x": np.asarray(raw["x"])[indices], "y": np.asarray(raw["y"])[indices]}

    data_utils.read_data = validation_read
    from flcore.servers.serverscd import FedSCD

    original_save = FedSCD.save_results

    def save_validation(server):
        original_save(server)
        scores = [float(value) for value in server.rs_test_acc]
        if not scores or not np.isfinite(scores).all():
            raise RuntimeError("Missing or non-finite validation scores")
        # Use final validation accuracy: every candidate has the same round budget.
        write_json(spec_path.with_suffix(".result.json"), {
            "dataset": spec["dataset"], "lr": spec["lr"], "lam": spec["lam"],
            "virt_local": spec.get("virt_local", 0.0), "temperature": spec.get("temperature", 0.1),
            "seed": spec.get("seed", 2006),
            "final_val_accuracy": scores[-1], "best_val_accuracy": max(scores),
            "validation_curve": scores,
        })

    FedSCD.save_results = save_validation
    os.chdir(SYSTEM)
    sys.argv = [str(SYSTEM / "main.py"), *spec["args"]]
    print("TUNING: all evaluation labelled 'test' below uses TRAIN HOLDOUT VALIDATION.", flush=True)
    runpy.run_path(str(SYSTEM / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100, help="main.py -gr (currently trains rounds 0..gr inclusive)")
    parser.add_argument("--seed", type=int, default=2006)
    parser.add_argument("--split-seed", type=int, default=2006)
    parser.add_argument("--extended", action="store_true", help="Eight trials/dataset: virtual local, lambda and temperature")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, help="New directory; existing directories are rejected")
    parser.add_argument("--check-only", action="store_true", help="Check environment and all client splits without training")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.rounds < 1 or not 0 < args.val_fraction < 1:
        parser.error("rounds must be >=1 and val-fraction must be between 0 and 1")

    import numpy as np
    import torch
    # Check every dependency imported by the actual entry point before a long run.
    import main as training_main  # noqa: F401

    if args.device == "cuda":
        if not torch.cuda.is_available() or not 0 <= args.device_id < torch.cuda.device_count():
            raise RuntimeError("Requested CUDA GPU is unavailable; refusing silent CPU fallback")
        print(f"GPU: {torch.cuda.get_device_name(args.device_id)}", flush=True)

    splits = {}
    for dataset in DATASETS:
        splits[dataset] = []
        for client in range(20):
            path = ROOT / "dataset" / dataset / "20" / "False_True_0.5" / "train" / f"{client}.npz"
            with np.load(path, allow_pickle=True) as archive:
                labels = archive["data"].tolist()["y"]
            train, val = split_indices(labels, args.val_fraction, args.split_seed + client)
            if len(train) < 32:
                raise ValueError(f"{dataset} client {client}: fewer than one full training batch")
            splits[dataset].append({"client": client, "train_indices": train, "validation_indices": val})
        print(f"Checked {dataset}: 20 train/validation splits", flush=True)
    if args.check_only:
        return

    output = (args.output or ROOT / "logs" / "tune_fedscd" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "splits.json", splits)
    write_json(output / "manifest.json", {
        "alpha": 0.5, "num_clients": 20, "rounds_argument": args.rounds,
        "actual_training_rounds": args.rounds + 1, "seed": args.seed,
        "split_seed": args.split_seed, "extended": args.extended,
        "validation_fraction_requested": args.val_fraction,
        "selection_metric": "final_val_accuracy", "python": sys.executable,
        "torch": torch.__version__,
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), SYSTEM / "main.py",
                                     SYSTEM / "flcore/clients/clientscd.py",
                                     SYSTEM / "flcore/servers/serverscd.py"]},
    })
    rows, winners = [], {}
    batch_id = output.name + "_" + hashlib.sha256(str(output).encode()).hexdigest()[:8]

    total_trials = 32 if args.extended else 20

    def trial(dataset, lr, lam, virt_local=0.0, temperature=0.1):
        name = f"{dataset}_lr{lr}_lam{lam}_vl{virt_local}_tau{temperature}"
        spec_path = output / f"{name}.json"
        command_args = [
            "-algo", "FedSCD", "-data", dataset, "-m", "cnn", "-nb", str(DATASETS[dataset]),
            "-nc", "20", "-jr", "1.0", "--noniid", "--alpha_dirich", "0.5",
            "-gr", str(args.rounds), "-ls", "1", "-lbs", "32", "-lr", str(lr),
            "-lam", str(lam), "-nct", str(temperature), "--lamda_compact", "1.0",
            "--lamda_ins", "0.05", "--lamda_virt_local", str(virt_local), "--lamda_virt_global", "0.05",
            "--proto_momentum", "0.1", "--phi_range", "0.1", "--bayes_shrink", "0.25",
            "--proto_warmup", "1", "--proto_rampup", "5", "--scd_aug", "1",
            "--mixup_alpha", "0.2", "--proj_mode", "shared",
            "-dev", args.device, "-did", str(args.device_id), "-sd", str(args.seed),
            "-t", "1", "--no_tsne", "-go", f"validation_{batch_id}_{name}",
        ]
        write_json(spec_path, {"dataset": dataset, "lr": lr, "lam": lam,
                              "virt_local": virt_local, "temperature": temperature, "seed": args.seed,
                              "val_fraction": args.val_fraction, "split_seed": args.split_seed,
                              "args": command_args})
        print(f"[{len(rows) + 1}/{total_trials}] {name} -> {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(spec_path)],
                           cwd=SYSTEM, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(spec_path.with_suffix(".result.json").read_text(encoding="utf-8"))
        rows.append({key: value for key, value in result.items() if key != "validation_curve"})
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Final validation accuracy: {result['final_val_accuracy']:.4%}", flush=True)
        return result

    for dataset in DATASETS:
        if args.extended:
            candidates = [trial(dataset, 0.01, 0.3), trial(dataset, 0.01, 0.1, 0.15)]
            candidates.extend(trial(dataset, 0.01, 0.3, vl) for vl in (0.05, 0.15, 0.3))
            best = max(candidates, key=lambda row: row["final_val_accuracy"])
            candidates.append(trial(dataset, 0.01, 0.6, best["virt_local"]))
            best = max(candidates, key=lambda row: row["final_val_accuracy"])
            candidates.extend(trial(dataset, 0.01, best["lam"], best["virt_local"], tau)
                              for tau in (0.05, 0.2))
        else:
            candidates = [trial(dataset, lr, 0.1) for lr in (0.001, 0.005, 0.01)]
            best_lr = max(candidates, key=lambda row: row["final_val_accuracy"])["lr"]
            candidates.extend(trial(dataset, best_lr, lam) for lam in (0.03, 0.3))
        winner = max(candidates, key=lambda row: row["final_val_accuracy"])
        winners[dataset] = {key: value for key, value in winner.items() if key != "validation_curve"}
        write_json(output / "best_configs.json", winners)
    print(f"Finished. Validation-selected configurations: {output / 'best_configs.json'}", flush=True)


if __name__ == "__main__":
    main()

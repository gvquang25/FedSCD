# Tune FedSCD: four datasets, alpha 0.5

## Extended search based on the supplied historical run

`python system/tune_fedscd.py --extended --seed 42 --split-seed 2006`
runs 32 trials (eight per dataset), retaining the original validation splits.
LR stays at 0.01. Each dataset first runs lambda 0.3/local virtual 0 as a
same-seed control, then lambda 0.1/local virtual 0.15 inspired by the screenshot.
It next tries local virtual weights 0.05, 0.15 and 0.3 at lambda 0.3;
then lambda 0.6 at the winning local virtual weight; finally temperatures 0.05
and 0.2 at the best lambda/local virtual pair. Temperature is otherwise 0.1.
Selection remains final validation accuracy. The screenshot does not show all
historical parameters or code, so this is not an exact reproduction of that run.
Seed 42 is the training seed; split seed 2006 preserves the previous holdout.

Run from the repository root with a working Python environment containing the
project dependencies and CUDA-enabled PyTorch:

```powershell
.\.venv\Scripts\python.exe system\tune_fedscd.py --check-only
.\.venv\Scripts\python.exe -u system\tune_fedscd.py
```

The runner performs 20 sequential trials, five for each of Cifar10, Cifar100,
Flowers102 and TinyImageNet. It uses the existing 20-client partitions at
`False_True_0.5`, CNN, full participation, batch size 32, one local epoch,
seed 2006, and `-gr 100`. The current server iterates from 0 through 100
inclusive, so that setting actually performs 101 training rounds per trial.

For each dataset it tries LR 0.001, 0.005 and 0.01 with lambda 0.1, then tries
lambda 0.03 and 0.3 at that dataset's winning LR. All other settings are fixed.
This is a limited sequential search, not a claim of globally optimal parameters.

Validation is drawn only from each client's training data, stratified by class,
with a requested fraction of 20%. Rounding and retaining singleton classes in
training mean the actual fraction varies. Splits are fixed across all candidates
and recorded in `splits.json`. Clients without validation samples or a full
training batch cause preflight to fail. Original data files are never rewritten.

Selection uses sample-weighted **final-round validation accuracy**, not the
maximum over rounds. Legacy console/HDF5 fields containing `test` refer to
validation inside these trials; the worker prints this explicitly. Original test
files are never read. This behavior is scoped to the tuning worker process.

Each invocation creates a fresh timestamped directory under
`logs/tune_fedscd/`, with command specifications, logs, validation curves,
`summary.csv`, `best_configs.json`, split indices and a manifest. HDF5 files use
unique `validation_...` names under `results/` so benchmark results are preserved.
WandB and t-SNE are disabled. The runner stops on a failed trial and retains
completed results; it does not resume a partial batch automatically.

Use `--rounds 1 --output logs/tune_smoke` for a short integration check (20
trials of two training rounds). Do not use its winners as final results.
`--output` must name a new directory. `--device cpu` is available explicitly;
the default fails if CUDA is unavailable instead of silently training on CPU.

After selection, retrain each chosen configuration on the full original training
data through `main.py`, evaluate on the untouched test data and repeat across
multiple seeds for reported benchmark results. This runner selects parameters;
it does not implement model checkpointing or final test evaluation.

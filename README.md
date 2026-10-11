# E3aj MAPPO

This repository provides the E3aj MAPPO algorithm for CDOBench F1-F15 and WSN F1-F5. It includes one pretrained checkpoint and the code/data required to run the algorithm.

## Setup

Use Linux/WSL with Python 3.11:

```bash
conda env create -f environment.yml
conda activate meta_
```

Alternatively, install `requirements.txt` into an existing Python 3.11 environment. The bundled checkpoint requires NumPy 2.x.

## Run

```bash
python release.py cdo
python release.py wsn
```

These commands run the packaged algorithm with the included checkpoint. To train from scratch using the packaged settings, run `python release.py train`. Training uses `epoch_start=0` and `epoch_end=6` (end exclusive), so it runs epochs 0–5. The scheduling axis uses `schedule_start_epoch=0` and `schedule_end_epoch=23` (end exclusive), so its active range is 0–22. Full training and evaluation use the configured function budgets and can take a long time. Add `--dry-run` to inspect a command without starting it. Extra command-line options are forwarded to the underlying entrypoint. `release.py` is the supported public entrypoint.

By default, each evaluation runs 30 evaluation seeds (42–71) per function with up to five workers. Use `--fun_ids`, `--repeat_times`, `--seed0`, and `--batch_size` to select a smaller run. For example:

```bash
python release.py cdo --fun_ids 1 --repeat_times 1 --batch_size 1 --cdo_bench_max_fes_list 1000
python release.py wsn --fun_ids 1 --repeat_times 1 --batch_size 1 --wsn_max_fes_list 1000
```

The per-function budget list takes precedence over `--max_fes`. To change a packaged CDO or WSN function budget, set `--cdo_bench_max_fes_list` or `--wsn_max_fes_list`, respectively. A short budget can still end above the requested FEs because initialization and one optimizer step have a minimum evaluation granularity.

To evaluate a checkpoint produced by a new training run, pass its actual path explicitly:

```bash
python release.py cdo --model_path outputs/train/ppo_model/RUN_NAME/mappo-epoch-5.pt
python release.py wsn --model_path outputs/train/ppo_model/RUN_NAME/mappo-epoch-5.pt
```

Replace `RUN_NAME` with the generated directory name from your training run. The default checkpoint stays fixed even when new training runs exist. Training writes checkpoints under `outputs/train/ppo_model/<generated_run_name>/`. Evaluation writes `trajectory.csv` and `result.json` under `outputs/<cdo-or-wsn>/eval/<generated_run_name>/f<function_id>/seed<seed>/`, with a per-function `summary.json`. See `outputs/README.md` for fields and cleanup. The benchmark layout and local optimizers are described in `benchmark/README.md` and `optimizers/README.md`.

The selected checkpoint is `checkpoints/mappo-epoch-5.pt`; its SHA256 is `6456de169bbe3c2f31214019108dcb90f47fd55619e764fd91bd973895754c91`. Algorithm defaults are summarized in `configs/algorithm.json`; the selected training settings are listed in `configs/training_e3aj.json`. These JSON files are reference documents, not parser input files; editing them does not change executable defaults. `release.py` and `options.py` supply the executable defaults.

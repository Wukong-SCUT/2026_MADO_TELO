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

These commands run the packaged algorithm with the included checkpoint. Runtime files are written under `outputs/`. To train from scratch using the packaged training settings, run `python release.py train`. This trains epochs 0–5 using the original 0–23 scheduling horizon. Full training and evaluation use the configured function budgets and can take a long time. Add `--dry-run` to inspect a command without starting it. Extra command-line options are forwarded to the underlying entrypoint. `release.py` is the supported public entrypoint. A newly trained checkpoint can be evaluated with `--model_path outputs/train/ppo_model/<run_name>/mappo-epoch-5.pt` after checking the actual checkpoint location.

Each evaluation runs 30 seeds (42–71) per function with up to five workers. To run a small check, use `python release.py cdo --fun_ids 1 --repeat_times 1 --batch_size 1 --cdo_bench_max_fes_list 1000`. Results and per-step trajectories are saved under `outputs/cdo/eval/` or `outputs/wsn/eval/`.

The selected checkpoint is `checkpoints/mappo-epoch-5.pt`; its SHA256 is `6456de169bbe3c2f31214019108dcb90f47fd55619e764fd91bd973895754c91`. Algorithm defaults are summarized in `configs/algorithm.json`.

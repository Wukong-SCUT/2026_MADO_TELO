# Release Manifest

## Identity

- Release: `C8c Compact10`
- Date: `2026-08-01`
- Source project: `MAPPO_based_Meta`
- Source baseline commit: `d71ad97`
- Supported benchmarks: `CDOBenchF1F15`, `WSNLocation`

## Checkpoints

| File | Role | SHA256 |
|---|---|---|
| `checkpoints/mappo-epoch-20.pt` | CDOBench F1-F15 | `311540165A02EBEC2C1DCB3286A2F995CD3B7D8C1C6A22B5C697FCEE30EAE270` |
| `checkpoints/mappo-epoch-24.pt` | WSN F1-F5 | `533FB6F6F51BAE1C622D53CBCF32795811489E4A45F0A9BFF62B0684F80B0A9E` |

Reproduction settings are stored in `checkpoints/training_config.json`. It contains no machine-specific paths.

## Included

- MAPPO training, two-part evaluation and release summary entry points.
- C8c actor/critic, trainer, buffer and runner.
- Compact10 state communication and graph consensus.
- MMES, VKD-CMA, CMA-ES and Sep-CMA-ES implementations under `optimizers/`.
- Two self-contained benchmark packages under `benchmarks/`.
- Paper objective references and 3e6-r5 time/communication references.
- Two representative pretrained checkpoints.

## Excluded

- A/B/C historical jobs, logs, notes and experiment outputs.
- CEC, DBO, CDOCompetition, Mixed and legacy actor public interfaces.
- 40-agent and 81-agent CDO data.
- Shell wrappers, job-runner integration and email notification tools.
- Runtime outputs and Python caches.

All generated files are written under `outputs/`, which is ignored by Git.

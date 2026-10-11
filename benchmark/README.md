# Benchmark interfaces

The supported benchmark entries are `CDOBenchF1F15` (function IDs 1–15) and `WSNLocation` (function IDs 1–5). Use `python release.py cdo` or `python release.py wsn` from the repository root. The corresponding environment modes are `cdo_objective` and `wsn_objective`.

`benchmark/cdo_bench_f1f15/benchmark.py` provides the CDO entry. F1–F14 use `benchmark/cdo_bench_f1f14/`; F15 uses the packaged synthetic WSN objective in `benchmark/_cdo_f15_shared/`. The CDO data files under `benchmark/cdo_bench_f1f14/data/` are the 20-agent files used by this entry.

`benchmark/wsn_localization/benchmark.py` provides the WSN entry. Its bundled data under `benchmark/wsn_localization/data/` support the 16-agent evaluation configuration. The benchmark objective and local objective are used by the environment through `env/optimizer/env_factory.py`.

The `--fun_ids` option selects function IDs for evaluation. The packaged entry points set the supported benchmark, agent count and data paths automatically.

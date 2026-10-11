# Runtime outputs

`release.py` writes generated files beneath `outputs/`. Only this README is tracked; generated run directories are ignored by Git. Training and evaluation create separate `outputs/train/`, `outputs/cdo/`, and `outputs/wsn/` trees. Each run has a generated name containing a timestamp.

For evaluation, the run directory is `outputs/<cdo-or-wsn>/eval/<generated_run_name>/`. Each `f<function_id>/seed<seed>/` contains:

- `trajectory.csv`: one row per completed optimization step, with `step`, cumulative `fes`, and reported `best_fitness`.
- `result.json`: benchmark name, function ID, seed, model path, final FEs and final reported best fitness.

Each `f<function_id>/summary.json` contains the run count, mean reported best fitness, and the individual run records. Training checkpoints are written beneath `outputs/train/ppo_model/<generated_run_name>/` when saving is enabled.

Remove a generated run directory when it is no longer needed. Keep any run data you want to retain outside this ignored runtime tree before cleaning it.

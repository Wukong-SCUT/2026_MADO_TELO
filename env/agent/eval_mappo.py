"""Evaluate the packaged MAPPO policy on CDOBench or WSN functions."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from options import get_options, resolve_mappo_obs_dim
from env.agent.mappo import MAPPOPolicy
from env.agent.mappo.checkpoint import torch_load_checkpoint
from env.agent.utils.utils import set_random_seed
from env.optimizer.env_factory import make_opt_env


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "checkpoints" / "mappo-epoch-5.pt"
CRITIC_SIGNATURE_FIELDS = {
    "n_agents", "critic_hidden_dim", "critic_agent_emb_dim", "critic_action_emb_dim"
}


def _parse_ids(value):
    return [int(part) for part in str(value).split(",") if part.strip()]


def _load_policy(opts, model_path, allow_agent_mismatch):
    obs_dim = resolve_mappo_obs_dim(opts)
    opts.feature_num_1 = obs_dim
    policy = MAPPOPolicy(
        opts, obs_dim=obs_dim, n_agents=int(opts.fixed_agent_num),
        action_dim=len(opts.optimizer_profile_candidates),
    )
    checkpoint = torch_load_checkpoint(str(model_path), map_location=policy.device)
    saved = checkpoint.get("policy_signature")
    current = policy.policy_signature
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint lacks a policy signature")
    if allow_agent_mismatch:
        saved = {k: v for k, v in saved.items() if k not in CRITIC_SIGNATURE_FIELDS}
        current = {k: v for k, v in current.items() if k not in CRITIC_SIGNATURE_FIELDS}
    if saved != current:
        raise ValueError(f"Checkpoint actor signature mismatch: saved={saved}, current={current}")
    policy.actor.load_state_dict(checkpoint["actor"])
    policy.actor.eval()
    return policy


def _budget(opts, fun_id):
    name = "wsn_max_fes_list" if opts.benchmark_name == "WSNLocation" else "cdo_bench_max_fes_list"
    values = getattr(opts, name, [])
    if isinstance(values, str):
        values = _parse_ids(values)
    return int(values[fun_id - 1]) if fun_id <= len(values) else int(opts.max_fes)


def evaluate_one(opts, model_path, fun_id, seed, allow_agent_mismatch, output_dir):
    opts.seed = int(seed)
    opts.max_fes = _budget(opts, fun_id)
    opts.fixed_agent_num = 16 if opts.benchmark_name == "WSNLocation" else 20
    opts.episode_steps = max(1, opts.max_fes // max(1, opts.fixed_agent_num * opts.fixed_subfes_per_agent))
    opts.no_cuda = True
    opts.use_cuda = 0
    set_random_seed(seed)
    policy = _load_policy(opts, model_path, allow_agent_mismatch)
    environment = make_opt_env(fun_id, opts_in=opts)
    state = environment.reset()
    rows = []
    while True:
        if state.shape[-1] != policy.obs_dim:
            raise ValueError(f"Observation width {state.shape[-1]} differs from policy width {policy.obs_dim}")
        observations = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        observations = torch.nan_to_num(observations)
        with torch.no_grad():
            actions, _, _ = policy.act(observations, deterministic=True)
        state, _, done, info = environment.step(actions.cpu().numpy()[0])
        rows.append({
            "step": len(rows),
            "fes": int(info.get("sumFEs", environment.sum_fes)),
            "best_fitness": float(getattr(environment, "report_best_f", info.get("gbest_f", np.nan))),
        })
        del environment.current_eval_fitness_record[:]
        del environment.current_eval_individual_record[:]
        if done:
            break
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "fes", "best_fitness"))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "benchmark": opts.benchmark_name,
        "function_id": fun_id,
        "seed": seed,
        "model": str(model_path),
        "final_fes": rows[-1]["fes"],
        "best_fitness": rows[-1]["best_fitness"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fun_ids", type=_parse_ids)
    parser.add_argument("--fun_id", type=int)
    parser.add_argument("--repeat_times", type=int, default=30)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--strategies", default="mappo_deterministic")
    parser.add_argument("--actor_only_eval_allow_agent_mismatch", action="store_true")
    parser.add_argument("--objective_split_auto_agent_num", action="store_true")
    args, other = parser.parse_known_args()
    if args.strategies != "mappo_deterministic":
        raise ValueError("This package exposes only mappo_deterministic")
    if args.repeat_times < 1 or args.batch_size < 1:
        raise ValueError("repeat_times and batch_size must be positive")
    opts = get_options(other)
    if opts.benchmark_name not in {"CDOBenchF1F15", "WSNLocation"}:
        raise ValueError("Supported benchmarks: CDOBenchF1F15 and WSNLocation")
    if (opts.benchmark_name == "CDOBenchF1F15" and opts.mappo_env_mode != "cdo_objective") or (
        opts.benchmark_name == "WSNLocation" and opts.mappo_env_mode != "wsn_objective"
    ):
        raise ValueError("Benchmark and environment mode must match")
    fun_ids = args.fun_ids or ([args.fun_id] if args.fun_id else [])
    max_id = 15 if opts.benchmark_name == "CDOBenchF1F15" else 5
    if not fun_ids or any(fun_id < 1 or fun_id > max_id for fun_id in fun_ids):
        raise ValueError(f"Function ids must be in 1..{max_id}")
    output_root = Path(opts.test_dir)
    for fun_id in fun_ids:
        jobs = [
            (
                opts, args.model_path, fun_id, args.seed0 + repeat,
                args.actor_only_eval_allow_agent_mismatch,
                output_root / f"f{fun_id}" / f"seed{args.seed0 + repeat}",
            )
            for repeat in range(args.repeat_times)
        ]
        if args.batch_size == 1:
            results = [evaluate_one(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.batch_size) as pool:
                futures = [pool.submit(evaluate_one, *job) for job in jobs]
                results = [future.result() for future in futures]
        with (output_root / f"f{fun_id}" / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump({"function_id": fun_id, "repeat_times": len(results),
                       "mean_best_fitness": float(np.mean([r["best_fitness"] for r in results])),
                       "runs": results}, stream, indent=2)


if __name__ == "__main__":
    main()

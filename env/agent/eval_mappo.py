import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

"""
MAPPO evaluation in original 2026_LSGOplatform style:
1) repeated tests with seed = seed0 + i
2) repeats run in parallel processes
3) use vector_env interaction
4) save outputs in original style files:
   - actions.csv
   - evaluation_curves.png
   - evaluation_curves_best_so_far.pdf
   - evaluation_curves_best_so_far.png
   - result_record.txt
   - running_data.h5 (extra, project native)
"""

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
import time
import uuid
import gc
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from options import (
    C8C_RELEASE_LATEST_TRAINING,
    RELEASE_ROOT,
    get_options,
    resolve_mappo_obs_dim,
)
from options import build_options_snapshot
from env.agent.utils.utils import set_random_seed
from env.agent.eval_utils import (
    plot_evaluation_curve,
    plot_evaluation_curve_best_so_far,
    running_data_record,
    run_parallel_task,
)
from env.optimizer.env_factory import make_opt_env
from env.agent.mappo import MAPPOPolicy


def _downsample_runs_for_plot(runs, max_points=2000):
    """
    Downsample only for plotting stability/performance.
    Keep raw runs intact for the result table and running-data record.
    """
    out = []
    for r in runs:
        n = len(r)
        if n <= max_points:
            out.append(r)
            continue
        idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
        out.append([r[i] for i in idx])
    return out


def _plot_runs_with_mean(
    runs,
    out_path,
    title="Evaluation Curves (Runs + Mean)",
    log_scale=True,
    maxfes: Optional[float] = None,
):
    """
    Plot all repeated runs as faint lines and the mean curve as a bold line.
    """
    if not runs:
        return
    min_len = min(len(r) for r in runs if len(r) > 0)
    if min_len <= 0:
        return

    arr = np.array([np.array(r[:min_len], dtype=np.float64) for r in runs], dtype=np.float64)
    mean_curve = np.mean(arr, axis=0)
    if maxfes is not None and float(maxfes) > 0:
        x = np.linspace(0, float(maxfes), min_len)
    else:
        x = np.arange(min_len)

    plt.figure(figsize=(9, 6))
    for i in range(arr.shape[0]):
        plt.plot(x, arr[i], color="#1f77b4", alpha=0.2, linewidth=1.0, label="single run" if i == 0 else None)
    plt.plot(x, mean_curve, color="#d62728", alpha=0.95, linewidth=2.4, label="mean")

    if log_scale:
        plt.yscale("log")
    plt.xlabel("FEs")
    plt.ylabel("Objective Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def _chunk_dir_for_eval(opts) -> str:
    p = Path(str(getattr(opts, "test_dir"))).expanduser().resolve() / "_stream_chunks"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _coerce_budget_list(raw) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = []
        for x in raw:
            try:
                out.append(int(float(x)))
            except Exception:
                continue
        return out
    out = []
    for x in str(raw).split(","):
        s = str(x).strip()
        if not s:
            continue
        try:
            out.append(int(float(s)))
        except Exception:
            continue
    return out


def _resolve_eval_max_fes_for_fun(opts, fun_id: int) -> int:
    benchmark = str(getattr(opts, "benchmark_name", ""))
    if benchmark == "WSNLocation":
        vals = _coerce_budget_list(getattr(opts, "wsn_max_fes_list", []))
    elif benchmark == "WSNLocationMASOIE":
        vals = _coerce_budget_list(getattr(opts, "masoie_wsn_max_fes_list", []))
    elif benchmark == "DBOF1F10":
        vals = _coerce_budget_list(getattr(opts, "dbof1f10_max_fes_list", []))
    elif benchmark == "CDOCompetition":
        vals = _coerce_budget_list(getattr(opts, "cdo_max_fes_list", []))
    elif benchmark in {"CDOBenchF1F14", "CDOBenchF1F15"}:
        vals = _coerce_budget_list(getattr(opts, "cdo_bench_max_fes_list", []))
    else:
        vals = []
    idx = int(fun_id) - 1
    if 0 <= idx < len(vals) and int(vals[idx]) > 0:
        return int(vals[idx])
    return int(getattr(opts, "max_fes", 0))


def _build_record_fes_list(max_fes: int) -> List[int]:
    max_fes = int(max(1, max_fes))
    anchors = [int(1.2e5), int(2e5), int(1e6), int(2e6), int(3e6)]
    out = [x for x in anchors if x < max_fes]
    if max_fes not in out:
        out.append(max_fes)
    return sorted(set(int(x) for x in out if int(x) > 0))


def _benchmark_info_for_fun(opts, fun_id: int) -> Dict:
    benchmark = str(getattr(opts, "benchmark_name", ""))
    if benchmark == "WSNLocation":
        from benchmarks.wsn_f1f5 import Benchmark
    elif benchmark == "WSNLocationMASOIE":
        raise ValueError("WSNLocationMASOIE is not included in the C8c release.")
    elif benchmark == "DBOF1F10":
        raise ValueError("DBOF1F10 is not included in the C8c release.")
    elif benchmark == "CDOCompetition":
        raise ValueError("CDOCompetition is not included in the C8c release.")
    elif benchmark == "CDOBenchF1F14":
        raise ValueError("CDOBenchF1F14 is internal to CDOBenchF1F15 in this release.")
    elif benchmark == "CDOBenchF1F15":
        from benchmarks.cdo_f1f15 import Benchmark
    else:
        return {}
    return dict(Benchmark(opts).get_info(int(fun_id)))


def _maybe_auto_align_objective_split_agent_num(opts, fun_id: int):
    if not bool(getattr(opts, "objective_split_auto_agent_num", False)):
        return
    info = _benchmark_info_for_fun(opts, int(fun_id))
    node_num = int(info.get("node_num", 0))
    if node_num <= 0:
        raise ValueError(
            f"--objective_split_auto_agent_num requires benchmark info node_num, "
            f"got benchmark={getattr(opts, 'benchmark_name', '')}, fun_id={fun_id}, info={info}"
        )
    old_agents = int(getattr(opts, "fixed_agent_num", node_num))
    opts.fixed_agent_num = int(node_num)
    opts.episode_steps = int(getattr(opts, "max_fes", 0)) // max(
        1,
        int(opts.fixed_agent_num) * int(getattr(opts, "fixed_subfes_per_agent", 1)),
    )
    if old_agents != int(node_num):
        print(
            "[ObjectiveSplit AutoAgent] "
            f"fun_id={int(fun_id)} benchmark={getattr(opts, 'benchmark_name', '')}: "
            f"fixed_agent_num {old_agents} -> {int(node_num)}"
        )


def _flush_fitness_chunk(chunk_dir: str, run_tag: str, chunk_idx: int, buf: list) -> Optional[str]:
    if not buf:
        return None
    arr = np.asarray(buf, dtype=np.float64)
    fp = os.path.join(chunk_dir, f"{run_tag}.fitness.{chunk_idx:05d}.npy")
    np.save(fp, arr, allow_pickle=False)
    return fp


def _flush_actions_chunk(chunk_dir: str, run_tag: str, chunk_idx: int, buf: list) -> Optional[str]:
    if not buf:
        return None
    fp = os.path.join(chunk_dir, f"{run_tag}.actions.{chunk_idx:05d}.jsonl")
    with open(fp, "w", encoding="utf-8") as f:
        for row in buf:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return fp


def _load_streamed_fitness(payload: dict) -> list:
    chunk_files = payload.get("fitness_chunk_files", []) or []
    if len(chunk_files) == 0:
        return list(payload.get("fitness_curve", []))
    out = []
    for fp in chunk_files:
        arr = np.load(fp, allow_pickle=False)
        out.extend(arr.tolist())
        with contextlib.suppress(Exception):
            os.remove(fp)
    return out


def _stream_run_summary(payload: dict, checkpoint_fes: List[int], max_plot_points: int = 2000) -> dict:
    """
    Build lightweight run summary from streamed chunks (or in-memory fallback):
      - best-so-far at target FE checkpoints
      - final raw / final best
      - downsampled raw & best curves for plotting
    Avoid materializing full long curve in memory.
    """
    checkpoint_fes = sorted([int(x) for x in checkpoint_fes if int(x) > 0])
    checkpoints = {int(x): float("nan") for x in checkpoint_fes}

    chunk_files = payload.get("fitness_chunk_files", []) or []
    inmem_curve = payload.get("fitness_curve", []) or []

    # Collect chunk lengths first to build fixed-index downsample grid.
    lengths: List[int] = []
    if len(chunk_files) > 0:
        for fp in chunk_files:
            arr = np.load(fp, allow_pickle=False, mmap_mode="r")
            lengths.append(int(arr.shape[0]))
    else:
        lengths.append(int(len(inmem_curve)))
    total_len = int(sum(lengths))

    if total_len <= 0:
        return {
            "checkpoints": checkpoints,
            "final_raw": float("nan"),
            "final_best": float("nan"),
            "raw_plot": [],
            "best_plot": [],
            "total_len": 0,
            "final_sum_fes": int(payload.get("final_sum_fes", 0) or 0),
        }

    n_plot = int(min(max(1, max_plot_points), total_len))
    plot_idx = np.linspace(0, total_len - 1, num=n_plot, dtype=np.int64)
    raw_plot = np.empty((n_plot,), dtype=np.float64)
    best_plot = np.empty((n_plot,), dtype=np.float64)

    best = float("inf")
    fe_idx = 0
    pptr = 0
    cptr = 0
    final_raw = float("nan")
    cp_keys = checkpoint_fes

    def _consume_values(vals: np.ndarray):
        nonlocal best, fe_idx, pptr, cptr, final_raw
        n = int(vals.shape[0])
        if n <= 0:
            return
        # Process sequentially so checkpoint semantics are exact.
        for i in range(n):
            v = float(vals[i])
            final_raw = v
            if v < best:
                best = v
            fe_idx += 1

            while cptr < len(cp_keys) and fe_idx >= cp_keys[cptr]:
                checkpoints[cp_keys[cptr]] = best
                cptr += 1

            # fill all plot slots mapping to current global index
            cur_global = fe_idx - 1  # 0-based
            while pptr < n_plot and int(plot_idx[pptr]) == cur_global:
                raw_plot[pptr] = v
                best_plot[pptr] = best
                pptr += 1

    if len(chunk_files) > 0:
        for fp in chunk_files:
            arr = np.load(fp, allow_pickle=False)
            _consume_values(np.asarray(arr, dtype=np.float64).reshape(-1))
            with contextlib.suppress(Exception):
                os.remove(fp)
    else:
        _consume_values(np.asarray(inmem_curve, dtype=np.float64).reshape(-1))

    # Backfill any unresolved checkpoints to final best.
    for k in cp_keys:
        if np.isnan(checkpoints[k]):
            checkpoints[k] = best
    # Backfill any remaining plot points (edge cases).
    while pptr < n_plot:
        raw_plot[pptr] = final_raw
        best_plot[pptr] = best
        pptr += 1

    return {
        "checkpoints": checkpoints,
        "final_raw": float(final_raw),
        "final_best": float(best),
        "raw_plot": raw_plot.tolist(),
        "best_plot": best_plot.tolist(),
        "total_len": int(total_len),
        "final_sum_fes": int(payload.get("final_sum_fes", 0) or 0),
    }


def _load_streamed_actions(payload: dict) -> list:
    chunk_files = payload.get("actions_chunk_files", []) or []
    if len(chunk_files) == 0:
        return list(payload.get("actions", []))
    out = []
    for fp in chunk_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        with contextlib.suppress(Exception):
            os.remove(fp)
    return out


def _write_per_run_values_txt(out_path, finals, repeat_times, average_time, strategy_name, fun_id):
    finals = np.array(finals, dtype=np.float64)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"fun_id: {int(fun_id)}\n")
        f.write(f"strategy: {strategy_name}\n")
        f.write(f"repeat_times: {int(repeat_times)}\n")
        f.write(f"avg_time: {float(average_time):.6f}\n")
        f.write(f"final_fitness_mean: {float(np.mean(finals)):.12f}\n")
        f.write(f"final_fitness_std: {float(np.std(finals)):.12f}\n")
        f.write("\nper_run_final_fitness:\n")
        for i, v in enumerate(finals.tolist()):
            f.write(f"run_{i}: {v:.12f}\n")


def _write_result_record_from_checkpoints(
    output_path: str,
    algorithm_name: str,
    run_summaries: List[dict],
    record_FEs_list: List[int],
    average_time: float,
):
    """
    Memory-safe result_record writer using per-run checkpoint summaries.
    """
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, "result_record.txt")
    record_FEs_list = sorted([int(x) for x in record_FEs_list if int(x) > 0])

    HEADER_FMT = "{:<20}{:<20}{:<25}{:<25}{:<25}{:<25}\n"
    DATA_FMT = "{:<20}{:<20}{:<25.6f}{:<25.3e}{:<25.6f}{:<25.3e}\n"
    FINAL_FMT = "{:<15}{:<25}{:<25.6f}{:<25.3e}{:<25.6f}{:<25.3e}\n"
    SEPARATOR = "-" * 140 + "\n"

    # Collect vectors at each checkpoint across runs.
    per_fe_values: Dict[int, List[float]] = {fe: [] for fe in record_FEs_list}
    final_vals: List[float] = []
    max_len_vals: List[int] = []
    final_fes_vals: List[int] = []
    for s in run_summaries:
        cps = s.get("checkpoints", {})
        for fe in record_FEs_list:
            v = float(cps.get(fe, float("nan")))
            if not np.isnan(v):
                per_fe_values[fe].append(v)
        fv = float(s.get("final_best", float("nan")))
        if not np.isnan(fv):
            final_vals.append(fv)
        max_len_vals.append(int(s.get("total_len", 0)))
        final_fes = int(s.get("final_sum_fes", 0) or 0)
        if final_fes > 0:
            final_fes_vals.append(final_fes)
    max_len = int(max(max_len_vals) if len(max_len_vals) > 0 else 0)
    max_final_fes = int(max(final_fes_vals) if len(final_fes_vals) > 0 else 0)

    def _dual_log(fobj, text: str):
        fobj.write(text)
        print(text, end="")

    with open(output_file_path, "w", encoding="utf-8") as f:
        _dual_log(f, SEPARATOR)
        _dual_log(f, HEADER_FMT.format("Algorithm", "Record Point", "Mean Fitness", "Mean Sci", "Std Dev", "Std Sci"))
        _dual_log(f, SEPARATOR)

        _dual_log(f, f"Algorithm: {algorithm_name}\n")
        for fe in record_FEs_list:
            vals = np.asarray(per_fe_values.get(fe, []), dtype=np.float64)
            fe_display = f"{fe:.1e}"
            if vals.size == 0:
                f.write(f"{'':<20}{fe_display:<20}{'N/A':<25}{'N/A':<25}{'N/A':<25}{'N/A':<25}\n")
            else:
                m = float(np.mean(vals))
                s = float(np.std(vals))
                f.write(DATA_FMT.format("", fe_display, m, m, s, s))

        if len(final_vals) > 0:
            arr = np.asarray(final_vals, dtype=np.float64)
            fm = float(np.mean(arr))
            fs = float(np.std(arr))
            final_count = max_final_fes if max_final_fes > 0 else max_len
            scale_str = f"{final_count:.0e}".replace("+0", "").replace("+", "") if final_count > 0 else "0e0"
            final_label = f"Final({scale_str}|{final_count:,})"
            _dual_log(f, FINAL_FMT.format("", final_label, fm, fm, fs, fs))

        _dual_log(f, f"{'':<15}{'Avg Time(s)':<25}{float(average_time):<25.6f}\n")
        _dual_log(f, SEPARATOR)

    print(f"Evaluation result records successfully saved to: {output_file_path}")


def _actor_only_signature_mismatches(ckpt_sig: dict, cur_sig: dict) -> Dict[str, Dict[str, object]]:
    ignore_keys = {
        "n_agents",
        "critic_hidden_dim",
        "critic_agent_emb_dim",
        "critic_action_emb_dim",
    }
    keys = sorted((set(ckpt_sig.keys()) | set(cur_sig.keys())) - ignore_keys)
    mismatches = {}
    for key in keys:
        ckpt_val = ckpt_sig.get(key, None)
        cur_val = cur_sig.get(key, None)
        if ckpt_val != cur_val:
            mismatches[key] = {
                "current": cur_val,
                "checkpoint": ckpt_val,
            }
    return mismatches


def _build_policy(opts, model_path: Optional[str]) -> Optional[MAPPOPolicy]:
    if model_path is None:
        return None
    obs_dim = resolve_mappo_obs_dim(opts)
    opts.feature_num_1 = obs_dim
    n_agents = int(opts.fixed_agent_num)
    action_dim = int(len(getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])))
    policy = MAPPOPolicy(opts, obs_dim=obs_dim, n_agents=n_agents, action_dim=action_dim)
    ckpt = torch.load(model_path, map_location=policy.device)
    ckpt_sig = ckpt.get("policy_signature", None)
    cur_sig = getattr(policy, "policy_signature", None)
    actor_only_allow_agent_mismatch = bool(getattr(opts, "actor_only_eval_allow_agent_mismatch", False))
    if ckpt_sig is None:
        raise RuntimeError(
            "[MAPPO Signature Mismatch] Loaded checkpoint has no policy_signature. "
            "This usually means an old architecture checkpoint. "
            f"\ncurrent_signature={getattr(policy, 'policy_signature_str', str(cur_sig))}\n"
            "checkpoint_signature=None"
        )
    if (not actor_only_allow_agent_mismatch) and ckpt_sig != cur_sig:
        raise RuntimeError(
            "[MAPPO Signature Mismatch] Checkpoint signature does not match current model architecture. "
            f"\ncurrent_signature={getattr(policy, 'policy_signature_str', str(cur_sig))}\n"
            f"checkpoint_signature={ckpt.get('policy_signature_str', str(ckpt_sig))}"
        )
    if actor_only_allow_agent_mismatch:
        mismatches = _actor_only_signature_mismatches(ckpt_sig, cur_sig)
        if mismatches:
            raise RuntimeError(
                "[MAPPO Actor-Only Signature Mismatch] "
                "--actor_only_eval_allow_agent_mismatch only ignores n_agents and critic-only fields. "
                "The actor/action semantics still differ, so this checkpoint is not safe to load. "
                f"\nactor_relevant_mismatches={json.dumps(mismatches, ensure_ascii=False, sort_keys=True)}"
                f"\ncurrent_signature={getattr(policy, 'policy_signature_str', str(cur_sig))}\n"
                f"checkpoint_signature={ckpt.get('policy_signature_str', str(ckpt_sig))}"
            )
        if "actor" not in ckpt:
            raise RuntimeError("[MAPPO Checkpoint Error] Checkpoint has no actor state_dict.")
        ckpt_agents = ckpt_sig.get("n_agents", None)
        cur_agents = cur_sig.get("n_agents", None) if isinstance(cur_sig, dict) else n_agents
        if ckpt_agents != cur_agents:
            print(
                "[MAPPO Actor-Only Eval] loading actor with agent-count mismatch: "
                f"checkpoint_n_agents={ckpt_agents}, current_n_agents={cur_agents}. "
                "Critic is intentionally not loaded and is unused during eval."
            )
        policy.actor.load_state_dict(ckpt["actor"])
    else:
        if "actor" in ckpt:
            policy.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            policy.critic.load_state_dict(ckpt["critic"])
    policy.actor.eval()
    policy.critic.eval()
    return policy


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [str(x).strip() for x in str(raw).split(",") if str(x).strip()]


def _force_eval_test_dir_prefix(opts):
    """
    Keep train artifacts under model_* and eval artifacts under eval_*.
    options.py currently derives directories from run_name (often model_*),
    so we normalize test_dir here for eval stage.
    """
    p = Path(str(getattr(opts, "test_dir"))).expanduser().resolve()
    name = p.name
    if name.startswith("eval_"):
        eval_name = name
    elif name.startswith("model_"):
        eval_name = "eval_" + name[len("model_"):]
    else:
        eval_name = "eval_" + name
    opts.test_dir = str((p.parent / eval_name).resolve())
    return opts


def _resolve_latest_model(model_dir: str) -> str:
    p = Path(model_dir).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {p}")
    cands = [x for x in p.rglob("*.pt") if x.is_file()]
    if len(cands) == 0:
        raise FileNotFoundError(f"No .pt model found under: {p}")
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cands[0])


def _resolve_model_path(
    model_path_arg: Optional[str],
    auto_latest_model: bool,
    model_dir: Optional[str],
    opts,
) -> Optional[str]:
    # priority 1: explicit concrete path
    if model_path_arg and str(model_path_arg).strip().lower() not in {"", "latest", "auto"}:
        return str(Path(model_path_arg).expanduser().resolve())

    # priority 2: explicit latest / auto switch
    use_latest = bool(auto_latest_model) or (str(model_path_arg).strip().lower() in {"latest", "auto"})
    if not use_latest:
        return None

    # default search root: output_modal_dir (root ckpt directory)
    search_dir = model_dir if (model_dir is not None and str(model_dir).strip()) else getattr(opts, "output_modal_dir", None)
    if search_dir is None:
        raise ValueError("Cannot auto-resolve latest model: model_dir is empty and opts.output_modal_dir is unavailable.")
    return _resolve_latest_model(search_dir)


def _resolve_profile_idx_from_name(strategy_name: str, profile_candidates: List[str], prefix: str) -> Optional[int]:
    if not strategy_name.startswith(prefix):
        return None
    name = str(strategy_name[len(prefix):]).strip().lower()
    if len(profile_candidates) == 0:
        return None
    if name not in [str(x).lower() for x in profile_candidates]:
        return None
    return int([str(x).lower() for x in profile_candidates].index(name))


def _extract_model_identity(model_path: Optional[str]) -> Dict[str, Optional[str]]:
    if model_path is None:
        return {
            "model_path": None,
            "model_file": None,
            "model_id": None,
            "model_time_tag": None,
            "model_epoch_tag": None,
        }
    p = Path(model_path).expanduser().resolve()
    model_id = None
    # Prefer directory id like model_xxx_timestamp, fallback to file stem/name.
    for cand in (p.parent.name, p.stem, p.name):
        if str(cand).startswith("model_"):
            model_id = str(cand)
            break
    if model_id is None:
        model_id = p.stem
    ts_match = re.search(r"(\d{8}T\d{6})", model_id)
    time_tag = None
    if ts_match:
        ts = ts_match.group(1)
        # 20260419T015948 -> 0419T0159
        time_tag = f"{ts[4:8]}T{ts[9:13]}"
    epoch_tag = None
    # e.g. mappo-epoch-59.pt, epoch-59.pt, .../epoch-59.pt
    for src in (p.name, p.stem, str(p)):
        m = re.search(r"epoch[-_]?(\d+)", str(src), flags=re.IGNORECASE)
        if m:
            epoch_tag = m.group(1)
            break
    return {
        "model_path": str(p),
        "model_file": p.name,
        "model_id": model_id,
        "model_time_tag": time_tag,
        "model_epoch_tag": epoch_tag,
    }


def _strategy_display_name(strategy_name: str, model_info: Dict[str, Optional[str]]) -> str:
    if strategy_name.startswith("mappo_"):
        tag = model_info.get("model_time_tag") or "unknown"
        ep = model_info.get("model_epoch_tag")
        base = f"mappo-{tag}-{ep}" if ep is not None else f"mappo-{tag}"
        if strategy_name == "mappo_deterministic":
            return base
        # Keep stochastic distinguishable if both are evaluated together.
        return f"{base}-{strategy_name.replace('mappo_', '')}"
    alias = {
        "random_opt_default": "random",
        "random_all": "random-all",
        "random_hierarchical": "random-all",
    }
    return alias.get(strategy_name, strategy_name)


def _normalize_strategy_name(strategy_name: str) -> str:
    s = str(strategy_name).strip().lower()
    legacy_map = {
        "fixed_sigma_0.01": "fixed_profile_conservative",
        "fixed_sigma_0.1": "fixed_profile_balanced",
        "fixed_sigma_1.0": "fixed_profile_aggressive",
        "fixed_vkd_sigma_0.01": "fixed_vkd_profile_conservative",
        "fixed_vkd_sigma_0.1": "fixed_vkd_profile_balanced",
        "fixed_vkd_sigma_1.0": "fixed_vkd_profile_aggressive",
    }
    return legacy_map.get(s, s)


def _parse_mappo_forced_optimizer(strategy_name: str) -> Tuple[Optional[str], bool]:
    """Return (optimizer_name, stochastic) for mappo_force_<opt> strategies."""
    s = str(strategy_name).strip().lower()
    prefix = "mappo_force_"
    if not s.startswith(prefix):
        return None, False

    rest = s[len(prefix):]
    stochastic = False
    for suffix, is_stochastic in (
        ("_deterministic", False),
        ("_stochastic", True),
    ):
        if rest.endswith(suffix):
            rest = rest[: -len(suffix)]
            stochastic = bool(is_stochastic)
            break

    if rest not in {"mmes", "vkd", "cmaes", "sepcmaes"}:
        raise ValueError(
            f"Unsupported forced MAPPO optimizer in strategy '{strategy_name}'. "
            "Supported forms: mappo_force_mmes, mappo_force_vkd, "
            "mappo_force_cmaes, mappo_force_sepcmaes, optionally suffixed "
            "with _deterministic or _stochastic."
        )
    return rest, stochastic


def _write_strategy_meta(
    strategy_out: str,
    run_name: str,
    fun_id: int,
    strategy_name: str,
    model_path: Optional[str],
    opts=None,
):
    model_info = _extract_model_identity(model_path) if strategy_name.startswith("mappo_") else {
        "model_path": None,
        "model_file": None,
        "model_id": None,
        "model_time_tag": None,
    }
    meta = {
        "display_name": _strategy_display_name(strategy_name, model_info),
        "eval_run_name": str(run_name),
        "fun_id": int(fun_id),
        "strategy_name_raw": str(strategy_name),
        "strategy_full_name": f"{run_name}/{strategy_name}",
        "strategy_dir": str(Path(strategy_out).resolve()),
        "is_model_based": bool(strategy_name.startswith("mappo_")),
        "loaded_model_id": model_info["model_id"],
        "loaded_model_file": model_info["model_file"],
        "loaded_model_path": model_info["model_path"],
    }
    if opts is not None:
        meta["objective_split_graph"] = {
            "consensus_mode": str(
                getattr(opts, "objective_split_consensus", "full_mean")
            ),
            "consensus_strength": float(
                getattr(opts, "objective_split_consensus_strength", 1.0)
            ),
            "comm_interval": int(
                getattr(opts, "objective_split_comm_interval", 1)
            ),
            "comm_rounds": int(
                getattr(opts, "objective_split_comm_rounds", 1)
            ),
            "graph_source": str(
                getattr(opts, "objective_split_graph_source", "benchmark_w")
            ),
            "weight_mode": str(
                getattr(opts, "objective_split_weight_mode", "metropolis")
            ),
            "neighbor_obs": bool(
                int(getattr(opts, "objective_split_neighbor_obs", 0))
            ),
            "neighbor_obs_mode": str(
                getattr(opts, "objective_split_neighbor_obs_mode", "none")
            ),
            "state_comm_mode": str(
                getattr(opts, "objective_split_state_comm_mode", "none")
            ),
            "state_comm_include_delta": bool(
                int(getattr(opts, "objective_split_state_comm_include_delta", 1))
            ),
            "state_comm_include_actual_fes": bool(
                int(getattr(opts, "objective_split_state_comm_include_actual_fes", 1))
            ),
            "consensus_reward_weight": float(
                getattr(opts, "objective_split_consensus_reward_weight", 0.0)
            ),
            "ccsa_momentum_decay": float(
                getattr(opts, "objective_split_ccsa_momentum_decay", 0.8)
            ),
            "ccsa_direction_lr": float(
                getattr(opts, "objective_split_ccsa_direction_lr", 0.5)
            ),
            "ccsa_scale_rate": float(
                getattr(opts, "objective_split_ccsa_scale_rate", 0.2)
            ),
            "ccsa_scale_min": float(
                getattr(opts, "objective_split_ccsa_scale_min", 0.5)
            ),
            "ccsa_scale_max": float(
                getattr(opts, "objective_split_ccsa_scale_max", 1.5)
            ),
            "ccsa_positive_improve": bool(
                int(getattr(opts, "objective_split_ccsa_positive_improve", 1))
            ),
            "masoie_velocity_decay": float(
                getattr(opts, "objective_split_masoie_velocity_decay", 0.5)
            ),
            "masoie_velocity_scale": float(
                getattr(opts, "objective_split_masoie_velocity_scale", 1.0)
            ),
            "masoie_velocity_clip_ratio": float(
                getattr(opts, "objective_split_masoie_velocity_clip_ratio", 1.0)
            ),
        }
    with open(os.path.join(strategy_out, "strategy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _eval_once(
    fun_id: int,
    strategy_name: str,
    model_path: Optional[str],
    seed0: int,
    seed_offset: int,
    opts_overrides: Optional[dict],
    repeat_idx: int,
):
    """
    Worker for run_parallel_task.
    Last arg repeat_idx is required by eval_utils.run_parallel_task.
    Returns: (payload_dict, elapsed_seconds)
    payload_dict contains: fitness_curve, actions
    """
    t0 = time.time()

    opts = get_options([])
    if isinstance(opts_overrides, dict):
        for k, v in opts_overrides.items():
            setattr(opts, k, v)
    opts.mappo_action_arch = str(getattr(opts, "mappo_action_arch", "current")).lower()
    if opts.mappo_action_arch == "pre_caf4a62":
        opts.mappo_cfg_param_num = 1
    # repeat seed style: seed = seed0 + i
    opts.seed = int(seed0 + seed_offset + repeat_idx)
    set_random_seed(int(opts.seed))
    # multiprocessing + cuda is fragile; keep evaluation workers on CPU.
    opts.no_cuda = True
    opts.use_cuda = 0
    _maybe_auto_align_objective_split_agent_num(opts, int(fun_id))

    need_policy = strategy_name.startswith("mappo_")
    policy = _build_policy(opts, model_path) if need_policy else None

    opt_names = [str(x).lower() for x in getattr(opts, "optimizer_candidates", ["mmes", "vkd", "cmaes", "sepcmaes"])]
    forced_mappo_optimizer, forced_mappo_stochastic = _parse_mappo_forced_optimizer(strategy_name)
    forced_mappo_opt_idx = None
    if forced_mappo_optimizer is not None:
        if forced_mappo_optimizer not in opt_names:
            raise ValueError(
                f"Strategy '{strategy_name}' requires optimizer '{forced_mappo_optimizer}', "
                f"but optimizer_candidates={opt_names}."
            )
        forced_mappo_opt_idx = int(opt_names.index(forced_mappo_optimizer))
    try:
        opt_idx_mmes = int(opt_names.index("mmes"))
    except ValueError:
        opt_idx_mmes = 0
    try:
        opt_idx_vkd = int(opt_names.index("vkd"))
    except ValueError:
        opt_idx_vkd = 0
    try:
        opt_idx_cmaes = int(opt_names.index("cmaes"))
    except ValueError:
        opt_idx_cmaes = 0
    try:
        opt_idx_sepcmaes = int(opt_names.index("sepcmaes"))
    except ValueError:
        opt_idx_sepcmaes = 0
    profile_candidates = [str(x).lower() for x in getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])]
    cfg_param_num = int(getattr(opts, "mappo_cfg_param_num", 4))
    comm_action_enable = bool(
        int(getattr(opts, "objective_split_comm_action_enable", 0))
    )
    raw_comm_candidates = getattr(opts, "objective_split_comm_round_candidates", [1, 2, 4, 8])
    if isinstance(raw_comm_candidates, str):
        raw_comm_candidates = [x.strip() for x in raw_comm_candidates.split(",") if x.strip()]
    comm_action_dim = (
        int(max(1, len(list(raw_comm_candidates))))
        if comm_action_enable
        else 1
    )
    action_cols = int(2 + cfg_param_num + (1 if comm_action_enable else 0))
    cfg_default = int(profile_candidates.index("balanced")) if "balanced" in profile_candidates else 0
    res_default = int(min(range(len(opts.resource_factors)), key=lambda i: abs(float(opts.resource_factors[i]) - 1.0)))

    fixed_idx_mmes = _resolve_profile_idx_from_name(strategy_name, profile_candidates, "fixed_profile_")
    fixed_idx_vkd = _resolve_profile_idx_from_name(strategy_name, profile_candidates, "fixed_vkd_profile_")
    fixed_idx_cmaes = _resolve_profile_idx_from_name(strategy_name, profile_candidates, "fixed_cmaes_profile_")
    fixed_idx_sepcmaes = _resolve_profile_idx_from_name(strategy_name, profile_candidates, "fixed_sepcmaes_profile_")
    fixed_idx_mmes_named = _resolve_profile_idx_from_name(strategy_name, profile_candidates, "fixed_mmes_profile_")
    if fixed_idx_mmes_named is not None:
        fixed_idx_mmes = fixed_idx_mmes_named

    env = make_opt_env(int(fun_id), opts_in=opts)

    flush_interval = int(float(getattr(opts, "eval_stream_flush_fes_interval", 0.0)))
    save_actions = bool(int(getattr(opts, "eval_save_actions", 1)))
    stream_mode = flush_interval > 0
    fitness_curve = []
    actions_record = []
    fitness_chunk_files: List[str] = []
    actions_chunk_files: List[str] = []
    chunk_idx = 0
    next_flush_fes = flush_interval if stream_mode else 0
    run_tag = f"f{int(fun_id)}_{strategy_name}_seed{int(opts.seed)}_{uuid.uuid4().hex[:8]}"
    chunk_dir = _chunk_dir_for_eval(opts) if stream_mode else ""

    state = env.reset()  # [A,F]
    if policy is not None and int(state.shape[-1]) != int(policy.obs_dim):
        raise ValueError(
            "Environment/policy observation mismatch: "
            f"env={state.shape[-1]}, policy={policy.obs_dim}."
        )
    state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)  # [1,A,F]
    state = torch.where(torch.isnan(state), torch.zeros_like(state), state)

    done = False
    final_sum_fes = 0
    graph_metrics_latest = {}
    graph_metric_series = {
        "pre_state_mean_disagreement": [],
        "pre_state_max_edge_disagreement": [],
        "proposal_mean_disagreement": [],
        "proposal_max_edge_disagreement": [],
        "post_mean_disagreement": [],
        "post_max_edge_disagreement": [],
        "consensus_improvement": [],
        "consensus_operator_improvement": [],
        "direction_agreement_mean": [],
        "neighbor_avg_improve_mean": [],
        "ccsa_momentum_norm_mean": [],
        "ccsa_scale_mean": [],
        "ccsa_scale_std": [],
        "masoie_velocity_norm_mean": [],
        "masoie_neighbor_pull_norm_mean": [],
        "comm_rounds_per_event": [],
        "comm_force_rounds": [],
        "last_comm_rounds_applied": [],
        "last_comm_rounds_requested": [],
        "comm_round_idx_mean": [],
        "comm_round_idx_max": [],
        "state_comm_enabled": [],
        "state_comm_msg_dim": [],
        "last_actual_fes_mean": [],
        "last_actual_fes_max": [],
        "cumulative_actual_fes_mean": [],
        "last_consensus_shift_norm_mean": [],
        "last_consensus_shift_norm_max": [],
        "step_comm_rounds_applied": [],
        "step_comm_events": [],
    }
    while not done:
        if fixed_idx_mmes is not None:
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_mmes)
            actions[..., 1 : 1 + cfg_param_num] = int(fixed_idx_mmes)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif fixed_idx_vkd is not None:
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_vkd)
            actions[..., 1 : 1 + cfg_param_num] = int(fixed_idx_vkd)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif fixed_idx_cmaes is not None:
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_cmaes)
            actions[..., 1 : 1 + cfg_param_num] = int(fixed_idx_cmaes)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif fixed_idx_sepcmaes is not None:
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_sepcmaes)
            actions[..., 1 : 1 + cfg_param_num] = int(fixed_idx_sepcmaes)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "fixed_mmes_default":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_mmes)
            actions[..., 1 : 1 + cfg_param_num] = int(cfg_default)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "fixed_vkd_default":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_vkd)
            actions[..., 1 : 1 + cfg_param_num] = int(cfg_default)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "fixed_cmaes_default":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_cmaes)
            actions[..., 1 : 1 + cfg_param_num] = int(cfg_default)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "fixed_sepcmaes_default":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_sepcmaes)
            actions[..., 1 : 1 + cfg_param_num] = int(cfg_default)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "random_opt_default":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = torch.randint(low=0, high=len(opts.optimizer_candidates), size=actions[..., 0].shape)
            actions[..., 1 : 1 + cfg_param_num] = int(cfg_default)
            actions[..., 1 + cfg_param_num] = int(res_default)
        elif strategy_name == "fixed_mmes_random_cfg_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_mmes)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name == "fixed_vkd_random_cfg_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_vkd)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name == "fixed_cmaes_random_cfg_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_cmaes)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name == "fixed_sepcmaes_random_cfg_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_sepcmaes)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name == "fixed_mmes_random_profile_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_mmes)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name == "fixed_vkd_random_profile_res":
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = int(opt_idx_vkd)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
        elif strategy_name in {"random_hierarchical", "random_all"}:
            actions = torch.zeros((state.shape[0], state.shape[1], action_cols), dtype=torch.int64)
            actions[..., 0] = torch.randint(low=0, high=len(opts.optimizer_candidates), size=actions[..., 0].shape)
            actions[..., 1 : 1 + cfg_param_num] = torch.randint(low=0, high=len(profile_candidates), size=actions[..., 1 : 1 + cfg_param_num].shape)
            actions[..., 1 + cfg_param_num] = torch.randint(low=0, high=len(opts.resource_factors), size=actions[..., 1 + cfg_param_num].shape)
            if comm_action_enable:
                actions[..., 2 + cfg_param_num] = torch.randint(
                    low=0,
                    high=comm_action_dim,
                    size=actions[..., 2 + cfg_param_num].shape,
                )
        elif strategy_name in {"mappo_deterministic", "mappo_stochastic"} or forced_mappo_opt_idx is not None:
            with torch.no_grad():
                deterministic = not (
                    strategy_name == "mappo_stochastic" or bool(forced_mappo_stochastic)
                )
                actions, _, _ = policy.act(
                    state,
                    deterministic=deterministic,
                    forced_opt_idx=forced_mappo_opt_idx,
                )  # [B,A,C]
        else:
            raise ValueError(f"Unsupported evaluation strategy: {strategy_name}")

        action_cpu = actions.detach().cpu().numpy()
        # record actions in rollout style
        if save_actions:
            actions_record.append(action_cpu[0].reshape(-1, action_cols).tolist())

        next_state, rewards, is_end, info = env.step(action_cpu[0].reshape(-1, action_cols))
        final_sum_fes = max(final_sum_fes, int(info.get("sumFEs", 0)))
        for metric_name in (
            "pre_state_mean_disagreement",
            "pre_state_max_edge_disagreement",
            "proposal_mean_disagreement",
            "proposal_max_edge_disagreement",
            "post_mean_disagreement",
            "post_max_edge_disagreement",
            "consensus_improvement",
            "consensus_operator_improvement",
            "direction_agreement_mean",
            "ccsa_momentum_norm_mean",
            "ccsa_scale_mean",
            "ccsa_scale_std",
            "masoie_velocity_norm_mean",
            "masoie_neighbor_pull_norm_mean",
            "local_search_fes",
            "agent_state_local_evals",
            "global_monitor_local_evals",
            "global_monitor_rounds",
            "graph_comm_rounds",
            "comm_rounds_per_event",
            "comm_force_rounds",
            "last_comm_rounds_applied",
            "last_comm_rounds_requested",
            "comm_round_idx_mean",
            "comm_round_idx_max",
            "state_comm_enabled",
            "state_comm_msg_dim",
            "last_actual_fes_mean",
            "last_actual_fes_max",
            "cumulative_actual_fes_mean",
            "last_consensus_shift_norm_mean",
            "last_consensus_shift_norm_max",
            "ccsa_lite_rounds",
            "masoie_lite_rounds",
            "centralized_full_mean_rounds",
            "total_comm_rounds_applied",
            "total_comm_events",
            "step_comm_rounds_applied",
            "step_comm_events",
            "graph_messages",
            "graph_transmitted_floats",
            "graph_transmitted_bytes",
            "graph_directed_edge_count",
            "graph_undirected_edge_count",
            "metric_directed_edge_count",
        ):
            if metric_name in info:
                graph_metrics_latest[metric_name] = float(info[metric_name])
        if "consensus_mode" in info:
            graph_metrics_latest["consensus_mode"] = str(info["consensus_mode"])
        for metric_name in graph_metric_series:
            if metric_name in info:
                graph_metric_series[metric_name].append(float(info[metric_name]))
        rec = info.get("fitness_record", [])
        if len(rec) > 0:
            fitness_curve.extend(np.asarray(rec, dtype=np.float64).tolist())

        if stream_mode:
            sum_fes = int(info.get("sumFEs", 0))
            if sum_fes >= next_flush_fes:
                fp = _flush_fitness_chunk(chunk_dir, run_tag, chunk_idx, fitness_curve)
                if fp is not None:
                    fitness_chunk_files.append(fp)
                if save_actions:
                    ap = _flush_actions_chunk(chunk_dir, run_tag, chunk_idx, actions_record)
                    if ap is not None:
                        actions_chunk_files.append(ap)
                fitness_curve.clear()
                actions_record.clear()
                chunk_idx += 1
                while next_flush_fes <= sum_fes:
                    next_flush_fes += flush_interval

        state = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)
        state = torch.where(torch.isnan(state), torch.zeros_like(state), state)

        if bool(is_end):
            done = True
            break

    elapsed = time.time() - t0
    if stream_mode:
        fp = _flush_fitness_chunk(chunk_dir, run_tag, chunk_idx, fitness_curve)
        if fp is not None:
            fitness_chunk_files.append(fp)
        if save_actions:
            ap = _flush_actions_chunk(chunk_dir, run_tag, chunk_idx, actions_record)
            if ap is not None:
                actions_chunk_files.append(ap)
        fitness_curve.clear()
        actions_record.clear()

    graph_metric_summary = dict(graph_metrics_latest)
    for metric_name, metric_values in graph_metric_series.items():
        if not metric_values:
            continue
        graph_metric_summary[f"mean_{metric_name}"] = float(
            np.mean(metric_values)
        )
        graph_metric_summary[f"final_{metric_name}"] = float(metric_values[-1])

    payload = {
        "fitness_curve": fitness_curve,
        "actions": actions_record,
        "fitness_chunk_files": fitness_chunk_files,
        "actions_chunk_files": actions_chunk_files,
        "elapsed": float(elapsed),
        "final_sum_fes": int(final_sum_fes),
        "graph_metrics": graph_metric_summary,
    }
    return payload, elapsed


def evaluate(
    fun_ids: List[int],
    repeat_times: int,
    seed0: int,
    model_path: Optional[str],
    batch_size: int,
    strategies: Optional[List[str]] = None,
    opts=None,
):
    if opts is None:
        opts = get_options([])
    opts.RL_agent = "mappo"
    batch_size = int(max(1, batch_size))
    run_name = os.path.basename(os.path.normpath(opts.test_dir))

    if strategies is None or len(strategies) == 0:
        strategies = [
            "fixed_profile_conservative",
            "fixed_profile_balanced",
            "fixed_profile_aggressive",
        ]
        if model_path is not None:
            strategies.append("mappo_deterministic")
    strategies = [_normalize_strategy_name(x) for x in strategies]
    if any(s.startswith("mappo_") for s in strategies) and model_path is None:
        raise ValueError("Strategies include mappo_* but --model_path is not provided.")

    opts_overrides = vars(opts).copy()

    batches_per_fun = int(math.ceil(float(repeat_times) / float(batch_size)))
    total_test_calls = int(len(fun_ids) * batches_per_fun * len(strategies))
    pbar = tqdm(total=total_test_calls, disable=bool(opts.no_progress_bar), desc="mappo-eval")

    os.makedirs(opts.test_dir, exist_ok=True)
    model_identity = _extract_model_identity(model_path)
    snap = build_options_snapshot(
        opts,
        extra={
            "stage": "eval_mappo",
            "fun_ids": [int(x) for x in fun_ids],
            "repeat_times": int(repeat_times),
            "batch_size": int(batch_size),
            "model_path": model_path,
            "loaded_model_id": model_identity.get("model_id"),
            "loaded_model_file": model_identity.get("model_file"),
            "actor_only_eval_allow_agent_mismatch": bool(
                getattr(opts, "actor_only_eval_allow_agent_mismatch", False)
            ),
            "objective_split_auto_agent_num": bool(
                getattr(opts, "objective_split_auto_agent_num", False)
            ),
            "loaded_policy_mode": (
                "actor_only_allow_agent_mismatch"
                if bool(getattr(opts, "actor_only_eval_allow_agent_mismatch", False))
                else "strict_full_policy"
            ),
            "run_name": run_name,
            "strategies": [str(x) for x in strategies],
        },
    )
    with open(os.path.join(opts.test_dir, "options_test.json"), "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    for fun_id in fun_ids:
        fun_max_fes = _resolve_eval_max_fes_for_fun(opts, int(fun_id))
        # opts.test_dir already includes run_name (contains timestamp). Avoid duplicating timestamp level.
        root_out = os.path.join(opts.test_dir, f"f{int(fun_id)}")
        os.makedirs(root_out, exist_ok=True)

        for strategy_name in strategies:
            run_summaries: List[dict] = []
            plot_runs_raw: List[List[float]] = []
            plot_runs_best: List[List[float]] = []
            actions_record = []
            elapsed_all = []
            graph_metrics_per_run = []
            save_actions = bool(int(getattr(opts, "eval_save_actions", 1)))
            save_running_data = bool(int(getattr(opts, "eval_save_running_data", 1)))
            record_fes_list = _build_record_fes_list(fun_max_fes)

            for bidx in range(batches_per_fun):
                cur_batch = int(min(batch_size, repeat_times - bidx * batch_size))
                if cur_batch <= 0:
                    continue
                print("\n" + "-" * 78)
                print(
                    f"[EvalBatch] run={run_name} | f={fun_id} | {strategy_name} | "
                    f"batch {bidx + 1}/{batches_per_fun} | size={cur_batch}"
                )
                print("-" * 78)
                seed_offset = int(bidx * batch_size)
                results_record, _ = run_parallel_task(
                    _eval_once,
                    parallel_num=cur_batch,
                    fun_id=int(fun_id),
                    strategy_name=strategy_name,
                    model_path=model_path,
                    seed0=int(seed0),
                    seed_offset=seed_offset,
                    opts_overrides=opts_overrides,
                )
                for x in results_record:
                    rs = _stream_run_summary(
                        x,
                        checkpoint_fes=record_fes_list,
                        max_plot_points=2000,
                    )
                    run_summaries.append(rs)
                    plot_runs_raw.append(rs["raw_plot"])
                    plot_runs_best.append(rs["best_plot"])
                    if save_actions:
                        actions_record.append(_load_streamed_actions(x))
                    elapsed_all.append(float(x.get("elapsed", 0.0)))
                    graph_metrics_per_run.append(dict(x.get("graph_metrics", {})))

                pbar.set_postfix_str(
                    f"f={fun_id} strategy={strategy_name} batch={bidx + 1}/{batches_per_fun}"
                )
                pbar.update(1)

            average_time = float(np.mean(elapsed_all)) if len(elapsed_all) > 0 else 0.0

            strategy_out = os.path.join(root_out, strategy_name)
            os.makedirs(strategy_out, exist_ok=True)
            _write_strategy_meta(
                strategy_out=strategy_out,
                run_name=run_name,
                fun_id=int(fun_id),
                strategy_name=strategy_name,
                model_path=model_path,
                opts=opts,
            )
            if any(graph_metrics_per_run):
                with open(
                    os.path.join(strategy_out, "graph_metrics_per_run.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(graph_metrics_per_run, f, ensure_ascii=False, indent=2)

            # Save in original project style.
            # Use lightweight cached runs for plotting.
            plot_data_raw: Dict[str, list] = {"LCC": plot_runs_raw, "LCC_time": [average_time]}
            plot_data_best: Dict[str, list] = {"LCC": plot_runs_best, "LCC_time": [average_time]}
            # Keep plotting/running-data logs quiet; keep result_record table output visible.
            with contextlib.redirect_stdout(io.StringIO()):
                plot_evaluation_curve(plot_data_raw, strategy_out + os.sep, 12, log_scale=True, show_variance=True)
                plot_evaluation_curve_best_so_far(
                    plot_data_best,
                    strategy_out + os.sep,
                    maxfes=fun_max_fes,
                    log_scale=True,
                    show_variance=True,
                )
                if save_running_data:
                    running_data_record(plot_data_best, strategy_out)
            _write_result_record_from_checkpoints(
                output_path=strategy_out,
                algorithm_name="LCC",
                run_summaries=run_summaries,
                record_FEs_list=record_fes_list,
                average_time=average_time,
            )

            # actions.csv
            if save_actions:
                pd.DataFrame(actions_record).to_csv(os.path.join(strategy_out, "actions.csv"), index=False, header=False)

            # additional figure: all runs (faint) + mean (bold)
            _plot_runs_with_mean(
                plot_data_best["LCC"],
                os.path.join(strategy_out, "evaluation_curves_runs_plus_mean.png"),
                title=f"{strategy_name} (f{int(fun_id)})",
                log_scale=True,
                maxfes=float(fun_max_fes),
            )

            finals = [float(s.get("final_best", float("nan"))) for s in run_summaries]
            finals = [x for x in finals if not np.isnan(x)]
            if finals:
                _write_per_run_values_txt(
                    os.path.join(strategy_out, "per_run_values.txt"),
                    finals=finals,
                    repeat_times=repeat_times,
                    average_time=average_time,
                    strategy_name=strategy_name,
                    fun_id=fun_id,
                )
                print(
                    f"[EvalSummary] run={run_name} | f={int(fun_id)} | {strategy_name} "
                    f"| repeats={int(repeat_times)} | batches={int(batches_per_fun)}\n\n\n\n"
                )
            # release strategy-level large buffers early
            run_summaries.clear()
            plot_runs_raw.clear()
            plot_runs_best.clear()
            actions_record.clear()
            elapsed_all.clear()
            graph_metrics_per_run.clear()
            gc.collect()

        print(f"[EvalDone] run={run_name} | f={int(fun_id)} finished")
    pbar.close()


def _build_release_eval_jobs(release_cli_args=None):
    raw = list(release_cli_args or [])
    release_opts = get_options(raw)
    explicit_dir = any(
        x == "--release_checkpoint_dir" or x.startswith("--release_checkpoint_dir=")
        for x in raw
    )
    source_mode = str(release_opts.release_checkpoint_source).strip().lower()
    checkpoint_source = "explicit"
    checkpoint_dir = Path(release_opts.release_checkpoint_dir).expanduser().resolve()

    if not explicit_dir and source_mode != "bundled":
        pointer_path = Path(C8C_RELEASE_LATEST_TRAINING)
        latest_error = None
        try:
            with pointer_path.open("r", encoding="utf-8") as f:
                pointer = json.load(f)
            candidate = Path(str(pointer["model_dir"])).expanduser()
            if not candidate.is_absolute():
                candidate = Path(RELEASE_ROOT) / candidate
            candidate = candidate.resolve()
            required = [
                candidate / str(release_opts.release_cdo_checkpoint),
                candidate / str(release_opts.release_wsn_checkpoint),
            ]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))
            checkpoint_dir = candidate
            checkpoint_source = "latest completed training"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            latest_error = exc
            if source_mode == "latest":
                raise FileNotFoundError(
                    f"No usable latest training record at {pointer_path}: {exc}"
                ) from exc
            checkpoint_dir = Path(release_opts.release_checkpoint_dir).expanduser().resolve()
            checkpoint_source = f"bundled fallback (latest unavailable: {latest_error})"
    elif source_mode == "bundled":
        checkpoint_source = "bundled"
    workers = max(1, int(release_opts.release_eval_agent_parallel_workers))

    common_overrides = [
        "--objective_split_agent_parallel_workers", str(workers),
        "--forced_optimizer_enable", "0",
        "--forced_optimizer_prob", "0",
        "--forced_optimizer_warmup_ratio", "0",
        "--objective_split_early_stop_mode", "none",
    ]
    cdo_args = raw + common_overrides + [
        "--run_name", "C8c_release_eval_epoch20",
        "--note", "C8c-release-epoch20-f1f15-3e6",
        "--benchmark_name", "CDOBenchF1F15",
        "--mappo_env_mode", "cdo_objective",
        "--fixed_agent_num", "20",
        "--max_fes", "3000000",
        "--cdo_bench_max_fes_list", ",".join(["3000000"] * 15),
    ]
    wsn_args = raw + common_overrides + [
        "--run_name", "C8c_release_eval_epoch24",
        "--note", "C8c-release-epoch24-wsn-3e6",
        "--benchmark_name", "WSNLocation",
        "--mappo_env_mode", "wsn_objective",
        "--fixed_agent_num", "16",
        "--wsn_node_num", "16",
        "--max_fes", "2600000",
        "--wsn_max_fes_list", ",".join(["3000000"] * 5),
    ]

    cdo_opts = _force_eval_test_dir_prefix(get_options(cdo_args))
    cdo_opts.actor_only_eval_allow_agent_mismatch = True
    cdo_opts.objective_split_auto_agent_num = True
    wsn_opts = _force_eval_test_dir_prefix(get_options(wsn_args))
    wsn_opts.actor_only_eval_allow_agent_mismatch = True
    wsn_opts.objective_split_auto_agent_num = False

    return [
        {
            "label": "CDOBenchF1F15 epoch-20",
            "opts": cdo_opts,
            "fun_ids": _parse_int_list(release_opts.release_cdo_fun_ids),
            "model_path": str(checkpoint_dir / release_opts.release_cdo_checkpoint),
            "checkpoint_source": checkpoint_source,
        },
        {
            "label": "WSNLocation epoch-24",
            "opts": wsn_opts,
            "fun_ids": _parse_int_list(release_opts.release_wsn_fun_ids),
            "model_path": str(checkpoint_dir / release_opts.release_wsn_checkpoint),
            "checkpoint_source": checkpoint_source,
        },
    ]


def _run_release_eval_suite(release_cli_args=None):
    jobs = _build_release_eval_jobs(release_cli_args)
    release_opts = get_options(list(release_cli_args or []))
    repeat_times = max(1, int(release_opts.release_eval_repeat_times))
    batch_size = max(1, int(release_opts.release_eval_batch_size))
    seed0 = int(release_opts.release_eval_seed0)
    strategies = [str(release_opts.release_eval_strategy)]

    print("[C8c Release] Running the default two-part evaluation suite.")
    for job in jobs:
        model_path = str(Path(job["model_path"]).expanduser().resolve())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Release checkpoint not found: {model_path}")
        print(
            f"[C8c Release] {job['label']} | source={job['checkpoint_source']} "
            f"| model={model_path}"
        )
        evaluate(
            fun_ids=job["fun_ids"],
            repeat_times=repeat_times,
            seed0=seed0,
            model_path=model_path,
            batch_size=batch_size,
            strategies=strategies,
            opts=job["opts"],
        )


def main():
    _run_release_eval_suite(sys.argv[1:])


if __name__ == "__main__":
    main()

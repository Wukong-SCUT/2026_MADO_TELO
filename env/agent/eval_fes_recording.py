"""Dependency-light helpers for evaluation FEs checkpoint reporting."""

import contextlib
import os
from typing import List, Optional

import numpy as np


def build_step_record_fes(pre_sum_fes: int, post_sum_fes: int, count: int) -> List[int]:
    """Map each fitness record in one environment step to cumulative FEs."""
    count = int(max(0, count))
    if count == 0:
        return []
    pre_sum_fes = int(max(0, pre_sum_fes))
    post_sum_fes = int(max(pre_sum_fes, post_sum_fes))
    delta = int(post_sum_fes - pre_sum_fes)
    if count == 1:
        return [post_sum_fes]
    if delta == count:
        return list(range(pre_sum_fes + 1, post_sum_fes + 1))
    raise ValueError(
        "Cannot assign exact reported FEs to a batched fitness record: "
        f"pre={pre_sum_fes}, post={post_sum_fes}, records={count}."
    )


def build_record_fes_list(
    max_fes: int,
    extra_fes: Optional[List[int]] = None,
) -> List[int]:
    """Merge opt-in checkpoints with legacy anchors and the final budget."""
    max_fes = int(max(1, max_fes))
    anchors = [int(1.2e5), int(2e5), int(1e6), int(2e6), int(3e6)]
    anchors.extend(int(x) for x in (extra_fes or []))
    out = [x for x in anchors if 0 < x < max_fes]
    if max_fes not in out:
        out.append(max_fes)
    return sorted(set(int(x) for x in out if int(x) > 0))


def stream_run_summary(
    payload: dict,
    checkpoint_fes: List[int],
    max_plot_points: int = 2000,
) -> dict:
    """Summarize streamed or in-memory fitness without loading a full curve."""
    checkpoint_fes = sorted([int(x) for x in checkpoint_fes if int(x) > 0])
    checkpoints = {int(x): float("nan") for x in checkpoint_fes}

    chunk_files = payload.get("fitness_chunk_files", []) or []
    inmem_curve = payload.get("fitness_curve", []) or []
    record_fes = [int(x) for x in (payload.get("fitness_fes_curve", []) or [])]

    lengths: List[int] = []
    if len(chunk_files) > 0:
        for fp in chunk_files:
            arr = np.load(fp, allow_pickle=False, mmap_mode="r")
            lengths.append(int(arr.shape[0]))
    else:
        lengths.append(int(len(inmem_curve)))
    total_len = int(sum(lengths))
    if record_fes and len(record_fes) != total_len:
        raise ValueError(
            "fitness_fes_curve length must match the streamed fitness curve: "
            f"fes={len(record_fes)}, fitness={total_len}."
        )
    exact_fes = len(record_fes) == total_len and total_len > 0

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
        for i in range(n):
            global_idx = fe_idx
            v = float(vals[i])
            current_fes = int(record_fes[global_idx]) if exact_fes else fe_idx + 1
            if exact_fes:
                while cptr < len(cp_keys) and cp_keys[cptr] < current_fes:
                    checkpoints[cp_keys[cptr]] = best if np.isfinite(best) else v
                    cptr += 1
            final_raw = v
            if v < best:
                best = v
            fe_idx += 1

            while cptr < len(cp_keys) and current_fes >= cp_keys[cptr]:
                checkpoints[cp_keys[cptr]] = best
                cptr += 1

            cur_global = fe_idx - 1
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

    for k in cp_keys:
        if np.isnan(checkpoints[k]):
            checkpoints[k] = best
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

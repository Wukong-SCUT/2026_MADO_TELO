"""Opt-in diagnostic forensics for the CMA family (observation only).

This module never raises into the caller, never draws randomness, never writes
optimizer state, and never changes control flow or exception conditions.  The
only entries it exposes are called from guarded sites, so disabling it leaves
the algorithm bit-identical.

Records live in bounded ring buffers (last 64 generations, last 16 commits per
active session).  Nothing is written during a normal run: a JSONL window is
flushed only when an evidence trigger fires, i.e.

* the evolution-path length exceeds 10x the expected length,
* sigma grows by more than 100x within one generation (compared in log10),
* a relocation-geometry / non-finite exception is about to be raised.

Each trigger class is flushed at most once per process.  Triggers are evidence
gates, not failure criteria: they never alter the run that produces them.

Diagnostic recomputation is reported separately from the values the algorithm
actually used (``naive_*`` vs ``scaled_*`` / ``*_log10`` fields).
"""

from __future__ import annotations

from collections import deque
import json
import os
import time

import numpy as np


GENERATION_RING = 64
COMMIT_RING = 16
PATH_LENGTH_TRIGGER = 10.0
SIGMA_GROWTH_TRIGGER_LOG10 = 2.0

# Per-process, so a repeated trigger class is recorded exactly once.
_PROCESS_FLUSHED = set()

JUMP_SIGMA_LOG10 = 2.0
JUMP_MAX_BYTES = 2_000_000
# Bounded window budget: how many jump windows one process may write for one
# optimizer (default 1 keeps the historical "first trigger only" contract; a
# larger value keeps the first trigger plus the rolling latest trigger).
JUMP_MAX_WINDOWS = 1
JUMP_MAX_LINES_PER_PROCESS = 8
# Staged capture: an early window and a mid window are selected by CUMULATIVE
# log10 sigma growth (or by a single-generation trigger), so a slow, steady
# amplification that never reaches a one-generation threshold is still captured.
# The pre-failure window has its own reserved budget and cannot be starved by
# the earlier stages.
JUMP_EARLY_MILESTONE_LOG10 = 0.5
JUMP_MID_MILESTONE_LOG10 = 2.0
JUMP_MAX_FAILURE_WINDOWS = 2
JUMP_FAILURE_LINE_RESERVE = 2
# Per-process window count per (pid, optimizer).  Kept under the historical name
# because the project checker resets it between cases.
_PROCESS_JUMP_WRITTEN = {}
_PROCESS_JUMP_FAILURE_WRITTEN = {}
# Total jump-window lines per (pid, optimizer), for the byte/volume cap only.
_PROCESS_JUMP_LINES = {}
# Object-scoped stage state: "cumulative growth" and the early/mid stages must
# persist across the per-event sessions that share one (pid, optimizer, object),
# otherwise weak early triggers spend the whole process budget and the real
# anomaly is never armed.
_OBJECT_JUMP_STATE = {}
# Live windows of this process, newest last, held by STRONG reference (bounded):
# the evaluation side builds its diagnostics after the event's optimizer core may
# already be released, so a weak reference is gone exactly when the failure line is
# needed (measured in the rev3/rev4 VKD runs: the hook ran, the registry was empty
# and nothing was written).  The window payloads are ~0.1 MB each; the cap keeps the
# retention bounded.  A window never keeps its optimizer core alive.
_JUMP_REGISTRY = []
JUMP_REGISTRY_MAX = 16


def register_jump_window(window) -> None:
    """Keep a bounded strong handle on a live window (no-op when unavailable)."""
    try:
        _JUMP_REGISTRY.append(window)
        excess = len(_JUMP_REGISTRY) - int(JUMP_REGISTRY_MAX)
        if excess > 0:
            del _JUMP_REGISTRY[:excess]
    except Exception:
        pass


def flush_registered_windows(reason: str, limit: int = 0) -> int:
    """Write the reserved pre-failure line of the newest window per optimizer.

    Returns the number of lines written.  Called on the failure path only, so a
    crash that happens outside an optimizer core still leaves the newest recorded
    generation on disk.  An attempt that has nothing to write does not stop the
    scan -- that starvation is exactly how the first VKD failure line went missing
    in the rev3 run -- and ``limit`` (0 = unlimited) counts successful writes only.
    A one-line diagnostic is written to stderr so the next run can tell "the window
    was already released" from "the window lives in another process".  Never raises
    and never changes optimizer state.
    """
    written = 0
    live = []
    try:
        live = [window for window in list(_JUMP_REGISTRY) if window is not None]
        seen = set()
        for window in reversed(live):
            key = (int(os.getpid()), str(getattr(window, "optimizer_name", "")))
            if key in seen:
                continue
            seen.add(key)
            try:
                wrote = bool(window.flush_latest(str(reason)))
            except Exception:
                wrote = False
            if wrote:
                written += 1
                if limit and written >= int(limit):
                    break
    except Exception:
        return written
    try:
        import sys

        sys.stderr.write(
            "[jump-registry] pid=%d reason=%s live=%d optimizers=%s "
            "with_latest=%s written=%d\n"
            % (int(os.getpid()), str(reason), len(live),
               ",".join(str(getattr(w, "optimizer_name", "?")) for w in live),
               ",".join(str(bool(getattr(w, "latest_any", None))) for w in live),
               written)
        )
        sys.stderr.flush()
    except Exception:
        pass
    return written


def jump_object_key(optimizer_name, identity=None) -> tuple:
    """Stable object identity for cross-session staging (env_step excluded)."""
    ident = identity if isinstance(identity, dict) else {}
    stable = tuple(
        (key, str(ident.get(key)))
        for key in ("problem_family", "function_id", "agent_id", "run_seed", "optimizer")
    )
    return (int(os.getpid()), str(optimizer_name), stable)
# Write a two-record line as soon as the trigger fires (opt-in): lets a bounded
# run be stopped early.  Default False keeps the historical write-once-at-next.
JUMP_EARLY_WRITE = False

# Fields a window must carry to be counted as "recompute-ready" by the Jobs
# checker and by env/optimizer/check_numeric_forensics_jump_light.py.
CMA_JUMP_REQUIRED_FIELDS = (
    "identity", "collection", "n_individuals", "n_parents",
    "sigma_before", "sigma_exp_arg_raw", "sigma_exp_arg_used", "sigma_after",
    "x_raw", "x_scored", "fitness_used_for_ranking", "ranking_order",
    "parent_weights", "effective_weights_w_o", "d_original", "d_sorted_used",
    "w_sorted_used", "mean_before", "mean_after", "wd", "whitened_wd",
    "path_p_s_before", "path_p_s_after", "path_p_c_before", "path_p_c_after",
    "h_s", "covariance_before", "covariance_updated_pre_eigh",
    "raw_eigenvalues", "eigenvalues_after_floor", "covariance_after_rebuilt",
    "coefficients",
)
VKD_JUMP_REQUIRED_FIELDS = (
    "identity", "collection", "lambda", "dimension", "mu", "cs", "ds",
    "score_positive", "score_negative", "rank_positive_zero_based",
    "rank_negative_zero_based", "alpha_act", "ary_positive", "ary_negative",
    "x_raw_positive", "x_raw_negative", "x_scored_positive", "x_scored_negative",
    "clip_distance_positive", "clip_distance_negative", "clip_coordinates",
    "candidate_coordinates", "ps_before", "ps_after", "sigma_consumer",
    "shape_consumer", "hsig", "sigma_before", "sigma_after", "sigma_log10_growth",
    "xmean_before", "dx_probe_used", "probe_direction_used", "probe_length",
    "probe_mahalanobis_norm", "sigma_used_for_probe", "fitness_all", "ranking_all",
)


def jump_target(options=None) -> dict:
    """Optional target identity filter (function/run-seed/agent) for jump windows.

    The seed filter matches the RUN seed (``run_seed``), not the per-step
    optimizer RNG seed that the forensic context also carries as ``seed``.
    """
    options = options or {}
    out = {}
    for key, field in (
        ("optimizer_numeric_forensics_target_function", "function_id"),
        ("optimizer_numeric_forensics_target_seed", "run_seed"),
        ("optimizer_numeric_forensics_target_agent", "agent_id"),
        ("optimizer_numeric_forensics_target_optimizer", "optimizer"),
    ):
        value = options.get(key, None)
        if value is None or str(value).strip() == "":
            continue
        try:
            out[field] = int(value)
        except Exception:
            out[field] = str(value).strip().lower()
    return out


def jump_max_windows(options=None) -> int:
    options = options or {}
    try:
        return max(1, min(32, int(options.get("optimizer_numeric_forensics_jump_max_windows",
                                             JUMP_MAX_WINDOWS))))
    except Exception:
        return JUMP_MAX_WINDOWS


def jump_early_write(options=None) -> bool:
    """Whether the trigger line is written before the following generation."""
    options = options or {}
    return bool(options.get("optimizer_numeric_forensics_jump_early_write",
                            JUMP_EARLY_WRITE))


def jump_alpha_gate(options=None) -> bool:
    """Whether |alpha|>=1 alone may open a VKD window (default off: too noisy)."""
    options = options or {}
    return bool(options.get("optimizer_numeric_forensics_jump_alpha_gate", False))


def jump_early_milestone(options=None) -> float:
    """Cumulative log10 sigma growth that selects the EARLY window."""
    options = options or {}
    try:
        return float(options.get("optimizer_numeric_forensics_jump_early_milestone_log10",
                                 JUMP_EARLY_MILESTONE_LOG10))
    except Exception:
        return JUMP_EARLY_MILESTONE_LOG10


def jump_mid_milestone(options=None) -> float:
    """Cumulative log10 sigma growth that selects the MID window."""
    options = options or {}
    try:
        return float(options.get("optimizer_numeric_forensics_jump_mid_milestone_log10",
                                 JUMP_MID_MILESTONE_LOG10))
    except Exception:
        return JUMP_MID_MILESTONE_LOG10


def jump_output_dir(options=None, fallback_dir: str = "") -> str:
    """Resolve the jump-window directory: option -> env var -> fallback."""
    options = options or {}
    for key in ("optimizer_numeric_forensics_dir", "numeric_forensics_dir"):
        value = str(options.get(key, "") or "").strip()
        if value:
            return value
    value = str(os.environ.get("MAPPO_NUMERIC_FORENSICS_DIR", "") or "").strip()
    return value or str(fallback_dir or "").strip()


def jump_threshold_log10(options=None) -> float:
    """Sigma-jump gate in log10 (default 2.0 == one generation x100)."""
    options = options or {}
    try:
        return float(
            options.get("optimizer_numeric_forensics_jump_log10", JUMP_SIGMA_LOG10)
        )
    except Exception:
        return JUMP_SIGMA_LOG10


class JumpWindow:
    """Keep only the previous/current/next record around the FIRST obvious jump.

    Diagnostic only: bounded to three records and one write per process per
    optimizer, with a hard byte cap.  It never raises into the caller and never
    changes optimizer state.
    """

    def __init__(self, optimizer_name: str, output_dir: str = "",
                 threshold_log10: float = JUMP_SIGMA_LOG10,
                 max_bytes: int = JUMP_MAX_BYTES, context=None,
                 target=None, max_windows: int = JUMP_MAX_WINDOWS,
                 alpha_gate: bool = False, early_write: bool = JUMP_EARLY_WRITE,
                 early_milestone_log10: float = JUMP_EARLY_MILESTONE_LOG10,
                 mid_milestone_log10: float = JUMP_MID_MILESTONE_LOG10):
        self.optimizer_name = str(optimizer_name)
        self.output_dir = str(output_dir or "").strip()
        self.threshold_log10 = float(threshold_log10)
        self.max_bytes = int(max_bytes)
        self.context = dict(context or {})
        self.target = dict(target or {})
        self.max_windows = int(max(1, int(max_windows)))
        self.alpha_gate = bool(alpha_gate)
        self.early_write = bool(early_write)
        self.early_milestone_log10 = float(early_milestone_log10)
        self.mid_milestone_log10 = float(mid_milestone_log10)
        self.previous = None
        self.latest = None
        # Newest payload seen at all, matching the target filter or not: the failure
        # line is written from this one, because on the failure path the state of the
        # crashing object matters more than the staged budget's target selection.
        self.latest_any = None
        self.pending = None
        self.written = False
        self.reason = ""
        self.notes = []
        self.windows_written = 0
        self.suppressed = 0
        # Load the object-scoped stage state so early/mid staging survives the
        # per-event session boundary; the newest known record is carried too, so a
        # failure in a fresh session can still flush a pre-failure line.
        self.object_key = jump_object_key(self.optimizer_name, self.context)
        state = _OBJECT_JUMP_STATE.get(self.object_key) or {}
        self.cumulative_log10 = float(state.get("cumulative_log10", 0.0))
        self.stages = set(state.get("stages", ()))
        if state.get("latest") is not None:
            self.latest = state["latest"]
        self.last_stage = ""
        register_jump_window(self)

    def _persist_state(self) -> None:
        _OBJECT_JUMP_STATE[self.object_key] = {
            "cumulative_log10": float(self.cumulative_log10),
            "stages": set(self.stages),
            "latest": self.latest,
        }

    def _matches(self, payload: dict) -> bool:
        """Target filter: an unrelated agent/function must not consume the budget."""
        if not self.target:
            return True
        ident = payload.get("identity")
        if not isinstance(ident, dict):
            ident = {}
        for field, want in self.target.items():
            got = ident.get(field, payload.get(field))
            if isinstance(want, int):
                try:
                    if int(got) != int(want):
                        return False
                except Exception:
                    return False
            elif str(got).strip().lower() != str(want):
                return False
        return True

    def _single_step_clears(self, growth_log10) -> bool:
        """True when this generation's own growth clears the one-step threshold."""
        try:
            value = float(growth_log10)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and value >= self.threshold_log10)

    def _next_stage(self, jump: bool, growth_log10=None):
        """Stage selection: early, then mid (cumulative), then jump (single step).

        The JUMP stage is gated on the generation's OWN growth clearing the
        one-step threshold, so a benign eigenvalue-floor generation cannot spend
        the budget, and it is not gated on the cumulative milestones: it arms even
        while the mid milestone is still out of reach.  It may be selected again on
        a later qualifying generation until the per-process window budget is spent
        (the caller enforces that budget).  With the ``jump`` flag false the
        selection is byte-for-byte the historical early-then-mid rule.
        """
        if "early" not in self.stages:
            if jump or self.cumulative_log10 >= self.early_milestone_log10:
                return "early"
            return None
        if "mid" not in self.stages and self.cumulative_log10 >= self.mid_milestone_log10:
            return "mid"
        if jump and self._single_step_clears(growth_log10):
            return "jump"
        return None

    def note(self, key, payload: dict, jump: bool, collection=None,
             growth_log10=None) -> bool:
        """Feed one generation record; True when a window line was written.

        Staged, bounded capture per (process, optimizer): an EARLY window (single
        step trigger or cumulative growth >= early milestone), a MID window
        (cumulative >= mid milestone), a JUMP window (one generation whose own
        growth clears the single-step threshold, repeatable until the window
        budget is spent) and a reserved FAILURE window written by
        :meth:`flush_latest`.  A slow sequence whose per-generation growth never
        reaches the single-step threshold is therefore still captured.  The
        thresholds are evidence gates only; they do not classify anything as
        anomalous, and they never change the run.
        """
        previous = self.previous
        self.previous = (str(key), payload)
        self.latest_any = (str(key), payload)
        if not self._matches(payload):
            self.suppressed += 1
            return False
        self.latest = (str(key), payload)
        step_log10 = None
        if growth_log10 is not None:
            try:
                value = float(growth_log10)
            except Exception:
                value = None
            if value is not None and np.isfinite(value) and value > 0.0:
                self.cumulative_log10 += value
            step_log10 = value
        self._persist_state()
        current = dict(payload)
        if collection is not None:
            current["collection"] = collection
        if self.pending is not None:
            stage = str(self.pending.get("stage") or "early")
            if str(key) != self.pending["key"]:
                self.notes.append("session changed before the next generation")
                self.pending.setdefault("records", []).append(None)
                written = self._write("session_changed", self.pending["records"],
                                      complete=True, stage=stage)
            else:
                self.pending["records"].append(current)
                written = self._write("next_generation", self.pending["records"],
                                      complete=True, stage=stage)
            self.stages.add(stage)
            self.pending = None
            self._persist_state()
            return bool(written)
        stage = self._next_stage(bool(jump), step_log10)
        if stage is None:
            if jump:
                self.suppressed += 1
            return False
        process_key = (int(os.getpid()), self.optimizer_name)
        if _PROCESS_JUMP_WRITTEN.get(process_key, 0) >= self.max_windows:
            self.suppressed += 1
            return False
        _PROCESS_JUMP_WRITTEN[process_key] = _PROCESS_JUMP_WRITTEN.get(process_key, 0) + 1
        records = [None, current]
        if previous is not None and previous[0] == str(key):
            records[0] = previous[1]
        else:
            self.notes.append("no previous generation in this session")
        self.pending = {"key": str(key), "records": records, "stage": stage}
        if self.early_write:
            self._write("trigger", records, complete=False, stage=stage)
            return True
        return False

    def flush_latest(self, reason: str, records=None, allow_bypass: bool = True) -> bool:
        """Reserved pre-failure line for the newest known state.

        Uses its own per-process budget and line reserve, so earlier stages can
        never exhaust the failure record.  The newest recorded generation is used
        even when it belongs to an object outside the target filter; such a line is
        marked ``target_bypass`` so the filter stays auditable.  The filter exists to
        keep the staged budget for the selected object, not to hide the crash.
        """
        process_key = (int(os.getpid()), self.optimizer_name)
        if _PROCESS_JUMP_FAILURE_WRITTEN.get(process_key, 0) >= JUMP_MAX_FAILURE_WINDOWS:
            self.suppressed += 1
            return False
        if records is None:
            records = list((self.pending or {}).get("records", []))
            if not records:
                newest = self.latest_any if self.latest_any is not None else self.latest
                if newest is None:
                    return False
                records = [None, newest[1]]
        current = records[-1] if records else None
        bypass = bool(isinstance(current, dict) and not self._matches(current))
        if bypass and not allow_bypass:
            return False
        written = self._write(str(reason), list(records), complete=False,
                              stage="failure", reserved=True, bypass=bypass)
        if written:
            _PROCESS_JUMP_FAILURE_WRITTEN[process_key] = (
                _PROCESS_JUMP_FAILURE_WRITTEN.get(process_key, 0) + 1
            )
            self.stages.add("failure")
            self._persist_state()
        return bool(written)

    def _payload(self, reason: str, records, stage=None, bypass: bool = False) -> dict:
        records = list(records or [])
        while len(records) < 3:
            records.append(None)
        return {
            "record_type": "numeric_forensics_jump",
            "optimizer": self.optimizer_name,
            "pid": int(os.getpid()),
            "reason": str(reason),
            "stage": None if stage is None else str(stage),
            "stages_written": sorted(self.stages),
            "cumulative_log10": float(self.cumulative_log10),
            "early_milestone_log10": float(self.early_milestone_log10),
            "mid_milestone_log10": float(self.mid_milestone_log10),
            "threshold_log10": float(self.threshold_log10),
            "alpha_gate": bool(self.alpha_gate),
            "target": dict(self.target),
            "target_bypass": bool(bypass),
            "identity": dict(self.context),
            "windows_written": int(self.windows_written),
            "suppressed_triggers": int(self.suppressed),
            "notes": list(self.notes),
            "previous": records[0],
            "current": records[1],
            "next": records[2],
        }

    def _write(self, reason: str, records, complete: bool, stage=None,
               reserved: bool = False, bypass: bool = False) -> bool:
        process_key = (int(os.getpid()), self.optimizer_name)
        line_cap = JUMP_MAX_LINES_PER_PROCESS + (
            JUMP_FAILURE_LINE_RESERVE if reserved else 0
        )
        if _PROCESS_JUMP_LINES.get(process_key, 0) >= line_cap:
            self.suppressed += 1
            return False
        if not self.output_dir:
            if not self.written:
                print(
                    "[numeric-forensics-jump] no output dir "
                    "(optimizer_numeric_forensics_dir / MAPPO_NUMERIC_FORENSICS_DIR); "
                    "window kept in memory",
                    flush=True,
                )
            self.written = True
            self.reason = str(reason)
            return False
        payload = self._payload(reason, records, stage, bypass)
        payload["complete"] = bool(complete)
        payload["next_pending"] = not bool(complete)
        payload["reserved_failure_line"] = bool(reserved)
        line = ""
        for stage_label, drop in (
            ("", ()),
            ("dropped_matrices", ("covariance_before", "covariance_updated_pre_eigh")),
            ("dropped_vectors", ("x_raw", "x_scored", "covariance_before",
                                 "covariance_updated_pre_eigh")),
            ("dropped_records", "__all__"),
        ):
            candidate = json.loads(json.dumps(_strict_json_safe(payload), default=str))
            for record in (candidate.get("previous"), candidate.get("current"),
                           candidate.get("next")):
                if not isinstance(record, dict):
                    continue
                if drop == "__all__":
                    keep = {
                        key: record[key]
                        for key in ("generation", "generation_neval", "collection")
                        if key in record
                    }
                    record.clear()
                    record.update(keep)
                else:
                    for field in drop:
                        record.pop(field, None)
            if stage_label:
                candidate["truncated"] = stage_label
            line = json.dumps(candidate, ensure_ascii=False, default=str)
            if len(line) <= self.max_bytes:
                break
        path = os.path.join(
            self.output_dir,
            f"numeric_forensics_jump_{self.optimizer_name}_worker_{os.getpid()}.jsonl",
        )
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            return False
        _PROCESS_JUMP_LINES[process_key] = _PROCESS_JUMP_LINES.get(process_key, 0) + 1
        self.windows_written += 1
        self.written = True
        self.reason = str(reason)
        self.last_stage = "" if stage is None else str(stage)
        return True


def _as_flat(value):
    """Return (raw_array, flat_float64_array) without mutating the input."""
    raw = np.asarray(value)
    return raw, np.asarray(raw, dtype=np.float64).reshape(-1)


def scale_probe(value) -> dict:
    """Overflow-safe magnitude diagnostics for a scalar/vector/matrix.

    ``scaled_norm`` / ``normalized_norm`` is the norm of the max-normalized
    vector (bounded by sqrt(size)) and is NOT the Euclidean norm; the Euclidean
    norm is reported as ``true_norm`` (may be ``inf``) with its log10 as
    ``true_norm_log10``.  ``naive_norm`` reproduces the control-path formula
    (``np.linalg.norm``) for evidence only.
    """
    if value is None:
        return None
    raw, flat = _as_flat(value)
    finite = np.isfinite(flat)
    out = {
        "shape": [int(x) for x in raw.shape],
        "size": int(flat.size),
        "nan_count": int(np.count_nonzero(np.isnan(flat))),
        "inf_count": int(np.count_nonzero(np.isinf(flat))),
        "nonfinite_count": int(flat.size - int(np.count_nonzero(finite))),
        "all_finite": bool(flat.size and bool(finite.all())),
    }
    if not finite.any():
        return out
    fin = flat[finite]
    max_abs = float(np.max(np.abs(fin)))
    out["finite_min"] = float(np.min(fin))
    out["finite_max"] = float(np.max(fin))
    out["max_abs"] = max_abs
    out["argmax_abs_index"] = int(np.argmax(np.abs(fin)))
    out["log10_max_abs"] = (
        float(np.log10(max_abs)) if 0.0 < max_abs < np.inf else None
    )
    tiny = np.abs(fin) < np.finfo(np.float64).tiny
    if tiny.any():
        out["argmin_abs_index"] = int(np.argmin(np.abs(fin)))
        out["min_abs"] = float(np.min(np.abs(fin)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        naive = float(np.linalg.norm(fin))
    out["naive_norm"] = naive
    out["naive_norm_is_finite"] = bool(np.isfinite(naive))
    if 0.0 < max_abs < np.inf:
        scaled = float(np.linalg.norm(fin / max_abs))
        out["scaled_norm"] = scaled
        out["normalized_norm"] = scaled
        out["norm_log10"] = (
            float(np.log10(max_abs) + np.log10(scaled)) if scaled > 0.0 else None
        )
        out["true_norm_log10"] = out["norm_log10"]
        out["true_norm"] = (
            float(max_abs * scaled) if out["norm_log10"] is None or out["norm_log10"] < 300.0
            else float("inf")
        )
    elif max_abs == 0.0:
        out["scaled_norm"] = 0.0
        out["normalized_norm"] = 0.0
        out["norm_log10"] = None
        out["true_norm_log10"] = None
        out["true_norm"] = 0.0
    else:
        out["scaled_norm"] = None
        out["normalized_norm"] = None
        out["norm_log10"] = None
        out["true_norm_log10"] = None
        out["true_norm"] = None
    return out


def direction_probe(value, include_values: bool = True) -> dict:
    """Magnitude probe plus a normalized direction and the dominant axis."""
    out = scale_probe(value)
    if out is None:
        return None
    if not out.get("all_finite", False) or not out["size"]:
        return out
    _, flat = _as_flat(value)
    max_abs = float(np.max(np.abs(flat)))
    if include_values:
        out["values"] = [float(x) for x in flat]
    if max_abs > 0.0 and max_abs < np.inf:
        out["unit"] = [float(x) for x in (flat / max_abs) / (np.linalg.norm(flat / max_abs))]
    return out


def path_decomposition(path_before, path_after, persist_factor: float) -> dict:
    """Split a path update into its persisted-history and fresh contributions.

    ``path_after == persist_factor * path_before + fresh`` holds by
    construction, so the fresh part is recovered by subtraction and no
    control-path expression is recomputed.
    """
    out = {"persist_factor": float(persist_factor)}
    before = None if path_before is None else np.asarray(
        path_before, dtype=np.float64
    ).reshape(-1)
    after = None if path_after is None else np.asarray(
        path_after, dtype=np.float64
    ).reshape(-1)
    if before is not None:
        out["before"] = scale_probe(before)
        persist = float(persist_factor) * before
        out["persist"] = scale_probe(persist)
    else:
        persist = None
    if after is not None:
        out["after"] = scale_probe(after)
        if persist is not None:
            out["fresh"] = scale_probe(after - persist)
    return out


def _strict_json_safe(value):
    """Recursively replace non-finite floats with explicit strings.

    ``json.dumps`` would otherwise emit bare ``NaN`` / ``Infinity`` tokens, which
    are not valid JSON and break strict parsers (jq, PowerShell).  Diagnostics
    keep an explicit marker instead: ``"nan"`` / ``"inf"`` / ``"-inf"``.
    """
    if isinstance(value, dict):
        return {str(k): _strict_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_strict_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number):
            return "nan"
        if np.isinf(number):
            return "inf" if number > 0.0 else "-inf"
        return number
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def cosine_probe(left, right):
    """Cosine similarity via the overflow-safe scaling path; diagnostics only."""
    if left is None or right is None:
        return None
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        return None
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return None
    scale = max(
        float(np.max(np.abs(a))) if a.size else 0.0,
        float(np.max(np.abs(b))) if b.size else 0.0,
    )
    if not (0.0 < scale < np.inf):
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sa = a / scale
        sb = b / scale
        na = float(np.linalg.norm(sa))
        nb = float(np.linalg.norm(sb))
        if na <= 0.0 or nb <= 0.0:
            return None
        return float(np.dot(sa, sb) / (na * nb))


def generation_record(
    *,
    family: str,
    generation: int,
    sigma_before,
    sigma_after_exp,
    sigma_after_guard,
    exp_arg_raw,
    exp_arg_used,
    c_s,
    d_sigma,
    mu_eff,
    e_chi,
    n_parents,
    n_individuals,
    path_before,
    path_after,
    path_persist_factor,
    wd,
    d_samples=None,
    cov_before=None,
    cov_after=None,
    axes_before=None,
    axes_after=None,
    negative_eigen_count=0,
    covariance_diag_repair_count=0,
    fitness=None,
    order=None,
    guide=None,
    extra=None,
) -> dict:
    """Build one per-generation CMA update record (no full matrices)."""
    record = {
        "record_type": "generation",
        "family": str(family),
        "generation": int(generation),
        "sigma_before": float(sigma_before),
        "sigma_after_exp": float(sigma_after_exp),
        "sigma_after_guard": float(sigma_after_guard),
        "sigma_growth_log10": (
            float(np.log10(sigma_after_guard) - np.log10(sigma_before))
            if sigma_before > 0.0 and sigma_after_guard > 0.0
            else None
        ),
        "exp_arg_raw": float(exp_arg_raw) if np.isfinite(float(exp_arg_raw)) else str(exp_arg_raw),
        "exp_arg_used": float(exp_arg_used),
        "exp_arg_clipped": bool(float(exp_arg_raw) != float(exp_arg_used)),
        "exp_arg_repair_nonfinite": bool(not np.isfinite(float(exp_arg_raw))),
        "c_s": float(c_s),
        "d_sigma": float(d_sigma),
        "mu_eff": float(mu_eff),
        "e_chi": float(e_chi),
        "n_parents": int(n_parents),
        "n_individuals": int(n_individuals),
    }
    record["path"] = path_decomposition(
        path_before, path_after, path_persist_factor
    )
    # The path length must be the Euclidean norm of p_s - the quantity the control
    # path feeds into exp_arg - computed in log space so that finite components
    # cannot overflow the ratio itself.
    after_probe = record["path"].get("after") or {}
    path_norm_log10 = after_probe.get("true_norm_log10")
    if float(e_chi) > 0.0 and path_norm_log10 is not None:
        log_ratio = float(path_norm_log10) - float(np.log10(float(e_chi)))
        record["path_length_ratio_log10"] = log_ratio
        record["path_length_ratio"] = (
            float(10.0 ** log_ratio) if log_ratio < 300.0 else float("inf")
        )
    elif after_probe.get("true_norm") == 0.0:
        record["path_length_ratio_log10"] = None
        record["path_length_ratio"] = 0.0
    else:
        record["path_length_ratio_log10"] = None
        record["path_length_ratio"] = None
    record["wd"] = direction_probe(wd)
    if d_samples is not None:
        record["samples_d"] = scale_probe(d_samples)
    if cov_before is not None or cov_after is not None:
        record["covariance"] = {
            "before": scale_probe(cov_before),
            "after": scale_probe(cov_after),
        }
    record["axes"] = {
        "before": scale_probe(axes_before),
        "after": scale_probe(axes_after),
    }
    for key, axes in (("before", axes_before), ("after", axes_after)):
        if axes is None:
            continue
        flat = np.asarray(axes, dtype=np.float64).reshape(-1)
        if flat.size and np.all(np.isfinite(flat)):
            record["axes"][f"{key}_min_axis"] = int(np.argmin(flat))
            record["axes"][f"{key}_min_value"] = float(np.min(flat))
            record["axes"][f"{key}_max_axis"] = int(np.argmax(flat))
            record["axes"][f"{key}_max_value"] = float(np.max(flat))
    record["eigen_replacement_occurred"] = bool(int(negative_eigen_count) > 0)
    record["negative_eigen_count"] = int(negative_eigen_count)
    record["covariance_diag_repair_count"] = int(covariance_diag_repair_count)
    if fitness is not None:
        record["fitness_used_for_ranking"] = [float(x) for x in np.asarray(
            fitness, dtype=np.float64
        ).reshape(-1)]
    if order is not None:
        record["ranking_order"] = [int(x) for x in np.asarray(order).reshape(-1)]
    record["guide"] = dict(guide) if guide else {}
    if extra:
        record.update(extra)
    return record


def commit_record(
    *,
    family: str,
    status: str,
    mode,
    metric,
    strength,
    sigma_used,
    axis_scales,
    rotation,
    center_before,
    raw_target,
    center_after,
    shift,
    ratio=None,
    retention=None,
    pre_paths=None,
    post_paths=None,
    generation=None,
    physical_fes=None,
    dimension=None,
    extra=None,
) -> dict:
    """Build one per-commit attenuation record.

    ``status`` is ``computed`` when ratio/retention were actually produced and
    ``not_computed`` when the caller raised before computing them; a
    ``not_computed`` record never carries a backfilled retention.
    """
    record = {
        "record_type": "commit",
        "family": str(family),
        "status": str(status),
        "mode": None if mode is None else str(mode),
        "metric": None if metric is None else str(metric),
        "strength": None if strength is None else float(strength),
        "dimension": None if dimension is None else int(dimension),
        "generation": None if generation is None else int(generation),
        "physical_fes": None if physical_fes is None else int(physical_fes),
        "center_before": None if center_before is None else [
            float(x) for x in np.asarray(center_before, dtype=np.float64).reshape(-1)
        ],
        "raw_target": None if raw_target is None else [
            float(x) for x in np.asarray(raw_target, dtype=np.float64).reshape(-1)
        ],
        "center_after": None if center_after is None else [
            float(x) for x in np.asarray(center_after, dtype=np.float64).reshape(-1)
        ],
        "shift": None if shift is None else [
            float(x) for x in np.asarray(shift, dtype=np.float64).reshape(-1)
        ],
        "sigma_used": None if sigma_used is None else float(sigma_used),
        "sigma_probe": scale_probe(sigma_used),
        "axis_scales": None if axis_scales is None else [
            float(x) for x in np.asarray(axis_scales, dtype=np.float64).reshape(-1)
        ],
        "axis_probe": scale_probe(axis_scales),
        "rotation": None if rotation is None else np.asarray(
            rotation, dtype=np.float64
        ).tolist(),
        "ratio": None if ratio is None else float(ratio),
        "retention": None if retention is None else float(retention),
        "pre_paths": {
            key: scale_probe(value) for key, value in (pre_paths or {}).items()
        },
        "post_paths": {
            key: scale_probe(value) for key, value in (post_paths or {}).items()
        },
    }
    if status != "computed":
        record["applied"] = "未算出、未执行"
    elif ratio is not None and float(ratio) <= 0.0:
        record["applied"] = "未执行(ratio=0)"
    else:
        record["applied"] = "已执行"
    if extra:
        record.update(extra)
    return record


class OptimizerNumericForensics:
    """Bounded, opt-in forensic recorder shared by one optimizer session."""

    def __init__(self, optimizer_name: str, options: dict):
        options = options or {}
        self.optimizer_name = str(optimizer_name)
        self.enabled = bool(options.get("optimizer_numeric_forensics_enable", False))
        self.output_dir = jump_output_dir(options)
        raw_context = options.get("optimizer_numeric_forensics_context", {})
        self.context = dict(raw_context) if isinstance(raw_context, dict) else {}
        self.generations = deque(maxlen=GENERATION_RING)
        self.commits = deque(maxlen=COMMIT_RING)
        self.scenes = deque(maxlen=COMMIT_RING)
        self.commit_index = 0
        self.flushed = []
        self.warned_no_dir = False
        self.missing_identity_fields = []
        self.scene = {}
        self.jump = None
        if self.enabled:
            self.jump = JumpWindow(
                self.optimizer_name,
                jump_output_dir(options, self.output_dir),
                jump_threshold_log10(options),
                context=self.context,
                target=jump_target(options),
                max_windows=jump_max_windows(options),
                alpha_gate=jump_alpha_gate(options),
                early_write=jump_early_write(options),
                early_milestone_log10=jump_early_milestone(options),
                mid_milestone_log10=jump_mid_milestone(options),
            )

    # -- observation -----------------------------------------------------
    def set_scene(self, **fields) -> None:
        if not self.enabled:
            return
        self.scene.update({k: v for k, v in fields.items() if v is not None})

    def note_sigma_provenance(self, **fields) -> None:
        if not self.enabled:
            return
        self.set_scene(sigma_provenance={k: v for k, v in fields.items()})

    def note_jump(self, key, payload: dict, jump: bool, collection=None,
                  growth_log10=None) -> bool:
        """Feed one generation into the bounded staged window (see :class:`JumpWindow`)."""
        if not self.enabled or self.jump is None:
            return False
        return self.jump.note(key, payload, jump, collection,
                              growth_log10=growth_log10)

    def record_generation(self, record: dict) -> None:
        if not self.enabled:
            return
        self.generations.append(record)
        ratio = record.get("path_length_ratio")
        if ratio is not None and float(ratio) > PATH_LENGTH_TRIGGER:
            self._trigger(
                "path_length",
                {"path_length_ratio": float(ratio),
                 "threshold": PATH_LENGTH_TRIGGER,
                 "generation": record.get("generation")},
            )
        growth = record.get("sigma_growth_log10")
        if growth is not None and float(growth) > SIGMA_GROWTH_TRIGGER_LOG10:
            self._trigger(
                "sigma_growth",
                {"sigma_growth_log10": float(growth),
                 "threshold": SIGMA_GROWTH_TRIGGER_LOG10,
                 "generation": record.get("generation")},
            )

    def record_commit(self, record: dict) -> None:
        if not self.enabled:
            return
        self.commit_index += 1
        record.setdefault("commit_index", int(self.commit_index))
        self.commits.append(record)

    def note_exception(self, tag: str, scene: dict) -> None:
        if not self.enabled:
            return
        self.scenes.append({"tag": str(tag), **scene})
        if self.jump is not None:
            # Pre-failure window: keep the newest state even when the sigma gate
            # never fired, bounded by the same per-process window budget.
            self.jump.flush_latest(f"exception:{tag}")
        self._trigger("nonfinite_geometry", {"tag": str(tag)})

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "generation_records": len(self.generations),
            "commit_records": len(self.commits),
            "scene_records": len(self.scenes),
            "flushed_classes": list(self.flushed),
            "output_dir": self.output_dir,
        }

    def window(self) -> dict:
        return {
            "generations": list(self.generations),
            "commits": list(self.commits),
            "scenes": list(self.scenes),
            "scene": dict(self.scene),
        }

    def force_flush(self, reason: str, detail=None) -> str:
        """Write the current window once; used by the light checker."""
        return self._write(reason, detail or {}, once=False)

    def _trigger(self, kind: str, detail: dict) -> None:
        key = (int(os.getpid()), str(kind))
        if key in _PROCESS_FLUSHED:
            return
        _PROCESS_FLUSHED.add(key)
        self.flushed.append(str(kind))
        self._write(str(kind), detail, once=True)

    def _write(self, reason: str, detail: dict, once: bool) -> str:
        if not self.enabled:
            return ""
        if not self.output_dir:
            if not self.warned_no_dir:
                self.warned_no_dir = True
                print(
                    "[numeric-forensics] no output dir configured "
                    "(optimizer_numeric_forensics_dir); window kept in memory",
                    flush=True,
                )
            return ""
        payload = {
            "record_type": "numeric_forensics_window",
            "reason": str(reason),
            "detail": detail,
            "once": bool(once),
            "time_unix": float(time.time()),
            "pid": int(os.getpid()),
            "optimizer": self.optimizer_name,
            **self.context,
            "generation_ring": list(self.generations),
            "commit_ring": list(self.commits),
            "scene_records": list(self.scenes),
            "scene": dict(self.scene),
        }
        try:
            data = _strict_json_safe(payload)
            line = (
                json.dumps(data, ensure_ascii=False, default=str, allow_nan=False)
                + "\n"
            ).encode("utf-8", errors="replace")
        except Exception:
            # Strict-JSON guarantee: never emit NaN/Infinity tokens, and never let a
            # serialization problem hide the fact that a window existed.
            line = (
                json.dumps(
                    {
                        "record_type": "numeric_forensics_window",
                        "reason": str(reason),
                        "serialization_error": True,
                        "generation_count": len(self.generations),
                        "commit_count": len(self.commits),
                        "scene": _strict_json_safe(dict(self.scene)),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8", errors="replace")
        path = os.path.join(
            self.output_dir, f"numeric_forensics_worker_{os.getpid()}.jsonl"
        )
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
        except Exception:
            # Observation must never alter optimizer behaviour or mask a failure.
            return ""
        return path

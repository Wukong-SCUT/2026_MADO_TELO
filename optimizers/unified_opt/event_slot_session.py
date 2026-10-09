"""Event-local, native-update sessions for cost-matched K-slot feedback.

The ordinary unified wrappers remain the production default.  These sessions
exist only for the opt-in objective-split event-slot path: one optimizer is
created at the start of a slow Actor event, advanced to K cumulative reported
FEs targets, exactly recentered after each intermediate commit, and destroyed
when the slow event returns.

No state in this module is an inter-event persistent bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Dict, List

import numpy as np

from optimizers.cmaes.cmaes import CMAES, CMAESNumericFailure
from optimizers.cmaes.numeric_forensics import (
    OptimizerNumericForensics,
    commit_record,
    scale_probe,
)
from optimizers.cmaes.sepcmaes import SEPCMAES
from optimizers.unified_opt.VkD_CMAES import VkdCma, default_option
from optimizers.unified_opt.cmaes_opt import CMAESOpt
from optimizers.unified_opt.mmes import MMES
from optimizers.unified_opt.sepcmaes_opt import SepCMAESOpt
from optimizers.unified_opt.vkd import VKD
from optimizers.unified_opt.boundary_compare_trace import trace_for


EVENT_SLOT_OPTIMIZERS = ("mmes", "vkd", "cmaes", "sepcmaes")


@dataclass(frozen=True)
class EventSlotPlan:
    optimizer_name: str
    requested_fes: int
    population: int
    slots: int
    minimum_population_units: int
    units_per_slot: tuple
    cumulative_reported_targets: tuple


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def build_event_slot_plan(
    optimizer_name: str,
    requested_fes: int,
    population: int,
    slots: int,
    dimension: int = None,
) -> EventSlotPlan:
    """Build a K-slot plan without adding objective calls.

    A slot may contain zero native population updates when the selected
    optimizer/resource pair cannot supply K complete updates. Repeated
    cumulative targets explicitly encode those communication-only slots;
    they must not become extra objective calls or a silent K change.

    MMES floor (F5 deadlock fix): when budget <= dimension the historical
    fresh-mean counter already saturates the reported budget, so MMES would
    otherwise run zero population units forever.  In that case exactly one
    minimum population unit is allocated; every other slot stays
    communication-only.  All other optimizer/budget combinations are
    bit-identical to the historical planner.
    """
    name = str(optimizer_name).strip().lower()
    if name not in EVENT_SLOT_OPTIMIZERS:
        raise ValueError(f"Unsupported event-slot optimizer: {optimizer_name}.")
    budget = int(requested_fes)
    lam = int(population)
    slot_count = int(slots)
    if budget <= 0:
        raise ValueError("Event-slot requested_fes must be positive.")
    if lam < 2:
        raise ValueError("Event-slot population must be at least 2.")
    if slot_count <= 0:
        raise ValueError("Event-slot count must be positive.")

    if name == "mmes":
        if dimension is None or int(dimension) <= 0:
            raise ValueError(
                "MMES event-slot planning requires a positive dimension "
                "to reproduce its historical fresh-mean FEs counter."
            )
        # Historical MMES evaluates one fresh mean, but its base evaluator
        # receives that mean as a 1-D vector and increments the protocol FEs
        # counter by len(mean)==dimension.  K1 must preserve this stopping
        # cadence even though the physical objective sample count is one.
        initial_native = int(dimension)
        minimum_units = (
            1
            if budget <= initial_native
            else _ceil_div(budget - initial_native, lam)
        )
    elif name == "sepcmaes":
        # SepCMAES normalizes the fresh mean to [1,D], so both its protocol
        # counter and physical sample count increase by one.
        initial_native = 1
        minimum_units = (
            0 if budget <= initial_native else _ceil_div(
                budget - initial_native, lam
            )
        )
    else:
        initial_native = 0
        minimum_units = _ceil_div(budget, lam)
    base_units, extra_units = divmod(minimum_units, slot_count)
    units_per_slot = [
        base_units + (1 if index < extra_units else 0)
        for index in range(slot_count)
    ]
    cumulative_units = np.cumsum(units_per_slot, dtype=np.int64)
    targets = tuple(
        min(budget, initial_native + int(units) * lam)
        for units in cumulative_units
    )
    if any(b < a for a, b in zip((0,) + targets[:-1], targets)):
        raise ValueError("Event-slot cumulative targets must be non-decreasing.")
    return EventSlotPlan(
        optimizer_name=name,
        requested_fes=budget,
        population=lam,
        slots=slot_count,
        minimum_population_units=int(minimum_units),
        units_per_slot=tuple(int(u) for u in units_per_slot),
        cumulative_reported_targets=targets,
    )


# Optional process-level override.  Normally the env injects the per-run path through
# the session option "nonfinite_dump_path", so the record lands next to the run's other
# logs (training: data_save_dir, evaluation: test_dir) instead of the repository root.
_NONFINITE_DUMP_PATH = os.environ.get("MAPPO_NONFINITE_DUMP") or None
_NONFINITE_DUMP_SEQ = 0
_NONFINITE_DUMP_WARNED = False


def _finite_probe(value):
    """Compact finiteness summary of a scalar/vector for the non-finite dump."""
    if value is None:
        return None
    raw = np.asarray(value)
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    out = {
        "shape": list(raw.shape),
        "size": int(arr.size),
        "all_finite": bool(arr.size and finite.all()),
        "nonfinite_count": int(arr.size - int(np.count_nonzero(finite))),
    }
    if finite.any():
        out["finite_max_abs"] = float(np.max(np.abs(arr[finite])))
    if 0 < arr.size <= 8:
        out["values"] = [float(x) for x in arr]
    return out


def _resolve_nonfinite_dump_path(session):
    """Prefer the path injected by the env, then the process-level override."""
    path = getattr(session, "_nonfinite_dump_path", None) or _NONFINITE_DUMP_PATH
    return str(path) if path else None


def _dump_nonfinite_geometry(tag: str, session, shift) -> None:
    """Append one JSONL record describing optimizer state at a non-finite relocation.

    Diagnostic only: never raises and never changes control flow - the caller still
    raises the original error immediately afterwards.  The record is written next to
    the run's other logs when the env supplied ``nonfinite_dump_path``.
    """
    global _NONFINITE_DUMP_SEQ, _NONFINITE_DUMP_WARNED
    try:
        path = _resolve_nonfinite_dump_path(session)
        if not path:
            if not _NONFINITE_DUMP_WARNED:
                _NONFINITE_DUMP_WARNED = True
                print(
                    "[nonfinite-dump] no dump path configured (session option "
                    "nonfinite_dump_path / MAPPO_NONFINITE_DUMP); record skipped",
                    flush=True,
                )
            return
        core = getattr(session, "core", None)
        _NONFINITE_DUMP_SEQ += 1
        payload = {
            "tag": tag,
            "time": time.time(),
            "pid": os.getpid(),
            "seq": _NONFINITE_DUMP_SEQ,
            "optimizer": getattr(session, "name", None) or type(session).__name__,
            "dimension": getattr(session, "dimension", None),
            "shift": _finite_probe(shift),
            "session_mean": _finite_probe(getattr(session, "_mean", None)),
            "sigma": _finite_probe(getattr(core, "sigma", None)),
            "ps": _finite_probe(getattr(core, "ps", None)),
            "p_s": _finite_probe(getattr(session, "_p_s", None)),
            "p_c": _finite_probe(getattr(session, "_p_c", None)),
            "s": _finite_probe(getattr(session, "_s", None)),
            "p": _finite_probe(getattr(session, "_p", None)),
            "w": _finite_probe(getattr(session, "_w", None)),
            "e_va": _finite_probe(getattr(session, "_e_va", None)),
            "d": _finite_probe(getattr(session, "_d", None)),
        }
        recorder = _forensics_recorder(session)
        if recorder is not None:
            payload["generation"] = int(getattr(core, "_n_generations", 0))
            payload["physical_fes"] = int(
                getattr(session, "physical_evaluations", 0)
            )
            payload["commit_index"] = int(getattr(recorder, "commit_index", 0))
            payload["forensics"] = recorder.summary()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if recorder is not None:
            # Unify the exception scene and flush the bounded window in one record.
            recorder.set_scene(
                exception_tag=str(tag),
                optimizer=str(getattr(session, "name", None) or type(session).__name__),
                dimension=getattr(session, "dimension", None),
                generation=int(getattr(core, "_n_generations", 0)),
                physical_fes=int(getattr(session, "physical_evaluations", 0)),
                commit_index=int(getattr(recorder, "commit_index", 0)),
                shift_probe=scale_probe(shift),
                session_mean_probe=scale_probe(getattr(session, "_mean", None)),
                sigma_probe=scale_probe(getattr(core, "sigma", None)),
            )
            recorder.note_exception(tag, dict(recorder.scene))
        core_flush = getattr(core, "flush_jump_window", None)
        if callable(core_flush):
            # VKD keeps its own bounded jump window (not the session recorder):
            # keep the newest state when the run is about to raise.
            core_flush(str(tag))
    except Exception:
        pass


def _forensics_recorder(session):
    """Return the enabled session recorder, else None (diagnostic only)."""
    recorder = getattr(session, "_forensics", None)
    if recorder is None or not getattr(recorder, "enabled", False):
        return None
    return recorder


def _attach_forensics(session) -> None:
    """Attach the session recorder to its native core and log sigma provenance."""
    recorder = _forensics_recorder(session)
    core = getattr(session, "core", None)
    if recorder is None or core is None:
        return
    if getattr(session, "_forensics_attached_to_core", False):
        return
    if hasattr(core, "_forensics"):
        core._forensics = recorder
        session._forensics_attached_to_core = True
    options = getattr(session, "options", {}) or {}
    context = dict(getattr(recorder, "context", {}) or {})
    recorder.note_sigma_provenance(
        optimizer=str(getattr(session, "optimizer_name", "") or type(session).__name__),
        core_family=type(core).__name__,
        sigma_option=(
            core.options.get("sigma") if hasattr(core, "options") else None
        ),
        sigma_used_at_session_start=float(getattr(core, "sigma", float("nan"))),
        sigma_inherit_enable=context.get(
            "sigma_inherit_enable",
            options.get("objective_split_sigma_inherit_enable"),
        ),
        previous_optimizer=context.get(
            "sigma_inherit_previous_optimizer",
            options.get("sigma_inherit_previous_optimizer"),
        ),
        reset_reason=options.get("sigma_reset_reason"),
        note="来源字段按会话 options / env 注入原样记录；缺失即留空，不推断",
    )


def _forensics_commit(session, status, sigma, axis_scales, rotation, center_before,
                      raw_target, center_after, shift, ratio=None, retention=None,
                      pre_paths=None, post_paths=None, extra=None):
    """Record one slot-commit attenuation step (diagnostic only)."""
    recorder = _forensics_recorder(session)
    if recorder is None:
        return None
    core = getattr(session, "core", None)
    record = commit_record(
        family=str(getattr(session, "optimizer_name", "") or type(session).__name__),
        status=str(status),
        mode=getattr(session, "ratio_path_mode", None),
        metric=getattr(session, "ratio_path_metric", None),
        strength=getattr(session, "ratio_path_strength", None),
        sigma_used=sigma,
        axis_scales=axis_scales,
        rotation=rotation,
        center_before=center_before,
        raw_target=raw_target,
        center_after=center_after,
        shift=shift,
        ratio=ratio,
        retention=retention,
        pre_paths=pre_paths,
        post_paths=post_paths,
        generation=(
            int(getattr(core, "_n_generations", 0)) if core is not None else None
        ),
        physical_fes=int(getattr(session, "physical_evaluations", 0)),
        dimension=int(getattr(session, "dimension", 0)),
        extra=extra,
    )
    recorder.record_commit(record)
    session._last_commit_forensics = record
    return record


def _finite_center(target, lower, upper, dimension: int) -> np.ndarray:
    center = np.asarray(target, dtype=np.float64).reshape(-1)
    if center.shape != (int(dimension),) or not np.all(np.isfinite(center)):
        raise ValueError("Event-slot commit centre must be finite and dimension-matched.")
    return np.clip(
        center,
        np.asarray(lower, dtype=np.float64).reshape(int(dimension)),
        np.asarray(upper, dtype=np.float64).reshape(int(dimension)),
    )


def _ratio_path_retention(shift, sigma, axis_scales, dimension: int, strength: float = 1.0,
                          metric: str = "rms", rotation=None, session=None,
                          raw_target=None, clipped_center=None):
    """Candidate path retention from total recentering in native search units."""
    shift_norm = float(np.linalg.norm(shift))
    if not np.isfinite(shift_norm):
        _forensics_commit(
            session, "not_computed", sigma, axis_scales, rotation,
            getattr(session, "_mean", None), raw_target, clipped_center, shift,
            extra={"failure": "nonfinite_shift", "shift_norm": str(shift_norm)},
        )
        _dump_nonfinite_geometry("cma_shift", session, shift)
        raise ValueError("Non-finite CMA path transition shift.")
    if shift_norm == 0.0:
        return 0.0, 1.0
    scale = abs(float(sigma)) * float(np.sqrt(np.mean(np.square(axis_scales))))
    if not np.isfinite(scale) or scale <= 0.0:
        _forensics_commit(
            session, "not_computed", sigma, axis_scales, rotation,
            getattr(session, "_mean", None), raw_target, clipped_center, shift,
            extra={"failure": "invalid_scale", "scale": str(scale)},
        )
        _dump_nonfinite_geometry("cma_scale", session, shift)
        raise ValueError("Invalid CMA path transition search scale.")
    if metric == "rms":
        ratio = shift_norm / (np.sqrt(float(dimension)) * scale)
    elif metric == "directional":
        axis = np.asarray(axis_scales, dtype=np.float64).reshape(-1)
        if (axis.shape != (dimension,) or not np.all(np.isfinite(axis))
                or np.any(axis <= 0.0) or np.asarray(shift).shape != (dimension,)):
            _forensics_commit(
                session, "not_computed", sigma, axis_scales, rotation,
                getattr(session, "_mean", None), raw_target, clipped_center, shift,
                extra={"failure": "invalid_axes"},
            )
            _dump_nonfinite_geometry("cma_axes", session, shift)
            raise ValueError("Invalid CMA directional search axes.")
        direction = np.asarray(shift, dtype=np.float64)
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64)
            if rotation.shape != (dimension, dimension) or not np.all(np.isfinite(rotation)):
                _forensics_commit(
                    session, "not_computed", sigma, axis_scales, rotation,
                    getattr(session, "_mean", None), raw_target, clipped_center, shift,
                    extra={"failure": "invalid_rotation"},
                )
                _dump_nonfinite_geometry("cma_rotation", session, shift)
                raise ValueError("Invalid CMA sampling rotation.")
            direction = rotation.T @ direction
        ratio = float(np.linalg.norm(direction / axis)
                      / (np.sqrt(float(dimension)) * abs(float(sigma))))
    else:
        raise ValueError("Invalid CMA path transition metric.")
    if not np.isfinite(ratio):
        _forensics_commit(
            session, "not_computed", sigma, axis_scales, rotation,
            getattr(session, "_mean", None), raw_target, clipped_center, shift,
            extra={"failure": "nonfinite_ratio", "ratio": str(ratio)},
        )
        _dump_nonfinite_geometry("cma_ratio", session, shift)
        raise ValueError("Non-finite CMA path transition ratio.")
    if strength == 1.0:
        return float(ratio), float(1.0 / (1.0 + ratio))
    return float(ratio), float(1.0 - strength * ratio / (1.0 + ratio))


def _mmes_native_shift_ratio(shift, sigma, q, v, gamma, c_a, session=None):
    """A-to-C distance in MMES's unbounded Gaussian mixture coordinates."""
    shift = np.asarray(shift, dtype=np.float64)
    if not np.all(np.isfinite(shift)):
        _dump_nonfinite_geometry("mmes_shift", session, shift)
        raise ValueError("Non-finite MMES center relocation.")
    if not np.any(shift):
        return 0.0
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.int64)
    m, dimension = q.shape
    if (shift.shape != (dimension,) or m < 1 or v.shape != (m,)
            or not np.array_equal(np.sort(v), np.arange(m))
            or not np.all(np.isfinite(q)) or not np.isfinite(sigma) or sigma <= 0
            or not np.isfinite(gamma) or not 0 <= gamma < 1
            or not np.isfinite(c_a) or not 0 < c_a <= 1):
        _dump_nonfinite_geometry("mmes_geometry", session, shift)
        raise ValueError("Invalid MMES sampling geometry for center relocation.")
    g = np.arange(1, m + 1)
    decay = 1.0 - c_a
    weights = c_a * decay ** (g - 1) / (1.0 - decay ** m)
    selected = q[v[(m - (g % m)) - 1]]
    b = (np.sqrt(gamma * weights)[:, None] * selected)
    isotropic = 1.0 - gamma
    projection = b @ shift
    gram = isotropic * np.eye(m) + b @ b.T
    squared = float((shift @ shift - projection @ np.linalg.solve(gram, projection)) / isotropic)
    if not np.isfinite(squared) or squared < -1e-9 * float(shift @ shift):
        _dump_nonfinite_geometry("mmes_squared", session, shift)
        raise ValueError("Invalid MMES relative center relocation.")
    return float(np.sqrt(max(0.0, squared)) / (np.sqrt(dimension) * sigma))


def _normalized_direction(raw, dimension: int):
    if raw is None:
        return None
    direction = np.asarray(raw, dtype=np.float64).reshape(-1)
    if direction.shape != (int(dimension),):
        return None
    norm = float(np.linalg.norm(direction))
    if (not np.isfinite(norm)) or norm <= 1e-12:
        return None
    return direction / norm


def _finite_norm(raw) -> float:
    value = np.asarray(raw, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    return norm if np.isfinite(norm) else 0.0


def _direction_cosine(raw, direction, dimension: int) -> float:
    left = np.asarray(raw, dtype=np.float64).reshape(-1)
    right = np.asarray(direction, dtype=np.float64).reshape(-1)
    if left.shape != (int(dimension),) or right.shape != (int(dimension),):
        return 0.0
    left_norm = _finite_norm(left)
    right_norm = _finite_norm(right)
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    if not np.isfinite(cosine):
        return 0.0
    return float(np.clip(cosine, -1.0, 1.0))


def _memory_summary(
    *,
    center,
    direction,
    primary,
    secondary,
    effective_scale_rms: float,
    guide,
    guide_strength: float,
    guide_mix_strength: float,
    dimension: int,
) -> Dict:
    """Build objective-free, mutation-free optimizer-memory telemetry."""
    primary_array = np.asarray(primary, dtype=np.float64).reshape(-1)
    secondary_array = np.asarray(secondary, dtype=np.float64).reshape(-1)
    if primary_array.shape != (int(dimension),):
        primary_array = np.zeros((int(dimension),), dtype=np.float64)
    if secondary_array.shape != (int(dimension),):
        secondary_array = np.zeros((int(dimension),), dtype=np.float64)
    guide_array = _normalized_direction(guide, dimension)
    guide_active = guide_array is not None
    if guide_array is None:
        guide_array = np.zeros((int(dimension),), dtype=np.float64)
    scale = float(effective_scale_rms)
    if not np.isfinite(scale) or scale < 0.0:
        scale = 0.0
    strength = float(guide_strength)
    mix_strength = float(guide_mix_strength)
    if not np.isfinite(strength) or strength < 0.0:
        strength = 0.0
    if not np.isfinite(mix_strength) or mix_strength < 0.0:
        mix_strength = 0.0
    if not guide_active:
        strength = 0.0
        mix_strength = 0.0
    return {
        "search_center": np.asarray(center, dtype=np.float64).reshape(
            int(dimension)
        ).copy(),
        "primary_memory_norm": _finite_norm(primary_array),
        "primary_memory_commit_cosine": _direction_cosine(
            primary_array, direction, dimension
        ),
        "secondary_memory_norm": _finite_norm(secondary_array),
        "secondary_memory_commit_cosine": _direction_cosine(
            secondary_array, direction, dimension
        ),
        "effective_scale_rms": scale,
        "guide_norm": _finite_norm(guide_array),
        "guide_strength": strength,
        "guide_mix_strength": mix_strength,
        "guide_commit_cosine": _direction_cosine(
            guide_array, direction, dimension
        ),
    }


class _BaseEventSlotSession:
    optimizer_name = ""

    def __init__(self, problem: Dict, options: Dict):
        self.problem = dict(problem)
        self.options = dict(options)
        self._nonfinite_dump_path = self.options.get("nonfinite_dump_path")
        self.requested_fes = int(self.options.get("max_function_evaluations", 0))
        self.dimension = int(self.problem["ndim_problem"])
        self.lower = np.asarray(
            self.problem["lower_boundary"], dtype=np.float64
        ).reshape(self.dimension)
        self.upper = np.asarray(
            self.problem["upper_boundary"], dtype=np.float64
        ).reshape(self.dimension)
        self.population_units = 0
        self.distribution_updates = 0
        self._terminated = False
        self._last_packet_best_x = None
        self._last_packet_best_y = np.inf
        self._last_packet_units = 0
        self._last_commit_shift = 0.0
        self._boundary_dir = str(self.options.get("boundary_compare_trace_dir", ""))
        self._boundary_identity = self.options.get("boundary_compare_identity")
        self._boundary_slot_id = -1
        # Diagnostic-only forensic recorder (opt-in).  Disabled -> inert object, every
        # forensic call site returns immediately and the control path is unchanged.
        self._forensics = OptimizerNumericForensics(self.optimizer_name, self.options)
        self._forensics_attached_to_core = False
        self._last_commit_forensics = None
        if getattr(self._forensics, "enabled", False):
            context = dict(self._forensics.context)
            self._forensics.set_scene(
                identity_present=dict(context),
                missing_identity_fields=[
                    key for key in (
                        "problem_family", "function_id", "env_step", "agent_id",
                        "subfes", "seed", "optimizer_action", "resource_action",
                        "cfg_levels", "train_epoch", "train_batch", "event_index",
                        "slot_index",
                    ) if key not in context
                ],
            )

    @property
    def population(self) -> int:
        raise NotImplementedError

    @property
    def physical_evaluations(self) -> int:
        raise NotImplementedError

    @property
    def native_evaluations(self) -> int:
        """Optimizer-native FEs counter used by the historical wrapper."""
        return int(self.physical_evaluations)

    @property
    def reported_evaluations(self) -> int:
        return int(min(self.native_evaluations, self.requested_fes))

    @property
    def terminated(self) -> bool:
        return bool(self._terminated)

    def _begin_packet(self) -> None:
        self._last_packet_best_x = None
        self._last_packet_best_y = np.inf
        self._last_packet_units = 0
        self._packet_native_first_generation = None

    def _observe_packet_population(self, x, y) -> None:
        points = np.asarray(x, dtype=np.float64)
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[0] != values.size or points.shape[1] != self.dimension:
            raise ValueError("Event-slot optimizer returned a malformed population.")
        index = int(np.argmin(values))
        value = float(values[index])
        if value < self._last_packet_best_y:
            self._last_packet_best_y = value
            self._last_packet_best_x = points[index].copy()

    def _boundary_wants_key(self):
        return bool(self._boundary_dir and self._boundary_identity and
                    trace_for(self._boundary_dir).wants_key(self._boundary_identity))

    def _boundary_note(self, before, after, raw, scored, fitness, sigma_before,
                       sigma_after, shape_scale, extra=None, replay=None):
        if not self._boundary_dir or not self._boundary_identity:
            return
        details = dict(extra or {})
        details.update(
            physical_evaluations=int(self.physical_evaluations),
            reported_evaluations=int(self.reported_evaluations),
            population_units=int(self.population_units),
            distribution_updates=int(self.distribution_updates),
            restart_id=int(getattr(self, "_restart_id", 0)),
        )
        core = getattr(self, "core", getattr(self, "_core", None))
        telemetry = getattr(core, "numeric_telemetry", None)
        if telemetry is not None:
            details["protection_counters"] = dict(telemetry.counters)
        trace_for(self._boundary_dir).note_generation(
            self._boundary_identity, self._boundary_slot_id, before, after,
            raw, scored, fitness, sigma_before, sigma_after, shape_scale,
            self.lower, self.upper, extra=details, replay=replay,
        )

    def boundary_commit(self, slot, internal_mean, proposal, committed, transition):
        if not self._boundary_dir or not self._boundary_identity:
            return
        trace_for(self._boundary_dir).note_commit(
            self._boundary_identity, slot, internal_mean, proposal, committed,
            self.evolved_sigma(), transition,
        )

    def advance_to_reported_target(self, target: int) -> Dict:
        raise NotImplementedError

    def synchronize_center(self, target) -> Dict:
        raise NotImplementedError

    def evolved_sigma(self):
        """事件末演化后 σ（14 卡 σ 继承写点用；不动 result() 既有字段语义）。
        基类无核心状态，返回 None；各会话覆写为真实演化源。"""
        return None

    def configure_guide(self, guide_options: Dict) -> None:
        """Update only the already-approved event-local guide parameters."""
        return

    def diagnostic_snapshot(self, commit_direction) -> Dict:
        """Read detached state summaries without changing optimizer state."""
        raise NotImplementedError

    def native_state_snapshot(self):
        """Optional detached raw state for evaluation diagnostics."""
        return None

    def packet_native_first_generation_snapshot(self):
        """Optional first evaluated generation from the current packet."""
        return None

    def _packet_result(self, *, communication_only: bool = False) -> Dict:
        if self._last_packet_best_x is None:
            raise RuntimeError("Event-slot packet completed without a candidate.")
        result = self.result()
        result.update(
            {
                "packet_best_x": self._last_packet_best_x.copy(),
                "packet_best_y": float(self._last_packet_best_y),
                "packet_population_units": int(self._last_packet_units),
                "population_units": int(self.population_units),
                "distribution_updates": int(self.distribution_updates),
                "physical_function_evaluations": int(self.physical_evaluations),
                "native_function_evaluations": int(self.native_evaluations),
                "reported_function_evaluations": int(self.reported_evaluations),
                "last_commit_shift": float(self._last_commit_shift),
                "packet_communication_only": bool(communication_only),
            }
        )
        return result

    def communication_only_packet(self, candidate, value: float) -> Dict:
        """Emit the current committed state without optimizer or objective work."""
        self._begin_packet()
        self._observe_packet_population(
            np.asarray(candidate, dtype=np.float64).reshape(1, self.dimension),
            np.asarray([float(value)], dtype=np.float64),
        )
        return self._packet_result(communication_only=True)

    def result(self) -> Dict:
        raise NotImplementedError


def _configure_native_guide(core, guide_options: Dict, dimension: int) -> None:
    opts = dict(guide_options or {})
    enabled = bool(opts.get("optimizer_guide_enable", False))
    direction = _normalized_direction(
        opts.get("optimizer_guide_direction"), dimension
    )
    enabled = bool(enabled and direction is not None)
    if hasattr(core, "optimizer_guide_enable"):
        core.optimizer_guide_enable = enabled
    if hasattr(core, "optimizer_guide_direction"):
        core.optimizer_guide_direction = direction if enabled else None
    for key in (
        "optimizer_guide_strength",
        "optimizer_guide_mix_strength",
        "optimizer_guide_injection_pairs",
        "optimizer_guide_use_negative_pair",
        "optimizer_guide_numeric_guard",
        "optimizer_guide_sigma_exp_clip",
        "optimizer_guide_sigma_clip_ratio",
        "optimizer_guide_sample_clip_ratio",
        "optimizer_guide_internal_mode",
        "optimizer_guide_internal_mean_lr",
        "optimizer_guide_internal_path_lr",
        "optimizer_guide_internal_cov_lr",
        "optimizer_guide_internal_agree_cos_min",
        "optimizer_guide_internal_max_step_ratio",
        "optimizer_guide_internal_max_rel_step",
        "optimizer_guide_internal_path_max_rel_norm",
        "optimizer_guide_internal_cov_rank1_clip",
        "optimizer_guide_internal_disable_sample_injection",
    ):
        if key in opts and hasattr(core, key):
            setattr(core, key, opts[key])


class CMAESEventSlotSession(_BaseEventSlotSession):
    optimizer_name = "cmaes"

    def __init__(self, problem: Dict, options: Dict):
        super().__init__(problem, options)
        self.ratio_path_mode = str(options.get("cma_sep_ratio_path_mode", "native")).lower()
        if self.ratio_path_mode not in {"native", "step_path", "shape_path", "both_paths"}:
            raise ValueError("Unsupported CMAES ratio path mode.")
        self.ratio_path_metric = str(options.get("cmaes_ratio_path_metric", "rms")).lower()
        if self.ratio_path_metric not in {"rms", "directional"}:
            raise ValueError("Invalid CMAES ratio path metric.")
        self.ratio_path_strength = float(options.get("cma_sep_ratio_path_strength", 1.0))
        if not np.isfinite(self.ratio_path_strength) or not 0.0 <= self.ratio_path_strength <= 1.0:
            raise ValueError("Invalid CMAES ratio path strength.")
        wrapper = CMAESOpt(problem, options)
        self.core = CMAES(wrapper.build_core_problem(), wrapper.build_core_options())
        self.core._boundary_capture = bool(self._boundary_dir)
        self.core.start_time = time.time()
        (
            self._x,
            self._mean,
            self._p_s,
            self._p_c,
            self._cm,
            self._e_ve,
            self._e_va,
            self._y,
            self._d,
        ) = self.core.initialize()

    @property
    def population(self) -> int:
        return int(self.core.n_individuals)

    @property
    def physical_evaluations(self) -> int:
        return int(self.core.n_function_evaluations)

    def evolved_sigma(self):
        # CMAES core.sigma 每代更新（cmaes.py），事件末即演化后 σ。
        return float(self.core.sigma)

    def advance_to_reported_target(self, target: int) -> Dict:
        target = int(target)
        if target <= self.reported_evaluations or target > self.requested_fes:
            raise ValueError("CMAES event-slot target must advance within the event budget.")
        _attach_forensics(self)
        self._begin_packet()
        capture_first = bool(self.options.get("record_native_first_generation", False))
        while self.reported_evaluations < target and not self._terminated:
            boundary_before = self._mean.copy() if self._boundary_dir else None
            boundary_sigma = float(self.core.sigma) if self._boundary_dir else None
            boundary_scale = (float(np.sqrt(np.mean(np.square(self._e_va))))
                              if self._boundary_dir else None)
            boundary_replay_before = self.native_state_snapshot() if self._boundary_wants_key() else None
            first_before = (
                self.native_state_snapshot()
                if capture_first and self._packet_native_first_generation is None
                else None
            )
            self._x, self._y, self._d = self.core.iterate(
                self._x,
                self._mean,
                self._e_ve,
                self._e_va,
                self._y,
                self._d,
            )
            if (
                first_before is not None
                and self.physical_evaluations > first_before["physical_fes"]
            ):
                self._packet_native_first_generation = {
                    "before_iterate": first_before,
                    "x": self._x.tolist(),
                    "y": self._y.tolist(),
                    "d": self._d.tolist(),
                    "generation_before_update": int(self.core._n_generations),
                    "after_update": None,
                }
            self.population_units += 1
            self._last_packet_units += 1
            self._observe_packet_population(self._x, self._y)
            if self.core._check_terminations():
                self._terminated = True
                break
            self.core._n_generations += 1
            try:
                (
                    self._mean,
                    self._p_s,
                    self._p_c,
                    self._cm,
                    self._e_ve,
                    self._e_va,
                ) = self.core.update_distribution(
                    self._x,
                    self._p_s,
                    self._p_c,
                    self._cm,
                    self._e_ve,
                    self._e_va,
                    self._y,
                    self._d,
                    mean_old=self._mean,
                )
                self.distribution_updates += 1
                if self._boundary_dir:
                    self._boundary_note(
                        boundary_before, self._mean, self._x,
                        np.clip(self._x, self.lower, self.upper), self._y,
                        boundary_sigma, self.core.sigma,
                        boundary_sigma * boundary_scale,
                        replay=({"before": boundary_replay_before,
                                 "sample_d": np.copy(self._d),
                                 "after": self.native_state_snapshot()}
                                if boundary_replay_before is not None else None),
                    )
                if (
                    first_before is not None
                    and self._packet_native_first_generation is not None
                ):
                    self._packet_native_first_generation["after_update"] = (
                        self.native_state_snapshot()
                    )
            except CMAESNumericFailure as exc:
                self.core.optimizer_numeric_fail_soft_triggered = True
                self.core.optimizer_numeric_fail_soft_reason = str(exc)
                self.core.optimizer_numeric_fail_soft_generation = int(
                    self.core._n_generations
                )
                self.core.optimizer_numeric_fail_soft_evaluations = int(
                    self.core.n_function_evaluations
                )
                self._terminated = True
                break
        if (
            self.reported_evaluations < target
            and self._last_packet_best_x is None
        ):
            reason = str(self.core.optimizer_numeric_fail_soft_reason)
            raise RuntimeError(
                "CMAES event-slot session terminated before its target"
                + (f": {reason}." if reason else ".")
            )
        return self._packet_result()

    def synchronize_center(self, target) -> Dict:
        center = _finite_center(target, self.lower, self.upper, self.dimension)
        shift = center - self._mean
        transition = None
        if self.ratio_path_mode != "native":
            ratio, retention = _ratio_path_retention(
                shift, self.core.sigma, self._e_va, self.dimension,
                self.ratio_path_strength, metric=self.ratio_path_metric,
                rotation=self._e_ve, session=self,
                raw_target=target, clipped_center=center,
            )
            before = {"p_s": float(np.linalg.norm(self._p_s)),
                      "p_c": float(np.linalg.norm(self._p_c))}
            pre_paths = {"p_s": np.copy(self._p_s), "p_c": np.copy(self._p_c)}
            if ratio > 0.0:
                if self.ratio_path_mode in {"step_path", "both_paths"}:
                    self._p_s = retention * self._p_s
                if self.ratio_path_mode in {"shape_path", "both_paths"}:
                    self._p_c = retention * self._p_c
            _forensics_commit(
                self, "computed", self.core.sigma, self._e_va, self._e_ve,
                self._mean, target, center, shift,
                ratio=ratio, retention=retention, pre_paths=pre_paths,
                post_paths={"p_s": np.copy(self._p_s), "p_c": np.copy(self._p_c)},
            )
            transition = {
                "mode": self.ratio_path_mode,
                "strength": self.ratio_path_strength,
                "metric": self.ratio_path_metric,
                "ratio": ratio,
                "retention": retention,
                "before_norm": before,
                "after_norm": {"p_s": float(np.linalg.norm(self._p_s)),
                               "p_c": float(np.linalg.norm(self._p_c))},
            }
        self._mean = center.copy()
        self.core.mean = center.copy()
        self._last_commit_shift = float(np.linalg.norm(shift))
        result = {"applied_norm": self._last_commit_shift, "exact": True}
        if transition is not None:
            result["transition"] = transition
        return result

    def configure_guide(self, guide_options: Dict) -> None:
        _configure_native_guide(self.core, guide_options, self.dimension)

    def diagnostic_snapshot(self, commit_direction) -> Dict:
        effective_scale = float(
            abs(float(self.core.sigma))
            * np.sqrt(np.mean(np.square(self._e_va)))
        )
        step_size_path_coordinate = (
            self._e_ve
            @ np.diag(self._e_va)
            @ self._e_ve.T
            @ self._p_s
        )
        return _memory_summary(
            center=self._mean,
            direction=commit_direction,
            primary=self._p_c,
            secondary=step_size_path_coordinate,
            effective_scale_rms=effective_scale,
            guide=getattr(self.core, "optimizer_guide_direction", None),
            guide_strength=getattr(
                self.core, "optimizer_guide_strength", 0.0
            ),
            guide_mix_strength=getattr(
                self.core, "optimizer_guide_mix_strength", 0.0
            ),
            dimension=self.dimension,
        )

    def native_state_snapshot(self) -> Dict:
        return {
            "optimizer": "cmaes",
            "mean": self._mean.tolist(),
            "core_mean": np.asarray(self.core.mean, dtype=np.float64).tolist(),
            "p_s": self._p_s.tolist(),
            "p_c": self._p_c.tolist(),
            "cm": self._cm.tolist(),
            "e_ve": self._e_ve.tolist(),
            "e_va": self._e_va.tolist(),
            "sigma": float(self.core.sigma),
            "physical_fes": int(self.physical_evaluations),
            "distribution_updates": int(self.distribution_updates),
        }

    def packet_native_first_generation_snapshot(self):
        return getattr(self, "_packet_native_first_generation", None)

    def result(self) -> Dict:
        best_x = np.asarray(self.core.best_so_far_x, dtype=np.float64).reshape(-1)
        return {
            "best_so_far_x": best_x.copy(),
            "best_so_far_y": float(self.core.best_so_far_y),
            "n_function_evaluations": int(self.reported_evaluations),
            "mean": best_x.copy(),
            "search_center": self._mean.copy(),
            "sigma": float(self.options["sigma"]),
            "optimizer_guide_internal_applied": float(
                self.core.optimizer_guide_internal_applied
            ),
            "optimizer_guide_internal_mean_step_norm": float(
                self.core.optimizer_guide_internal_mean_step_norm
            ),
            "optimizer_guide_internal_alignment": float(
                self.core.optimizer_guide_internal_alignment
            ),
            "optimizer_anchor_applied": float(self.core.optimizer_anchor_applied),
            "optimizer_anchor_mean_step_norm": float(
                self.core.optimizer_anchor_mean_step_norm
            ),
            "optimizer_anchor_sample_applied": float(
                self.core.optimizer_anchor_sample_applied
            ),
            "optimizer_anchor_dist": float(self.core.optimizer_anchor_dist),
            "optimizer_numeric_telemetry": self.core.numeric_telemetry.result_records(),
            "optimizer_numeric_guard_counters": self.core.numeric_telemetry.result_counters(),
            "optimizer_numeric_fail_soft_enabled": bool(
                self.core.optimizer_numeric_fail_soft
            ),
            "optimizer_numeric_fail_soft_triggered": bool(
                self.core.optimizer_numeric_fail_soft_triggered
            ),
            "optimizer_numeric_fail_soft_reason": str(
                self.core.optimizer_numeric_fail_soft_reason
            ),
            "optimizer_numeric_fail_soft_generation": int(
                self.core.optimizer_numeric_fail_soft_generation
            ),
            "optimizer_numeric_fail_soft_evaluations": int(
                self.core.optimizer_numeric_fail_soft_evaluations
            ),
        }


class MMESEventSlotSession(_BaseEventSlotSession):
    optimizer_name = "mmes"

    def __init__(self, problem: Dict, options: Dict):
        super().__init__(problem, options)
        self.core = MMES(problem, options)
        self.core._boundary_capture = bool(self._boundary_dir)
        self.core.start_time = time.time()
        (
            self._x,
            self._mean,
            self._p,
            self._w,
            self._q,
            self._t,
            self._v,
            self._y,
        ) = self.core.initialize()
        self._initial_candidate_pending = True
        self._neutralize_success_once = False
        self.mmes_state_transition_mode = str(options.get("mmes_state_transition_mode", "native")).lower()
        if self.mmes_state_transition_mode not in {"native", "neutralize_credit"}:
            raise ValueError(f"Unsupported MMES state transition mode: {self.mmes_state_transition_mode}")
        self.ratio_success_mode = str(options.get("mmes_ratio_success_mode", "native")).lower()
        if self.ratio_success_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid MMES ratio success mode.")
        if self.ratio_success_mode != "native" and self.mmes_state_transition_mode != "native":
            raise ValueError("MMES ratio success and diagnostic credit modes cannot be combined.")
        self.ratio_success_metric = str(options.get("mmes_ratio_success_metric", "sigma")).lower()
        if self.ratio_success_metric not in {"sigma", "full"}:
            raise ValueError("Invalid MMES ratio success metric.")
        self.ratio_success_strength = float(options.get("mmes_ratio_success_strength", 0.1))
        if not np.isfinite(self.ratio_success_strength) or not 0.0 <= self.ratio_success_strength <= 1.0:
            raise ValueError("Invalid MMES ratio success strength.")
        self.ratio_direction_mode = str(options.get("mmes_ratio_direction_mode", "native")).lower()
        if self.ratio_direction_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid MMES ratio direction mode.")
        if self.ratio_direction_mode != "native" and self.mmes_state_transition_mode != "native":
            raise ValueError("MMES direction trial cannot combine with diagnostic credit mode.")
        if self.ratio_direction_mode != "native" and self.ratio_success_mode != "native" and self.ratio_success_metric != "full":
            raise ValueError("Combined MMES w/p trial requires the same full sampling metric.")
        self.ratio_direction_strength = float(options.get("mmes_ratio_direction_strength", 0.1))
        if not np.isfinite(self.ratio_direction_strength) or not 0.0 <= self.ratio_direction_strength <= 1.0:
            raise ValueError("Invalid MMES ratio direction strength.")
        self._pending_mmes_stale_shift_norm = 0.0
        self._begin_packet()

    def _begin_packet(self) -> None:
        super()._begin_packet()
        self._packet_mmes_neutralized_updates = 0
        self._packet_mmes_paired_updates = 0
        self._packet_mmes_incomparable_updates = 0
        self._packet_mmes_comparable_updates = 0
        self._packet_mmes_previous_slot_shift_norm = float(self._last_commit_shift)
        self._packet_mmes_stale_pending_at_start = int(self._neutralize_success_once)
        self._packet_mmes_stale_source_shift_norm = float(self._pending_mmes_stale_shift_norm)
        self._packet_mmes_sigma_before = float(self.core.sigma)
        self._packet_mmes_success_stat_before = float(self._w)

    @property
    def population(self) -> int:
        return int(self.core.n_individuals)

    def evolved_sigma(self):
        # MMES core.sigma 随 _update_distribution 演化（事件末即演化后 σ）。
        return float(self.core.sigma)

    @property
    def physical_evaluations(self) -> int:
        # The historical MMES base evaluator over-counts the one fresh-mean
        # objective sample as D protocol evaluations.  Population evaluations
        # are counted correctly.  Keep protocol and physical ledgers separate.
        return int(1 + self.population_units * self.population)

    @property
    def native_evaluations(self) -> int:
        return int(self.core.n_function_evaluations)

    def advance_to_reported_target(
        self, target: int, force: bool = False
    ) -> Dict:
        target = int(target)
        if target > self.requested_fes:
            raise ValueError("MMES event-slot target must advance within the event budget.")
        if not force and target <= self.reported_evaluations:
            raise ValueError("MMES event-slot target must advance within the event budget.")
        self._begin_packet()
        if self._initial_candidate_pending:
            self._observe_packet_population(
                self._mean.reshape(1, -1),
                np.asarray([self._y[0]], dtype=np.float64),
            )
            self._initial_candidate_pending = False
        # 保底最少评估（F5 死锁修复）：force=True 时即使 reported 已达预算
        # （B<=D 的饥饿情形）也执行恰好一批完整种群评估与一次分布更新。
        # 复用 mmes.py 的 _force_one_iteration 让 iterate 跳过其内部终止检查，
        # 并让本循环跳过批后终止检查，保证分布更新真实执行一次。
        while (
            force or (self.reported_evaluations < target)
        ) and not self._terminated:
            boundary_before = self._mean.copy() if self._boundary_dir else None
            boundary_sigma = float(self.core.sigma) if self._boundary_dir else None
            boundary_replay_before = (
                {"mean": np.copy(self._mean), "sigma": boundary_sigma,
                 "p": np.copy(self._p), "w": float(self._w),
                 "q": np.copy(self._q), "t": np.copy(self._t),
                 "v": np.copy(self._v)} if self._boundary_wants_key() else None
            )
            y_bak = np.copy(self._y)
            forced_batch = bool(force)
            force = False
            if forced_batch:
                self.core._force_one_iteration = True
            self._x, self._y = self.core.iterate(
                self._x,
                self._mean,
                self._q,
                self._v,
            )
            if forced_batch:
                self.core._force_one_iteration = False
            if self._y is None:
                self._terminated = True
                break
            boundary_y = np.copy(self._y) if self._boundary_dir else None
            self.population_units += 1
            self._last_packet_units += 1
            self._observe_packet_population(self._x, self._y)
            if self.core._check_terminations() and not forced_batch:
                self._terminated = True
                break
            stale_y_bak = bool(self._neutralize_success_once)
            neutralize_success = bool(
                stale_y_bak
                or (
                    self.mmes_state_transition_mode == "neutralize_credit"
                    and self._last_commit_shift > 1e-15
                )
            )
            (
                self._mean,
                self._p,
                self._w,
                self._q,
                self._t,
                self._v,
            ) = self.core._update_distribution(
                self._x,
                self._mean,
                self._p,
                self._w,
                self._q,
                self._t,
                self._v,
                self._y,
                y_bak,
                neutralize_success=neutralize_success,
            )
            if neutralize_success:
                self._packet_mmes_neutralized_updates += 1
            else:
                self._packet_mmes_paired_updates += 1
            if stale_y_bak:
                self._packet_mmes_incomparable_updates += 1
            else:
                self._packet_mmes_comparable_updates += 1
            self._neutralize_success_once = False
            self._pending_mmes_stale_shift_norm = 0.0
            self.core._n_generations += 1
            self.distribution_updates += 1
            if self._boundary_dir:
                self._boundary_note(
                    boundary_before, self._mean, self.core._boundary_raw,
                    self._x, boundary_y, boundary_sigma, self.core.sigma,
                    boundary_sigma, extra={"shape_scale": "MMES q/v sampling; use raw_distance_rms"},
                    replay=({"before": boundary_replay_before, "previous_fitness": y_bak,
                             "after": {"mean": np.copy(self._mean),
                                       "sigma": float(self.core.sigma),
                                       "p": np.copy(self._p), "w": float(self._w),
                                       "q": np.copy(self._q), "t": np.copy(self._t),
                                       "v": np.copy(self._v)}}
                            if boundary_replay_before is not None else None),
                )
        if (
            self.reported_evaluations < target
            and self._last_packet_best_x is None
        ):
            raise RuntimeError("MMES event-slot session terminated before its target.")
        return self._packet_result()

    def synchronize_center(self, target) -> Dict:
        center = _finite_center(target, self.lower, self.upper, self.dimension)
        shift = center - self._mean
        shift_norm = float(np.linalg.norm(shift))
        transition = None
        if self.ratio_success_mode == "attenuate":
            ratio, retention = 0.0, 1.0
            before_w = float(self._w)
            if shift_norm > 0.0:
                sigma = float(self.core.sigma)
                if not np.isfinite(sigma) or sigma <= 0.0:
                    _dump_nonfinite_geometry("mmes_step_scale", self, shift)
                    raise ValueError("Invalid MMES step scale for center relocation.")
                if self.ratio_success_metric == "full":
                    ratio = _mmes_native_shift_ratio(
                        shift, sigma, self._q, self._v,
                        float(self.core.gamma), float(self.core.c_a),
                        session=self,
                    )
                else:
                    ratio = float(shift_norm / (np.sqrt(self.dimension) * sigma))
                    if not np.isfinite(ratio):
                        _dump_nonfinite_geometry("mmes_ratio", self, shift)
                        raise ValueError("Non-finite MMES relative center relocation.")
                retention = float(1.0 - self.ratio_success_strength * ratio / (1.0 + ratio))
                self._w *= retention
            transition = {
                "mode": self.ratio_success_mode, "strength": self.ratio_success_strength,
                "metric": self.ratio_success_metric, "ratio": ratio, "retention": retention,
                "before_w": before_w, "after_w": float(self._w),
            }
        if self.ratio_direction_mode == "attenuate":
            before_p_norm = float(np.linalg.norm(self._p))
            ratio = _mmes_native_shift_ratio(
                shift, float(self.core.sigma), self._q, self._v,
                float(self.core.gamma), float(self.core.c_a),
            )
            retention = float(1.0 - self.ratio_direction_strength * ratio / (1.0 + ratio))
            if ratio > 0.0:
                self._p *= retention
            direction_transition = {
                "mode": "attenuate_direction", "strength": self.ratio_direction_strength,
                "ratio": ratio, "retention": retention,
                "before_p_norm": before_p_norm, "after_p_norm": float(np.linalg.norm(self._p)),
            }
            transition = ({"mode": "attenuate_both", "success": transition,
                           "direction": direction_transition}
                          if transition is not None else direction_transition)
        self._mean = center.copy()
        self.core.mean = center.copy()
        self._last_commit_shift = shift_norm
        if self._last_commit_shift > 1e-15:
            self._neutralize_success_once = True
            self._pending_mmes_stale_shift_norm = self._last_commit_shift
        result = {
            "applied_norm": self._last_commit_shift,
            "exact": True,
            "next_success_credit": (
                "decay_only" if self._neutralize_success_once else "paired"
            ),
        }
        if transition is not None:
            result["transition"] = transition
        return result

    def configure_guide(self, guide_options: Dict) -> None:
        _configure_native_guide(self.core, guide_options, self.dimension)

    def diagnostic_snapshot(self, commit_direction) -> Dict:
        archive = (
            self._q[0]
            if np.asarray(self._q).ndim == 2 and self._q.shape[0] > 0
            else np.zeros((self.dimension,), dtype=np.float64)
        )
        summary = _memory_summary(
            center=self._mean,
            direction=commit_direction,
            primary=self._p,
            secondary=archive,
            effective_scale_rms=abs(float(self.core.sigma)),
            guide=getattr(self.core, "optimizer_guide_direction", None),
            guide_strength=getattr(self.core, "optimizer_guide_strength", 0.0),
            guide_mix_strength=getattr(self.core, "optimizer_guide_mix_strength", 0.0),
            dimension=self.dimension,
        )
        summary.update({
            "mmes_state_transition_mode": self.mmes_state_transition_mode,
            "mmes_neutralize_success": int(self._packet_mmes_neutralized_updates > 0),
            "mmes_credit_action": (
                "mixed" if self._packet_mmes_neutralized_updates and self._packet_mmes_paired_updates
                else "neutralize_credit" if self._packet_mmes_neutralized_updates
                else "paired" if self._packet_mmes_paired_updates
                else "none"
            ),
            "mmes_y_bak_comparable": (
                0 if self._packet_mmes_incomparable_updates else
                1 if self._packet_mmes_comparable_updates else None
            ),
            "mmes_previous_slot_shift_norm": self._packet_mmes_previous_slot_shift_norm,
            "mmes_stale_credit_pending_at_start": self._packet_mmes_stale_pending_at_start,
            "mmes_stale_source_shift_norm": self._packet_mmes_stale_source_shift_norm,
            "mmes_y_bak_incomparable_updates": self._packet_mmes_incomparable_updates,
            "mmes_y_bak_comparable_updates": self._packet_mmes_comparable_updates,
            "mmes_neutralized_updates": self._packet_mmes_neutralized_updates,
            "mmes_paired_updates": self._packet_mmes_paired_updates,
            "mmes_sigma_before": self._packet_mmes_sigma_before,
            "mmes_sigma_after": float(self.core.sigma),
            "mmes_success_stat_before": self._packet_mmes_success_stat_before,
            "mmes_success_stat_after": float(self._w),
            "mmes_sigma": float(self.core.sigma),
            "mmes_primary_norm": float(np.linalg.norm(self._p)),
            "mmes_secondary_norm": float(np.linalg.norm(archive)),
        })
        return summary

    def result(self) -> Dict:
        best_x = np.asarray(self.core.best_so_far_x, dtype=np.float64).reshape(-1)
        return {
            "best_so_far_x": best_x.copy(),
            "best_so_far_y": float(self.core.best_so_far_y),
            "n_function_evaluations": int(self.reported_evaluations),
            "mean": self._mean.copy(),
            "search_center": self._mean.copy(),
            "sigma": float(self.core.sigma),
            "x": np.asarray(self._x, dtype=np.float64).copy(),
            "y": np.asarray(self._y, dtype=np.float64).copy(),
            "optimizer_anchor_applied": float(self.core.optimizer_anchor_applied),
            "optimizer_anchor_mean_step_norm": float(
                self.core.optimizer_anchor_mean_step_norm
            ),
            "optimizer_anchor_sample_applied": float(
                self.core.optimizer_anchor_sample_applied
            ),
            "optimizer_anchor_dist": float(self.core.optimizer_anchor_dist),
        }


class SepCMAESEventSlotSession(_BaseEventSlotSession):
    optimizer_name = "sepcmaes"

    def __init__(self, problem: Dict, options: Dict):
        super().__init__(problem, options)
        self.ratio_path_mode = str(options.get("cma_sep_ratio_path_mode", "native")).lower()
        if self.ratio_path_mode not in {"native", "step_path", "shape_path", "both_paths"}:
            raise ValueError("Unsupported SepCMAES ratio path mode.")
        self.ratio_path_metric = str(options.get("sepcmaes_ratio_path_metric", "rms")).lower()
        if self.ratio_path_metric not in {"rms", "directional"}:
            raise ValueError("Invalid SepCMAES ratio path metric.")
        self.ratio_path_strength = float(options.get("cma_sep_ratio_path_strength", 1.0))
        if not np.isfinite(self.ratio_path_strength) or not 0.0 <= self.ratio_path_strength <= 1.0:
            raise ValueError("Invalid SepCMAES ratio path strength.")
        wrapper = SepCMAESOpt(problem, options)
        self.core = SEPCMAES(
            wrapper.build_core_problem(), wrapper.build_core_options()
        )
        self.core._boundary_capture = bool(self._boundary_dir)
        self.core.start_time = time.time()
        (
            self._z,
            self._x,
            self._mean,
            self._s,
            self._p,
            self._c,
            self._d,
            self._y,
        ) = self.core.initialize()
        self._initial_candidate_pending = True

    @property
    def population(self) -> int:
        return int(self.core.n_individuals)

    def evolved_sigma(self):
        # SepCMAES core.sigma 每代演化（事件末即演化后 σ）。
        return float(self.core.sigma)

    @property
    def physical_evaluations(self) -> int:
        return int(self.core.n_function_evaluations)

    def advance_to_reported_target(self, target: int) -> Dict:
        target = int(target)
        if target <= self.reported_evaluations or target > self.requested_fes:
            raise ValueError("SepCMAES event-slot target must advance within the event budget.")
        _attach_forensics(self)
        self._begin_packet()
        capture_first = bool(self.options.get("record_native_first_generation", False))
        if self._initial_candidate_pending:
            self._observe_packet_population(
                self._mean.reshape(1, -1),
                np.asarray([self._y[0]], dtype=np.float64),
            )
            self._initial_candidate_pending = False
        while self.reported_evaluations < target and not self._terminated:
            boundary_before = self._mean.copy() if self._boundary_dir else None
            boundary_sigma = float(self.core.sigma) if self._boundary_dir else None
            boundary_scale = (float(np.sqrt(np.mean(np.square(self._d))))
                              if self._boundary_dir else None)
            boundary_replay_before = self.native_state_snapshot() if self._boundary_wants_key() else None
            first_before = (
                self.native_state_snapshot()
                if capture_first and self._packet_native_first_generation is None
                else None
            )
            self._z, self._x, self._y = self.core.iterate(
                self._z,
                self._x,
                self._mean,
                self._d,
                self._y,
            )
            if (
                first_before is not None
                and self.physical_evaluations > first_before["physical_fes"]
            ):
                self._packet_native_first_generation = {
                    "before_iterate": first_before,
                    "z": self._z.tolist(),
                    "x": self._x.tolist(),
                    "y": self._y.tolist(),
                    "generation_before_update": int(self.core._n_generations),
                    "after_update": None,
                }
            self.population_units += 1
            self._last_packet_units += 1
            self._observe_packet_population(self._x, self._y)
            if self.core._check_terminations():
                self._terminated = True
                break
            self._mean, self._s, self._p, self._c, self._d = (
                self.core._update_distribution(
                    self._z,
                    self._x,
                    self._s,
                    self._p,
                    self._c,
                    self._d,
                    self._y,
                    mean_old=self._mean,
                )
            )
            self.core._n_generations += 1
            self.distribution_updates += 1
            if self._boundary_dir:
                self._boundary_note(
                    boundary_before, self._mean, self.core._boundary_raw,
                    self._x, self._y, boundary_sigma, self.core.sigma,
                    boundary_sigma * boundary_scale,
                    replay=({"before": boundary_replay_before,
                             "sample_z": np.copy(self._z),
                             "after": self.native_state_snapshot()}
                            if boundary_replay_before is not None else None),
                )
            if (
                first_before is not None
                and self._packet_native_first_generation is not None
            ):
                self._packet_native_first_generation["after_update"] = (
                    self.native_state_snapshot()
                )
        if (
            self.reported_evaluations < target
            and self._last_packet_best_x is None
        ):
            raise RuntimeError("SepCMAES event-slot session terminated before its target.")
        return self._packet_result()

    def synchronize_center(self, target) -> Dict:
        center = _finite_center(target, self.lower, self.upper, self.dimension)
        shift = center - self._mean
        transition = None
        if self.ratio_path_mode != "native":
            ratio, retention = _ratio_path_retention(
                shift, self.core.sigma, self._d, self.dimension,
                self.ratio_path_strength, metric=self.ratio_path_metric,
                session=self, raw_target=target, clipped_center=center,
            )
            before = {"s": float(np.linalg.norm(self._s)),
                      "p": float(np.linalg.norm(self._p))}
            pre_paths = {"s": np.copy(self._s), "p": np.copy(self._p)}
            if ratio > 0.0:
                if self.ratio_path_mode in {"step_path", "both_paths"}:
                    self._s = retention * self._s
                if self.ratio_path_mode in {"shape_path", "both_paths"}:
                    self._p = retention * self._p
            _forensics_commit(
                self, "computed", self.core.sigma, self._d, None,
                self._mean, target, center, shift,
                ratio=ratio, retention=retention, pre_paths=pre_paths,
                post_paths={"s": np.copy(self._s), "p": np.copy(self._p)},
            )
            transition = {
                "mode": self.ratio_path_mode,
                "strength": self.ratio_path_strength,
                "metric": self.ratio_path_metric,
                "ratio": ratio,
                "retention": retention,
                "before_norm": before,
                "after_norm": {"s": float(np.linalg.norm(self._s)),
                               "p": float(np.linalg.norm(self._p))},
            }
        self._mean = center.copy()
        self.core.mean = center.copy()
        self._last_commit_shift = float(np.linalg.norm(shift))
        result = {"applied_norm": self._last_commit_shift, "exact": True}
        if transition is not None:
            result["transition"] = transition
        return result

    def configure_guide(self, guide_options: Dict) -> None:
        _configure_native_guide(self.core, guide_options, self.dimension)

    def diagnostic_snapshot(self, commit_direction) -> Dict:
        effective_scale = float(
            abs(float(self.core.sigma))
            * np.sqrt(np.mean(np.square(self._d)))
        )
        return _memory_summary(
            center=self._mean,
            direction=commit_direction,
            primary=self._p,
            secondary=self._d * self._s,
            effective_scale_rms=effective_scale,
            guide=getattr(self.core, "optimizer_guide_direction", None),
            guide_strength=getattr(
                self.core, "optimizer_guide_strength", 0.0
            ),
            guide_mix_strength=getattr(
                self.core, "optimizer_guide_mix_strength", 0.0
            ),
            dimension=self.dimension,
        )

    def native_state_snapshot(self) -> Dict:
        return {
            "optimizer": "sepcmaes",
            "mean": self._mean.tolist(),
            "core_mean": np.asarray(self.core.mean, dtype=np.float64).tolist(),
            "s": self._s.tolist(),
            "p": self._p.tolist(),
            "c": self._c.tolist(),
            "d": self._d.tolist(),
            "sigma": float(self.core.sigma),
            "physical_fes": int(self.physical_evaluations),
            "distribution_updates": int(self.distribution_updates),
        }

    def packet_native_first_generation_snapshot(self):
        return getattr(self, "_packet_native_first_generation", None)

    def result(self) -> Dict:
        best_x = np.asarray(self.core.best_so_far_x, dtype=np.float64).reshape(-1)
        return {
            "best_so_far_x": best_x.copy(),
            "best_so_far_y": float(self.core.best_so_far_y),
            "n_function_evaluations": int(self.reported_evaluations),
            "mean": self._mean.copy(),
            "search_center": self._mean.copy(),
            "sigma": float(self.core.sigma),
            "x": np.asarray(self._x, dtype=np.float64).copy(),
            "y": np.asarray(self._y, dtype=np.float64).copy(),
            "optimizer_guide_internal_applied": float(
                self.core.optimizer_guide_internal_applied
            ),
            "optimizer_guide_internal_mean_step_norm": float(
                self.core.optimizer_guide_internal_mean_step_norm
            ),
            "optimizer_guide_internal_alignment": float(
                self.core.optimizer_guide_internal_alignment
            ),
            "optimizer_anchor_applied": float(self.core.optimizer_anchor_applied),
            "optimizer_anchor_mean_step_norm": float(
                self.core.optimizer_anchor_mean_step_norm
            ),
            "optimizer_anchor_sample_applied": float(
                self.core.optimizer_anchor_sample_applied
            ),
            "optimizer_anchor_dist": float(self.core.optimizer_anchor_dist),
            "optimizer_numeric_telemetry": self.core.numeric_telemetry.result_records(),
            "optimizer_numeric_guard_counters": self.core.numeric_telemetry.result_counters(),
        }


class VKDEventSlotSession(_BaseEventSlotSession):
    optimizer_name = "vkd"

    def __init__(self, problem: Dict, options: Dict):
        super().__init__(problem, options)
        self.wrapper = VKD(problem, options)
        self._lam = max(2, int(self.wrapper.n_individuals))
        self._restart_id = 0
        self._completed_restart_evals = 0
        self._last_mean = self.wrapper.mean.copy()
        self._last_sigma = float(self.wrapper.sigma)
        self._core = None
        self._best_x = self.wrapper.mean.copy()
        self._best_y = np.inf
        self._final_x = np.tile(self.wrapper.mean, (self._lam, 1))
        self._final_y = np.full((self._lam,), np.inf, dtype=np.float64)
        self._last_population_observed = False
        self._guide_options = dict(options)
        self.ratio_ps_mode = str(options.get("vkd_ratio_ps_mode", "native")).lower()
        if self.ratio_ps_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid VKD ratio ps mode.")
        self.ratio_ps_strength = float(options.get("vkd_ratio_ps_strength", 0.1))
        if not np.isfinite(self.ratio_ps_strength) or not 0.0 <= self.ratio_ps_strength <= 1.0:
            raise ValueError("Invalid VKD ratio ps strength.")
        global_state = np.random.get_state()
        try:
            np.random.seed(int(self.wrapper.seed_rng))
            self._rng_state = np.random.get_state()
        finally:
            np.random.set_state(global_state)

    @property
    def population(self) -> int:
        return int(self._lam if self._core is None else self._core.lam)

    def evolved_sigma(self):
        # VKD：core 存活取 core.sigma，重启间隙取 _last_sigma（既有采集点）。
        core = self._core
        return float(self._last_sigma if core is None else core.sigma)

    @property
    def physical_evaluations(self) -> int:
        current = 0 if self._core is None else int(self._core.neval)
        return int(self._completed_restart_evals + current)

    def _begin_packet(self) -> None:
        super()._begin_packet()
        if self.options.get("vkd_record_detail", False):
            self._vkd_generations = []

    def state_trace_snapshot(self) -> Dict:
        """Detached arrays; never initialize a core or evaluate an objective."""
        core = self._core
        result = {
            "core_available": core is not None,
            "restart_id": int(self._restart_id),
            "distribution_updates": int(self.distribution_updates),
            "physical_evaluations": int(self.physical_evaluations),
            "reported_evaluations": int(self.reported_evaluations),
            "mean": np.asarray(self._last_mean if core is None else core.xmean).tolist(),
            "sigma": float(self._last_sigma if core is None else core.sigma),
        }
        if core is not None:
            for field in ("ps", "dx", "dz", "pc", "D", "V", "S", "k", "k_active", "neval", "flg_injection"):
                value = getattr(core, field, None)
                result[field] = np.asarray(value).tolist() if value is not None else None
            result["vkd_boundary_update_mode"] = str(
                getattr(core, "vkd_boundary_update_mode", "native")
            )
        return result

    def _record_vkd_generation(self, before: Dict, error=None) -> None:
        # A recording failure is evidence, not a reason to alter the optimizer.
        try:
            core = self._core
            record = {
                "before": before, "after": self.state_trace_snapshot(),
                "candidates": np.asarray(core.arx).tolist(),
                "fitness": np.asarray(core.arf).tolist(),
                "tpa_active": bool(before.get("flg_injection", False)),
                "alpha": float(core.diag_last_alpha) if before.get("flg_injection") else None,
                "hsig": bool(core.diag_last_hsig),
                "sigma_consumer": float(core.diag_last_sigma_consumer),
                "shape_consumer": float(core.diag_last_shape_consumer),
                "mode": str(core.vkd_ps_outlet_mode),
                "native_evals_delta": int(core.neval) - int(before.get("neval", 0)),
                "raw_candidate_finite": getattr(core, "diag_raw_candidate_finite", None),
                "clip_coordinates": getattr(core, "diag_clip_coordinates", None),
                "candidate_coordinates": getattr(core, "diag_candidate_coordinates", None),
                "generation_trace": list(getattr(self, "_vkd_generations", [])),
                "failure": str(error) if error is not None else None,
            }
        except Exception as exc:
            record = {"capture_error": str(exc), "failure": str(error) if error is not None else None}
        if not hasattr(self, "_vkd_generations"):
            self._vkd_generations = []
        self._vkd_generations.append(record)

    def _core_options(self, remaining: int) -> Dict:
        opts = default_option(
            self.dimension,
            int(remaining),
            lb=self.lower,
            ub=self.upper,
        )
        opts["lam"] = int(self._lam)
        opts["seed_rng"] = int(self.wrapper.seed_rng + self._restart_id)
        opts["batch_evaluation"] = False
        opts["k_init"] = int(
            max(0, min(self.wrapper.k_init, self.dimension - 1))
        )
        opts["kmax"] = int(
            max(0, min(self.wrapper.kmax, self.dimension - 1))
        )
        if opts["k_init"] > opts["kmax"]:
            opts["k_init"] = int(opts["kmax"])
        for key in (
            "kmin",
            "k_inc_cond",
            "k_dec_cond",
            "k_adapt_factor",
            "factor_sigma_slope",
            "factor_diag_slope",
            "cs",
            "ds",
            "vkd_ps_outlet_mode",
            "vkd_boundary_update_mode",
            "vkd_ratio_ps_mode",
            "vkd_ratio_ps_strength",
            "vkd_record_detail",
            "optimizer_guide_enable",
            "optimizer_guide_direction",
            "optimizer_guide_strength",
            "optimizer_guide_mix_strength",
            "optimizer_anchor_enable",
            "optimizer_anchor_point",
            "optimizer_anchor_strength",
            "optimizer_anchor_mix_strength",
            # Diagnostic-only jump window plumbing for the VKD core.  Without these
            # keys the core cannot create its bounded window, so the capture would
            # silently produce nothing.  They stay inert unless enable=1.
            "optimizer_numeric_forensics_enable",
            "optimizer_numeric_forensics_dir",
            "optimizer_numeric_forensics_context",
            "optimizer_numeric_forensics_jump_log10",
            "optimizer_numeric_forensics_jump_max_windows",
            "optimizer_numeric_forensics_jump_early_write",
            "optimizer_numeric_forensics_jump_alpha_gate",
            "optimizer_numeric_forensics_target_function",
            "optimizer_numeric_forensics_target_seed",
            "optimizer_numeric_forensics_target_agent",
            "vkd_origin_trace_dir",
            "vkd_origin_trace_identity",
            "vkd_origin_objective",
        ):
            if key in self._guide_options:
                opts[key] = self._guide_options[key]
        return opts

    def _ensure_core(self) -> None:
        if self._core is not None:
            return
        remaining = int(self.requested_fes - self._completed_restart_evals)
        if remaining <= 0:
            self._terminated = True
            return
        self._core = VkdCma(
            self.wrapper._fitness_single,
            self._last_mean,
            self._last_sigma,
            np.tile(self._last_mean, (self._lam, 1)),
            **self._core_options(remaining),
        )
        self._core._boundary_capture = bool(self._boundary_dir)

    def _run_one_population(self):
        self._ensure_core()
        if self._core is None:
            raise RuntimeError("VKD event-slot session has no remaining population.")
        if getattr(self._core, "_origin_trace", None) is not None:
            self._core._origin_slot_id = int(getattr(self, "_origin_slot_id", -1))
        capture = bool(self.options.get("vkd_record_detail", False))
        boundary_before = np.copy(self._core.xmean) if self._boundary_dir else None
        boundary_sigma = float(self._core.sigma) if self._boundary_dir else None
        boundary_scale = (
            float(np.sqrt(np.mean(np.square(self._core.D)
                * (1.0 + np.sum(self._core.S[:self._core.k_active, None]
                                 * np.square(self._core.V[:self._core.k_active]), axis=0)))))
            if self._boundary_dir else None
        )
        boundary_replay_before = self.state_trace_snapshot() if self._boundary_wants_key() else None
        before = self.state_trace_snapshot() if capture else None
        global_state = np.random.get_state()
        try:
            np.random.set_state(self._rng_state)
            self._core._onestep()
            self._rng_state = np.random.get_state()
        except Exception as exc:
            if capture:
                self._record_vkd_generation(before, error=exc)
            raise
        finally:
            np.random.set_state(global_state)
        self.population_units += 1
        self.distribution_updates += 1
        self._last_packet_units += 1
        self._final_x = np.asarray(self._core.arx, dtype=np.float64).copy()
        self._final_y = np.asarray(self._core.arf, dtype=np.float64).copy()
        if self._boundary_dir:
            self._boundary_note(
                boundary_before, self._core.xmean, self._core._boundary_raw,
                self._final_x, self._final_y, boundary_sigma, self._core.sigma,
                boundary_sigma * boundary_scale,
                extra={"alpha": self._core.diag_last_alpha,
                       "ps": self._core.ps, "hsig": self._core.diag_last_hsig},
                replay=({"before": boundary_replay_before,
                         "after": self.state_trace_snapshot()}
                        if boundary_replay_before is not None else None),
            )
        self._last_population_observed = False
        if capture:
            self._record_vkd_generation(before)
        satisfied, condition = self._core._check()
        if capture:
            self._vkd_generations[-1]["termination_condition"] = str(condition)
        if not satisfied:
            return

        self._observe_packet_population(self._final_x, self._final_y)
        self._last_population_observed = True
        index = int(np.argmin(self._final_y))
        value = float(self._final_y[index])
        if value < self._best_y:
            self._best_y = value
            self._best_x = self._final_x[index].copy()
        self._last_mean = np.asarray(self._core.xmean, dtype=np.float64).copy()
        self._last_sigma = float(self._core.sigma)
        if condition == "maxeval":
            self._terminated = True
            return
        self._completed_restart_evals += int(self._core.neval)
        self._lam = min(int(self._lam * 2), 100)
        self._restart_id += 1
        self._core = None

    def advance_to_reported_target(self, target: int) -> Dict:
        target = int(target)
        if target <= self.reported_evaluations or target > self.requested_fes:
            raise ValueError("VKD event-slot target must advance within the event budget.")
        self._begin_packet()
        while self.reported_evaluations < target and not self._terminated:
            self._run_one_population()
        if (
            self.reported_evaluations < target
            and self._last_packet_best_x is None
        ):
            raise RuntimeError("VKD event-slot session terminated before its target.")
        if not self._last_population_observed:
            self._observe_packet_population(self._final_x, self._final_y)
            self._last_population_observed = True
        return self._packet_result()

    def synchronize_center(self, target) -> Dict:
        center = _finite_center(target, self.lower, self.upper, self.dimension)
        current = self._last_mean if self._core is None else self._core.xmean
        shift = center - np.asarray(current, dtype=np.float64)
        transition = None
        if self.ratio_ps_mode == "attenuate":
            ratio, retention = 0.0, 1.0
            before_ps = None if self._core is None else float(self._core.ps)
            if self._core is not None and np.any(shift != 0.0):
                squared = float(self._core._mahalanobis_square_norm(shift))
                sigma = float(self._core.sigma)
                if not np.isfinite(squared) or squared < -1e-10 or not np.isfinite(sigma) or sigma <= 0.0:
                    _dump_nonfinite_geometry("vkd_geometry", self, shift)
                    raise ValueError("Invalid VKD search geometry for center relocation.")
                ratio = float(np.sqrt(max(0.0, squared)) / (np.sqrt(self.dimension) * sigma))
                if not np.isfinite(ratio):
                    _dump_nonfinite_geometry("vkd_ratio", self, shift)
                    raise ValueError("Non-finite VKD relative center relocation.")
                retention = float(1.0 - self.ratio_ps_strength * ratio / (1.0 + ratio))
                self._core.ps *= retention
            transition = {
                "mode": self.ratio_ps_mode, "strength": self.ratio_ps_strength,
                "ratio": ratio, "retention": retention,
                "before_ps": before_ps,
                "after_ps": None if self._core is None else float(self._core.ps),
                "core_available": self._core is not None,
            }
        if self._core is None:
            self._last_mean = center.copy()
        else:
            self._core.xmean = center.copy()
        self._last_commit_shift = float(np.linalg.norm(shift))
        result = {"applied_norm": self._last_commit_shift, "exact": True}
        if transition is not None:
            result["transition"] = transition
        return result

    def configure_guide(self, guide_options: Dict) -> None:
        opts = dict(guide_options or {})
        self._guide_options.update(opts)
        if self._core is not None:
            _configure_native_guide(self._core, opts, self.dimension)

    def diagnostic_snapshot(self, commit_direction) -> Dict:
        core = self._core
        if core is None:
            center = self._last_mean
            primary = np.zeros((self.dimension,), dtype=np.float64)
            axis_variance = np.ones((self.dimension,), dtype=np.float64)
            sigma = float(self._last_sigma)
            guide = self._guide_options.get("optimizer_guide_direction")
            guide_strength = self._guide_options.get(
                "optimizer_guide_strength", 0.0
            )
            guide_mix_strength = self._guide_options.get(
                "optimizer_guide_mix_strength", 0.0
            )
        else:
            center = np.asarray(core.xmean, dtype=np.float64)
            primary = np.asarray(core.pc, dtype=np.float64)
            axis_variance = np.square(
                np.asarray(core.D, dtype=np.float64)
            )
            active = int(max(0, getattr(core, "k_active", 0)))
            if active > 0:
                shape = 1.0 + np.sum(
                    np.asarray(core.S[:active], dtype=np.float64)[:, None]
                    * np.square(
                        np.asarray(core.V[:active], dtype=np.float64)
                    ),
                    axis=0,
                )
                axis_variance = axis_variance * np.maximum(shape, 0.0)
            sigma = float(core.sigma)
            guide = getattr(core, "optimizer_guide_direction", None)
            guide_strength = getattr(
                core, "optimizer_guide_strength", 0.0
            )
            guide_mix_strength = getattr(
                core, "optimizer_guide_mix_strength", 0.0
            )
        effective_scale = float(
            abs(sigma) * np.sqrt(np.mean(np.maximum(axis_variance, 0.0)))
        )
        summary = _memory_summary(
            center=center,
            direction=commit_direction,
            primary=primary,
            # VKD's TPA accumulator is scalar; do not pretend it is a
            # direction vector comparable to CMA evolution paths.
            secondary=np.zeros((self.dimension,), dtype=np.float64),
            effective_scale_rms=effective_scale,
            guide=guide,
            guide_strength=guide_strength,
            guide_mix_strength=guide_mix_strength,
            dimension=self.dimension,
        )
        if not self.options.get("vkd_record_state", False):
            return summary
        if core is not None:
            summary.update({
                "vkd_core_available": 1,
                "vkd_sigma": float(core.sigma),
                "vkd_ps_outlet_mode": str(getattr(core, "vkd_ps_outlet_mode", "native")),
                "vkd_boundary_update_mode": str(getattr(core, "vkd_boundary_update_mode", "native")),
                "vkd_ps": float(getattr(core, "ps", 0.0)),
                "vkd_alpha": float(getattr(core, "diag_last_alpha", 0.0)),
                "vkd_hsig": int(bool(getattr(core, "diag_last_hsig", True))),
                "vkd_sigma_consumer": float(getattr(core, "diag_last_sigma_consumer", getattr(core, "ps", 0.0))),
                "vkd_shape_consumer": float(getattr(core, "diag_last_shape_consumer", getattr(core, "ps", 0.0))),
                "vkd_pc_norm": float(np.linalg.norm(np.asarray(core.pc, dtype=np.float64))),
                "vkd_D_rms": float(np.sqrt(np.mean(np.square(np.asarray(core.D, dtype=np.float64))))),
                "vkd_S_norm": float(np.linalg.norm(np.asarray(core.S, dtype=np.float64))),
                "vkd_V_norm": float(np.linalg.norm(np.asarray(core.V, dtype=np.float64))),
                "vkd_state_finite": int(all(np.all(np.isfinite(np.asarray(getattr(core, name)))) for name in ("xmean", "sigma", "ps", "dx", "dz", "pc", "D", "V", "S"))),
                "vkd_candidate_finite": int(np.all(np.isfinite(np.asarray(core.arx, dtype=np.float64)))),
                "vkd_fitness_finite": int(np.all(np.isfinite(np.asarray(core.arf, dtype=np.float64)))),
                "vkd_objective_calls": int(self.physical_evaluations),
            })
        else:
            summary.update({
                "vkd_core_available": 0,
                "vkd_sigma": float(self._last_sigma),
                "vkd_ps_outlet_mode": str(self._guide_options.get("vkd_ps_outlet_mode", "native")),
                "vkd_boundary_update_mode": str(self._guide_options.get("vkd_boundary_update_mode", "native")),
                "vkd_ps": float("nan"), "vkd_alpha": float("nan"), "vkd_hsig": -1,
                "vkd_sigma_consumer": float("nan"), "vkd_shape_consumer": float("nan"),
                "vkd_pc_norm": float("nan"), "vkd_D_rms": float("nan"),
                "vkd_S_norm": float("nan"), "vkd_V_norm": float("nan"),
                "vkd_state_finite": float("nan"), "vkd_candidate_finite": float("nan"),
                "vkd_fitness_finite": float("nan"), "vkd_objective_calls": int(self.physical_evaluations),
            })
        return summary

    def result(self) -> Dict:
        core = self._core
        current_mean = (
            self._last_mean
            if core is None
            else np.asarray(core.xmean, dtype=np.float64)
        )
        current_sigma = self._last_sigma if core is None else float(core.sigma)
        return {
            "best_so_far_x": np.asarray(self._best_x, dtype=np.float64).copy(),
            "best_so_far_y": float(self._best_y),
            "n_function_evaluations": int(self.reported_evaluations),
            "mean": np.asarray(current_mean, dtype=np.float64).copy(),
            "search_center": np.asarray(
                current_mean, dtype=np.float64
            ).copy(),
            "sigma": float(current_sigma),
            "x": self._final_x.copy(),
            "y": self._final_y.copy(),
            "n_individuals": int(self.population),
            "lam": int(self.population),
            "k": int(getattr(core, "k", self.wrapper.k_init)) if core is not None else int(self.wrapper.k_init),
            "k_active": int(getattr(core, "k_active", 0)) if core is not None else 0,
            "optimizer_anchor_applied": float(
                getattr(core, "optimizer_anchor_applied", 0.0)
            ),
            "optimizer_anchor_mean_step_norm": float(
                getattr(core, "optimizer_anchor_mean_step_norm", 0.0)
            ),
            "optimizer_anchor_sample_applied": float(
                getattr(core, "optimizer_anchor_sample_applied", 0.0)
            ),
            "optimizer_anchor_dist": float(
                getattr(core, "optimizer_anchor_dist", 0.0)
            ),
            "vkd_boundary_update_mode": str(
                getattr(core, "vkd_boundary_update_mode", self._guide_options.get("vkd_boundary_update_mode", "native"))
            ),
        }


def create_event_slot_session(
    optimizer_name: str,
    problem: Dict,
    options: Dict,
) -> _BaseEventSlotSession:
    name = str(optimizer_name).strip().lower()
    classes = {
        "mmes": MMESEventSlotSession,
        "vkd": VKDEventSlotSession,
        "cmaes": CMAESEventSlotSession,
        "sepcmaes": SepCMAESEventSlotSession,
    }
    if name not in classes:
        raise ValueError(f"Unsupported event-slot optimizer: {optimizer_name}.")
    return classes[name](problem, options)


def build_multiagent_event_slot_plans(
    optimizer_names: List[str],
    requested_fes: List[int],
    populations: List[int],
    slots: int,
    dimensions: List[int] = None,
) -> List[EventSlotPlan]:
    if not (
        len(optimizer_names) == len(requested_fes) == len(populations)
    ):
        raise ValueError("Event-slot multi-agent plan fields must have equal length.")
    if dimensions is None:
        dimensions = [None for _ in optimizer_names]
    if len(dimensions) != len(optimizer_names):
        raise ValueError(
            "Event-slot multi-agent dimensions must match optimizer names."
        )
    return [
        build_event_slot_plan(
            name,
            budget,
            population,
            slots,
            dimension=dimension,
        )
        for name, budget, population, dimension in zip(
            optimizer_names, requested_fes, populations, dimensions
        )
    ]

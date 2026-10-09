"""Strict-local persistent SepCMAES bank helpers.

The objective-split environment owns the bank.  Workers receive and return
detached state dictionaries so serial and ProcessPool execution have identical
lifecycle semantics.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, Optional, Tuple

import numpy as np

from optimizers.unified_opt import SepCMAESOpt, StatefulSepCMAES


PERSISTENT_SEPCMAES_SIGNATURE_VERSION = 2
PERSISTENT_SEPCMAES_BANK_STATE_VERSION = 1


def _build_persistent_core_options(wrapper: SepCMAESOpt) -> Dict:
    """Build a bounded, counter-audited numeric lifecycle for persistent use."""
    core_options = wrapper.build_core_options()
    core_options["stateful_evaluate_initial_mean"] = False
    # The legacy option name predates persistent experts.  For a state that can
    # live for thousands of generations this is an invariant, not an optional
    # guide feature: keep sigma finite and bounded by the search-domain scale.
    core_options["optimizer_guide_numeric_guard"] = True
    core_options["optimizer_numeric_counter_enable"] = True
    exp_clip = float(
        core_options.get("optimizer_guide_sigma_exp_clip", 20.0)
    )
    sigma_clip_ratio = float(
        core_options.get("optimizer_guide_sigma_clip_ratio", 0.5)
    )
    if not np.isfinite(exp_clip) or exp_clip <= 0.0:
        raise ValueError(
            "Persistent SepCMAES requires a finite positive sigma exp clip."
        )
    if not np.isfinite(sigma_clip_ratio) or sigma_clip_ratio <= 0.0:
        raise ValueError(
            "Persistent SepCMAES requires a finite positive sigma clip ratio."
        )
    core_options["optimizer_guide_sigma_exp_clip"] = exp_clip
    core_options["optimizer_guide_sigma_clip_ratio"] = sigma_clip_ratio
    return core_options


def build_persistent_sepcmaes_signature(wrapper: SepCMAESOpt) -> Dict:
    """Return fields whose change makes a saved distribution incompatible."""
    core_options = _build_persistent_core_options(wrapper)
    probe = StatefulSepCMAES(
        wrapper.build_core_problem(),
        core_options,
    )
    probe.initialize()
    return {
        "version": int(PERSISTENT_SEPCMAES_SIGNATURE_VERSION),
        "dimension": int(wrapper.ndim_problem),
        "lower_boundary": wrapper.lower_boundary.astype(
            np.float64, copy=True
        ).tolist(),
        "upper_boundary": wrapper.upper_boundary.astype(
            np.float64, copy=True
        ).tolist(),
        "n_individuals": int(core_options["n_individuals"]),
        "n_parents": int(core_options["n_parents"]),
        "initial_sigma": float(core_options["sigma"]),
        "c_s": float(probe.core.c_s),
        "c_cov": float(probe.core.c_cov),
        "numeric_guard": bool(
            core_options["optimizer_guide_numeric_guard"]
        ),
        "numeric_counter_enable": bool(
            core_options["optimizer_numeric_counter_enable"]
        ),
        "sigma_exp_clip": float(
            core_options["optimizer_guide_sigma_exp_clip"]
        ),
        "sigma_clip_ratio": float(
            core_options["optimizer_guide_sigma_clip_ratio"]
        ),
        "nonfinite_penalty": float(wrapper.nonfinite_penalty),
    }


def _persistent_result(
    stateful: StatefulSepCMAES,
    tranche_result: Dict,
    evaluations: int,
    elapsed: float,
) -> Dict:
    advance_x = tranche_result.get("advance_best_x")
    advance_y = float(tranche_result.get("advance_best_y", np.inf))
    if advance_x is None or not np.isfinite(advance_y):
        raise RuntimeError(
            "Persistent SepCMAES tranche completed without a finite "
            "tranche-local best."
        )
    core = stateful.core
    return {
        # The environment must propose from this tranche, not from the
        # optimizer's lifetime incumbent at an obsolete center.
        "best_so_far_x": np.asarray(
            advance_x, dtype=np.float64
        ).copy(),
        "best_so_far_y": advance_y,
        # Report physical calls in this event; lifetime counts remain telemetry.
        "n_function_evaluations": int(evaluations),
        "time_function_evaluations": float(elapsed),
        "x": np.asarray(advance_x, dtype=np.float64).copy(),
        "y": np.asarray([advance_y], dtype=np.float64),
        "mean": np.asarray(
            tranche_result["mean"], dtype=np.float64
        ).copy(),
        "sigma": float(tranche_result["sigma"]),
        "optimizer_guide_internal_applied": float(
            getattr(core, "optimizer_guide_internal_applied", 0.0)
        ),
        "optimizer_guide_internal_mean_step_norm": float(
            getattr(core, "optimizer_guide_internal_mean_step_norm", 0.0)
        ),
        "optimizer_guide_internal_alignment": float(
            getattr(core, "optimizer_guide_internal_alignment", 0.0)
        ),
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
        "optimizer_numeric_telemetry": list(
            core.numeric_telemetry.result_records()
        ),
        "optimizer_numeric_guard_counters": dict(
            core.numeric_telemetry.result_counters()
        ),
        "persistent_lifetime_function_evaluations": int(
            tranche_result["n_function_evaluations"]
        ),
        "persistent_lifetime_generations": int(
            tranche_result["n_generations"]
        ),
    }


def run_persistent_sepcmaes(
    problem: Dict,
    options: Dict,
    x_base,
    previous_state: Optional[Dict],
    previous_signature: Optional[Dict],
    recenter_max_shift: float,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Advance one persistent expert by a complete-generation tranche."""
    wrapper = SepCMAESOpt(problem, dict(options))
    signature = build_persistent_sepcmaes_signature(wrapper)
    has_previous = previous_state is not None
    compatible = bool(
        has_previous
        and previous_signature is not None
        and previous_signature == signature
    )
    config_reset = bool(has_previous and not compatible)

    core_options = _build_persistent_core_options(wrapper)
    stateful = StatefulSepCMAES(
        wrapper.build_core_problem(),
        core_options,
    )
    if compatible:
        stateful.import_state(copy.deepcopy(previous_state))

    recenter = stateful.recenter(
        np.asarray(x_base, dtype=np.float64),
        max_shift=float(recenter_max_shift),
    )
    evaluations = int(wrapper.max_function_evaluations)
    calls_before = int(stateful.n_function_evaluations)
    started = time.time()
    tranche_result = stateful.advance_evaluations(evaluations)
    elapsed = float(time.time() - started)
    calls_after = int(stateful.n_function_evaluations)
    if calls_after - calls_before != evaluations:
        raise RuntimeError(
            "Persistent SepCMAES physical-call accounting mismatch: "
            f"{calls_after} - {calls_before} != {evaluations}."
        )

    result = _persistent_result(
        stateful,
        tranche_result,
        evaluations=evaluations,
        elapsed=elapsed,
    )
    telemetry = {
        "active": True,
        "fresh": bool(not compatible),
        "resumed": bool(compatible),
        "config_reset": config_reset,
        "recenter_requested_norm": float(recenter["requested_norm"]),
        "recenter_applied_norm": float(recenter["applied_norm"]),
        "recenter_remaining_norm": float(recenter["remaining_norm"]),
        "lifetime_function_evaluations": int(
            tranche_result["n_function_evaluations"]
        ),
        "lifetime_generations": int(tranche_result["n_generations"]),
        "numeric_guard": True,
        "sigma_exp_clip": float(
            core_options["optimizer_guide_sigma_exp_clip"]
        ),
        "sigma_clip_ratio": float(
            core_options["optimizer_guide_sigma_clip_ratio"]
        ),
        "sigma_max": float(
            stateful.core._optimizer_guide_sigma_max()
        ),
        "numeric_guard_counters": dict(
            stateful.core.numeric_telemetry.result_counters()
        ),
    }
    return (
        result,
        stateful.export_state(),
        copy.deepcopy(signature),
        telemetry,
    )


def validate_persistent_sepcmaes_snapshot(
    state: Dict,
    signature: Dict,
) -> None:
    """Validate one detached bank entry without objective evaluations."""
    if not isinstance(state, dict):
        raise TypeError("Persistent SepCMAES state must be a dictionary.")
    if not isinstance(signature, dict):
        raise TypeError("Persistent SepCMAES signature must be a dictionary.")
    if int(signature.get("version", -1)) != PERSISTENT_SEPCMAES_SIGNATURE_VERSION:
        raise ValueError("Unsupported persistent SepCMAES signature version.")
    dim = int(signature.get("dimension", -1))
    lower = np.asarray(signature.get("lower_boundary"), dtype=np.float64)
    upper = np.asarray(signature.get("upper_boundary"), dtype=np.float64)
    if (
        dim <= 0
        or lower.shape != (dim,)
        or upper.shape != (dim,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Persistent SepCMAES signature bounds are invalid.")
    n_individuals = int(signature.get("n_individuals", -1))
    n_parents = int(signature.get("n_parents", -1))
    initial_sigma = float(signature.get("initial_sigma", np.nan))
    c_s = float(signature.get("c_s", np.nan))
    c_cov = float(signature.get("c_cov", np.nan))
    numeric_guard = bool(signature.get("numeric_guard", False))
    numeric_counter_enable = bool(
        signature.get("numeric_counter_enable", False)
    )
    sigma_exp_clip = float(signature.get("sigma_exp_clip", np.nan))
    sigma_clip_ratio = float(signature.get("sigma_clip_ratio", np.nan))
    nonfinite_penalty = float(
        signature.get("nonfinite_penalty", np.nan)
    )
    if (
        n_individuals < 2
        or n_parents < 1
        or n_parents > n_individuals
        or not np.isfinite(initial_sigma)
        or initial_sigma <= 0.0
        or not np.isfinite(c_s)
        or c_s <= 0.0
        or not np.isfinite(c_cov)
        or c_cov <= 0.0
        or not numeric_guard
        or not numeric_counter_enable
        or not np.isfinite(sigma_exp_clip)
        or sigma_exp_clip <= 0.0
        or not np.isfinite(sigma_clip_ratio)
        or sigma_clip_ratio <= 0.0
        or not np.isfinite(nonfinite_penalty)
    ):
        raise ValueError(
            "Persistent SepCMAES signature optimizer fields are invalid."
        )

    metadata = state.get("metadata", {}) if isinstance(state, dict) else {}
    for key in ("dimension", "n_individuals", "n_parents"):
        if int(metadata.get(key, -1)) != int(signature.get(key, -2)):
            raise ValueError(
                f"Persistent SepCMAES state/signature mismatch for {key}."
            )
    for key, expected in (
        ("lower_boundary", lower),
        ("upper_boundary", upper),
    ):
        value = np.asarray(metadata.get(key), dtype=np.float64)
        if value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError(
                f"Persistent SepCMAES state/signature mismatch for {key}."
            )
    for key, signature_key in (
        ("c_s", "c_s"),
        ("c_cov", "c_cov"),
        ("optimizer_guide_sigma_exp_clip", "sigma_exp_clip"),
        ("optimizer_guide_sigma_clip_ratio", "sigma_clip_ratio"),
    ):
        expected = signature.get(signature_key)
        if expected is None or not np.isclose(
            float(metadata.get(key, np.nan)),
            float(expected),
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError(
                f"Persistent SepCMAES state/signature mismatch for {key}."
            )
    for key, signature_key in (
        ("optimizer_guide_numeric_guard", "numeric_guard"),
        ("optimizer_numeric_counter_enable", "numeric_counter_enable"),
    ):
        if bool(metadata.get(key, False)) != bool(
            signature.get(signature_key, False)
        ):
            raise ValueError(
                f"Persistent SepCMAES state/signature mismatch for {key}."
            )

    def no_eval(x):
        arr = np.asarray(x, dtype=np.float64)
        return np.zeros((arr.reshape(-1, dim).shape[0],), dtype=np.float64)

    verifier = StatefulSepCMAES(
        {
            "fitness_function": no_eval,
            "ndim_problem": dim,
            "lower_boundary": lower,
            "upper_boundary": upper,
        },
        {
            "max_function_evaluations": 0,
            "seed_rng": 0,
            "mean": np.asarray(
                state.get("arrays", {}).get("mean"), dtype=np.float64
            ),
            "sigma": float(signature["initial_sigma"]),
            "n_individuals": n_individuals,
            "n_parents": n_parents,
            "c_s": c_s,
            "c_cov": c_cov,
            "optimizer_guide_numeric_guard": numeric_guard,
            "optimizer_numeric_counter_enable": numeric_counter_enable,
            "optimizer_guide_sigma_exp_clip": sigma_exp_clip,
            "optimizer_guide_sigma_clip_ratio": sigma_clip_ratio,
            "is_restart": False,
            "verbose": False,
            "stateful_evaluate_initial_mean": False,
        },
    )
    verifier.import_state(copy.deepcopy(state))

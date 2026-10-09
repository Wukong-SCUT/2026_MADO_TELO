"""Commit-conditioned credit for a persistent strict-local SepCMAES state.

The optimizer proposes from agent-local objective evaluations.  The
target-block field then produces the state that is actually committed after
explicit graph communication.  Under commit lock, that cooperative commit is
the next optimizer center, so the optimizer's paths and scale must not keep
full credit for a proposal that the cooperative layer rejected or shortened.

This module is deliberately objective-free and communication-free.  It only
reconciles one detached optimizer snapshot with three states already local to
the receiving agent: previous center, local proposal, and actual commit.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import numpy as np


PERSISTENT_SEPCMAES_COMMIT_CREDIT_MODES = {
    "off",
    "path",
    "scale",
    "joint",
}


def _finite_vector(name: str, value, dimension: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (dimension,):
        raise ValueError(
            f"{name} has shape {array.shape}, expected {(dimension,)}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity.")
    return array.copy()


def _finite_norm(value: np.ndarray) -> float:
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm):
        raise ValueError("Commit-credit vector norm is non-finite.")
    return norm


def reconcile_persistent_sepcmaes_commit(
    state: Dict,
    *,
    previous_center,
    proposal,
    committed,
    mode: str,
    copy_state: bool = True,
) -> Tuple[Dict, Dict]:
    """Return a reconciled detached snapshot and objective-free telemetry.

    ``copy_state=True`` keeps this helper functional for isolated checks.
    The owning environment may pass ``False`` after a worker has returned a
    detached state, avoiding another copy of lifetime histories.

    Path credit is continuous:

        max(0, cos(proposal_step, commit_step))
        * min(1, ||commit_step|| / ||proposal_step||).

    It scales both SepCMAES evolution paths: ``s`` (step-size path) and ``p``
    (covariance path).  A zero proposal or zero commit provides no evidence
    that the optimizer path caused the accepted transition and therefore
    receives zero path credit.

    Scale credit compares the committed per-coordinate RMS displacement with
    the current RMS effective axis standard deviation ``sigma * d``.  It can
    only contract ``sigma``; covariance shape ``c/d`` is retained.  A
    domain-scaled numerical floor based on sqrt(machine epsilon) prevents an
    exact zero commit from creating an invalid non-positive sigma.  The floor
    is a numeric invariant, not a research parameter.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in PERSISTENT_SEPCMAES_COMMIT_CREDIT_MODES:
        raise ValueError(
            "Unsupported persistent SepCMAES commit-credit mode: "
            f"{normalized_mode}."
        )
    if not isinstance(state, dict):
        raise TypeError("Persistent SepCMAES state must be a dictionary.")

    reconciled = copy.deepcopy(state) if bool(copy_state) else state
    metadata = reconciled.get("metadata", {})
    arrays = reconciled.get("arrays", {})
    progress = reconciled.get("progress", {})
    dimension = int(metadata.get("dimension", -1))
    if dimension <= 0:
        raise ValueError(
            "Persistent SepCMAES commit credit requires a positive dimension."
        )

    previous = _finite_vector(
        "previous_center", previous_center, dimension
    )
    proposed = _finite_vector("proposal", proposal, dimension)
    accepted = _finite_vector("committed", committed, dimension)
    sigma_path = _finite_vector("state arrays.s", arrays.get("s"), dimension)
    covariance_path = _finite_vector(
        "state arrays.p", arrays.get("p"), dimension
    )
    covariance = _finite_vector(
        "state arrays.c", arrays.get("c"), dimension
    )
    axis_shape = _finite_vector("state arrays.d", arrays.get("d"), dimension)
    if np.any(covariance <= 0.0) or np.any(axis_shape <= 0.0):
        raise ValueError(
            "Persistent SepCMAES commit credit requires positive covariance "
            "and axis-shape entries."
        )
    sigma_before = float(progress.get("sigma", np.nan))
    if not np.isfinite(sigma_before) or sigma_before <= 0.0:
        raise ValueError(
            "Persistent SepCMAES commit credit requires finite positive sigma."
        )

    lower = _finite_vector(
        "state metadata.lower_boundary",
        metadata.get("lower_boundary"),
        dimension,
    )
    upper = _finite_vector(
        "state metadata.upper_boundary",
        metadata.get("upper_boundary"),
        dimension,
    )
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError(
            "Persistent SepCMAES commit credit requires valid search bounds."
        )

    proposal_step = proposed - previous
    commit_step = accepted - previous
    correction = accepted - proposed
    proposal_norm = _finite_norm(proposal_step)
    commit_norm = _finite_norm(commit_step)
    correction_norm = _finite_norm(correction)

    numeric_zero = float(
        np.finfo(np.float64).eps
        * max(1.0, proposal_norm, commit_norm)
    )
    cosine = 0.0
    credit = 0.0
    if proposal_norm > numeric_zero and commit_norm > numeric_zero:
        cosine = float(
            np.clip(
                np.dot(proposal_step, commit_step)
                / (proposal_norm * commit_norm),
                -1.0,
                1.0,
            )
        )
        credit = float(
            max(0.0, cosine)
            * min(1.0, commit_norm / proposal_norm)
        )

    effective_axis_before = sigma_before * axis_shape
    if not np.all(np.isfinite(effective_axis_before)):
        raise ValueError(
            "Persistent SepCMAES effective axis scale is non-finite."
        )
    effective_axis_rms_before = float(
        np.sqrt(np.mean(np.square(effective_axis_before)))
    )
    if (
        not np.isfinite(effective_axis_rms_before)
        or effective_axis_rms_before <= 0.0
    ):
        raise ValueError(
            "Persistent SepCMAES effective axis RMS must be positive."
        )

    path_retention = (
        credit if normalized_mode in {"path", "joint"} else 1.0
    )
    scale_retention = 1.0
    sigma_after = sigma_before
    commit_rms = float(commit_norm / np.sqrt(float(dimension)))
    if normalized_mode in {"scale", "joint"}:
        requested_scale_retention = float(
            min(1.0, commit_rms / effective_axis_rms_before)
        )
        span_rms = float(np.sqrt(np.mean(np.square(span))))
        axis_floor = float(
            np.sqrt(np.finfo(np.float64).eps) * span_rms
        )
        shape_rms = float(np.sqrt(np.mean(np.square(axis_shape))))
        sigma_floor = float(axis_floor / shape_rms)
        sigma_after = float(
            min(
                sigma_before,
                max(
                    sigma_before * requested_scale_retention,
                    sigma_floor,
                    np.finfo(np.float64).tiny,
                ),
            )
        )
        if not np.isfinite(sigma_after) or sigma_after <= 0.0:
            raise ValueError(
                "Commit-conditioned scale credit produced invalid sigma."
            )
        scale_retention = float(sigma_after / sigma_before)

    sigma_path_before_norm = _finite_norm(sigma_path)
    covariance_path_before_norm = _finite_norm(covariance_path)
    arrays["s"] = sigma_path * path_retention
    arrays["p"] = covariance_path * path_retention
    progress["sigma"] = sigma_after
    effective_axis_after = sigma_after * axis_shape

    telemetry = {
        "active": bool(normalized_mode != "off"),
        "mode": normalized_mode,
        "credit": float(credit),
        "cosine": float(cosine),
        "proposal_norm": float(proposal_norm),
        "commit_norm": float(commit_norm),
        "correction_norm": float(correction_norm),
        "commit_rms": float(commit_rms),
        "path_retention": float(path_retention),
        "scale_retention": float(scale_retention),
        "sigma_before": float(sigma_before),
        "sigma_after": float(sigma_after),
        "effective_axis_rms_before": float(
            effective_axis_rms_before
        ),
        "effective_axis_rms_after": float(
            np.sqrt(np.mean(np.square(effective_axis_after)))
        ),
        "sigma_path_norm_before": float(sigma_path_before_norm),
        "sigma_path_norm_after": float(
            _finite_norm(np.asarray(arrays["s"], dtype=np.float64))
        ),
        "covariance_path_norm_before": float(
            covariance_path_before_norm
        ),
        "covariance_path_norm_after": float(
            _finite_norm(np.asarray(arrays["p"], dtype=np.float64))
        ),
    }
    return reconciled, telemetry

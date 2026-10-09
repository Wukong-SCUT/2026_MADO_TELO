"""Strict-local WSN target-block cooperative field.

The field consumes only each agent's own per-target residual change along its
local optimizer move.  One graph round jointly mixes proposal coordinates,
block secants, and the persistent cooperative path.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from env.optimizer.target_block_dormancy_recovery import (
    TargetBlockDormancyStep,
    apply_target_block_dormancy_recovery,
)


TARGET_BLOCK_FIELD_STATE_VERSION = 1
CCSA_WSN_PATH_DECAY = 0.9
CCSA_OUTER_STEP_RATE = 0.01
TARGET_BLOCK_RADIUS_MIN_RATIO = 1e-6
TARGET_BLOCK_RADIUS_MAX_RATIO = 0.05


@dataclass(frozen=True)
class TargetBlockFieldStep:
    committed_states: np.ndarray
    secant: np.ndarray
    secant_valid: np.ndarray
    secant_field_alignment: np.ndarray
    field_direction: np.ndarray
    field_valid: np.ndarray
    source_diversity: np.ndarray
    path: np.ndarray
    path_initialized: np.ndarray
    path_norm: np.ndarray
    path_alignment: np.ndarray
    conflict: np.ndarray
    radius: np.ndarray
    radius_expand: np.ndarray
    radius_shrink: np.ndarray
    radius_clip_min: np.ndarray
    radius_clip_max: np.ndarray
    requested_commit_norm: np.ndarray
    applied_commit_norm: np.ndarray
    boundary_clipped: np.ndarray
    path_decay: float
    path_injection: float
    path_angle_degrees: float
    dormancy_recovery: TargetBlockDormancyStep | None


def ccsa_wsn_path_coefficients(
    budget_progress: float,
    *,
    path_decay: float = CCSA_WSN_PATH_DECAY,
) -> Tuple[float, float, float]:
    """Return the CCSA-WSN path coefficients without a parameter grid.

    The reference WSN implementation linearly moves the target angle from
    90 degrees to 0 degrees over the evaluation budget and solves the
    unit-resultant constraint for the injected direction coefficient.
    """

    beta = float(path_decay)
    if not np.isfinite(beta) or beta < 0.0 or beta >= 1.0:
        raise ValueError("path_decay must be finite and in [0,1).")
    progress = float(np.clip(budget_progress, 0.0, 1.0))
    angle = 0.5 * np.pi * (1.0 - progress)
    cosine = float(np.cos(angle))
    radicand = max(0.0, 1.0 - beta * beta * (1.0 - cosine * cosine))
    injection = -beta * cosine + float(np.sqrt(radicand))
    return beta, injection, float(np.degrees(angle))


def _bounds_vector(value, dimension: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((dimension,), float(arr), dtype=np.float64)
    try:
        return np.broadcast_to(arr, (dimension,)).astype(
            np.float64, copy=True
        )
    except ValueError as exc:
        raise ValueError(
            f"{name} cannot broadcast to dimension {dimension}: {arr.shape}."
        ) from exc


def _unit_blocks(values: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=2)
    units = np.divide(
        values,
        norms[:, :, None],
        out=np.zeros_like(values),
        where=norms[:, :, None] > eps,
    )
    return units, norms


def build_target_block_cooperative_field(
    *,
    base_states: np.ndarray,
    proposal_states: np.ndarray,
    commit_base_states: np.ndarray,
    base_residuals: np.ndarray,
    proposal_residuals: np.ndarray,
    previous_path: np.ndarray,
    previous_radius: np.ndarray,
    previous_initialized: np.ndarray,
    weight: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    lower_bound,
    upper_bound,
    budget_progress: float,
    path_decay: float = CCSA_WSN_PATH_DECAY,
    step_rate: float = CCSA_OUTER_STEP_RATE,
    radius_min_ratio: float = TARGET_BLOCK_RADIUS_MIN_RATIO,
    radius_max_ratio: float = TARGET_BLOCK_RADIUS_MAX_RATIO,
    eps: float = 1e-12,
    dormancy_recovery_enable: bool = False,
    previous_recovery_residual_reference: np.ndarray = None,
    previous_recovery_reserve_radius: np.ndarray = None,
    previous_recovery_initialized: np.ndarray = None,
    previous_recovery_floor_age: np.ndarray = None,
    previous_recovery_stagnation_age: np.ndarray = None,
    previous_recovery_cooldown: np.ndarray = None,
    previous_recovery_activation_count: np.ndarray = None,
) -> TargetBlockFieldStep:
    """Build one persistent one-hop target-block field event."""

    target_num = int(target_num)
    coordinate_dim = int(coordinate_dim)
    if target_num <= 0 or coordinate_dim <= 0:
        raise ValueError("target_num and coordinate_dim must be positive.")
    dimension = target_num * coordinate_dim
    bases = np.asarray(base_states, dtype=np.float64)
    proposals = np.asarray(proposal_states, dtype=np.float64)
    commit_bases = np.asarray(commit_base_states, dtype=np.float64)
    if bases.ndim != 2 or bases.shape[1] != dimension:
        raise ValueError(
            f"base_states must have shape [A,{dimension}], got {bases.shape}."
        )
    n_agents = int(bases.shape[0])
    expected_state_shape = (n_agents, dimension)
    for name, value in (
        ("proposal_states", proposals),
        ("commit_base_states", commit_bases),
    ):
        if value.shape != expected_state_shape:
            raise ValueError(
                f"{name} must have shape {expected_state_shape}, got {value.shape}."
            )
    residual_shape = (n_agents, target_num)
    base_r = np.asarray(base_residuals, dtype=np.float64)
    proposal_r = np.asarray(proposal_residuals, dtype=np.float64)
    if base_r.shape != residual_shape or proposal_r.shape != residual_shape:
        raise ValueError(
            "base/proposal residuals must have shape "
            f"{residual_shape}, got {base_r.shape}/{proposal_r.shape}."
        )
    block_shape = (n_agents, target_num, coordinate_dim)
    old_path = np.asarray(previous_path, dtype=np.float64)
    old_radius = np.asarray(previous_radius, dtype=np.float64)
    old_initialized = np.asarray(previous_initialized, dtype=bool)
    if old_path.shape != block_shape:
        raise ValueError(
            f"previous_path must have shape {block_shape}, got {old_path.shape}."
        )
    if old_radius.shape != residual_shape:
        raise ValueError(
            f"previous_radius must have shape {residual_shape}, got {old_radius.shape}."
        )
    if old_initialized.shape != residual_shape:
        raise ValueError(
            "previous_initialized must have shape "
            f"{residual_shape}, got {old_initialized.shape}."
        )
    w = np.asarray(weight, dtype=np.float64)
    if w.shape != (n_agents, n_agents):
        raise ValueError(
            f"weight must have shape {(n_agents, n_agents)}, got {w.shape}."
        )
    if (
        not np.all(np.isfinite(w))
        or np.any(w < -eps)
        or not np.allclose(np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("weight must be finite, non-negative, and row stochastic.")
    if not np.isfinite(step_rate) or step_rate < 0.0:
        raise ValueError("step_rate must be finite and non-negative.")
    if (
        not np.isfinite(radius_min_ratio)
        or not np.isfinite(radius_max_ratio)
        or radius_min_ratio <= 0.0
        or radius_max_ratio < radius_min_ratio
    ):
        raise ValueError("radius ratios are invalid.")

    lb = _bounds_vector(lower_bound, dimension, "lower_bound")
    ub = _bounds_vector(upper_bound, dimension, "upper_bound")
    if (
        not np.all(np.isfinite(lb))
        or not np.all(np.isfinite(ub))
        or np.any(ub <= lb)
    ):
        raise ValueError("Bounds must be finite and strictly ordered.")
    if not (
        np.all(np.isfinite(bases))
        and np.all(np.isfinite(proposals))
        and np.all(np.isfinite(commit_bases))
    ):
        raise ValueError("State tensors must be finite.")

    base_blocks = bases.reshape(block_shape)
    proposal_blocks = proposals.reshape(block_shape)
    moves = proposal_blocks - base_blocks
    move_norm_sq = np.sum(moves * moves, axis=2)
    response = base_r - proposal_r
    secant_valid = (
        np.isfinite(response)
        & np.isfinite(move_norm_sq)
        & (move_norm_sq > eps * eps)
    )
    secant = np.divide(
        response[:, :, None] * moves,
        move_norm_sq[:, :, None] + eps,
        out=np.zeros_like(moves),
        where=secant_valid[:, :, None],
    )
    finite_secant = np.all(np.isfinite(secant), axis=2)
    secant_valid &= finite_secant
    secant[~secant_valid] = 0.0

    merged = np.einsum("ij,jtk->itk", w, secant)
    field_direction, merged_norm = _unit_blocks(merged, eps)
    field_valid = np.isfinite(merged_norm) & (merged_norm > eps)
    field_direction[~field_valid] = 0.0
    source_diversity = np.sum(
        (w[:, :, None] > eps) & secant_valid[None, :, :],
        axis=1,
        dtype=np.int64,
    )
    local_secant_unit, local_secant_norm = _unit_blocks(secant, eps)
    secant_field_alignment = np.sum(
        local_secant_unit * field_direction, axis=2
    )
    secant_field_alignment[
        ~(secant_valid & field_valid & (local_secant_norm > eps))
    ] = 0.0

    beta, gamma, angle_degrees = ccsa_wsn_path_coefficients(
        budget_progress,
        path_decay=path_decay,
    )
    safe_old_path = np.nan_to_num(
        old_path, nan=0.0, posinf=0.0, neginf=0.0
    )
    diffused_path = np.einsum("ij,jtk->itk", w, safe_old_path)
    diffused_unit, diffused_norm = _unit_blocks(diffused_path, eps)
    path_alignment = np.sum(diffused_unit * field_direction, axis=2)
    alignment_valid = (diffused_norm > eps) & field_valid
    path_alignment[~alignment_valid] = 0.0
    conflict = alignment_valid & (path_alignment < 0.0)

    path = beta * diffused_path + gamma * field_direction
    cold = (~old_initialized) & field_valid
    path[cold] = field_direction[cold]
    path = np.nan_to_num(path, nan=0.0, posinf=0.0, neginf=0.0)
    path_unit, path_norm = _unit_blocks(path, eps)
    path_initialized = (
        old_initialized | field_valid | (diffused_norm > eps)
    )
    inactive = ~path_initialized
    path[inactive] = 0.0
    path_unit[inactive] = 0.0
    path_norm[inactive] = 0.0

    block_span = (ub - lb).reshape(target_num, coordinate_dim)
    block_diagonal = np.linalg.norm(block_span, axis=1)
    radius_min = radius_min_ratio * block_diagonal
    radius_max = radius_max_ratio * block_diagonal
    move_norm = np.sqrt(np.maximum(move_norm_sq, 0.0))
    seeded_radius = np.clip(move_norm, radius_min[None, :], radius_max[None, :])
    radius_base = np.where(
        old_initialized
        & np.isfinite(old_radius)
        & (old_radius > 0.0),
        old_radius,
        seeded_radius,
    )
    exponent = np.clip(
        float(step_rate) * (path_norm - 1.0),
        -50.0,
        50.0,
    )
    raw_radius = radius_base * np.exp(exponent)
    radius = np.clip(raw_radius, radius_min[None, :], radius_max[None, :])
    radius[inactive] = radius_min[None, :].repeat(n_agents, axis=0)[inactive]
    tolerance = 1e-15
    ordinary_radius_clip_min = path_initialized & (
        raw_radius <= radius_min[None, :] + tolerance
    )
    dormancy_recovery = None
    if bool(dormancy_recovery_enable):
        recovery_inputs = (
            previous_recovery_residual_reference,
            previous_recovery_reserve_radius,
            previous_recovery_initialized,
            previous_recovery_floor_age,
            previous_recovery_stagnation_age,
            previous_recovery_cooldown,
            previous_recovery_activation_count,
        )
        if any(value is None for value in recovery_inputs):
            raise ValueError(
                "Enabled target-block dormancy recovery requires complete "
                "previous recovery state."
            )
        dormancy_recovery = apply_target_block_dormancy_recovery(
            base_residuals=base_r,
            proposal_residuals=proposal_r,
            proposal_states=proposals,
            weight=w,
            field_initialized=path_initialized,
            field_valid=field_valid,
            source_diversity=source_diversity,
            secant_field_alignment=secant_field_alignment,
            ordinary_radius=radius,
            ordinary_radius_clip_min=ordinary_radius_clip_min,
            radius_min=radius_min,
            radius_max=radius_max,
            block_diagonal=block_diagonal,
            previous_residual_reference=(
                previous_recovery_residual_reference
            ),
            previous_reserve_radius=previous_recovery_reserve_radius,
            previous_initialized=previous_recovery_initialized,
            previous_floor_age=previous_recovery_floor_age,
            previous_stagnation_age=(
                previous_recovery_stagnation_age
            ),
            previous_cooldown=previous_recovery_cooldown,
            previous_activation_count=(
                previous_recovery_activation_count
            ),
            target_num=target_num,
            coordinate_dim=coordinate_dim,
            eps=eps,
        )
        radius = dormancy_recovery.radius.copy()

    radius_expand = path_initialized & (radius > radius_base + tolerance)
    radius_shrink = path_initialized & (radius < radius_base - tolerance)
    if bool(dormancy_recovery_enable):
        radius_clip_min = path_initialized & (
            radius <= radius_min[None, :] + tolerance
        )
        radius_clip_max = path_initialized & (
            radius >= radius_max[None, :] - tolerance
        )
    else:
        # Preserve the exact pre-NT079 telemetry path when recovery is off.
        radius_clip_min = path_initialized & (
            raw_radius <= radius_min[None, :] + tolerance
        )
        radius_clip_max = path_initialized & (
            raw_radius >= radius_max[None, :] - tolerance
        )

    requested_delta = radius[:, :, None] * path_unit
    requested_delta[~path_initialized] = 0.0
    requested_commit_norm = np.linalg.norm(requested_delta, axis=2)
    requested_states = commit_bases.reshape(block_shape) + requested_delta
    clipped_states = np.clip(
        requested_states,
        lb.reshape(target_num, coordinate_dim)[None, :, :],
        ub.reshape(target_num, coordinate_dim)[None, :, :],
    )
    applied_delta = clipped_states - commit_bases.reshape(block_shape)
    applied_commit_norm = np.linalg.norm(applied_delta, axis=2)
    boundary_clipped = np.any(
        np.abs(clipped_states - requested_states) > 1e-12,
        axis=2,
    )
    committed = clipped_states.reshape(n_agents, dimension)
    if not np.all(np.isfinite(committed)):
        raise FloatingPointError("Target-block field emitted non-finite states.")

    return TargetBlockFieldStep(
        committed_states=committed,
        secant=secant,
        secant_valid=secant_valid,
        secant_field_alignment=secant_field_alignment,
        field_direction=field_direction,
        field_valid=field_valid,
        source_diversity=source_diversity,
        path=path,
        path_initialized=path_initialized,
        path_norm=path_norm,
        path_alignment=path_alignment,
        conflict=conflict,
        radius=radius,
        radius_expand=radius_expand,
        radius_shrink=radius_shrink,
        radius_clip_min=radius_clip_min,
        radius_clip_max=radius_clip_max,
        requested_commit_norm=requested_commit_norm,
        applied_commit_norm=applied_commit_norm,
        boundary_clipped=boundary_clipped,
        path_decay=float(beta),
        path_injection=float(gamma),
        path_angle_degrees=float(angle_degrees),
        dormancy_recovery=dormancy_recovery,
    )

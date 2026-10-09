"""Strict-local selective recovery for dormant WSN target blocks.

The recovery state is local to one receiver agent and one three-dimensional
target block.  It consumes only own-local base/proposal residuals plus
coordinates and field metadata already present in the NT073 graph event.
It performs no objective call and adds no communication payload.
"""

from dataclasses import dataclass

import numpy as np


TARGET_BLOCK_DORMANCY_STATE_VERSION = 1
TARGET_BLOCK_DORMANCY_PROGRESS_RATIO = 0.01
TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO = 0.01
TARGET_BLOCK_DORMANCY_PATIENCE = 32
TARGET_BLOCK_DORMANCY_COOLDOWN = 64
TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY = 2


@dataclass(frozen=True)
class TargetBlockDormancyStep:
    radius: np.ndarray
    residual_reference: np.ndarray
    reserve_radius: np.ndarray
    initialized: np.ndarray
    floor_age: np.ndarray
    stagnation_age: np.ndarray
    cooldown: np.ndarray
    activation_count: np.ndarray
    active: np.ndarray
    unresolved: np.ndarray
    material_progress: np.ndarray
    reliable_scale: np.ndarray
    residual_ratio: np.ndarray
    progress_ratio: np.ndarray
    disagreement: np.ndarray
    disagreement_ratio: np.ndarray
    restore_radius: np.ndarray


def _float_array(name: str, value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}, got {array.shape}."
        )
    return array.copy()


def _nonnegative_int_array(name: str, value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.shape != shape or np.any(array < 0):
        raise ValueError(
            f"{name} must be non-negative with shape {shape}, "
            f"got {array.shape}."
        )
    return array.copy()


def _increment_age(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    limit = np.iinfo(np.int64).max
    incremented = np.minimum(values, limit - 1) + 1
    return np.where(mask, incremented, 0).astype(np.int64, copy=False)


def apply_target_block_dormancy_recovery(
    *,
    base_residuals: np.ndarray,
    proposal_residuals: np.ndarray,
    proposal_states: np.ndarray,
    weight: np.ndarray,
    field_initialized: np.ndarray,
    field_valid: np.ndarray,
    source_diversity: np.ndarray,
    secant_field_alignment: np.ndarray,
    ordinary_radius: np.ndarray,
    ordinary_radius_clip_min: np.ndarray,
    radius_min: np.ndarray,
    radius_max: np.ndarray,
    block_diagonal: np.ndarray,
    previous_residual_reference: np.ndarray,
    previous_reserve_radius: np.ndarray,
    previous_initialized: np.ndarray,
    previous_floor_age: np.ndarray,
    previous_stagnation_age: np.ndarray,
    previous_cooldown: np.ndarray,
    previous_activation_count: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    progress_ratio: float = TARGET_BLOCK_DORMANCY_PROGRESS_RATIO,
    unresolved_ratio: float = TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO,
    patience: int = TARGET_BLOCK_DORMANCY_PATIENCE,
    cooldown_events: int = TARGET_BLOCK_DORMANCY_COOLDOWN,
    min_source_diversity: int = (
        TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY
    ),
    eps: float = 1e-12,
) -> TargetBlockDormancyStep:
    """Apply one block-selective dormant-radius recovery event.

    Residual magnitudes are normalized by a receiver-local running reference.
    A second unresolved signal comes from the W-weighted dispersion of the
    already communicated proposal coordinates.  Recovery is possible only
    after both the ordinary field radius and local progress have remained
    dormant for ``patience`` consecutive events.
    """

    target_num = int(target_num)
    coordinate_dim = int(coordinate_dim)
    if target_num <= 0 or coordinate_dim <= 0:
        raise ValueError("target_num and coordinate_dim must be positive.")
    if (
        not np.isfinite(progress_ratio)
        or progress_ratio <= 0.0
        or not np.isfinite(unresolved_ratio)
        or unresolved_ratio <= 0.0
        or int(patience) <= 0
        or int(cooldown_events) < 0
        or int(min_source_diversity) <= 0
        or not np.isfinite(eps)
        or eps <= 0.0
    ):
        raise ValueError("Dormancy recovery constants are invalid.")

    bases = np.asarray(base_residuals, dtype=np.float64)
    proposals_r = np.asarray(proposal_residuals, dtype=np.float64)
    if bases.ndim != 2 or proposals_r.shape != bases.shape:
        raise ValueError(
            "base/proposal residuals must have matching [A,T] shapes."
        )
    n_agents = int(bases.shape[0])
    block_shape = (n_agents, target_num)
    if bases.shape != block_shape:
        raise ValueError(
            f"Residuals must have shape {block_shape}, got {bases.shape}."
        )

    dimension = target_num * coordinate_dim
    proposals = _float_array(
        "proposal_states",
        proposal_states,
        (n_agents, dimension),
    ).reshape(n_agents, target_num, coordinate_dim)
    w = _float_array("weight", weight, (n_agents, n_agents))
    if (
        np.any(w < -eps)
        or not np.allclose(
            np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-10
        )
    ):
        raise ValueError(
            "weight must be non-negative and row stochastic."
        )

    path_initialized = np.asarray(field_initialized, dtype=bool)
    valid_field = np.asarray(field_valid, dtype=bool)
    clip_min = np.asarray(ordinary_radius_clip_min, dtype=bool)
    for name, value in (
        ("field_initialized", path_initialized),
        ("field_valid", valid_field),
        ("ordinary_radius_clip_min", clip_min),
    ):
        if value.shape != block_shape:
            raise ValueError(
                f"{name} must have shape {block_shape}, got {value.shape}."
            )
    diversity = _nonnegative_int_array(
        "source_diversity", source_diversity, block_shape
    )
    alignment = _float_array(
        "secant_field_alignment",
        secant_field_alignment,
        block_shape,
    )
    if (
        np.any(diversity > n_agents)
        or np.any(alignment < -1.0 - 1e-10)
        or np.any(alignment > 1.0 + 1e-10)
    ):
        raise ValueError(
            "Dormancy recovery field evidence is outside its contract."
        )
    ordinary = _float_array(
        "ordinary_radius", ordinary_radius, block_shape
    )
    min_radius = _float_array(
        "radius_min", radius_min, (target_num,)
    )
    max_radius = _float_array(
        "radius_max", radius_max, (target_num,)
    )
    diagonal = _float_array(
        "block_diagonal", block_diagonal, (target_num,)
    )
    if (
        np.any(min_radius <= 0.0)
        or np.any(max_radius < min_radius)
        or np.any(diagonal <= 0.0)
        or np.any(ordinary < min_radius[None, :] - 1e-12)
        or np.any(ordinary > max_radius[None, :] + 1e-12)
    ):
        raise ValueError("Dormancy recovery radius bounds are invalid.")

    old_reference = _float_array(
        "previous_residual_reference",
        previous_residual_reference,
        block_shape,
    )
    old_reserve = _float_array(
        "previous_reserve_radius",
        previous_reserve_radius,
        block_shape,
    )
    old_initialized = np.asarray(previous_initialized, dtype=bool)
    if old_initialized.shape != block_shape:
        raise ValueError(
            "previous_initialized must have shape "
            f"{block_shape}, got {old_initialized.shape}."
        )
    if (
        np.any(old_reference < 0.0)
        or np.any(old_reserve < 0.0)
        or np.any(old_initialized & (old_reference <= 0.0))
    ):
        raise ValueError("Dormancy recovery floating state is invalid.")
    old_floor_age = _nonnegative_int_array(
        "previous_floor_age", previous_floor_age, block_shape
    )
    old_stagnation_age = _nonnegative_int_array(
        "previous_stagnation_age",
        previous_stagnation_age,
        block_shape,
    )
    old_cooldown = _nonnegative_int_array(
        "previous_cooldown", previous_cooldown, block_shape
    )
    old_activation_count = _nonnegative_int_array(
        "previous_activation_count",
        previous_activation_count,
        block_shape,
    )
    if np.any(old_cooldown > int(cooldown_events)):
        raise ValueError("Dormancy recovery cooldown exceeds its contract.")
    if np.any(
        (~old_initialized)
        & (
            (old_reference != 0.0)
            | (old_reserve != 0.0)
            | (old_floor_age != 0)
            | (old_stagnation_age != 0)
            | (old_cooldown != 0)
            | (old_activation_count != 0)
        )
    ):
        raise ValueError(
            "Uninitialized dormancy recovery blocks must have zero state."
        )

    residual_valid = (
        np.isfinite(bases)
        & np.isfinite(proposals_r)
        & (bases >= 0.0)
        & (proposals_r >= 0.0)
    )
    observed_scale = np.maximum(
        np.where(residual_valid, bases, 0.0),
        np.where(residual_valid, proposals_r, 0.0),
    )
    initialized = old_initialized | residual_valid
    residual_reference = np.where(
        old_initialized,
        np.maximum(old_reference, observed_scale),
        np.maximum(observed_scale, eps),
    )
    residual_reference[~initialized] = 0.0

    response = np.where(
        residual_valid,
        np.maximum(0.0, bases - proposals_r),
        0.0,
    )
    progress = np.divide(
        response,
        residual_reference,
        out=np.zeros_like(response),
        where=residual_reference > eps,
    )
    residual = np.divide(
        np.where(residual_valid, bases, 0.0),
        residual_reference,
        out=np.zeros_like(bases),
        where=residual_reference > eps,
    )
    material_progress = residual_valid & (
        progress >= float(progress_ratio)
    )

    mixed = np.einsum("ij,jtk->itk", w, proposals)
    offsets = proposals[None, :, :, :] - mixed[:, None, :, :]
    squared_offsets = np.sum(offsets * offsets, axis=3)
    disagreement = np.sqrt(
        np.maximum(
            np.einsum("ij,ijt->it", w, squared_offsets),
            0.0,
        )
    )
    disagreement_ratio_value = disagreement / diagonal[None, :]

    reserve_seed = np.clip(
        ordinary,
        min_radius[None, :],
        max_radius[None, :],
    )
    reserve = np.where(
        old_initialized,
        old_reserve,
        0.0,
    )
    reliable_scale = (
        material_progress
        & valid_field
        & (diversity >= int(min_source_diversity))
        & (alignment > 0.0)
    )
    reserve = np.where(
        reliable_scale,
        np.maximum(reserve, reserve_seed),
        reserve,
    )
    reserve = np.where(
        reserve > 0.0,
        np.clip(
            reserve,
            min_radius[None, :],
            max_radius[None, :],
        ),
        0.0,
    )

    restore_radius = np.where(
        reserve > 0.0,
        np.clip(
            np.sqrt(min_radius[None, :] * reserve),
            min_radius[None, :],
            max_radius[None, :],
        ),
        0.0,
    )
    unresolved = (
        (residual_valid & (residual >= float(unresolved_ratio)))
        | (disagreement > restore_radius)
    )

    floor_age = _increment_age(
        old_floor_age,
        path_initialized & clip_min,
    )
    stagnation_age = _increment_age(
        old_stagnation_age,
        initialized & residual_valid & ~material_progress,
    )
    cooldown = np.maximum(old_cooldown - 1, 0)
    active = (
        initialized
        & residual_valid
        & path_initialized
        & valid_field
        & (diversity >= int(min_source_diversity))
        & clip_min
        & (floor_age >= int(patience))
        & (stagnation_age >= int(patience))
        & unresolved
        & (old_cooldown == 0)
        & (reserve > min_radius[None, :] + 1e-15)
    )

    radius = ordinary.copy()
    radius[active] = np.maximum(
        radius[active],
        restore_radius[active],
    )
    radius = np.clip(
        radius,
        min_radius[None, :],
        max_radius[None, :],
    )
    cooldown[active] = int(cooldown_events)
    floor_age[active] = 0
    stagnation_age[active] = 0
    activation_count = old_activation_count.copy()
    activation_count[active] = np.minimum(
        activation_count[active],
        np.iinfo(np.int64).max - 1,
    ) + 1

    return TargetBlockDormancyStep(
        radius=radius,
        residual_reference=residual_reference,
        reserve_radius=reserve,
        initialized=initialized,
        floor_age=floor_age,
        stagnation_age=stagnation_age,
        cooldown=cooldown,
        activation_count=activation_count,
        active=active,
        unresolved=unresolved,
        material_progress=material_progress,
        reliable_scale=reliable_scale,
        residual_ratio=residual,
        progress_ratio=progress,
        disagreement=disagreement,
        disagreement_ratio=disagreement_ratio_value,
        restore_radius=restore_radius,
    )

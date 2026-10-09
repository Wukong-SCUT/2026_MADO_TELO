"""Strict-local shadow diagnosis for dormant WSN target-block directions.

The module only constructs bounded probe geometry and summarizes receiver-local
target residuals.  It never changes an optimizer state or a cooperative commit.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


TARGET_BLOCK_DIRECTION_SHADOW_STATE_VERSION = 1
TARGET_BLOCK_DIRECTION_SHADOW_SOURCES: Tuple[str, ...] = (
    "current_path",
    "elite_orthogonal",
    "proposal_disagreement_orthogonal",
)
TARGET_BLOCK_DIRECTION_SHADOW_MAX_BLOCKS_PER_AGENT = 1


@dataclass(frozen=True)
class TargetBlockDirectionShadowPlan:
    selected_block: np.ndarray
    eligible: np.ndarray
    candidate_valid: np.ndarray
    candidate_direction: np.ndarray
    candidate_angle_degrees: np.ndarray
    probe_states: Tuple[np.ndarray, ...]
    probe_plus_index: np.ndarray
    probe_minus_index: np.ndarray
    local_evals: int


@dataclass(frozen=True)
class TargetBlockDirectionShadowResult:
    candidate_response: np.ndarray
    candidate_positive: np.ndarray
    best_response: np.ndarray
    best_source: np.ndarray
    best_sign: np.ndarray
    best_direction: np.ndarray
    response_field: np.ndarray
    support: np.ndarray
    conflict: np.ndarray


def _finite_array(name: str, value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}, got {array.shape}."
        )
    return array.copy()


def _unit(vector: np.ndarray, eps: float) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        return None
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        return None
    return value / norm


def _orthogonal_unit(
    vector: np.ndarray,
    reference: np.ndarray,
    eps: float,
) -> np.ndarray | None:
    candidate = np.asarray(vector, dtype=np.float64)
    ref = _unit(reference, eps)
    if ref is not None:
        candidate = candidate - float(np.dot(candidate, ref)) * ref
    return _unit(candidate, eps)


def _elite_displacement(
    state: Dict | None,
    dimension: int,
) -> np.ndarray | None:
    if not isinstance(state, dict):
        return None
    arrays = state.get("arrays", {})
    if not isinstance(arrays, dict):
        return None
    try:
        population = np.asarray(arrays.get("x"), dtype=np.float64)
        fitness = np.asarray(arrays.get("y"), dtype=np.float64)
        mean = np.asarray(arrays.get("mean"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (
        population.ndim != 2
        or population.shape[1:] != (dimension,)
        or fitness.shape != (population.shape[0],)
        or mean.shape != (dimension,)
        or population.shape[0] <= 0
        or not np.all(np.isfinite(population))
        or not np.all(np.isfinite(fitness))
        or not np.all(np.isfinite(mean))
    ):
        return None
    return population[int(np.argmin(fitness))] - mean


def build_target_block_direction_shadow_plan(
    *,
    commit_base_states: np.ndarray,
    proposal_states: np.ndarray,
    persistent_states: Sequence[Dict | None],
    path: np.ndarray,
    probe_radius: np.ndarray,
    eligible: np.ndarray,
    residual_ratio: np.ndarray,
    weight: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    lower_bound,
    upper_bound,
    max_blocks_per_agent: int = (
        TARGET_BLOCK_DIRECTION_SHADOW_MAX_BLOCKS_PER_AGENT
    ),
    eps: float = 1e-12,
) -> TargetBlockDirectionShadowPlan:
    """Build symmetric, non-actuating probes for at most one block per agent."""

    target_num = int(target_num)
    coordinate_dim = int(coordinate_dim)
    max_blocks_per_agent = int(max_blocks_per_agent)
    if (
        target_num <= 0
        or coordinate_dim != 3
        or max_blocks_per_agent != 1
        or not np.isfinite(eps)
        or eps <= 0.0
    ):
        raise ValueError("Direction-shadow geometry constants are invalid.")
    dimension = target_num * coordinate_dim
    bases = np.asarray(commit_base_states, dtype=np.float64)
    if bases.ndim != 2 or bases.shape[1:] != (dimension,):
        raise ValueError("commit_base_states must have shape [A,D].")
    n_agents = int(bases.shape[0])
    bases = _finite_array(
        "commit_base_states", bases, (n_agents, dimension)
    )
    proposals = _finite_array(
        "proposal_states",
        proposal_states,
        (n_agents, dimension),
    ).reshape(n_agents, target_num, coordinate_dim)
    path_value = _finite_array(
        "path",
        path,
        (n_agents, target_num, coordinate_dim),
    )
    radius = _finite_array(
        "probe_radius", probe_radius, (n_agents, target_num)
    )
    ratio = _finite_array(
        "residual_ratio", residual_ratio, (n_agents, target_num)
    )
    active = np.asarray(eligible, dtype=bool)
    if active.shape != (n_agents, target_num):
        raise ValueError("eligible must have shape [A,T].")
    if np.any(radius < 0.0) or np.any(ratio < 0.0):
        raise ValueError("Direction-shadow radius/ratio must be non-negative.")
    w = _finite_array("weight", weight, (n_agents, n_agents))
    if np.any(w < -eps) or not np.allclose(
        np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("weight must be non-negative and row stochastic.")
    if len(persistent_states) != n_agents:
        raise ValueError("persistent_states must contain one state per agent.")

    lower = np.broadcast_to(
        np.asarray(lower_bound, dtype=np.float64), (dimension,)
    ).copy()
    upper = np.broadcast_to(
        np.asarray(upper_bound, dtype=np.float64), (dimension,)
    ).copy()
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Direction-shadow bounds are invalid.")

    source_num = len(TARGET_BLOCK_DIRECTION_SHADOW_SOURCES)
    selected_block = np.full((n_agents,), -1, dtype=np.int64)
    candidate_valid = np.zeros(
        (n_agents, target_num, source_num), dtype=bool
    )
    candidate_direction = np.zeros(
        (n_agents, target_num, source_num, coordinate_dim),
        dtype=np.float64,
    )
    candidate_angle = np.zeros(
        (n_agents, target_num, source_num), dtype=np.float64
    )
    plus_index = np.full(
        (n_agents, target_num, source_num), -1, dtype=np.int64
    )
    minus_index = np.full_like(plus_index, -1)
    mixed_proposals = np.einsum("ij,jtk->itk", w, proposals)
    all_probe_states = []
    total_evals = 0

    for agent_id in range(n_agents):
        active_blocks = np.flatnonzero(active[agent_id])
        if active_blocks.size == 0:
            all_probe_states.append(
                np.empty((0, dimension), dtype=np.float64)
            )
            continue
        block_id = int(
            active_blocks[
                np.argmax(ratio[agent_id, active_blocks])
            ]
        )
        selected_block[agent_id] = block_id
        path_unit = _unit(path_value[agent_id, block_id], eps)
        elite = _elite_displacement(
            persistent_states[agent_id], dimension
        )
        elite_block = (
            None
            if elite is None
            else elite.reshape(target_num, coordinate_dim)[block_id]
        )
        disagreement = (
            mixed_proposals[agent_id, block_id]
            - proposals[agent_id, block_id]
        )
        directions = (
            path_unit,
            (
                None
                if elite_block is None
                else _orthogonal_unit(elite_block, path_value[agent_id, block_id], eps)
            ),
            _orthogonal_unit(
                disagreement,
                path_value[agent_id, block_id],
                eps,
            ),
        )
        center = bases[agent_id].copy()
        probes = [center]
        block_slice = slice(
            block_id * coordinate_dim,
            (block_id + 1) * coordinate_dim,
        )
        for source_id, direction in enumerate(directions):
            if direction is None or radius[agent_id, block_id] <= eps:
                continue
            plus = center.copy()
            minus = center.copy()
            delta = radius[agent_id, block_id] * direction
            plus[block_slice] = np.clip(
                plus[block_slice] + delta,
                lower[block_slice],
                upper[block_slice],
            )
            minus[block_slice] = np.clip(
                minus[block_slice] - delta,
                lower[block_slice],
                upper[block_slice],
            )
            if (
                np.linalg.norm(plus - center) <= eps
                or np.linalg.norm(minus - center) <= eps
                or np.linalg.norm(plus - minus) <= eps
            ):
                continue
            candidate_valid[agent_id, block_id, source_id] = True
            candidate_direction[
                agent_id, block_id, source_id
            ] = direction
            if path_unit is not None:
                cosine = float(
                    np.clip(np.dot(direction, path_unit), -1.0, 1.0)
                )
                candidate_angle[
                    agent_id, block_id, source_id
                ] = float(np.degrees(np.arccos(cosine)))
            plus_index[agent_id, block_id, source_id] = len(probes)
            probes.append(plus)
            minus_index[agent_id, block_id, source_id] = len(probes)
            probes.append(minus)
        if len(probes) == 1:
            selected_block[agent_id] = -1
            all_probe_states.append(
                np.empty((0, dimension), dtype=np.float64)
            )
        else:
            probe_array = np.asarray(probes, dtype=np.float64)
            all_probe_states.append(probe_array)
            total_evals += int(probe_array.shape[0])

    return TargetBlockDirectionShadowPlan(
        selected_block=selected_block,
        eligible=active.copy(),
        candidate_valid=candidate_valid,
        candidate_direction=candidate_direction,
        candidate_angle_degrees=candidate_angle,
        probe_states=tuple(all_probe_states),
        probe_plus_index=plus_index,
        probe_minus_index=minus_index,
        local_evals=int(total_evals),
    )


def summarize_target_block_direction_shadow(
    *,
    plan: TargetBlockDirectionShadowPlan,
    probe_residuals: Sequence[np.ndarray],
    weight: np.ndarray,
    min_improvement_ratio: float = 1e-12,
    eps: float = 1e-12,
) -> TargetBlockDirectionShadowResult:
    """Select the best own-local response and summarize one-hop agreement."""

    if (
        not np.isfinite(min_improvement_ratio)
        or min_improvement_ratio < 0.0
        or not np.isfinite(eps)
        or eps <= 0.0
    ):
        raise ValueError("Direction-shadow response constants are invalid.")
    n_agents, target_num, source_num = plan.candidate_valid.shape
    if len(probe_residuals) != n_agents:
        raise ValueError("probe_residuals must contain one array per agent.")
    w = _finite_array("weight", weight, (n_agents, n_agents))
    response = np.zeros(
        (n_agents, target_num, source_num), dtype=np.float64
    )
    positive = np.zeros_like(response, dtype=bool)
    best_response = np.zeros((n_agents, target_num), dtype=np.float64)
    best_source = np.full(
        (n_agents, target_num), -1, dtype=np.int64
    )
    best_sign = np.zeros((n_agents, target_num), dtype=np.int64)
    best_direction = np.zeros(
        (n_agents, target_num, 3), dtype=np.float64
    )

    for agent_id in range(n_agents):
        block_id = int(plan.selected_block[agent_id])
        if block_id < 0:
            if np.asarray(probe_residuals[agent_id]).size != 0:
                raise ValueError("Inactive shadow agent returned residuals.")
            continue
        residuals = np.asarray(
            probe_residuals[agent_id], dtype=np.float64
        )
        expected_rows = int(plan.probe_states[agent_id].shape[0])
        if (
            residuals.shape != (expected_rows, target_num)
            or not np.all(np.isfinite(residuals))
            or np.any(residuals < 0.0)
        ):
            raise ValueError(
                "Shadow probe residuals violate the [N,T] contract."
            )
        center = float(residuals[0, block_id])
        scale = max(center, eps)
        for source_id in range(source_num):
            if not plan.candidate_valid[
                agent_id, block_id, source_id
            ]:
                continue
            plus = float(
                residuals[
                    plan.probe_plus_index[
                        agent_id, block_id, source_id
                    ],
                    block_id,
                ]
            )
            minus = float(
                residuals[
                    plan.probe_minus_index[
                        agent_id, block_id, source_id
                    ],
                    block_id,
                ]
            )
            plus_gain = (center - plus) / scale
            minus_gain = (center - minus) / scale
            signed_gain = plus_gain if plus_gain >= minus_gain else minus_gain
            sign = 1 if plus_gain >= minus_gain else -1
            response[agent_id, block_id, source_id] = max(
                0.0, signed_gain
            )
            positive[agent_id, block_id, source_id] = bool(
                signed_gain > min_improvement_ratio
            )
            if signed_gain > best_response[agent_id, block_id]:
                best_response[agent_id, block_id] = max(
                    0.0, signed_gain
                )
                best_source[agent_id, block_id] = source_id
                best_sign[agent_id, block_id] = sign
                best_direction[agent_id, block_id] = (
                    sign
                    * plan.candidate_direction[
                        agent_id, block_id, source_id
                    ]
                )

    response_field = (
        best_response[:, :, None] * best_direction
    )
    support = np.zeros((n_agents, target_num), dtype=np.float64)
    conflict = np.zeros_like(support)
    for agent_id in range(n_agents):
        for block_id in range(target_num):
            own = _unit(best_direction[agent_id, block_id], eps)
            if own is None:
                continue
            neighbor_mass = 0.0
            positive_mass = 0.0
            negative_mass = 0.0
            for neighbor_id in range(n_agents):
                if neighbor_id == agent_id or w[agent_id, neighbor_id] <= 0.0:
                    continue
                amplitude = float(
                    best_response[neighbor_id, block_id]
                )
                neighbor = _unit(
                    best_direction[neighbor_id, block_id], eps
                )
                if neighbor is None or amplitude <= 0.0:
                    continue
                mass = float(w[agent_id, neighbor_id]) * amplitude
                cosine = float(np.dot(own, neighbor))
                neighbor_mass += mass
                positive_mass += mass * max(0.0, cosine)
                negative_mass += mass * max(0.0, -cosine)
            if neighbor_mass > eps:
                support[agent_id, block_id] = (
                    positive_mass / neighbor_mass
                )
                conflict[agent_id, block_id] = (
                    negative_mass / neighbor_mass
                )

    return TargetBlockDirectionShadowResult(
        candidate_response=response,
        candidate_positive=positive,
        best_response=best_response,
        best_source=best_source,
        best_sign=best_sign,
        best_direction=best_direction,
        response_field=response_field,
        support=support,
        conflict=conflict,
    )

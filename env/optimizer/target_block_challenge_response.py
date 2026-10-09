"""Block-addressed one-hop challenge/response for dormant WSN blocks.

Each dormant proposer names one target block and broadcasts fixed candidate
directions.  Every W-visible receiver evaluates those same directions using
only its own local target residual.  The proposer compares the neighborhood
weighted response of the alternatives against the current cooperative path.
No global objective value is constructed or queried here.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from env.optimizer.target_block_direction_shadow import (
    TARGET_BLOCK_DIRECTION_SHADOW_SOURCES,
    build_target_block_direction_shadow_plan,
)


TARGET_BLOCK_CHALLENGE_RESPONSE_STATE_VERSION = 1
TARGET_BLOCK_CHALLENGE_RESPONSE_MODES = frozenset(
    {"off", "shadow", "actuate"}
)
TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES = (
    TARGET_BLOCK_DIRECTION_SHADOW_SOURCES
)


@dataclass(frozen=True)
class TargetBlockChallengePlan:
    selected_block: np.ndarray
    eligible: np.ndarray
    candidate_valid: np.ndarray
    candidate_direction: np.ndarray
    candidate_angle_degrees: np.ndarray
    receiver_probe_states: Tuple[np.ndarray, ...]
    probe_plus_index: np.ndarray
    probe_minus_index: np.ndarray
    response_available: np.ndarray
    local_evals: int
    challenges: int
    directed_responses: int


@dataclass(frozen=True)
class TargetBlockChallengeResult:
    candidate_score: np.ndarray
    candidate_sign: np.ndarray
    path_score: np.ndarray
    alternative_score: np.ndarray
    alternative_margin: np.ndarray
    best_source: np.ndarray
    best_sign: np.ndarray
    best_direction: np.ndarray
    response_coverage: np.ndarray
    positive_sources: np.ndarray
    neighbor_positive_sources: np.ndarray
    support: np.ndarray
    conflict: np.ndarray
    actuation_eligible: np.ndarray


@dataclass(frozen=True)
class TargetBlockChallengeActuation:
    committed_states: np.ndarray
    applied: np.ndarray
    applied_norm: np.ndarray
    boundary_clipped: np.ndarray


def _finite_array(name: str, value, shape) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must be finite with shape {shape}, got {array.shape}."
        )
    return array.copy()


def build_target_block_challenge_plan(
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
    eps: float = 1e-12,
) -> TargetBlockChallengePlan:
    """Build receiver-local probes for every W-visible named challenge."""

    target_num = int(target_num)
    coordinate_dim = int(coordinate_dim)
    if target_num <= 0 or coordinate_dim != 3:
        raise ValueError(
            "Block-addressed challenge response requires 3D target blocks."
        )
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    dimension = target_num * coordinate_dim
    bases = np.asarray(commit_base_states, dtype=np.float64)
    if bases.ndim != 2 or bases.shape[1:] != (dimension,):
        raise ValueError("commit_base_states must have shape [A,D].")
    n_agents = int(bases.shape[0])
    bases = _finite_array(
        "commit_base_states", bases, (n_agents, dimension)
    )
    w = _finite_array("weight", weight, (n_agents, n_agents))
    if np.any(w < -eps) or not np.allclose(
        np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("weight must be non-negative and row stochastic.")

    # Reuse the already-qualified NT080 source construction and deterministic
    # one-block selection.  Its local probe geometry is not evaluated here;
    # NT081 rebuilds receiver-addressed probes below.
    source_plan = build_target_block_direction_shadow_plan(
        commit_base_states=bases,
        proposal_states=proposal_states,
        persistent_states=persistent_states,
        path=path,
        probe_radius=probe_radius,
        eligible=eligible,
        residual_ratio=residual_ratio,
        weight=w,
        target_num=target_num,
        coordinate_dim=coordinate_dim,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        eps=eps,
    )
    source_num = len(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES)
    radius = _finite_array(
        "probe_radius", probe_radius, (n_agents, target_num)
    )
    if np.any(radius < 0.0):
        raise ValueError("probe_radius must be non-negative.")
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
        raise ValueError("Challenge-response bounds are invalid.")

    plus_index = np.full(
        (n_agents, n_agents, source_num), -1, dtype=np.int64
    )
    minus_index = np.full_like(plus_index, -1)
    response_available = np.zeros_like(plus_index, dtype=bool)
    receiver_batches = []
    local_evals = 0
    directed_responses = 0

    for receiver_id in range(n_agents):
        probes = [bases[receiver_id].copy()]
        responded_proposers = set()
        for proposer_id in range(n_agents):
            block_id = int(source_plan.selected_block[proposer_id])
            if block_id < 0 or w[proposer_id, receiver_id] <= eps:
                continue
            block_slice = slice(
                block_id * coordinate_dim,
                (block_id + 1) * coordinate_dim,
            )
            center = bases[receiver_id]
            for source_id in range(source_num):
                if not source_plan.candidate_valid[
                    proposer_id, block_id, source_id
                ]:
                    continue
                direction = source_plan.candidate_direction[
                    proposer_id, block_id, source_id
                ]
                delta = radius[proposer_id, block_id] * direction
                plus = center.copy()
                minus = center.copy()
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
                plus_index[
                    proposer_id, receiver_id, source_id
                ] = len(probes)
                probes.append(plus)
                minus_index[
                    proposer_id, receiver_id, source_id
                ] = len(probes)
                probes.append(minus)
                response_available[
                    proposer_id, receiver_id, source_id
                ] = True
                responded_proposers.add(proposer_id)
        if len(probes) == 1:
            receiver_batches.append(
                np.empty((0, dimension), dtype=np.float64)
            )
        else:
            batch = np.asarray(probes, dtype=np.float64)
            receiver_batches.append(batch)
            local_evals += int(batch.shape[0])
            directed_responses += int(
                sum(
                    proposer_id != receiver_id
                    for proposer_id in responded_proposers
                )
            )

    challenges = int(np.sum(source_plan.selected_block >= 0))
    return TargetBlockChallengePlan(
        selected_block=source_plan.selected_block.copy(),
        eligible=source_plan.eligible.copy(),
        candidate_valid=source_plan.candidate_valid.copy(),
        candidate_direction=source_plan.candidate_direction.copy(),
        candidate_angle_degrees=(
            source_plan.candidate_angle_degrees.copy()
        ),
        receiver_probe_states=tuple(receiver_batches),
        probe_plus_index=plus_index,
        probe_minus_index=minus_index,
        response_available=response_available,
        local_evals=int(local_evals),
        challenges=challenges,
        directed_responses=int(directed_responses),
    )


def summarize_target_block_challenge_response(
    *,
    plan: TargetBlockChallengePlan,
    probe_residuals: Sequence[np.ndarray],
    weight: np.ndarray,
    eps: float = 1e-12,
) -> TargetBlockChallengeResult:
    """Aggregate continuous receiver-local responses for named challenges."""

    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    n_agents, target_num, source_num = plan.candidate_valid.shape
    if len(probe_residuals) != n_agents:
        raise ValueError("probe_residuals must contain one batch per receiver.")
    w = _finite_array("weight", weight, (n_agents, n_agents))
    if np.any(w < -eps) or not np.allclose(
        np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("weight must be non-negative and row stochastic.")

    gains = np.zeros(
        (n_agents, n_agents, source_num, 2), dtype=np.float64
    )
    for receiver_id in range(n_agents):
        residuals = np.asarray(
            probe_residuals[receiver_id], dtype=np.float64
        )
        expected = int(plan.receiver_probe_states[receiver_id].shape[0])
        if expected == 0:
            if residuals.size != 0:
                raise ValueError("Inactive receiver returned residuals.")
            continue
        if (
            residuals.shape != (expected, target_num)
            or not np.all(np.isfinite(residuals))
            or np.any(residuals < 0.0)
        ):
            raise ValueError(
                "Receiver residuals violate the finite non-negative [N,T] "
                "contract."
            )
        for proposer_id in range(n_agents):
            block_id = int(plan.selected_block[proposer_id])
            if block_id < 0:
                continue
            center = float(residuals[0, block_id])
            scale = max(center, eps)
            for source_id in range(source_num):
                if not plan.response_available[
                    proposer_id, receiver_id, source_id
                ]:
                    continue
                plus = float(
                    residuals[
                        plan.probe_plus_index[
                            proposer_id, receiver_id, source_id
                        ],
                        block_id,
                    ]
                )
                minus = float(
                    residuals[
                        plan.probe_minus_index[
                            proposer_id, receiver_id, source_id
                        ],
                        block_id,
                    ]
                )
                gains[proposer_id, receiver_id, source_id, 0] = (
                    center - plus
                ) / scale
                gains[proposer_id, receiver_id, source_id, 1] = (
                    center - minus
                ) / scale

    candidate_score = np.zeros(
        (n_agents, target_num, source_num), dtype=np.float64
    )
    candidate_sign = np.zeros_like(candidate_score, dtype=np.int64)
    path_score = np.zeros((n_agents, target_num), dtype=np.float64)
    alternative_score = np.zeros_like(path_score)
    alternative_margin = np.zeros_like(path_score)
    best_source = np.full_like(path_score, -1, dtype=np.int64)
    best_sign = np.zeros_like(path_score, dtype=np.int64)
    best_direction = np.zeros(
        (n_agents, target_num, 3), dtype=np.float64
    )
    coverage = np.zeros_like(path_score)
    positive_sources = np.zeros_like(path_score, dtype=np.int64)
    neighbor_positive_sources = np.zeros_like(
        path_score, dtype=np.int64
    )
    support = np.zeros_like(path_score)
    conflict = np.zeros_like(path_score)
    actuation_eligible = np.zeros_like(path_score, dtype=bool)

    for proposer_id in range(n_agents):
        block_id = int(plan.selected_block[proposer_id])
        if block_id < 0:
            continue
        source_best_sign = np.zeros((source_num,), dtype=np.int64)
        source_best_score = np.full(
            (source_num,), -np.inf, dtype=np.float64
        )
        source_coverage = np.zeros((source_num,), dtype=np.float64)
        for source_id in range(source_num):
            available = plan.response_available[
                proposer_id, :, source_id
            ]
            if not np.any(available):
                continue
            weights = np.where(available, w[proposer_id], 0.0)
            source_coverage[source_id] = float(np.sum(weights))
            if source_coverage[source_id] <= eps:
                continue
            signed_scores = np.sum(
                weights[:, None] * gains[proposer_id, :, source_id, :],
                axis=0,
            )
            sign_index = int(np.argmax(signed_scores))
            source_best_sign[source_id] = 1 if sign_index == 0 else -1
            source_best_score[source_id] = float(signed_scores[sign_index])
            candidate_score[
                proposer_id, block_id, source_id
            ] = source_best_score[source_id]
            candidate_sign[
                proposer_id, block_id, source_id
            ] = source_best_sign[source_id]

        path_value = (
            source_best_score[0]
            if np.isfinite(source_best_score[0])
            else 0.0
        )
        path_score[proposer_id, block_id] = path_value
        alternative_ids = [
            source_id
            for source_id in range(1, source_num)
            if np.isfinite(source_best_score[source_id])
        ]
        if not alternative_ids:
            continue
        source_id = max(
            alternative_ids,
            key=lambda value: source_best_score[value],
        )
        alt_value = float(source_best_score[source_id])
        sign = int(source_best_sign[source_id])
        alternative_score[proposer_id, block_id] = alt_value
        alternative_margin[proposer_id, block_id] = alt_value - path_value
        best_source[proposer_id, block_id] = source_id
        best_sign[proposer_id, block_id] = sign
        best_direction[proposer_id, block_id] = (
            sign
            * plan.candidate_direction[
                proposer_id, block_id, source_id
            ]
        )
        coverage[proposer_id, block_id] = source_coverage[source_id]

        sign_index = 0 if sign > 0 else 1
        selected_gains = gains[
            proposer_id, :, source_id, sign_index
        ]
        available = plan.response_available[
            proposer_id, :, source_id
        ]
        positive = available & (selected_gains > eps)
        negative = available & (selected_gains < -eps)
        positive_sources[proposer_id, block_id] = int(np.sum(positive))
        neighbor_positive_sources[proposer_id, block_id] = int(
            np.sum(
                positive
                & (np.arange(n_agents, dtype=np.int64) != proposer_id)
            )
        )
        positive_mass = float(
            np.sum(
                w[proposer_id]
                * np.where(positive, selected_gains, 0.0)
            )
        )
        negative_mass = float(
            np.sum(
                w[proposer_id]
                * np.where(negative, -selected_gains, 0.0)
            )
        )
        total_mass = positive_mass + negative_mass
        if total_mass > eps:
            support[proposer_id, block_id] = positive_mass / total_mass
            conflict[proposer_id, block_id] = negative_mass / total_mass
        actuation_eligible[proposer_id, block_id] = bool(
            alt_value > max(0.0, path_value) + eps
            and positive_sources[proposer_id, block_id] >= 2
            and neighbor_positive_sources[proposer_id, block_id] >= 1
            and source_coverage[source_id]
            > float(w[proposer_id, proposer_id]) + eps
        )

    return TargetBlockChallengeResult(
        candidate_score=candidate_score,
        candidate_sign=candidate_sign,
        path_score=path_score,
        alternative_score=alternative_score,
        alternative_margin=alternative_margin,
        best_source=best_source,
        best_sign=best_sign,
        best_direction=best_direction,
        response_coverage=coverage,
        positive_sources=positive_sources,
        neighbor_positive_sources=neighbor_positive_sources,
        support=support,
        conflict=conflict,
        actuation_eligible=actuation_eligible,
    )


def apply_target_block_challenge_actuator(
    *,
    commit_base_states: np.ndarray,
    fallback_committed_states: np.ndarray,
    probe_radius: np.ndarray,
    result: TargetBlockChallengeResult,
    lower_bound,
    upper_bound,
    target_num: int,
    coordinate_dim: int,
    eps: float = 1e-12,
) -> TargetBlockChallengeActuation:
    """Replace only eligible dormant-block path commits with the winner."""

    target_num = int(target_num)
    coordinate_dim = int(coordinate_dim)
    if target_num <= 0 or coordinate_dim != 3:
        raise ValueError("Challenge actuator requires 3D target blocks.")
    dimension = target_num * coordinate_dim
    fallback = np.asarray(fallback_committed_states, dtype=np.float64)
    if fallback.ndim != 2 or fallback.shape[1:] != (dimension,):
        raise ValueError("fallback_committed_states must have shape [A,D].")
    n_agents = int(fallback.shape[0])
    fallback = _finite_array(
        "fallback_committed_states", fallback, (n_agents, dimension)
    )
    bases = _finite_array(
        "commit_base_states",
        commit_base_states,
        (n_agents, dimension),
    )
    radius = _finite_array(
        "probe_radius", probe_radius, (n_agents, target_num)
    )
    if np.any(radius < 0.0):
        raise ValueError("probe_radius must be non-negative.")
    if result.actuation_eligible.shape != (n_agents, target_num):
        raise ValueError("Challenge result has incompatible block shape.")
    direction = _finite_array(
        "best_direction",
        result.best_direction,
        (n_agents, target_num, coordinate_dim),
    )
    lower = np.broadcast_to(
        np.asarray(lower_bound, dtype=np.float64), (dimension,)
    ).reshape(target_num, coordinate_dim)
    upper = np.broadcast_to(
        np.asarray(upper_bound, dtype=np.float64), (dimension,)
    ).reshape(target_num, coordinate_dim)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Challenge actuator bounds are invalid.")

    committed = fallback.reshape(
        n_agents, target_num, coordinate_dim
    ).copy()
    base_blocks = bases.reshape(n_agents, target_num, coordinate_dim)
    applied = np.zeros((n_agents, target_num), dtype=bool)
    applied_norm = np.zeros((n_agents, target_num), dtype=np.float64)
    boundary_clipped = np.zeros((n_agents, target_num), dtype=bool)
    for agent_id, block_id in np.argwhere(result.actuation_eligible):
        requested = (
            base_blocks[agent_id, block_id]
            + radius[agent_id, block_id]
            * direction[agent_id, block_id]
        )
        clipped = np.clip(requested, lower[block_id], upper[block_id])
        delta = clipped - base_blocks[agent_id, block_id]
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(norm) or norm <= eps:
            continue
        committed[agent_id, block_id] = clipped
        applied[agent_id, block_id] = True
        applied_norm[agent_id, block_id] = norm
        boundary_clipped[agent_id, block_id] = bool(
            np.any(np.abs(clipped - requested) > 1e-12)
        )
    flat = committed.reshape(n_agents, dimension)
    if not np.all(np.isfinite(flat)):
        raise FloatingPointError(
            "Challenge-response actuator emitted non-finite states."
        )
    return TargetBlockChallengeActuation(
        committed_states=flat,
        applied=applied,
        applied_norm=applied_norm,
        boundary_clipped=boundary_clipped,
    )

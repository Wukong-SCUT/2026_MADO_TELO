"""Pure helpers for D3-P local-SPSA candidate-response consensus.

The helpers in this module intentionally do not own an environment, mutate a
global RNG, or call a global objective.  Environment code remains responsible
for invoking each agent-local residual closure and for charging FEs and
communication.
"""

from dataclasses import dataclass
import hashlib
from typing import Tuple

import numpy as np

from env.optimizer.committee_verification import (
    aggregate_candidate_scores,
    build_rank_tensor,
    select_verified_candidates,
)


@dataclass(frozen=True)
class SPSAProbeGeometry:
    direction: np.ndarray
    trust_radius: np.ndarray
    probe_radius: np.ndarray
    plus_x: np.ndarray
    minus_x: np.ndarray
    probe_active: np.ndarray


@dataclass(frozen=True)
class SPSAGeneration:
    candidate_x: np.ndarray
    response_delta: np.ndarray
    confidence: np.ndarray
    active_mask: np.ndarray


@dataclass(frozen=True)
class CandidateResponseSelection:
    verified_x: np.ndarray
    selected_source: np.ndarray
    selected_score: np.ndarray
    selection_confidence: np.ndarray
    support_ratio: np.ndarray
    accepted_mask: np.ndarray


@dataclass(frozen=True)
class ActuatorResult:
    actuated_x: np.ndarray
    requested_shift_norm: np.ndarray
    applied_shift_norm: np.ndarray
    applied_mask: np.ndarray
    rollback_mask: np.ndarray


@dataclass(frozen=True)
class ProbeSecantSamples:
    step: np.ndarray
    response: np.ndarray
    center: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class MultisecantDirection:
    direction: np.ndarray
    active_mask: np.ndarray
    history_size: np.ndarray
    effective_rank: np.ndarray
    condition_proxy: np.ndarray
    gradient_norm: np.ndarray
    predicted_response: np.ndarray
    fallback_reason: np.ndarray


def _validate_target_layout(
    states: np.ndarray,
    target_num: int,
    coordinate_dim: int,
) -> Tuple[np.ndarray, int, int, int]:
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Candidate-response states must be [agent,D], got {x.shape}.")
    n_agents, dimension = x.shape
    target = int(target_num)
    coord = int(coordinate_dim)
    if target <= 0 or coord <= 0 or dimension != target * coord:
        raise ValueError(
            "Candidate-response target layout mismatch: "
            f"states={x.shape}, target_num={target}, coordinate_dim={coord}."
        )
    return x, n_agents, target, coord


def deterministic_rademacher(
    *,
    seed: int,
    function_id: int,
    event_id: int,
    target_num: int,
    coordinate_dim: int,
) -> np.ndarray:
    """Return a stable {-1,+1} target-block direction without global RNG use."""
    target = int(target_num)
    coord = int(coordinate_dim)
    if target <= 0 or coord <= 0:
        raise ValueError("target_num and coordinate_dim must be positive.")
    key = (
        f"d3p-spsa-v1|{int(seed)}|{int(function_id)}|{int(event_id)}|"
        f"{target}|{coord}"
    ).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=16).digest()
    local_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(local_seed)
    bits = rng.integers(0, 2, size=(target, coord), dtype=np.int8)
    return (2 * bits - 1).astype(np.float64, copy=False)


def _canonical_block_directions(
    direction: np.ndarray,
    *,
    n_agents: int,
    target_num: int,
    coordinate_dim: int,
) -> np.ndarray:
    """Return [agent,target,coord] directions without altering legacy ±1 blocks."""
    z = np.asarray(direction, dtype=np.float64)
    target = int(target_num)
    coord = int(coordinate_dim)
    if z.shape == (target, coord):
        return np.broadcast_to(
            z[None, :, :],
            (int(n_agents), target, coord),
        ).copy()
    expected = (int(n_agents), target, coord)
    if z.shape != expected:
        raise ValueError(
            "Candidate-response direction shape mismatch: "
            f"expected {(target, coord)} or {expected}, got {z.shape}."
        )
    return z.copy()


def normalize_probe_directions(
    direction: np.ndarray,
    *,
    coordinate_dim: int,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize continuous block directions to the legacy Rademacher L2 norm.

    A coordinate-wise {-1,+1} block has L2 norm sqrt(coordinate_dim).  Keeping
    this norm preserves the existing probe and pre-verifier candidate
    displacement scales while changing only the direction source.
    """
    z = np.asarray(direction, dtype=np.float64)
    if z.ndim != 3 or z.shape[2] != int(coordinate_dim):
        raise ValueError(
            "Continuous candidate directions must be [agent,target,coord], "
            f"got {z.shape}."
        )
    norms = np.linalg.norm(z, axis=2)
    valid = np.all(np.isfinite(z), axis=2) & np.isfinite(norms) & (
        norms > max(float(epsilon), 1e-30)
    )
    target_norm = float(np.sqrt(max(1, int(coordinate_dim))))
    out = np.zeros_like(z)
    out[valid] = (
        z[valid]
        * np.divide(
            target_norm,
            norms[valid],
            out=np.zeros_like(norms[valid]),
            where=norms[valid] > 0.0,
        )[:, None]
    )
    return out, valid


def build_probe_secant_samples(
    *,
    geometry: SPSAProbeGeometry,
    residual_plus: np.ndarray,
    residual_minus: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    epsilon: float = 1e-12,
) -> ProbeSecantSamples:
    """Build normalized, agent-local secants from already-paid two-sided probes."""
    plus = np.asarray(geometry.plus_x, dtype=np.float64)
    minus = np.asarray(geometry.minus_x, dtype=np.float64)
    r_plus = np.asarray(residual_plus, dtype=np.float64)
    r_minus = np.asarray(residual_minus, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError(
            f"Probe secant state shape mismatch: plus={plus.shape}, minus={minus.shape}."
        )
    n_agents = int(plus.shape[0])
    target_num = int(r_plus.shape[1]) if r_plus.ndim == 2 else 0
    if (
        r_plus.shape != r_minus.shape
        or r_plus.ndim != 2
        or target_num <= 0
        or plus.shape[1] % target_num != 0
    ):
        raise ValueError(
            "Probe secant residual/layout mismatch: "
            f"plus={plus.shape}, residual_plus={r_plus.shape}, "
            f"residual_minus={r_minus.shape}."
        )
    coordinate_dim = int(plus.shape[1] // target_num)
    span = float(upper_bound) - float(lower_bound)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("Probe secant bounds must define a positive finite span.")
    plus_blocks = plus.reshape(n_agents, target_num, coordinate_dim)
    minus_blocks = minus.reshape(n_agents, target_num, coordinate_dim)
    step = (plus_blocks - minus_blocks) / span
    center = 0.5 * (plus_blocks + minus_blocks)
    delta = r_plus - r_minus
    denom = np.abs(r_plus) + np.abs(r_minus) + max(float(epsilon), 1e-30)
    response = np.divide(
        delta,
        denom,
        out=np.zeros_like(delta),
        where=np.isfinite(denom) & (denom > 0.0),
    )
    step_norm = np.linalg.norm(step, axis=2)
    valid = (
        np.asarray(geometry.probe_active, dtype=bool)
        & np.all(np.isfinite(step), axis=2)
        & np.all(np.isfinite(center), axis=2)
        & np.isfinite(r_plus)
        & np.isfinite(r_minus)
        & np.isfinite(delta)
        & np.isfinite(denom)
        & (denom > 0.0)
        & np.isfinite(response)
        & np.isfinite(step_norm)
        & (step_norm > max(float(epsilon), 1e-30))
    )
    return ProbeSecantSamples(
        step=np.where(valid[:, :, None], step, 0.0),
        response=np.where(valid, response, 0.0),
        center=np.where(valid[:, :, None], center, 0.0),
        valid_mask=valid,
    )


def build_multisecant_directions(
    *,
    history_step: np.ndarray,
    history_response: np.ndarray,
    history_center: np.ndarray,
    history_event_id: np.ndarray,
    history_valid: np.ndarray,
    base_states: np.ndarray,
    current_event_id: int,
    target_num: int,
    coordinate_dim: int,
    lower_bound: float,
    upper_bound: float,
    min_rank: int,
    rank_tolerance: float,
    condition_max: float,
    center_distance_max_ratio: float,
    max_age: int,
    gradient_min_norm: float,
    epsilon: float = 1e-12,
) -> MultisecantDirection:
    """Recover one guarded local residual direction per agent and target."""
    bases, n_agents, target, coord = _validate_target_layout(
        base_states,
        target_num,
        coordinate_dim,
    )
    steps = np.asarray(history_step, dtype=np.float64)
    responses = np.asarray(history_response, dtype=np.float64)
    centers = np.asarray(history_center, dtype=np.float64)
    event_ids = np.asarray(history_event_id, dtype=np.int64)
    valid = np.asarray(history_valid, dtype=bool)
    if steps.ndim != 4:
        raise ValueError(f"Multisecant history_step must be [A,T,K,C], got {steps.shape}.")
    history_size_limit = int(steps.shape[2])
    expected_step = (n_agents, target, history_size_limit, coord)
    expected_scalar = (n_agents, target, history_size_limit)
    if (
        steps.shape != expected_step
        or centers.shape != expected_step
        or responses.shape != expected_scalar
        or event_ids.shape != expected_scalar
        or valid.shape != expected_scalar
    ):
        raise ValueError(
            "Multisecant history layout mismatch: "
            f"step={steps.shape}, response={responses.shape}, "
            f"center={centers.shape}, event={event_ids.shape}, valid={valid.shape}."
        )

    span = float(upper_bound) - float(lower_bound)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("Multisecant bounds must define a positive finite span.")
    base_blocks = bases.reshape(n_agents, target, coord)
    direction = np.zeros((n_agents, target, coord), dtype=np.float64)
    active = np.zeros((n_agents, target), dtype=bool)
    sizes = np.zeros((n_agents, target), dtype=np.float64)
    ranks = np.zeros((n_agents, target), dtype=np.float64)
    conditions = np.full((n_agents, target), np.inf, dtype=np.float64)
    gradient_norm = np.zeros((n_agents, target), dtype=np.float64)
    predicted = np.zeros((n_agents, target), dtype=np.float64)
    # 0=active, 1=cold/insufficient, 2=stale, 3=rank, 4=condition,
    # 5=zero-gradient, 6=nonfinite.
    fallback_reason = np.ones((n_agents, target), dtype=np.int64)
    requested_rank = int(np.clip(int(min_rank), 1, coord))
    rank_tol = max(float(rank_tolerance), 0.0)
    condition_limit = max(float(condition_max), 1.0)
    max_center_distance = (
        max(float(center_distance_max_ratio), 0.0)
        * span
        * float(np.sqrt(max(1, coord)))
    )
    age_limit = max(0, int(max_age))
    grad_floor = max(float(gradient_min_norm), max(float(epsilon), 1e-30))

    for agent_id in range(n_agents):
        for target_id in range(target):
            mask = valid[agent_id, target_id].copy()
            initial_count = int(np.count_nonzero(mask))
            finite_mask = (
                np.all(
                    np.isfinite(steps[agent_id, target_id]),
                    axis=1,
                )
                & np.isfinite(responses[agent_id, target_id])
                & np.all(
                    np.isfinite(centers[agent_id, target_id]),
                    axis=1,
                )
            )
            mask &= finite_mask
            finite_count = int(np.count_nonzero(mask))
            removed_nonfinite = finite_count < initial_count
            removed_stale = False
            if age_limit > 0:
                ages = int(current_event_id) - event_ids[agent_id, target_id]
                before_age = int(np.count_nonzero(mask))
                mask &= (ages >= 0) & (ages <= age_limit)
                removed_stale |= int(np.count_nonzero(mask)) < before_age
            if max_center_distance > 0.0:
                distances = np.linalg.norm(
                    centers[agent_id, target_id]
                    - base_blocks[agent_id, target_id][None, :],
                    axis=1,
                )
                before_distance = int(np.count_nonzero(mask))
                mask &= np.isfinite(distances) & (
                    distances <= max_center_distance
                )
                removed_stale |= (
                    int(np.count_nonzero(mask)) < before_distance
                )
            sample_count = int(np.count_nonzero(mask))
            sizes[agent_id, target_id] = float(sample_count)
            if sample_count < requested_rank:
                if removed_stale:
                    fallback_reason[agent_id, target_id] = 2
                elif removed_nonfinite:
                    fallback_reason[agent_id, target_id] = 6
                continue
            S = steps[agent_id, target_id, mask]
            y = responses[agent_id, target_id, mask]
            try:
                u, singular, vt = np.linalg.svd(S, full_matrices=False)
            except np.linalg.LinAlgError:
                fallback_reason[agent_id, target_id] = 6
                continue
            if (
                not np.all(np.isfinite(u))
                or not np.all(np.isfinite(singular))
                or not np.all(np.isfinite(vt))
                or singular.size == 0
                or singular[0] <= max(float(epsilon), 1e-30)
            ):
                fallback_reason[agent_id, target_id] = 6
                continue
            threshold = max(
                max(float(epsilon), 1e-30),
                rank_tol * float(singular[0]),
            )
            effective_rank = int(np.count_nonzero(singular > threshold))
            ranks[agent_id, target_id] = float(effective_rank)
            if effective_rank < requested_rank:
                fallback_reason[agent_id, target_id] = 3
                continue
            retained = singular[:effective_rank]
            condition = float(retained[0] / retained[-1])
            conditions[agent_id, target_id] = condition
            if not np.isfinite(condition) or condition > condition_limit:
                fallback_reason[agent_id, target_id] = 4
                continue
            coeff = (u[:, :effective_rank].T @ y) / retained
            gradient = vt[:effective_rank].T @ coeff
            norm = float(np.linalg.norm(gradient))
            gradient_norm[agent_id, target_id] = norm
            if not np.all(np.isfinite(gradient)) or not np.isfinite(norm):
                fallback_reason[agent_id, target_id] = 6
                continue
            if norm <= grad_floor:
                fallback_reason[agent_id, target_id] = 5
                continue
            raw_direction = -gradient
            target_norm = float(np.sqrt(max(1, coord)))
            direction[agent_id, target_id] = raw_direction * (
                target_norm / norm
            )
            predicted[agent_id, target_id] = float(
                np.dot(gradient, direction[agent_id, target_id])
            )
            active[agent_id, target_id] = True
            fallback_reason[agent_id, target_id] = 0

    direction, direction_finite = normalize_probe_directions(
        direction,
        coordinate_dim=coord,
        epsilon=epsilon,
    )
    active &= direction_finite
    fallback_reason[(fallback_reason == 0) & ~active] = 6
    return MultisecantDirection(
        direction=direction,
        active_mask=active,
        history_size=sizes,
        effective_rank=ranks,
        condition_proxy=conditions,
        gradient_norm=gradient_norm,
        predicted_response=predicted,
        fallback_reason=fallback_reason,
    )


def build_probe_geometry(
    *,
    base_states: np.ndarray,
    optimizer_base_states: np.ndarray,
    proposal_states: np.ndarray,
    direction: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    lower_bound: float,
    upper_bound: float,
    trust_scale: float,
    trust_min_ratio: float,
    trust_max_ratio: float,
    probe_scale: float,
    probe_min_ratio: float,
    probe_max_ratio: float,
) -> SPSAProbeGeometry:
    bases, n_agents, target, coord = _validate_target_layout(
        base_states, target_num, coordinate_dim
    )
    opt_bases = np.asarray(optimizer_base_states, dtype=np.float64)
    proposals = np.asarray(proposal_states, dtype=np.float64)
    if opt_bases.shape != bases.shape or proposals.shape != bases.shape:
        raise ValueError(
            "Candidate-response proposal/base shape mismatch: "
            f"base={bases.shape}, optimizer_base={opt_bases.shape}, "
            f"proposal={proposals.shape}."
        )
    raw_direction = np.asarray(direction, dtype=np.float64)
    z = _canonical_block_directions(
        direction,
        n_agents=n_agents,
        target_num=target,
        coordinate_dim=coord,
    )
    direction_norm = np.linalg.norm(z, axis=2)
    direction_valid = (
        np.all(np.isfinite(z), axis=2)
        & np.isfinite(direction_norm)
        & (direction_norm > 1e-30)
    )
    lower = float(lower_bound)
    upper = float(upper_bound)
    span = upper - lower
    if not np.isfinite(lower) or not np.isfinite(upper) or span <= 0.0:
        raise ValueError(f"Invalid candidate-response bounds: [{lower}, {upper}].")

    old_blocks = opt_bases.reshape(n_agents, target, coord)
    proposal_blocks = proposals.reshape(n_agents, target, coord)
    displacement = np.linalg.norm(proposal_blocks - old_blocks, axis=2)
    trust_min = max(0.0, float(trust_min_ratio)) * span
    trust_max = max(trust_min, float(trust_max_ratio) * span)
    trust = np.clip(max(0.0, float(trust_scale)) * displacement, trust_min, trust_max)

    probe_min = max(0.0, float(probe_min_ratio)) * span
    probe_max = max(probe_min, float(probe_max_ratio) * span)
    probe = np.clip(max(0.0, float(probe_scale)) * trust, probe_min, probe_max)
    finite_geometry = (
        np.isfinite(trust)
        & np.isfinite(probe)
        & (trust > 0.0)
        & (probe > 0.0)
        & direction_valid
    )
    trust = np.where(finite_geometry, trust, 0.0)
    probe = np.where(finite_geometry, probe, 0.0)

    base_blocks = bases.reshape(n_agents, target, coord)
    signed_probe = probe[:, :, None] * z
    plus = np.clip(base_blocks + signed_probe, lower, upper)
    minus = np.clip(base_blocks - signed_probe, lower, upper)
    separation = np.linalg.norm(plus - minus, axis=2)
    probe_active = finite_geometry & np.isfinite(separation) & (separation > 1e-15 * span)
    plus = np.where(probe_active[:, :, None], plus, base_blocks)
    minus = np.where(probe_active[:, :, None], minus, base_blocks)

    return SPSAProbeGeometry(
        direction=(
            raw_direction.copy()
            if raw_direction.shape == (target, coord)
            else z.copy()
        ),
        trust_radius=trust,
        probe_radius=probe,
        plus_x=plus.reshape(n_agents, target * coord),
        minus_x=minus.reshape(n_agents, target * coord),
        probe_active=probe_active,
    )


def build_spsa_candidates(
    *,
    base_states: np.ndarray,
    residual_plus: np.ndarray,
    residual_minus: np.ndarray,
    geometry: SPSAProbeGeometry,
    target_num: int,
    coordinate_dim: int,
    lower_bound: float,
    upper_bound: float,
    confidence_min: float,
    generator_enabled_mask: np.ndarray = None,
    epsilon: float = 1e-12,
) -> SPSAGeneration:
    bases, n_agents, target, coord = _validate_target_layout(
        base_states, target_num, coordinate_dim
    )
    r_plus = np.asarray(residual_plus, dtype=np.float64)
    r_minus = np.asarray(residual_minus, dtype=np.float64)
    if r_plus.shape != (n_agents, target) or r_minus.shape != (n_agents, target):
        raise ValueError(
            "SPSA residual shape mismatch: "
            f"plus={r_plus.shape}, minus={r_minus.shape}, expected={(n_agents, target)}."
        )
    if geometry.trust_radius.shape != (n_agents, target):
        raise ValueError("SPSA geometry does not match generation layout.")

    delta = r_plus - r_minus
    denom = np.abs(r_plus) + np.abs(r_minus) + max(float(epsilon), 1e-30)
    confidence = np.divide(
        np.abs(delta),
        denom,
        out=np.zeros_like(delta),
        where=np.isfinite(denom) & (denom > 0.0),
    )
    confidence = np.clip(
        np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )
    active = (
        np.asarray(geometry.probe_active, dtype=bool)
        & np.isfinite(r_plus)
        & np.isfinite(r_minus)
        & np.isfinite(delta)
        & (np.abs(delta) > 0.0)
        & (confidence >= max(0.0, float(confidence_min)))
    )
    if generator_enabled_mask is not None:
        enabled = np.asarray(generator_enabled_mask, dtype=bool)
        if enabled.shape != (n_agents, target):
            raise ValueError(
                "Candidate generator-enabled mask shape mismatch: "
                f"expected {(n_agents, target)}, got {enabled.shape}."
            )
        active &= enabled

    base_blocks = bases.reshape(n_agents, target, coord)
    z = _canonical_block_directions(
        geometry.direction,
        n_agents=n_agents,
        target_num=target,
        coordinate_dim=coord,
    )
    step = (
        -geometry.trust_radius[:, :, None]
        * np.sign(delta)[:, :, None]
        * z
    )
    candidate_blocks = np.clip(
        base_blocks + np.where(active[:, :, None], step, 0.0),
        float(lower_bound),
        float(upper_bound),
    )
    block_finite = np.all(np.isfinite(candidate_blocks), axis=2)
    active &= block_finite
    candidate_blocks = np.where(active[:, :, None], candidate_blocks, base_blocks)
    return SPSAGeneration(
        candidate_x=candidate_blocks.reshape(n_agents, target * coord),
        response_delta=np.where(np.isfinite(delta), delta, 0.0),
        confidence=confidence,
        active_mask=active,
    )


def select_supported_target_blocks(
    *,
    base_states: np.ndarray,
    candidate_states: np.ndarray,
    base_residuals: np.ndarray,
    candidate_residuals: np.ndarray,
    generator_active_mask: np.ndarray,
    closed_mask: np.ndarray,
    weight: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    support_min: float,
    comparison_tolerance: float = 0.0,
) -> CandidateResponseSelection:
    bases, n_agents, target, coord = _validate_target_layout(
        base_states, target_num, coordinate_dim
    )
    candidates = np.asarray(candidate_states, dtype=np.float64)
    base_r = np.asarray(base_residuals, dtype=np.float64)
    cand_r = np.asarray(candidate_residuals, dtype=np.float64)
    active = np.asarray(generator_active_mask, dtype=bool)
    closed = np.asarray(closed_mask, dtype=bool)
    w = np.asarray(weight, dtype=np.float64)
    if candidates.shape != bases.shape:
        raise ValueError("Candidate-response candidate shape does not match bases.")
    expected_residual = (n_agents, n_agents, target)
    if base_r.shape != expected_residual or cand_r.shape != expected_residual:
        raise ValueError(
            f"Candidate-response residual tensor must be {expected_residual}, "
            f"got base={base_r.shape}, candidate={cand_r.shape}."
        )
    if active.shape != (n_agents, target):
        raise ValueError("Candidate-response active mask shape mismatch.")
    if closed.shape != (n_agents, n_agents) or w.shape != (n_agents, n_agents):
        raise ValueError("Candidate-response graph/weight shape mismatch.")

    ranks = build_rank_tensor(cand_r, closed)
    scores = aggregate_candidate_scores(ranks, w, closed)
    raw_selection = select_verified_candidates(
        candidates,
        scores,
        closed,
        coordinate_dim=coord,
        selection="target_block",
    )
    verified = bases.copy().reshape(n_agents, target, coord)
    source_ids = np.full((n_agents, target), -1, dtype=np.int64)
    selected_scores = np.full((n_agents, target), np.inf, dtype=np.float64)
    support = np.zeros((n_agents, target), dtype=np.float64)
    accepted = np.zeros((n_agents, target), dtype=bool)
    candidate_blocks = candidates.reshape(n_agents, target, coord)
    tolerance = max(0.0, float(comparison_tolerance))
    support_threshold = float(np.clip(support_min, 0.0, 1.0))

    for receiver in range(n_agents):
        for target_id in range(target):
            source = int(raw_selection.source_ids[receiver, target_id])
            verifiers = np.flatnonzero(closed[receiver] & closed[:, source])
            if verifiers.size == 0:
                continue
            candidate_values = cand_r[verifiers, source, target_id]
            base_values = base_r[verifiers, receiver, target_id]
            valid = np.isfinite(candidate_values) & np.isfinite(base_values)
            if not np.any(valid):
                continue
            verifier_ids = verifiers[valid]
            verifier_weights = np.maximum(w[receiver, verifier_ids], 0.0)
            denom = float(np.sum(verifier_weights))
            if denom <= 1e-12:
                verifier_weights = np.ones((verifier_ids.size,), dtype=np.float64)
                denom = float(verifier_ids.size)
            better = (
                cand_r[verifier_ids, source, target_id]
                < base_r[verifier_ids, receiver, target_id] - tolerance
            )
            support_value = float(np.dot(verifier_weights, better.astype(np.float64)) / denom)
            own_candidate = float(cand_r[receiver, source, target_id])
            own_base = float(base_r[receiver, receiver, target_id])
            own_nonworse = bool(
                np.isfinite(own_candidate)
                and np.isfinite(own_base)
                and own_candidate <= own_base + tolerance
            )
            source_active = bool(active[source, target_id])
            block_finite = bool(np.all(np.isfinite(candidate_blocks[source, target_id])))
            is_accepted = bool(
                source_active
                and block_finite
                and own_nonworse
                and support_value >= support_threshold
            )
            support[receiver, target_id] = support_value
            if not is_accepted:
                continue
            verified[receiver, target_id] = candidate_blocks[source, target_id]
            source_ids[receiver, target_id] = source
            selected_scores[receiver, target_id] = float(scores[source, target_id])
            accepted[receiver, target_id] = True

    return CandidateResponseSelection(
        verified_x=verified.reshape(n_agents, target * coord),
        selected_source=source_ids,
        selected_score=selected_scores,
        selection_confidence=np.asarray(raw_selection.confidence, dtype=np.float64),
        support_ratio=support,
        accepted_mask=accepted,
    )


def apply_bounded_actuator(
    *,
    base_states: np.ndarray,
    verified_states: np.ndarray,
    accepted_mask: np.ndarray,
    trust_radius: np.ndarray,
    target_num: int,
    coordinate_dim: int,
    beta,
    lower_bound: float,
    upper_bound: float,
) -> ActuatorResult:
    bases, n_agents, target, coord = _validate_target_layout(
        base_states, target_num, coordinate_dim
    )
    verified = np.asarray(verified_states, dtype=np.float64)
    accepted = np.asarray(accepted_mask, dtype=bool)
    trust = np.asarray(trust_radius, dtype=np.float64)
    if verified.shape != bases.shape:
        raise ValueError("Actuator verified-state shape mismatch.")
    if accepted.shape != (n_agents, target) or trust.shape != (n_agents, target):
        raise ValueError("Actuator target mask/radius shape mismatch.")
    beta_arr = np.asarray(beta, dtype=np.float64)
    if beta_arr.ndim == 0:
        beta_values = np.full(
            (n_agents,),
            float(np.clip(beta_arr.item(), 0.0, 1.0)),
            dtype=np.float64,
        )
    else:
        beta_values = np.clip(beta_arr.reshape(n_agents), 0.0, 1.0)
    base_blocks = bases.reshape(n_agents, target, coord)
    verified_blocks = verified.reshape(n_agents, target, coord)
    raw_shift = verified_blocks - base_blocks
    raw_norm = np.linalg.norm(raw_shift, axis=2)
    safe_trust = np.where(np.isfinite(trust) & (trust > 0.0), trust, 0.0)
    scale = np.ones_like(raw_norm)
    nonzero = raw_norm > 1e-30
    scale[nonzero] = np.minimum(1.0, safe_trust[nonzero] / raw_norm[nonzero])
    requested = np.where(accepted[:, :, None], raw_shift * scale[:, :, None], 0.0)
    requested_norm = np.linalg.norm(requested, axis=2)
    actuated_blocks = np.clip(
        base_blocks + beta_values[:, None, None] * requested,
        float(lower_bound),
        float(upper_bound),
    )
    rollback = ~np.all(np.isfinite(actuated_blocks), axis=(1, 2))
    if np.any(rollback):
        actuated_blocks[rollback] = base_blocks[rollback]
    applied_shift = actuated_blocks - base_blocks
    applied_norm = np.linalg.norm(applied_shift, axis=2)
    applied = accepted & (applied_norm > 1e-15 * max(1.0, float(upper_bound) - float(lower_bound)))
    applied[rollback] = False
    return ActuatorResult(
        actuated_x=actuated_blocks.reshape(n_agents, target * coord),
        requested_shift_norm=requested_norm,
        applied_shift_norm=applied_norm,
        applied_mask=applied,
        rollback_mask=rollback,
    )

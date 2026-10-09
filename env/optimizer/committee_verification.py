from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CommitteeSelection:
    verified_x: np.ndarray
    source_ids: np.ndarray
    selected_scores: np.ndarray
    confidence: np.ndarray


@dataclass
class CommitteeAcceptance:
    accepted: bool
    log_improvement: float
    effective_beta: np.ndarray


def _safe_log_improvement(previous: float, current: float) -> float:
    prev = float(previous)
    cur = float(current)
    if not np.isfinite(prev) or not np.isfinite(cur):
        return float("nan")
    scale = max(1.0, abs(prev), abs(cur))
    eps = 1e-8 * scale
    shift = max(0.0, -min(prev, cur)) + eps
    numerator = prev + shift
    denominator = cur + shift
    if numerator <= 0.0 or denominator <= 0.0:
        return float("nan")
    return float(np.log(numerator / denominator))


def evaluate_report_improve_acceptance(
    candidate_report_f: float,
    report_f: float,
    confidence: np.ndarray,
    mix_strength: float,
    min_log_improve: float = 0.0,
) -> CommitteeAcceptance:
    """Apply the C13-r2 event-level acceptance rule without environment state."""
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    clean_conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
    clean_conf = np.clip(clean_conf, 0.0, 1.0)
    threshold = max(0.0, float(min_log_improve))
    log_improvement = _safe_log_improvement(report_f, candidate_report_f)
    accepted = bool(
        np.isfinite(log_improvement) and log_improvement > threshold
    )
    if accepted:
        beta = np.clip(float(mix_strength) * clean_conf, 0.0, 1.0)
    else:
        beta = np.zeros_like(clean_conf)
    return CommitteeAcceptance(
        accepted=accepted,
        log_improvement=float(log_improvement),
        effective_beta=beta,
    )


def build_closed_neighborhood_mask(adjacency: np.ndarray) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Committee adjacency must be square, got {adj.shape}.")
    closed = adj.copy()
    np.fill_diagonal(closed, True)
    return closed


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.empty((0,), dtype=np.float64)
    clean = np.nan_to_num(arr, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(clean, kind="mergesort")
    ranks = np.empty((arr.size,), dtype=np.float64)
    cursor = 0
    while cursor < arr.size:
        end = cursor + 1
        value = clean[order[cursor]]
        while end < arr.size and clean[order[end]] == value:
            end += 1
        average_rank = 0.5 * float(cursor + end - 1)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    if arr.size > 1:
        ranks /= float(arr.size - 1)
    else:
        ranks.fill(0.0)
    return ranks


def build_rank_tensor(
    residual_tensor: np.ndarray,
    closed_mask: np.ndarray,
) -> np.ndarray:
    residuals = np.asarray(residual_tensor, dtype=np.float64)
    closed = np.asarray(closed_mask, dtype=bool)
    if residuals.ndim != 3:
        raise ValueError(
            "Committee residual_tensor must have shape [verifier,candidate,target]."
        )
    n_verifier, n_candidate, _ = residuals.shape
    if n_verifier != n_candidate or closed.shape != (n_verifier, n_candidate):
        raise ValueError(
            f"Committee shape mismatch: residuals={residuals.shape}, closed={closed.shape}."
        )
    ranks = np.full_like(residuals, np.nan, dtype=np.float64)
    for verifier in range(n_verifier):
        candidate_ids = np.flatnonzero(closed[verifier])
        for target in range(residuals.shape[2]):
            ranks[verifier, candidate_ids, target] = average_tie_ranks(
                residuals[verifier, candidate_ids, target]
            )
    return ranks


def aggregate_candidate_scores(
    rank_tensor: np.ndarray,
    weight: np.ndarray,
    closed_mask: np.ndarray,
) -> np.ndarray:
    ranks = np.asarray(rank_tensor, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    closed = np.asarray(closed_mask, dtype=bool)
    n_agents, n_candidates, n_targets = ranks.shape
    if n_agents != n_candidates or w.shape != (n_agents, n_agents):
        raise ValueError(
            f"Committee aggregation shape mismatch: ranks={ranks.shape}, weight={w.shape}."
        )
    scores = np.full((n_agents, n_targets), np.inf, dtype=np.float64)
    for candidate in range(n_agents):
        verifier_ids = np.flatnonzero(closed[candidate])
        candidate_weights = np.maximum(w[candidate, verifier_ids], 0.0)
        for target in range(n_targets):
            vals = ranks[verifier_ids, candidate, target]
            valid = np.isfinite(vals)
            if not np.any(valid):
                continue
            weights_valid = candidate_weights[valid]
            denom = float(np.sum(weights_valid))
            if denom <= 1e-12:
                weights_valid = np.ones((int(np.sum(valid)),), dtype=np.float64)
                denom = float(weights_valid.size)
            scores[candidate, target] = float(
                np.dot(weights_valid, vals[valid]) / denom
            )
    return scores


def _select_index_with_self_tie(
    candidate_ids: np.ndarray,
    values: np.ndarray,
    self_id: int,
) -> int:
    ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if ids.size == 0 or vals.size != ids.size:
        raise ValueError("Committee selection requires aligned non-empty candidates.")
    vals = np.nan_to_num(vals, nan=np.inf, posinf=np.inf, neginf=-np.inf)
    best = float(np.min(vals))
    tied = ids[vals == best]
    if int(self_id) in tied:
        return int(self_id)
    return int(np.min(tied))


def _relative_margin(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = np.sort(np.nan_to_num(vals, nan=np.inf, posinf=np.inf, neginf=-np.inf))
    if vals.size < 2 or not np.isfinite(vals[0]) or not np.isfinite(vals[1]):
        return 0.0
    denom = max(abs(float(vals[1])), 1e-12)
    return float(np.clip((float(vals[1]) - float(vals[0])) / denom, 0.0, 1.0))


def select_verified_candidates(
    proposals: np.ndarray,
    candidate_scores: np.ndarray,
    closed_mask: np.ndarray,
    coordinate_dim: int,
    selection: str,
) -> CommitteeSelection:
    x = np.asarray(proposals, dtype=np.float64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    closed = np.asarray(closed_mask, dtype=bool)
    n_agents, dimension = x.shape
    if scores.ndim != 2 or scores.shape[0] != n_agents:
        raise ValueError(
            f"Committee score shape mismatch: proposals={x.shape}, scores={scores.shape}."
        )
    n_targets = int(scores.shape[1])
    coord = int(coordinate_dim)
    if coord <= 0 or dimension != n_targets * coord:
        raise ValueError(
            f"Committee target layout mismatch: D={dimension}, targets={n_targets}, coord={coord}."
        )
    mode = str(selection).lower()
    if mode not in {"whole", "target_block"}:
        raise ValueError(f"Unsupported committee selection: {selection}.")

    verified = np.empty_like(x)
    source_ids = np.empty((n_agents, n_targets), dtype=np.int64)
    selected_scores = np.empty((n_agents, n_targets), dtype=np.float64)
    confidence = np.zeros((n_agents,), dtype=np.float64)

    if mode == "whole":
        whole_scores = np.mean(scores, axis=1)
        for agent in range(n_agents):
            candidates = np.flatnonzero(closed[agent])
            vals = whole_scores[candidates]
            winner = _select_index_with_self_tie(candidates, vals, agent)
            verified[agent] = x[winner]
            source_ids[agent].fill(winner)
            selected_scores[agent] = scores[winner]
            confidence[agent] = _relative_margin(vals)
        return CommitteeSelection(
            verified_x=verified,
            source_ids=source_ids,
            selected_scores=selected_scores,
            confidence=confidence,
        )

    for agent in range(n_agents):
        candidates = np.flatnonzero(closed[agent])
        target_confidence = np.zeros((n_targets,), dtype=np.float64)
        for target in range(n_targets):
            vals = scores[candidates, target]
            winner = _select_index_with_self_tie(candidates, vals, agent)
            left = target * coord
            right = left + coord
            verified[agent, left:right] = x[winner, left:right]
            source_ids[agent, target] = winner
            selected_scores[agent, target] = scores[winner, target]
            target_confidence[target] = _relative_margin(vals)
        confidence[agent] = float(np.mean(target_confidence))
    return CommitteeSelection(
        verified_x=verified,
        source_ids=source_ids,
        selected_scores=selected_scores,
        confidence=confidence,
    )


def summarize_selection(
    selection: CommitteeSelection,
) -> Dict[str, float]:
    sources = np.asarray(selection.source_ids, dtype=np.int64)
    n_agents = int(sources.shape[0])
    self_hits = 0
    total = int(sources.size)
    diversity = np.empty((n_agents,), dtype=np.float64)
    for agent in range(n_agents):
        self_hits += int(np.sum(sources[agent] == agent))
        diversity[agent] = float(np.unique(sources[agent]).size)
    return {
        "self_source_ratio": float(self_hits / max(1, total)),
        "source_diversity_mean": float(np.mean(diversity)),
        "source_diversity_max": float(np.max(diversity)),
        "best_score_mean": float(np.mean(selection.selected_scores)),
        "best_score_max": float(np.max(selection.selected_scores)),
        "confidence_mean": float(np.mean(selection.confidence)),
        "confidence_max": float(np.max(selection.confidence)),
    }

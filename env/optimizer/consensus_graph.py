from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ConsensusGraph:
    weight: np.ndarray
    adjacency: np.ndarray
    source: str
    weight_mode: str
    directed_edge_count: int
    undirected_edge_count: int


def build_ring_adjacency(n_agents: int) -> np.ndarray:
    n = int(n_agents)
    if n < 2:
        raise ValueError("Ring consensus requires at least two agents.")
    adjacency = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adjacency[i, (i - 1) % n] = True
        adjacency[i, (i + 1) % n] = True
    np.fill_diagonal(adjacency, False)
    return adjacency


def adjacency_from_weight(raw_weight: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
    weight = np.asarray(raw_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"Consensus weight must be square, got shape={weight.shape}.")
    adjacency = np.abs(weight) > float(max(0.0, threshold))
    np.fill_diagonal(adjacency, False)
    return np.logical_or(adjacency, adjacency.T)


def _is_connected(adjacency: np.ndarray) -> bool:
    n = int(adjacency.shape[0])
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(adjacency[i]):
            jj = int(j)
            if jj not in seen:
                seen.add(jj)
                stack.append(jj)
    return len(seen) == n


def validate_adjacency(adjacency: np.ndarray, n_agents: Optional[int] = None) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=bool)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Consensus adjacency must be square, got shape={adj.shape}.")
    if n_agents is not None and adj.shape != (int(n_agents), int(n_agents)):
        raise ValueError(
            f"Consensus adjacency shape mismatch: expected {(int(n_agents), int(n_agents))}, "
            f"got {adj.shape}."
        )
    adj = np.logical_or(adj, adj.T)
    np.fill_diagonal(adj, False)
    if not _is_connected(adj):
        raise ValueError("Consensus graph must be connected.")
    return adj


def validate_weight_matrix(
    raw_weight: np.ndarray,
    n_agents: Optional[int] = None,
    tolerance: float = 1e-8,
) -> np.ndarray:
    weight = np.asarray(raw_weight, dtype=np.float64)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError(f"Consensus weight must be square, got shape={weight.shape}.")
    if n_agents is not None and weight.shape != (int(n_agents), int(n_agents)):
        raise ValueError(
            f"Consensus weight shape mismatch: expected {(int(n_agents), int(n_agents))}, "
            f"got {weight.shape}."
        )
    if not np.all(np.isfinite(weight)):
        raise ValueError("Consensus weight contains NaN or infinity.")
    tol = float(max(0.0, tolerance))
    if float(np.min(weight)) < -tol:
        raise ValueError("Consensus weight contains negative entries.")
    if not np.allclose(weight, weight.T, atol=tol, rtol=0.0):
        raise ValueError("Consensus weight must be symmetric in the first implementation.")
    if not np.allclose(np.sum(weight, axis=1), 1.0, atol=tol, rtol=0.0):
        raise ValueError("Consensus weight rows must sum to one.")
    if not np.allclose(np.sum(weight, axis=0), 1.0, atol=tol, rtol=0.0):
        raise ValueError("Consensus weight columns must sum to one.")
    if np.any(np.diag(weight) <= tol):
        raise ValueError("Consensus weight must include a positive self-weight at every node.")
    adjacency = adjacency_from_weight(weight, threshold=tol)
    validate_adjacency(adjacency, n_agents=weight.shape[0])
    return np.maximum(weight, 0.0)


def build_metropolis_weight(adjacency: np.ndarray) -> np.ndarray:
    adj = validate_adjacency(adjacency)
    n = int(adj.shape[0])
    degree = np.sum(adj, axis=1).astype(np.float64)
    weight = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            jj = int(j)
            weight[i, jj] = 1.0 / (1.0 + max(degree[i], degree[jj]))
        weight[i, i] = 1.0 - float(np.sum(weight[i]))
    return validate_weight_matrix(weight, n_agents=n)


def resolve_consensus_graph(
    fun,
    n_agents: int,
    graph_source: str,
    weight_mode: str,
    threshold: float = 1e-12,
) -> ConsensusGraph:
    source = str(graph_source).strip().lower()
    mode = str(weight_mode).strip().lower()
    n = int(n_agents)

    if source == "benchmark_w":
        if not hasattr(fun, "W"):
            raise ValueError(
                "The selected benchmark function has no W matrix. "
                "Use --objective_split_graph_source ring for an explicit fallback topology."
            )
        raw_weight = np.asarray(fun.W, dtype=np.float64)
        if raw_weight.shape != (n, n):
            raise ValueError(
                f"Benchmark W shape {raw_weight.shape} does not match fixed_agent_num={n}."
            )
        adjacency = validate_adjacency(
            adjacency_from_weight(raw_weight, threshold=threshold),
            n_agents=n,
        )
        if mode == "raw_validated":
            weight = validate_weight_matrix(raw_weight, n_agents=n)
        elif mode == "metropolis":
            weight = build_metropolis_weight(adjacency)
        else:
            raise ValueError(f"Unsupported objective_split_weight_mode: {mode}.")
    elif source == "ring":
        adjacency = build_ring_adjacency(n)
        if mode != "metropolis":
            raise ValueError(
                "Ring topology has no benchmark raw W; use "
                "--objective_split_weight_mode metropolis."
            )
        weight = build_metropolis_weight(adjacency)
    else:
        raise ValueError(f"Unsupported objective_split_graph_source: {source}.")

    directed = int(np.count_nonzero(adjacency))
    return ConsensusGraph(
        weight=weight,
        adjacency=adjacency,
        source=source,
        weight_mode=mode,
        directed_edge_count=directed,
        undirected_edge_count=directed // 2,
    )

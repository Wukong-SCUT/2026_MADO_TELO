from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

from optimizers.unified_opt import create_optimizer
from env.optimizer.consensus_graph import (
    adjacency_from_weight,
    build_ring_adjacency,
    resolve_consensus_graph,
    validate_adjacency,
)
from env.optimizer.opt_ma import _safe_log_improvement, opt_ma


GRAPH_CONSENSUS_MODES = {"graph_mean", "relaxed_graph_mean"}
LITE_CONSENSUS_MODES = {"ccsa_lite", "masoie_lite", "ccsa_masoie_lite"}
OBJECTIVE_SPLIT_CONSENSUS_MODES = (
    {"full_mean", "none"} | GRAPH_CONSENSUS_MODES | LITE_CONSENSUS_MODES
)
OBJECTIVE_SPLIT_EARLY_STOP_MODES = {
    "none",
    "mean_disagreement",
    "max_edge_disagreement",
    "consensus_shift",
    "masoie_velocity",
}


def _objective_split_agent_optimize_worker(
    agent_id: int,
    optimizer_name: str,
    fun,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    x_base: np.ndarray,
    options: Dict,
):
    aid = int(agent_id)
    d = int(dimension)

    def fitness_local(z_batch: np.ndarray):
        z_batch = np.asarray(z_batch, dtype=np.float64)
        if z_batch.ndim == 1:
            z_batch = z_batch[None, :]
        elif z_batch.ndim == 3 and z_batch.shape[1] == 1:
            z_batch = np.squeeze(z_batch, axis=1)
        if z_batch.ndim != 2 or z_batch.shape[1] != d:
            raise ValueError(f"fitness_local expected [N,{d}], got {z_batch.shape}")
        return np.asarray(fun.local_eval_batch(aid, z_batch), dtype=np.float64).reshape(-1)

    problem = {
        "fitness_function": fitness_local,
        "ndim_problem": d,
        "lower_boundary": float(lower_bound) * np.ones((d,), dtype=np.float64),
        "upper_boundary": float(upper_bound) * np.ones((d,), dtype=np.float64),
    }
    optimizer = create_optimizer(str(optimizer_name), problem, options)
    res = optimizer.optimize()
    return int(aid), str(optimizer_name), np.asarray(x_base, dtype=np.float64), options, res


class opt_ma_objective_split(opt_ma):
    """
    Generic objective-decomposition MAPPO environment.

    Agents do not own disjoint decision dimensions. Each agent owns one local
    objective f_i(x), optimizes a full copy of x, and then the round commits a
    simple mean-consensus solution for global evaluation.
    """

    supported_problem_families = {"WSNLocation", "DBOF1F10", "CDOCompetition", "CDOBenchF1F14", "CDOBenchF1F15", "WSNLocationMASOIE"}
    env_mode_name = "objective_split"

    def __init__(self, question: int, opts_in=None):
        super().__init__(question, opts_in=opts_in)
        if self.problem_family not in self.supported_problem_families:
            raise ValueError(
                f"{self.__class__.__name__} supports only {sorted(self.supported_problem_families)}, "
                f"got benchmark_name={self.benchmark_name}, family={self.problem_family}."
            )
        node_num = int(self.info.get("node_num", self.n_agents))
        if self.n_agents != node_num:
            raise ValueError(
                f"{self.env_mode_name} requires one agent per local objective: "
                f"fixed_agent_num={self.n_agents}, node_num={node_num}. "
                "Please align --fixed_agent_num with the benchmark node count."
            )
        if not hasattr(self.fun, "local_eval_batch") or not hasattr(self.fun, "local_eval_all_batch"):
            raise ValueError(
                f"{self.problem_family} function does not provide local objective batch interfaces."
            )

        self.full_dims = list(range(self.D))
        # Keep opt_ma's inherited observation machinery valid while overriding
        # the dimension-ratio feature to 1.0 in _build_obs.
        self.grouping_result = [self.full_dims for _ in range(self.n_agents)]
        self.agent_x = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.agent_local_f = np.full((self.n_agents,), np.inf, dtype=np.float64)
        self.consensus_mode = self._resolve_consensus_mode()
        if self.consensus_mode not in OBJECTIVE_SPLIT_CONSENSUS_MODES:
            raise ValueError(f"Unsupported consensus mode for {self.env_mode_name}: {self.consensus_mode}.")
        self.consensus_strength = float(
            getattr(self.opts, "objective_split_consensus_strength", 1.0)
        )
        if self.consensus_mode == "graph_mean":
            self.consensus_strength = 1.0
        elif self.consensus_mode == "none":
            self.consensus_strength = 0.0
        self.comm_interval = int(max(1, getattr(self.opts, "objective_split_comm_interval", 1)))
        self.comm_rounds_per_event = int(
            max(1, getattr(self.opts, "objective_split_comm_rounds", 1))
        )
        self.comm_force_rounds = int(
            max(0, getattr(self.opts, "objective_split_comm_force_rounds", 0))
        )
        self.comm_action_enable = bool(
            int(getattr(self.opts, "objective_split_comm_action_enable", 0))
        )
        raw_comm_candidates = getattr(
            self.opts, "objective_split_comm_round_candidates", [1, 2, 4, 8]
        )
        if isinstance(raw_comm_candidates, str):
            raw_comm_candidates = [
                x.strip() for x in raw_comm_candidates.split(",") if x.strip()
            ]
        self.comm_round_candidates = [
            int(max(1, int(float(x)))) for x in list(raw_comm_candidates)
        ]
        if len(self.comm_round_candidates) == 0:
            self.comm_round_candidates = [1]
        self.comm_action_reduce = str(
            getattr(self.opts, "objective_split_comm_action_reduce", "max")
        ).lower()
        self.neighbor_obs_mode = str(
            getattr(self.opts, "objective_split_neighbor_obs_mode", "auto")
        ).lower()
        if self.neighbor_obs_mode == "auto":
            self.neighbor_obs_mode = (
                "full"
                if bool(int(getattr(self.opts, "objective_split_neighbor_obs", 0)))
                else "none"
            )
        self.neighbor_obs_dim = {
            "none": 0,
            "improve": 1,
            "improve_disagreement": 2,
            "full": 3,
        }.get(self.neighbor_obs_mode, -1)
        if self.neighbor_obs_dim < 0:
            raise ValueError(
                f"Unsupported objective_split_neighbor_obs_mode: {self.neighbor_obs_mode}."
            )
        self.neighbor_obs_enabled = self.neighbor_obs_dim > 0
        self.state_comm_mode = str(
            getattr(self.opts, "objective_split_state_comm_mode", "none")
        ).lower()
        if self.state_comm_mode not in {"none", "full_mean", "graph_mean"}:
            raise ValueError(
                f"Unsupported objective_split_state_comm_mode: {self.state_comm_mode}."
            )
        self.state_comm_enabled = self.state_comm_mode != "none"
        self.state_comm_include_delta = bool(
            int(getattr(self.opts, "objective_split_state_comm_include_delta", 0))
        )
        self.state_comm_include_actual_fes = bool(
            int(getattr(self.opts, "objective_split_state_comm_include_actual_fes", 0))
        )
        self.consensus_reward_weight = float(
            max(0.0, getattr(self.opts, "objective_split_consensus_reward_weight", 0.0))
        )
        self.ccsa_momentum_decay = float(
            getattr(self.opts, "objective_split_ccsa_momentum_decay", 0.8)
        )
        self.ccsa_direction_lr = float(
            getattr(self.opts, "objective_split_ccsa_direction_lr", 0.5)
        )
        self.ccsa_scale_rate = float(
            getattr(self.opts, "objective_split_ccsa_scale_rate", 0.2)
        )
        self.ccsa_scale_min = float(
            getattr(self.opts, "objective_split_ccsa_scale_min", 0.5)
        )
        self.ccsa_scale_max = float(
            getattr(self.opts, "objective_split_ccsa_scale_max", 1.5)
        )
        self.ccsa_positive_improve = bool(
            int(getattr(self.opts, "objective_split_ccsa_positive_improve", 1))
        )
        self.masoie_velocity_decay = float(
            getattr(self.opts, "objective_split_masoie_velocity_decay", 0.5)
        )
        self.masoie_velocity_scale = float(
            getattr(self.opts, "objective_split_masoie_velocity_scale", 1.0)
        )
        self.masoie_velocity_clip_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_masoie_velocity_clip_ratio", 1.0))
        )
        self.record_comm_cost = bool(
            int(getattr(self.opts, "objective_split_record_comm_cost", 1))
        )
        self.agent_parallel_workers = int(
            max(1, getattr(self.opts, "objective_split_agent_parallel_workers", 1))
        )
        self.early_stop_mode = str(
            getattr(self.opts, "objective_split_early_stop_mode", "none")
        ).lower()
        if self.early_stop_mode not in OBJECTIVE_SPLIT_EARLY_STOP_MODES:
            raise ValueError(
                f"Unsupported objective_split_early_stop_mode: {self.early_stop_mode}."
            )
        if (
            self.early_stop_mode == "masoie_velocity"
            and self.consensus_mode not in {"masoie_lite", "ccsa_masoie_lite"}
        ):
            raise ValueError(
                "objective_split_early_stop_mode=masoie_velocity requires "
                "objective_split_consensus=masoie_lite or ccsa_masoie_lite."
            )
        self.early_stop_threshold = float(
            max(0.0, getattr(self.opts, "objective_split_early_stop_threshold", 1e-10))
        )
        self.early_stop_patience = int(
            max(1, getattr(self.opts, "objective_split_early_stop_patience", 1))
        )
        self.early_stop_check_interval = int(
            max(1, getattr(self.opts, "objective_split_early_stop_check_interval", 1))
        )
        self.early_stop_min_steps = int(
            max(0, getattr(self.opts, "objective_split_early_stop_min_steps", 0))
        )
        self.early_stop_counter = 0
        self.last_early_stop_metric = float("inf")
        self.last_early_stop_triggered = False
        self.last_early_stop_reason = ""
        self.base_obs_dim = 16
        self.state_msg_dim = (
            1 + len(self.optimizer_candidates) + 5
            if self.state_comm_enabled
            else 0
        )
        self.obs_dim = self.base_obs_dim + self.neighbor_obs_dim
        if self.state_comm_enabled:
            self.obs_dim += self.state_msg_dim
            if self.state_comm_include_delta:
                self.obs_dim += self.state_msg_dim
        self.observation_space = self.observation_space.__class__(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )

        self.graph = None
        self.consensus_weight = None
        self.consensus_adjacency = None
        if self._needs_consensus_graph() or self._needs_state_comm_graph():
            self.graph = resolve_consensus_graph(
                fun=self.fun,
                n_agents=self.n_agents,
                graph_source=getattr(self.opts, "objective_split_graph_source", "benchmark_w"),
                weight_mode=getattr(self.opts, "objective_split_weight_mode", "metropolis"),
                threshold=float(
                    getattr(self.opts, "objective_split_graph_threshold", 1e-12)
                ),
            )
            self.consensus_weight = self.graph.weight
            self.consensus_adjacency = self.graph.adjacency
        self.metric_adjacency = self._resolve_metric_adjacency()

        self.last_neighbor_summary = np.zeros((self.n_agents, 3), dtype=np.float64)
        self.agent_initial_local_f = np.zeros((self.n_agents,), dtype=np.float64)
        self.agent_best_local_f = np.full((self.n_agents,), np.inf, dtype=np.float64)
        self.last_sigma_value = np.full(
            (self.n_agents,), float(max(1e-12, getattr(self.opts, "sigma", 0.3))),
            dtype=np.float64,
        )
        self.last_actual_fes_per_agent = np.zeros((self.n_agents,), dtype=np.float64)
        self.cumulative_actual_fes_per_agent = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_consensus_shift_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_state_neighbor_message = np.zeros((self.n_agents, 16), dtype=np.float64)
        self.last_state_delta_message = np.zeros((self.n_agents, 16), dtype=np.float64)
        self.ccsa_direction_momentum = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.masoie_velocity = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.last_ccsa_scale = np.ones((self.n_agents,), dtype=np.float64)
        self.last_masoie_neighbor_pull = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.last_consensus_metrics = self._empty_consensus_metrics()
        self.local_search_fes = 0
        self.agent_state_local_evals = 0
        self.global_monitor_local_evals = 0
        self.global_monitor_rounds = 0
        self.graph_comm_rounds = 0
        self.last_comm_rounds_applied = 0
        self.last_comm_rounds_requested = int(self.comm_rounds_per_event)
        self.last_comm_round_idx = np.zeros((self.n_agents,), dtype=np.int64)
        self.ccsa_lite_rounds = 0
        self.masoie_lite_rounds = 0
        self.centralized_full_mean_rounds = 0
        self.total_comm_rounds_applied = 0
        self.total_comm_events = 0
        self.step_comm_rounds_applied = 0
        self.step_comm_events = 0
        self.graph_messages = 0
        self.graph_transmitted_floats = 0

    def _resolve_consensus_mode(self) -> str:
        mode = getattr(self.opts, "objective_split_consensus", "")
        if str(mode).strip():
            return str(mode).lower()
        return str(getattr(self.opts, "wsn_objective_consensus", "full_mean")).lower()

    def _needs_consensus_graph(self) -> bool:
        return self.consensus_mode in (GRAPH_CONSENSUS_MODES | LITE_CONSENSUS_MODES)

    def _needs_state_comm_graph(self) -> bool:
        return self.state_comm_mode == "graph_mean"

    def _resolve_metric_adjacency(self):
        if self.consensus_adjacency is not None:
            return self.consensus_adjacency
        source = str(
            getattr(self.opts, "objective_split_graph_source", "benchmark_w")
        ).lower()
        if source == "ring":
            return build_ring_adjacency(self.n_agents)
        if source == "benchmark_w" and hasattr(self.fun, "W"):
            try:
                return validate_adjacency(
                    adjacency_from_weight(
                        np.asarray(self.fun.W, dtype=np.float64),
                        threshold=float(
                            getattr(
                                self.opts,
                                "objective_split_graph_threshold",
                                1e-12,
                            )
                        ),
                    ),
                    n_agents=self.n_agents,
                )
            except ValueError:
                return None
        return None

    def _eval_all_local(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.fun.local_eval_all_batch(x), dtype=np.float64).reshape(-1)

    def _eval_global_and_all_local(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        local_vals = self._eval_all_local(x)
        return float(np.mean(local_vals)), local_vals

    def _eval_agent_local_states(self, agent_x: np.ndarray) -> np.ndarray:
        states = np.asarray(agent_x, dtype=np.float64).reshape(self.n_agents, self.D)
        vals = np.empty((self.n_agents,), dtype=np.float64)
        for i in range(self.n_agents):
            vals[i] = float(
                np.asarray(
                    self.fun.local_eval_batch(i, states[i]),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
        return vals

    @staticmethod
    def _empty_consensus_metrics() -> Dict[str, float]:
        return {
            "pre_state_mean_disagreement": 0.0,
            "pre_state_max_edge_disagreement": 0.0,
            "proposal_mean_disagreement": 0.0,
            "proposal_max_edge_disagreement": 0.0,
            "post_mean_disagreement": 0.0,
            "post_max_edge_disagreement": 0.0,
            "consensus_improvement": 0.0,
            "consensus_operator_improvement": 0.0,
            "direction_agreement_mean": 0.0,
            "ccsa_momentum_norm_mean": 0.0,
            "ccsa_scale_mean": 1.0,
            "ccsa_scale_std": 0.0,
            "masoie_velocity_norm_mean": 0.0,
            "masoie_neighbor_pull_norm_mean": 0.0,
        }

    def _state_disagreement(
        self,
        states: np.ndarray,
        adjacency: np.ndarray = None,
    ) -> Tuple[float, float, np.ndarray]:
        x = np.asarray(states, dtype=np.float64).reshape(self.n_agents, self.D)
        normalizer = max(
            1e-12,
            float(np.sqrt(self.D) * max(1e-12, self.ub - self.lb)),
        )
        center = np.mean(x, axis=0)
        node_to_mean = np.linalg.norm(x - center[None, :], axis=1) / normalizer
        mean_disagreement = float(np.mean(node_to_mean))

        adj = adjacency
        if adj is None:
            adj = np.ones((self.n_agents, self.n_agents), dtype=bool)
            np.fill_diagonal(adj, False)
        edge_i, edge_j = np.where(np.triu(np.asarray(adj, dtype=bool), k=1))
        if edge_i.size:
            edge_vals = np.linalg.norm(x[edge_i] - x[edge_j], axis=1) / normalizer
            max_edge = float(np.max(edge_vals))
        else:
            max_edge = 0.0
        return mean_disagreement, max_edge, node_to_mean

    def _resolve_early_stop_metric(
        self,
        post_mean_disagreement: float,
        post_max_edge_disagreement: float,
    ) -> Tuple[float, str]:
        mode = self.early_stop_mode
        if mode == "none":
            return float("inf"), "none"
        if mode == "mean_disagreement":
            return float(post_mean_disagreement), "post_mean_disagreement"
        if mode == "max_edge_disagreement":
            return float(post_max_edge_disagreement), "post_max_edge_disagreement"
        if mode == "consensus_shift":
            return float(np.max(self.last_consensus_shift_norm)), "last_consensus_shift_norm_max"
        if mode == "masoie_velocity":
            return float(np.mean(np.linalg.norm(self.masoie_velocity, axis=1))), "masoie_velocity_norm_mean"
        raise ValueError(f"Unsupported objective_split_early_stop_mode: {mode}.")

    def _neighbor_summaries(
        self,
        local_improvements: np.ndarray,
        base_states: np.ndarray,
        proposal_states: np.ndarray,
        next_states: np.ndarray,
        weight: np.ndarray,
    ) -> np.ndarray:
        w = np.asarray(weight, dtype=np.float64)
        local_arr = np.asarray(local_improvements, dtype=np.float64).reshape(self.n_agents)
        directions = np.asarray(proposal_states, dtype=np.float64) - np.asarray(
            base_states, dtype=np.float64
        )
        direction_norm = np.linalg.norm(directions, axis=1, keepdims=True)
        direction_unit = np.divide(
            directions,
            direction_norm,
            out=np.zeros_like(directions),
            where=direction_norm > 1e-12,
        )
        pair_agreement = np.clip(
            (direction_unit @ direction_unit.T + 1.0) * 0.5,
            0.0,
            1.0,
        )
        normalizer = max(
            1e-12,
            float(np.sqrt(self.D) * max(1e-12, self.ub - self.lb)),
        )
        pair_distance = (
            np.linalg.norm(
                next_states[:, None, :] - next_states[None, :, :],
                axis=2,
            )
            / normalizer
        )
        summaries = np.zeros((self.n_agents, 3), dtype=np.float64)
        summaries[:, 0] = w @ local_arr
        summaries[:, 1] = np.sum(w * pair_distance, axis=1)
        summaries[:, 2] = np.sum(w * pair_agreement, axis=1)
        return summaries

    @staticmethod
    def _safe_unit_vectors(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1)
        units = np.divide(
            arr,
            norms[:, None],
            out=np.zeros_like(arr),
            where=norms[:, None] > 1e-12,
        )
        return units, norms

    @staticmethod
    def _clip_row_norms(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float64)
        limit = float(max_norm)
        if limit <= 0.0 or not np.isfinite(limit):
            return arr
        norms = np.linalg.norm(arr, axis=1)
        factors = np.ones_like(norms)
        mask = norms > limit
        factors[mask] = limit / np.maximum(norms[mask], 1e-12)
        return arr * factors[:, None]

    def _record_graph_communication(self, floats_per_message: int = None, rounds: int = 1):
        rounds = int(max(1, rounds))
        self.graph_comm_rounds += rounds
        self._record_communication_event(rounds=rounds)
        if self.record_comm_cost and self.graph is not None:
            messages = int(self.graph.directed_edge_count)
            self.graph_messages += messages * rounds
            if floats_per_message is None:
                floats_per_message = int(2 * self.D + 1)
            self.graph_transmitted_floats += (
                messages * rounds * int(max(1, floats_per_message))
            )

    def _record_communication_event(self, rounds: int = 1):
        rounds = int(max(1, rounds))
        self.total_comm_rounds_applied += rounds
        self.total_comm_events += 1
        self.step_comm_rounds_applied = rounds
        self.step_comm_events = 1

    def _ccsa_adjust_proposals(
        self,
        base_states: np.ndarray,
        proposals: np.ndarray,
        local_improvements: np.ndarray,
    ) -> np.ndarray:
        base = np.asarray(base_states, dtype=np.float64).reshape(self.n_agents, self.D)
        props = np.asarray(proposals, dtype=np.float64).reshape(self.n_agents, self.D)
        directions = props - base
        direction_units, step_norms = self._safe_unit_vectors(directions)

        improvements = np.nan_to_num(
            np.asarray(local_improvements, dtype=np.float64).reshape(self.n_agents),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.ccsa_positive_improve:
            strengths = np.maximum(improvements, 0.0)
        else:
            strengths = improvements
        weighted_direction_signal = direction_units * strengths[:, None]
        merged = self.consensus_weight @ weighted_direction_signal
        merged_units, merged_norms = self._safe_unit_vectors(merged)

        fallback = self.consensus_weight @ direction_units
        fallback_units, _ = self._safe_unit_vectors(fallback)
        empty_rows = merged_norms <= 1e-12
        if np.any(empty_rows):
            merged_units[empty_rows] = fallback_units[empty_rows]

        self.ccsa_direction_momentum = (
            float(self.ccsa_momentum_decay) * (self.consensus_weight @ self.ccsa_direction_momentum)
            + float(self.ccsa_direction_lr) * merged_units
        )
        self.ccsa_direction_momentum = np.nan_to_num(
            self.ccsa_direction_momentum,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        momentum_norms = np.linalg.norm(self.ccsa_direction_momentum, axis=1)
        raw_scale = np.exp(float(self.ccsa_scale_rate) * (momentum_norms - 1.0))
        scale_min = min(self.ccsa_scale_min, self.ccsa_scale_max)
        scale_max = max(self.ccsa_scale_min, self.ccsa_scale_max)
        self.last_ccsa_scale = np.clip(raw_scale, scale_min, scale_max)

        # Keep zero local movement at zero, even when the cooperative scale is non-one.
        adjusted = base + self.last_ccsa_scale[:, None] * directions
        adjusted[step_norms <= 1e-12] = props[step_norms <= 1e-12]
        self.ccsa_lite_rounds += 1
        return adjusted

    def _masoie_apply_velocity(self, proposals: np.ndarray) -> np.ndarray:
        props = np.asarray(proposals, dtype=np.float64).reshape(self.n_agents, self.D)
        neighbor_pull = self.consensus_weight @ props - props
        self.last_masoie_neighbor_pull = np.nan_to_num(
            neighbor_pull,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.masoie_velocity = (
            float(self.masoie_velocity_decay) * self.masoie_velocity
            + float(self.masoie_velocity_scale) * self.last_masoie_neighbor_pull
        )
        self.masoie_velocity = np.nan_to_num(
            self.masoie_velocity,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if self.masoie_velocity_clip_ratio > 0.0:
            pull_norms = np.linalg.norm(self.last_masoie_neighbor_pull, axis=1)
            clip_base = float(np.mean(pull_norms)) if pull_norms.size else 0.0
            if clip_base > 1e-12:
                self.masoie_velocity = self._clip_row_norms(
                    self.masoie_velocity,
                    float(self.masoie_velocity_clip_ratio) * clip_base,
                )
        self.masoie_lite_rounds += 1
        return props + float(self.consensus_strength) * self.masoie_velocity

    def _resolve_comm_rounds_from_actions(self, comm_actions=None) -> int:
        if self.comm_force_rounds > 0:
            self.last_comm_rounds_requested = int(self.comm_force_rounds)
            if self.comm_action_enable and comm_actions is not None:
                idx = np.clip(
                    np.asarray(comm_actions, dtype=np.int64).reshape(self.n_agents),
                    0,
                    len(self.comm_round_candidates) - 1,
                )
                self.last_comm_round_idx = idx.copy()
            else:
                self.last_comm_round_idx.fill(0)
            return int(self.comm_force_rounds)

        if (not self.comm_action_enable) or comm_actions is None:
            self.last_comm_round_idx.fill(0)
            self.last_comm_rounds_requested = int(self.comm_rounds_per_event)
            return int(self.comm_rounds_per_event)

        idx = np.clip(
            np.asarray(comm_actions, dtype=np.int64).reshape(self.n_agents),
            0,
            len(self.comm_round_candidates) - 1,
        )
        self.last_comm_round_idx = idx.copy()
        vals = np.asarray(
            [self.comm_round_candidates[int(i)] for i in idx],
            dtype=np.float64,
        )
        reduce_mode = str(self.comm_action_reduce).lower()
        if reduce_mode == "min":
            rounds = int(np.min(vals))
        elif reduce_mode == "median":
            rounds = int(round(float(np.median(vals))))
        elif reduce_mode == "mean_round":
            rounds = int(round(float(np.mean(vals))))
        else:
            rounds = int(np.max(vals))
        rounds = int(max(1, rounds))
        self.last_comm_rounds_requested = rounds
        return rounds

    def _apply_consensus(
        self,
        base_states: np.ndarray,
        proposal_states: np.ndarray,
        local_improvements: np.ndarray,
        comm_rounds: int = None,
    ) -> Tuple[np.ndarray, bool]:
        proposals = np.asarray(proposal_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        if self.consensus_mode == "full_mean":
            x_mean = np.mean(proposals, axis=0)
            next_states = np.repeat(x_mean[None, :], self.n_agents, axis=0)
            self.centralized_full_mean_rounds += 1
            self.last_comm_rounds_applied = 1
            self._record_communication_event(rounds=1)
            return np.clip(next_states, self.lb, self.ub), True

        do_comm = (
            self._needs_consensus_graph()
            and ((self.step_count + 1) % self.comm_interval == 0)
        )
        if not do_comm:
            self.last_comm_rounds_applied = 0
            self.step_comm_rounds_applied = 0
            self.step_comm_events = 0
            return np.clip(proposals, self.lb, self.ub), False

        eta = float(self.consensus_strength)
        if self.consensus_mode in GRAPH_CONSENSUS_MODES:
            rounds = int(max(1, comm_rounds if comm_rounds is not None else self.comm_rounds_per_event))
            next_states = proposals
            for _ in range(rounds):
                mixed = self.consensus_weight @ next_states
                next_states = (1.0 - eta) * next_states + eta * mixed
            self.last_comm_rounds_applied = rounds
            self._record_graph_communication(rounds=rounds)
            return np.clip(next_states, self.lb, self.ub), True

        if self.consensus_mode == "ccsa_lite":
            adjusted = self._ccsa_adjust_proposals(
                base_states=base_states,
                proposals=proposals,
                local_improvements=local_improvements,
            )
            mixed = self.consensus_weight @ adjusted
            next_states = (1.0 - eta) * adjusted + eta * mixed
            self.last_comm_rounds_applied = 1
            self._record_graph_communication(floats_per_message=int(4 * self.D + 2))
            return np.clip(next_states, self.lb, self.ub), True

        if self.consensus_mode == "masoie_lite":
            next_states = self._masoie_apply_velocity(proposals)
            self.last_comm_rounds_applied = 1
            self._record_graph_communication(floats_per_message=int(2 * self.D))
            return np.clip(next_states, self.lb, self.ub), True

        if self.consensus_mode == "ccsa_masoie_lite":
            adjusted = self._ccsa_adjust_proposals(
                base_states=base_states,
                proposals=proposals,
                local_improvements=local_improvements,
            )
            next_states = self._masoie_apply_velocity(adjusted)
            self.last_comm_rounds_applied = 1
            self._record_graph_communication(floats_per_message=int(5 * self.D + 2))
            return np.clip(next_states, self.lb, self.ub), True

        next_states = proposals
        self.last_comm_rounds_applied = 0
        self.step_comm_rounds_applied = 0
        self.step_comm_events = 0
        return np.clip(next_states, self.lb, self.ub), True

    def _normalize_candidate_x(self, raw_x, agent_id: int, x_fallback: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Normalize optimizer outputs to one full decision vector of length D.
        """
        arr = np.asarray(raw_x, dtype=np.float64)
        if arr.size == self.D:
            x = arr.reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, False

        if arr.ndim >= 2 and arr.shape[-1] == self.D:
            candidates = arr.reshape(-1, self.D)
        elif arr.size > 0 and arr.size % self.D == 0:
            candidates = arr.reshape(-1, self.D)
        else:
            x = np.asarray(x_fallback, dtype=np.float64).reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, True

        candidates = np.clip(candidates, self.lb, self.ub)
        vals = np.asarray(self.fun.local_eval_batch(agent_id, candidates), dtype=np.float64).reshape(-1)
        if vals.size != candidates.shape[0] or not np.any(np.isfinite(vals)):
            x = np.asarray(x_fallback, dtype=np.float64).reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, True

        vals = np.nan_to_num(vals, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        idx = int(np.argmin(vals))
        return candidates[idx].copy(), float(vals[idx]), True

    @staticmethod
    def _rank_ratio(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n <= 1:
            return np.zeros((n,), dtype=np.float64)
        arr = np.nan_to_num(arr, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty((n,), dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / float(max(1, n - 1))

    def _sigma_log_norm_values(self) -> np.ndarray:
        sigma_map = self._sigma_level_map()
        anchor = self._sigma_anchor()
        explicit = [
            float(max(1e-12, anchor * float(v)))
            for v in sigma_map.values()
            if np.isfinite(float(v)) and float(v) > 0
        ]
        if len(explicit) == 0:
            explicit = [float(max(1e-12, getattr(self.opts, "sigma", 0.3)))]
        lo = float(max(1e-12, min(explicit)))
        hi = float(max(lo, max(explicit)))
        denom = float(np.log(hi) - np.log(lo))
        vals = np.asarray(self.last_sigma_value, dtype=np.float64).reshape(self.n_agents)
        vals = np.maximum(vals, 1e-12)
        if abs(denom) <= 1e-12:
            return np.zeros((self.n_agents,), dtype=np.float64)
        out = (np.log(vals) - np.log(lo)) / denom
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    def _last_resource_factor_norm(self) -> np.ndarray:
        factors = np.asarray(self.resource_factors, dtype=np.float64)
        max_factor = float(max(1e-12, np.max(factors))) if factors.size else 1.0
        idx = np.clip(
            np.asarray(self.last_resource_idx, dtype=np.int64),
            0,
            max(0, len(self.resource_factors) - 1),
        )
        vals = np.asarray([self.resource_factors[int(i)] for i in idx], dtype=np.float64)
        return np.clip(vals / max_factor, 0.0, np.inf)

    def _last_actual_fes_step_norm(self) -> np.ndarray:
        max_factor = float(max(1e-12, max(self.resource_factors)))
        max_step_fes = float(max(1.0, self.subfes_per_agent * max_factor))
        vals = np.asarray(self.last_actual_fes_per_agent, dtype=np.float64)
        return np.clip(vals / max_step_fes, 0.0, np.inf)

    def _comm_round_value_norm(self) -> np.ndarray:
        if self.comm_action_enable:
            k_max = float(max(1, max(self.comm_round_candidates)))
            if self.comm_force_rounds > 0:
                vals = np.full((self.n_agents,), float(self.comm_force_rounds), dtype=np.float64)
            else:
                idx = np.clip(
                    np.asarray(self.last_comm_round_idx, dtype=np.int64),
                    0,
                    len(self.comm_round_candidates) - 1,
                )
                vals = np.asarray(
                    [self.comm_round_candidates[int(i)] for i in idx],
                    dtype=np.float64,
                )
            return np.clip(vals / k_max, 0.0, np.inf)
        avg_share = float(max(1.0, self.max_fes / max(1, self.n_agents)))
        return np.clip(
            np.asarray(self.cumulative_actual_fes_per_agent, dtype=np.float64) / avg_share,
            0.0,
            np.inf,
        )

    def _local_progress_rank_ratio(self) -> np.ndarray:
        cur = np.asarray(self.agent_local_f, dtype=np.float64).reshape(self.n_agents)
        init = np.asarray(self.agent_initial_local_f, dtype=np.float64).reshape(self.n_agents)
        best = np.asarray(self.agent_best_local_f, dtype=np.float64).reshape(self.n_agents)
        denom_best = np.abs(init - best)
        denom_init = np.maximum(1.0, np.abs(init))
        denom = np.where(denom_best > 1e-12, denom_best, denom_init)
        z = (cur - best) / (denom + 1e-12)
        return self._rank_ratio(z)

    def _build_objective_split_base_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.base_obs_dim), dtype=np.float32)
        rem_ratio = max(0.0, (self.max_fes - self.sum_fes) / max(1, self.max_fes))
        gbest_norm = float(np.tanh(np.log10(abs(self.gbest_f) + 1.0) / 10.0))
        sigma_norm = self._sigma_log_norm_values()
        actual_fes_norm = self._last_actual_fes_step_norm()
        comm_or_fes_norm = self._comm_round_value_norm()
        resource_norm = self._last_resource_factor_norm()
        progress_rank = self._local_progress_rank_ratio()

        for i in range(self.n_agents):
            local_last = self.last_local_improve[i]
            local_mean = (
                float(np.mean(self.local_improve_hist[i][-5:]))
                if self.local_improve_hist[i]
                else 0.0
            )
            opt_ratio = self.last_optimizer_idx[i] / max(1, len(self.optimizer_candidates) - 1)

            obs[i, 0] = float(progress_rank[i])
            obs[i, 1] = float(sigma_norm[i])
            obs[i, 2] = float(local_last)
            obs[i, 3] = float(actual_fes_norm[i])
            obs[i, 4] = float(rem_ratio)
            obs[i, 5] = float(comm_or_fes_norm[i])
            obs[i, 6] = float(local_mean)
            obs[i, 7] = float(self.last_team_improve)
            obs[i, 8] = float(gbest_norm)
            obs[i, 9] = float(opt_ratio)
            obs[i, 10] = float(resource_norm[i])
            obs[i, 11] = float(self.last_commit_success)
            obs[i, 12] = float(self.last_joint_vs_mean_local_gap)
            obs[i, 13] = float(self.last_local_vs_joint_gap[i])
            obs[i, 14] = float(self.last_consensus_shift_norm[i])
            obs[i, 15] = float(i) / float(max(1, self.n_agents - 1))

        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _build_state_message(self, base_obs: np.ndarray) -> np.ndarray:
        msg = np.zeros((self.n_agents, self.state_msg_dim), dtype=np.float64)
        if not self.state_comm_enabled:
            return msg
        msg[:, 0] = base_obs[:, 1]  # sigma log norm
        opt_idx = np.clip(
            np.asarray(self.last_optimizer_idx, dtype=np.int64),
            0,
            len(self.optimizer_candidates) - 1,
        )
        onehot_width = min(len(self.optimizer_candidates), max(0, self.state_msg_dim - 1))
        for i, oi in enumerate(opt_idx):
            if int(oi) < onehot_width:
                msg[i, 1 + int(oi)] = 1.0
        cursor = 1 + onehot_width
        fields = [
            base_obs[:, 10],  # resource factor norm
            base_obs[:, 5] if self.comm_action_enable else None,
            base_obs[:, 2],   # local last
            base_obs[:, 6],   # local mean
            base_obs[:, 14],  # consensus shift
        ]
        for field in fields:
            if cursor >= self.state_msg_dim:
                break
            if field is None:
                msg[:, cursor] = 0.0
            else:
                msg[:, cursor] = np.asarray(field, dtype=np.float64)
            cursor += 1
        return np.nan_to_num(msg, nan=0.0, posinf=1e6, neginf=-1e6)

    def _state_comm_weight(self) -> np.ndarray:
        if self.state_comm_mode == "full_mean":
            return np.full(
                (self.n_agents, self.n_agents),
                1.0 / float(max(1, self.n_agents)),
                dtype=np.float64,
            )
        if self.state_comm_mode == "graph_mean":
            if self.consensus_weight is None:
                raise ValueError("State communication graph_mean requires a resolved consensus graph.")
            return self.consensus_weight
        return np.eye(self.n_agents, dtype=np.float64)

    def _build_obs(self) -> np.ndarray:
        if not self.state_comm_enabled:
            obs = super()._build_obs()
            obs[:, 0] = 1.0
            if self.neighbor_obs_enabled:
                obs[:, 16 : 16 + self.neighbor_obs_dim] = self.last_neighbor_summary[
                    :, : self.neighbor_obs_dim
                ].astype(np.float32, copy=False)
            return obs

        base_obs = self._build_objective_split_base_obs()
        parts = [base_obs]
        if self.neighbor_obs_enabled:
            parts.append(
                self.last_neighbor_summary[:, : self.neighbor_obs_dim].astype(
                    np.float32, copy=False
                )
            )
        if self.state_comm_enabled:
            msg = self._build_state_message(base_obs)
            neighbor_msg = self._state_comm_weight() @ msg
            delta_msg = neighbor_msg - msg
            self.last_state_neighbor_message = neighbor_msg.copy()
            self.last_state_delta_message = delta_msg.copy()
            parts.append(neighbor_msg.astype(np.float32, copy=False))
            if self.state_comm_include_delta:
                parts.append(delta_msg.astype(np.float32, copy=False))
        obs = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def reset(self):
        self.agent_x = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.current_x = np.mean(self.agent_x, axis=0)
        self.current_f, self.agent_local_f = self._eval_global_and_all_local(self.current_x)
        self.agent_initial_local_f = self.agent_local_f.copy()
        self.agent_best_local_f = self.agent_local_f.copy()
        self.gbest_x = self.current_x.copy()
        self.gbest_f = self.current_f

        self.sum_fes = 0
        self.step_count = 0
        self.last_action_idx.fill(0)
        self.last_optimizer_idx.fill(0)
        self.last_resource_idx.fill(0)
        self.last_local_improve.fill(0.0)
        self.last_team_improve = 0.0
        self.local_improve_hist = [[] for _ in range(self.n_agents)]
        self.last_commit_success = 0.0
        self.last_joint_vs_mean_local_gap = 0.0
        self.last_local_vs_joint_gap.fill(0.0)
        self.last_local_centered.fill(0.0)
        del self.current_eval_fitness_record[:]
        del self.current_eval_individual_record[:]
        self.param_state_cache = [dict() for _ in range(self.n_agents)]
        self.last_neighbor_summary.fill(0.0)
        self.last_sigma_value.fill(float(max(1e-12, getattr(self.opts, "sigma", 0.3))))
        self.last_actual_fes_per_agent.fill(0.0)
        self.cumulative_actual_fes_per_agent.fill(0.0)
        self.last_consensus_shift_norm.fill(0.0)
        self.last_state_neighbor_message.fill(0.0)
        self.last_state_delta_message.fill(0.0)
        self.ccsa_direction_momentum.fill(0.0)
        self.masoie_velocity.fill(0.0)
        self.last_ccsa_scale.fill(1.0)
        self.last_masoie_neighbor_pull.fill(0.0)
        self.last_consensus_metrics = self._empty_consensus_metrics()
        self.local_search_fes = 0
        self.agent_state_local_evals = 0
        self.global_monitor_local_evals = 0
        self.global_monitor_rounds = 0
        self.graph_comm_rounds = 0
        self.last_comm_rounds_applied = 0
        self.last_comm_rounds_requested = int(self.comm_rounds_per_event)
        self.last_comm_round_idx.fill(0)
        self.ccsa_lite_rounds = 0
        self.masoie_lite_rounds = 0
        self.centralized_full_mean_rounds = 0
        self.total_comm_rounds_applied = 0
        self.total_comm_events = 0
        self.step_comm_rounds_applied = 0
        self.step_comm_events = 0
        self.graph_messages = 0
        self.graph_transmitted_floats = 0
        self.early_stop_counter = 0
        self.last_early_stop_metric = float("inf")
        self.last_early_stop_triggered = False
        self.last_early_stop_reason = ""

        return self._build_obs()

    def export_state(self) -> Dict:
        state = super().export_state()
        state.update(
            {
                "agent_x": self.agent_x.astype(np.float64, copy=True).tolist(),
                "agent_local_f": self.agent_local_f.astype(np.float64, copy=True).tolist(),
                "agent_initial_local_f": self.agent_initial_local_f.astype(
                    np.float64, copy=True
                ).tolist(),
                "agent_best_local_f": self.agent_best_local_f.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_neighbor_summary": self.last_neighbor_summary.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_sigma_value": self.last_sigma_value.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_actual_fes_per_agent": self.last_actual_fes_per_agent.astype(
                    np.float64, copy=True
                ).tolist(),
                "cumulative_actual_fes_per_agent": self.cumulative_actual_fes_per_agent.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_consensus_shift_norm": self.last_consensus_shift_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_state_neighbor_message": self.last_state_neighbor_message.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_state_delta_message": self.last_state_delta_message.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_consensus_metrics": dict(self.last_consensus_metrics),
                "ccsa_direction_momentum": self.ccsa_direction_momentum.astype(
                    np.float64, copy=True
                ).tolist(),
                "masoie_velocity": self.masoie_velocity.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_ccsa_scale": self.last_ccsa_scale.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_masoie_neighbor_pull": self.last_masoie_neighbor_pull.astype(
                    np.float64, copy=True
                ).tolist(),
                "local_search_fes": int(self.local_search_fes),
                "agent_state_local_evals": int(self.agent_state_local_evals),
                "global_monitor_local_evals": int(
                    self.global_monitor_local_evals
                ),
                "global_monitor_rounds": int(self.global_monitor_rounds),
                "graph_comm_rounds": int(self.graph_comm_rounds),
                "last_comm_rounds_applied": int(self.last_comm_rounds_applied),
                "last_comm_rounds_requested": int(self.last_comm_rounds_requested),
                "last_comm_round_idx": self.last_comm_round_idx.astype(
                    np.int64, copy=True
                ).tolist(),
                "ccsa_lite_rounds": int(self.ccsa_lite_rounds),
                "masoie_lite_rounds": int(self.masoie_lite_rounds),
                "centralized_full_mean_rounds": int(self.centralized_full_mean_rounds),
                "total_comm_rounds_applied": int(self.total_comm_rounds_applied),
                "total_comm_events": int(self.total_comm_events),
                "step_comm_rounds_applied": int(self.step_comm_rounds_applied),
                "step_comm_events": int(self.step_comm_events),
                "graph_messages": int(self.graph_messages),
                "graph_transmitted_floats": int(self.graph_transmitted_floats),
                "early_stop_counter": int(self.early_stop_counter),
                "last_early_stop_metric": float(self.last_early_stop_metric),
                "last_early_stop_triggered": bool(self.last_early_stop_triggered),
                "last_early_stop_reason": str(self.last_early_stop_reason),
                "graph_config": {
                    "consensus_mode": self.consensus_mode,
                    "consensus_strength": float(self.consensus_strength),
                    "comm_interval": int(self.comm_interval),
                    "comm_rounds": int(self.comm_rounds_per_event),
                    "comm_force_rounds": int(self.comm_force_rounds),
                    "comm_action_enable": bool(self.comm_action_enable),
                    "comm_round_candidates": [int(x) for x in self.comm_round_candidates],
                    "comm_action_reduce": self.comm_action_reduce,
                    "neighbor_obs_mode": self.neighbor_obs_mode,
                    "state_comm_mode": self.state_comm_mode,
                    "state_comm_include_delta": bool(self.state_comm_include_delta),
                    "state_comm_include_actual_fes": bool(
                        self.state_comm_include_actual_fes
                    ),
                    "graph_source": (
                        self.graph.source if self.graph is not None else None
                    ),
                    "weight_mode": (
                        self.graph.weight_mode if self.graph is not None else None
                    ),
                    "ccsa_momentum_decay": float(self.ccsa_momentum_decay),
                    "ccsa_direction_lr": float(self.ccsa_direction_lr),
                    "ccsa_scale_rate": float(self.ccsa_scale_rate),
                    "ccsa_scale_min": float(self.ccsa_scale_min),
                    "ccsa_scale_max": float(self.ccsa_scale_max),
                    "ccsa_positive_improve": bool(self.ccsa_positive_improve),
                    "masoie_velocity_decay": float(self.masoie_velocity_decay),
                    "masoie_velocity_scale": float(self.masoie_velocity_scale),
                    "masoie_velocity_clip_ratio": float(
                        self.masoie_velocity_clip_ratio
                    ),
                    "early_stop_mode": self.early_stop_mode,
                    "early_stop_threshold": float(self.early_stop_threshold),
                    "early_stop_patience": int(self.early_stop_patience),
                    "early_stop_check_interval": int(
                        self.early_stop_check_interval
                    ),
                    "early_stop_min_steps": int(self.early_stop_min_steps),
                },
            }
        )
        return state

    def import_state(self, state: Dict):
        super().import_state(state)
        self.agent_x = np.asarray(
            state.get(
                "agent_x",
                np.repeat(self.current_x[None, :], self.n_agents, axis=0),
            ),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        saved_local = state.get("agent_local_f", None)
        if saved_local is None:
            saved_local = self._eval_agent_local_states(self.agent_x)
        self.agent_local_f = np.asarray(saved_local, dtype=np.float64).reshape(
            self.n_agents
        )
        self.agent_initial_local_f = np.asarray(
            state.get("agent_initial_local_f", self.agent_initial_local_f),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.agent_best_local_f = np.asarray(
            state.get("agent_best_local_f", self.agent_best_local_f),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_neighbor_summary = np.asarray(
            state.get("last_neighbor_summary", self.last_neighbor_summary),
            dtype=np.float64,
        ).reshape(self.n_agents, 3)
        self.last_sigma_value = np.asarray(
            state.get("last_sigma_value", self.last_sigma_value),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_actual_fes_per_agent = np.asarray(
            state.get("last_actual_fes_per_agent", self.last_actual_fes_per_agent),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.cumulative_actual_fes_per_agent = np.asarray(
            state.get(
                "cumulative_actual_fes_per_agent",
                self.cumulative_actual_fes_per_agent,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_consensus_shift_norm = np.asarray(
            state.get("last_consensus_shift_norm", self.last_consensus_shift_norm),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_state_neighbor_message = np.asarray(
            state.get("last_state_neighbor_message", self.last_state_neighbor_message),
            dtype=np.float64,
        ).reshape(self.n_agents, 16)
        self.last_state_delta_message = np.asarray(
            state.get("last_state_delta_message", self.last_state_delta_message),
            dtype=np.float64,
        ).reshape(self.n_agents, 16)
        metrics = state.get("last_consensus_metrics", None)
        if isinstance(metrics, dict):
            self.last_consensus_metrics = {
                key: float(metrics.get(key, 0.0))
                for key in self._empty_consensus_metrics()
            }
        self.ccsa_direction_momentum = np.asarray(
            state.get("ccsa_direction_momentum", self.ccsa_direction_momentum),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.masoie_velocity = np.asarray(
            state.get("masoie_velocity", self.masoie_velocity),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.last_ccsa_scale = np.asarray(
            state.get("last_ccsa_scale", self.last_ccsa_scale),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_masoie_neighbor_pull = np.asarray(
            state.get("last_masoie_neighbor_pull", self.last_masoie_neighbor_pull),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.local_search_fes = int(state.get("local_search_fes", self.local_search_fes))
        self.agent_state_local_evals = int(
            state.get("agent_state_local_evals", self.agent_state_local_evals)
        )
        self.global_monitor_local_evals = int(
            state.get(
                "global_monitor_local_evals",
                self.global_monitor_local_evals,
            )
        )
        self.global_monitor_rounds = int(
            state.get("global_monitor_rounds", self.global_monitor_rounds)
        )
        self.graph_comm_rounds = int(
            state.get("graph_comm_rounds", self.graph_comm_rounds)
        )
        self.last_comm_rounds_applied = int(
            state.get("last_comm_rounds_applied", self.last_comm_rounds_applied)
        )
        self.last_comm_rounds_requested = int(
            state.get("last_comm_rounds_requested", self.last_comm_rounds_requested)
        )
        self.last_comm_round_idx = np.asarray(
            state.get("last_comm_round_idx", self.last_comm_round_idx),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.ccsa_lite_rounds = int(
            state.get("ccsa_lite_rounds", self.ccsa_lite_rounds)
        )
        self.masoie_lite_rounds = int(
            state.get("masoie_lite_rounds", self.masoie_lite_rounds)
        )
        self.centralized_full_mean_rounds = int(
            state.get(
                "centralized_full_mean_rounds",
                self.centralized_full_mean_rounds,
            )
        )
        self.total_comm_rounds_applied = int(
            state.get("total_comm_rounds_applied", self.total_comm_rounds_applied)
        )
        self.total_comm_events = int(
            state.get("total_comm_events", self.total_comm_events)
        )
        self.step_comm_rounds_applied = int(
            state.get("step_comm_rounds_applied", self.step_comm_rounds_applied)
        )
        self.step_comm_events = int(
            state.get("step_comm_events", self.step_comm_events)
        )
        self.graph_messages = int(state.get("graph_messages", self.graph_messages))
        self.graph_transmitted_floats = int(
            state.get("graph_transmitted_floats", self.graph_transmitted_floats)
        )
        self.early_stop_counter = int(
            state.get("early_stop_counter", self.early_stop_counter)
        )
        self.last_early_stop_metric = float(
            state.get("last_early_stop_metric", self.last_early_stop_metric)
        )
        self.last_early_stop_triggered = bool(
            state.get("last_early_stop_triggered", self.last_early_stop_triggered)
        )
        self.last_early_stop_reason = str(
            state.get("last_early_stop_reason", self.last_early_stop_reason)
        )

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        actions = np.asarray(actions)
        comm_actions = None
        if actions.ndim == 2 and actions.shape == (self.n_agents, 3):
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions = np.clip(actions[:, 1].astype(np.int64), 0, len(self.profile_candidates) - 1)
            res_actions = np.clip(actions[:, 2].astype(np.int64), 0, len(self.resource_factors) - 1)
            cfg_actions_block = np.repeat(cfg_actions[:, None], self.cfg_param_num, axis=1)
        elif actions.ndim == 2 and actions.shape == (self.n_agents, 2 + self.cfg_param_num):
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions_block = np.clip(
                actions[:, 1 : 1 + self.cfg_param_num].astype(np.int64),
                0,
                len(self.profile_candidates) - 1,
            )
            cfg_actions = cfg_actions_block[:, 0].copy()
            res_actions = np.clip(actions[:, 1 + self.cfg_param_num].astype(np.int64), 0, len(self.resource_factors) - 1)
        elif actions.ndim == 2 and actions.shape == (self.n_agents, 3 + self.cfg_param_num):
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions_block = np.clip(
                actions[:, 1 : 1 + self.cfg_param_num].astype(np.int64),
                0,
                len(self.profile_candidates) - 1,
            )
            cfg_actions = cfg_actions_block[:, 0].copy()
            res_actions = np.clip(actions[:, 1 + self.cfg_param_num].astype(np.int64), 0, len(self.resource_factors) - 1)
            comm_actions = actions[:, 2 + self.cfg_param_num].astype(np.int64)
        else:
            raise ValueError(
                f"Expected actions shape [A,3], [A,{2 + self.cfg_param_num}], or [A,{3 + self.cfg_param_num}] with A={self.n_agents}, got {actions.shape}."
            )

        self.last_optimizer_idx = opt_actions.copy()
        self.last_action_idx = cfg_actions.copy()
        self.last_resource_idx = res_actions.copy()
        requested_comm_rounds = self._resolve_comm_rounds_from_actions(comm_actions)

        agent_base_x = self.agent_x.copy()
        local_prev = self.agent_local_f.copy()
        f_prev = float(self.current_f)
        proposal_xs: List[np.ndarray] = []
        local_improvements: List[float] = []
        total_evals = 0
        candidate_shape_fallbacks = 0

        agent_tasks = []
        for i in range(self.n_agents):
            optimizer_name = self.optimizer_candidates[int(opt_actions[i])]
            subfes_i = max(1, int(round(self.subfes_per_agent * float(self.resource_factors[int(res_actions[i])]))))
            seed = int(self.opts.seed + self.step_count * 1000 + i)
            x_base_i = agent_base_x[i].copy()
            options = self._build_optimizer_options(
                agent_id=i,
                optimizer_name=optimizer_name,
                cfg_levels=cfg_actions_block[i].tolist(),
                dims=self.full_dims,
                x_base=x_base_i,
                subfes_i=subfes_i,
                seed=seed,
            )
            agent_tasks.append(
                (
                    int(i),
                    str(optimizer_name),
                    self.fun,
                    int(self.D),
                    float(self.lb),
                    float(self.ub),
                    x_base_i,
                    options,
                )
            )

        if self.agent_parallel_workers <= 1 or len(agent_tasks) <= 1:
            agent_results = [
                _objective_split_agent_optimize_worker(*task) for task in agent_tasks
            ]
        else:
            worker_num = int(min(self.agent_parallel_workers, len(agent_tasks)))
            with ProcessPoolExecutor(max_workers=worker_num) as executor:
                agent_results = list(
                    executor.map(
                        _objective_split_agent_optimize_worker,
                        [task[0] for task in agent_tasks],
                        [task[1] for task in agent_tasks],
                        [task[2] for task in agent_tasks],
                        [task[3] for task in agent_tasks],
                        [task[4] for task in agent_tasks],
                        [task[5] for task in agent_tasks],
                        [task[6] for task in agent_tasks],
                        [task[7] for task in agent_tasks],
                    )
                )
        agent_results = sorted(agent_results, key=lambda x: int(x[0]))

        for i, optimizer_name, x_base_i, options, res in agent_results:
            best_x, best_local, used_shape_fallback = self._normalize_candidate_x(
                res.get("best_so_far_x", x_base_i),
                agent_id=i,
                x_fallback=x_base_i,
            )
            n_eval = int(res["n_function_evaluations"])
            sigma_used = float(options.get("sigma", self._sigma_ref(i, optimizer_name)))
            if np.isfinite(sigma_used) and sigma_used > 0:
                self.last_sigma_value[i] = sigma_used
            self.last_actual_fes_per_agent[i] = float(max(0, n_eval))
            self.cumulative_actual_fes_per_agent[i] += float(max(0, n_eval))
            if not np.isfinite(best_local):
                best_local = float(res.get("best_so_far_y", np.inf))
            if used_shape_fallback:
                candidate_shape_fallbacks += 1
            local_improve = _safe_log_improvement(float(local_prev[i]), best_local)

            int_keys = {"n_individuals", "m", "ms", "k_init", "kmax", "lam", "distance"}
            float_keys = {"sigma", "c_s", "a_z", "c_a", "gamma", "cs", "ds", "k_inc_cond", "k_dec_cond"}
            if not self._uses_pre_caf4a62_arch():
                float_keys = set(float_keys) | {"cov_lr_scale", "c_s_scale"}
            rec = {}
            for k, v in options.items():
                if k in int_keys:
                    rec[k] = int(v)
                elif k in float_keys:
                    rec[k] = float(v)
            self.param_state_cache[int(i)][str(optimizer_name).lower()] = rec

            proposal_xs.append(best_x)
            local_improvements.append(float(local_improve))
            total_evals += int(n_eval)

        proposal_states = np.stack(proposal_xs, axis=0)
        if not np.all(np.isfinite(proposal_states)):
            raise FloatingPointError("Objective-split proposals contain NaN or infinity.")
        metric_adjacency = self.metric_adjacency
        pre_state_mean_disagreement, pre_state_max_edge, _ = self._state_disagreement(
            agent_base_x,
            adjacency=metric_adjacency,
        )
        proposal_mean_disagreement, proposal_max_edge, _ = self._state_disagreement(
            proposal_states,
            adjacency=metric_adjacency,
        )
        next_agent_x, communication_applied = self._apply_consensus(
            base_states=agent_base_x,
            proposal_states=proposal_states,
            local_improvements=np.asarray(local_improvements, dtype=np.float64),
            comm_rounds=requested_comm_rounds,
        )
        if communication_applied:
            if self.consensus_mode == "full_mean":
                summary_weight = np.full(
                    (self.n_agents, self.n_agents),
                    1.0 / float(self.n_agents),
                    dtype=np.float64,
                )
            else:
                summary_weight = self.consensus_weight
            self.last_neighbor_summary = self._neighbor_summaries(
                local_improvements=np.asarray(local_improvements, dtype=np.float64),
                base_states=agent_base_x,
                proposal_states=proposal_states,
                next_states=next_agent_x,
                weight=summary_weight,
            )
        if not np.all(np.isfinite(next_agent_x)):
            raise FloatingPointError("Objective-split agent states contain NaN or infinity.")
        if np.any(next_agent_x < self.lb - 1e-10) or np.any(
            next_agent_x > self.ub + 1e-10
        ):
            raise FloatingPointError("Objective-split agent states escaped problem bounds.")
        shift_norm = np.linalg.norm(next_agent_x - proposal_states, axis=1)
        shift_denom = max(1e-12, float(np.sqrt(self.D) * max(1e-12, self.ub - self.lb)))
        self.last_consensus_shift_norm = np.nan_to_num(
            shift_norm / shift_denom,
            nan=0.0,
            posinf=1e6,
            neginf=0.0,
        )
        if not np.all(np.isfinite(self.last_neighbor_summary)):
            raise FloatingPointError("Neighbor summaries contain NaN or infinity.")
        post_mean_disagreement, post_max_edge, _ = self._state_disagreement(
            next_agent_x,
            adjacency=metric_adjacency,
        )
        disagreement_eps = 1e-12
        consensus_improvement = float(
            np.log(
                (pre_state_mean_disagreement + disagreement_eps)
                / (post_mean_disagreement + disagreement_eps)
            )
        )
        consensus_operator_improvement = float(
            np.log(
                (proposal_mean_disagreement + disagreement_eps)
                / (post_mean_disagreement + disagreement_eps)
            )
        )

        if self.consensus_mode == "full_mean":
            x_report = next_agent_x[0].copy()
            f_report, report_local = self._eval_global_and_all_local(x_report)
            next_agent_local = report_local.copy()
            agent_state_eval_count = 0
        else:
            next_agent_local = self._eval_agent_local_states(next_agent_x)
            agent_state_eval_count = self.n_agents
            x_report = np.mean(next_agent_x, axis=0)
            f_report, report_local = self._eval_global_and_all_local(x_report)

        self.agent_x = next_agent_x
        self.agent_local_f = next_agent_local
        self.agent_best_local_f = np.minimum(self.agent_best_local_f, next_agent_local)
        self.current_x = x_report
        self.current_f = f_report
        if self.current_f <= self.gbest_f:
            self.gbest_f = self.current_f
            self.gbest_x = self.current_x.copy()

        self.last_team_improve = _safe_log_improvement(f_prev, self.current_f)
        self.last_commit_success = 1.0 if f_report <= f_prev else 0.0
        direction_agreement_mean = float(np.mean(self.last_neighbor_summary[:, 2]))
        ccsa_momentum_norms = np.linalg.norm(self.ccsa_direction_momentum, axis=1)
        masoie_velocity_norms = np.linalg.norm(self.masoie_velocity, axis=1)
        masoie_pull_norms = np.linalg.norm(self.last_masoie_neighbor_pull, axis=1)
        self.last_consensus_metrics = {
            "pre_state_mean_disagreement": float(pre_state_mean_disagreement),
            "pre_state_max_edge_disagreement": float(pre_state_max_edge),
            "proposal_mean_disagreement": float(proposal_mean_disagreement),
            "proposal_max_edge_disagreement": float(proposal_max_edge),
            "post_mean_disagreement": float(post_mean_disagreement),
            "post_max_edge_disagreement": float(post_max_edge),
            "consensus_improvement": float(consensus_improvement),
            "consensus_operator_improvement": float(
                consensus_operator_improvement
            ),
            "direction_agreement_mean": direction_agreement_mean,
            "ccsa_momentum_norm_mean": float(np.mean(ccsa_momentum_norms)),
            "ccsa_scale_mean": float(np.mean(self.last_ccsa_scale)),
            "ccsa_scale_std": float(np.std(self.last_ccsa_scale)),
            "masoie_velocity_norm_mean": float(np.mean(masoie_velocity_norms)),
            "masoie_neighbor_pull_norm_mean": float(np.mean(masoie_pull_norms)),
        }

        for i, li in enumerate(local_improvements):
            self.last_local_improve[i] = float(li)
            self.local_improve_hist[i].append(float(li))

        local_arr = np.asarray(local_improvements, dtype=np.float64)
        mean_local_improve = float(np.mean(local_arr)) if local_arr.size else 0.0
        team_improve = float(self.last_team_improve)
        self.last_joint_vs_mean_local_gap = float(team_improve - mean_local_improve)
        self.last_local_vs_joint_gap = (local_arr - team_improve).astype(np.float64, copy=False)
        self.last_local_centered = (local_arr - mean_local_improve).astype(np.float64, copy=False)

        self.local_search_fes += int(total_evals)
        self.agent_state_local_evals += int(agent_state_eval_count)
        self.global_monitor_local_evals += int(self.n_agents)
        self.global_monitor_rounds += 1
        # Preserve historical budget semantics: local optimizer evaluations plus
        # one reported global-monitor round, irrespective of diagnostic local calls.
        reported_step_evals = int(total_evals) + 1
        self.sum_fes += reported_step_evals
        self.step_count += 1

        budget_done = self.sum_fes >= self.max_fes
        early_metric, early_metric_name = self._resolve_early_stop_metric(
            post_mean_disagreement=post_mean_disagreement,
            post_max_edge_disagreement=post_max_edge,
        )
        early_check_due = (
            self.early_stop_mode != "none"
            and self.step_count >= self.early_stop_min_steps
            and self.step_count % self.early_stop_check_interval == 0
        )
        if early_check_due and np.isfinite(early_metric) and early_metric <= self.early_stop_threshold:
            self.early_stop_counter += 1
        elif early_check_due:
            self.early_stop_counter = 0
        early_done = (
            early_check_due
            and self.early_stop_mode != "none"
            and self.early_stop_counter >= self.early_stop_patience
        )
        self.last_early_stop_metric = float(early_metric)
        self.last_early_stop_triggered = bool(early_done)
        self.last_early_stop_reason = (
            early_metric_name if early_done else ("max_fes" if budget_done else "")
        )
        done = bool(budget_done or early_done)
        obs_next = self._build_obs()
        step_fitness_record = np.asarray([f_report], dtype=np.float64)
        if self.record_eval_individual:
            step_individual_record = x_report.astype(np.float32, copy=False)[None, :]
        else:
            step_individual_record = np.empty((0, self.D), dtype=np.float32)

        local_centered = self.last_local_centered.astype(np.float32, copy=False)
        mixed_reward = (
            float(self.reward_team_weight) * team_improve
            + float(self.reward_local_weight) * local_centered
        ).astype(np.float32, copy=False)
        if self.consensus_reward_weight > 0.0:
            mixed_reward = (
                mixed_reward
                + float(self.consensus_reward_weight) * float(consensus_improvement)
            ).astype(np.float32, copy=False)
        info = {
            "team_improve": self.last_team_improve,
            "local_improve": np.asarray(local_improvements, dtype=np.float32),
            "current_f": self.current_f,
            "report_x_fitness": self.current_f,
            "gbest_fitness": self.gbest_f,
            "sumFEs": self.sum_fes,
            "reported_sum_fes": self.sum_fes,
            "budget_done": bool(budget_done),
            "early_stop_triggered": bool(early_done),
            "early_stop_mode": self.early_stop_mode,
            "early_stop_reason": str(self.last_early_stop_reason),
            "early_stop_metric_name": str(early_metric_name),
            "early_stop_metric": float(early_metric),
            "early_stop_threshold": float(self.early_stop_threshold),
            "early_stop_counter": int(self.early_stop_counter),
            "early_stop_patience": int(self.early_stop_patience),
            "early_stop_check_due": bool(early_check_due),
            "early_stop_check_interval": int(self.early_stop_check_interval),
            "early_stop_min_steps": int(self.early_stop_min_steps),
            "step": self.step_count,
            "fitness_record": step_fitness_record,
            "individual_record": step_individual_record,
            "commit_success_flag": float(self.last_commit_success),
            "joint_vs_mean_local_gap": float(self.last_joint_vs_mean_local_gap),
            "local_vs_joint_gap": self.last_local_vs_joint_gap.astype(np.float32, copy=False),
            "local_centered": self.last_local_centered.astype(np.float32, copy=False),
            "team_reward": float(team_improve),
            "mixed_reward": mixed_reward,
            "env_mode": self.env_mode_name,
            "agent_parallel_workers": int(self.agent_parallel_workers),
            "candidate_shape_fallbacks": int(candidate_shape_fallbacks),
            "consensus_mode": self.consensus_mode,
            "consensus_strength": float(self.consensus_strength),
            "communication_applied": bool(communication_applied),
            "did_communicate": bool(communication_applied),
            "state_comm_mode": self.state_comm_mode,
            "state_comm_enabled": bool(self.state_comm_enabled),
            "state_comm_include_delta": bool(self.state_comm_include_delta),
            "state_comm_include_actual_fes": bool(self.state_comm_include_actual_fes),
            "state_comm_msg_dim": int(self.state_msg_dim),
            "last_actual_fes_mean": float(np.mean(self.last_actual_fes_per_agent)),
            "last_actual_fes_max": float(np.max(self.last_actual_fes_per_agent)),
            "cumulative_actual_fes_mean": float(
                np.mean(self.cumulative_actual_fes_per_agent)
            ),
            "last_consensus_shift_norm_mean": float(
                np.mean(self.last_consensus_shift_norm)
            ),
            "last_consensus_shift_norm_max": float(
                np.max(self.last_consensus_shift_norm)
            ),
            **self.last_consensus_metrics,
            "mean_disagreement": float(post_mean_disagreement),
            "max_edge_disagreement": float(post_max_edge),
            "proposal_disagreement": float(proposal_mean_disagreement),
            "consensus_improve": float(consensus_improvement),
            "consensus_operator_improve": float(
                consensus_operator_improvement
            ),
            "local_search_fes": int(self.local_search_fes),
            "agent_state_local_evals": int(self.agent_state_local_evals),
            "global_monitor_local_evals": int(self.global_monitor_local_evals),
            "global_monitor_rounds": int(self.global_monitor_rounds),
            "graph_comm_rounds": int(self.graph_comm_rounds),
            "comm_rounds": int(self.graph_comm_rounds),
            "comm_rounds_per_event": int(self.comm_rounds_per_event),
            "comm_force_rounds": int(self.comm_force_rounds),
            "last_comm_rounds_applied": int(self.last_comm_rounds_applied),
            "last_comm_rounds_requested": int(self.last_comm_rounds_requested),
            "comm_action_enable": bool(self.comm_action_enable),
            "comm_action_reduce": self.comm_action_reduce,
            "comm_round_idx_mean": float(np.mean(self.last_comm_round_idx)),
            "comm_round_idx_max": int(np.max(self.last_comm_round_idx)),
            "comm_round_idx_min": int(np.min(self.last_comm_round_idx)),
            "ccsa_lite_rounds": int(self.ccsa_lite_rounds),
            "masoie_lite_rounds": int(self.masoie_lite_rounds),
            "centralized_full_mean_rounds": int(self.centralized_full_mean_rounds),
            "total_comm_rounds_applied": int(self.total_comm_rounds_applied),
            "total_comm_events": int(self.total_comm_events),
            "step_comm_rounds_applied": int(self.step_comm_rounds_applied),
            "step_comm_events": int(self.step_comm_events),
            "graph_messages": int(self.graph_messages),
            "comm_directed_messages": int(self.graph_messages),
            "graph_transmitted_floats": int(self.graph_transmitted_floats),
            "comm_float_count": int(self.graph_transmitted_floats),
            "graph_transmitted_bytes": int(self.graph_transmitted_floats * 8),
            "comm_byte_count": int(self.graph_transmitted_floats * 8),
            "mean_local_improve": float(mean_local_improve),
            "neighbor_avg_improve_mean": float(
                np.mean(self.last_neighbor_summary[:, 0])
            ),
            "reward_consensus_monitor": float(consensus_improvement),
            "reward_final_used": mixed_reward,
            "comm_interval": int(self.comm_interval),
            "objective_split_comm_rounds": int(self.comm_rounds_per_event),
            "objective_split_comm_force_rounds": int(self.comm_force_rounds),
            "objective_split_comm_action_enable": int(self.comm_action_enable),
            "objective_split_comm_rounds_requested": int(self.last_comm_rounds_requested),
            "per_agent_local_search_fes": float(
                self.local_search_fes / max(1, self.n_agents)
            ),
            "graph_directed_edge_count": int(
                self.graph.directed_edge_count if self.graph is not None else 0
            ),
            "graph_undirected_edge_count": int(
                self.graph.undirected_edge_count if self.graph is not None else 0
            ),
            "metric_directed_edge_count": int(
                np.count_nonzero(self.metric_adjacency)
                if self.metric_adjacency is not None
                else self.n_agents * max(0, self.n_agents - 1)
            ),
        }
        return obs_next, mixed_reward, bool(done), info


class opt_ma_wsn_objective(opt_ma_objective_split):
    supported_problem_families = {"WSNLocation"}
    env_mode_name = "wsn_objective"


class opt_ma_dbo_objective(opt_ma_objective_split):
    supported_problem_families = {"DBOF1F10"}
    env_mode_name = "dbo_objective"


class opt_ma_cdo_objective(opt_ma_objective_split):
    supported_problem_families = {"CDOCompetition", "CDOBenchF1F14", "CDOBenchF1F15"}
    env_mode_name = "cdo_objective"


class opt_ma_masoie_wsn_objective(opt_ma_objective_split):
    supported_problem_families = {"WSNLocationMASOIE"}
    env_mode_name = "masoie_wsn_objective"

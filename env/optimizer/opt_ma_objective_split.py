from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from optimizers.unified_opt import create_optimizer
from optimizers.unified_opt.event_slot_session import (
    build_event_slot_plan,
    create_event_slot_session,
)
from env.optimizer.consensus_graph import (
    adjacency_from_weight,
    build_ring_adjacency,
    resolve_consensus_graph,
    validate_adjacency,
)
from env.optimizer.event_slot_diagnostics import (
    EVENT_SLOT_COMMIT_AUDIT_FIELDS,
    EVENT_SLOT_FORENSIC_FIELDS,
    EVENT_SLOT_VKD_FIELDS,
    MMES_FIELDS,
    FORENSIC_INFO_PREFIX,
)
from env.optimizer.committee_verification import (
    CommitteeSelection,
    build_closed_neighborhood_mask,
    build_rank_tensor,
    aggregate_candidate_scores,
    select_verified_candidates,
    summarize_selection,
    evaluate_report_improve_acceptance,
)
from env.optimizer.candidate_response_consensus import (
    apply_bounded_actuator,
    build_multisecant_directions,
    build_probe_geometry,
    build_probe_secant_samples,
    build_spsa_candidates,
    deterministic_rademacher,
    select_supported_target_blocks,
)
from env.optimizer.opt_ma import _safe_log_improvement, opt_ma
from env.optimizer.persistent_sepcmaes_bank import (
    PERSISTENT_SEPCMAES_BANK_STATE_VERSION,
    run_persistent_sepcmaes,
    validate_persistent_sepcmaes_snapshot,
)
from env.optimizer.persistent_sepcmaes_commit_credit import (
    PERSISTENT_SEPCMAES_COMMIT_CREDIT_MODES,
    reconcile_persistent_sepcmaes_commit,
)
from env.optimizer.target_block_cooperative_field import (
    CCSA_OUTER_STEP_RATE,
    CCSA_WSN_PATH_DECAY,
    TARGET_BLOCK_FIELD_STATE_VERSION,
    TARGET_BLOCK_RADIUS_MAX_RATIO,
    TARGET_BLOCK_RADIUS_MIN_RATIO,
    build_target_block_cooperative_field,
)
from env.optimizer.target_block_dual_clock import (
    TARGET_BLOCK_DUAL_CLOCK_STATE_VERSION,
    build_target_block_dual_clock_plan,
)
from env.optimizer.target_block_dormancy_recovery import (
    TARGET_BLOCK_DORMANCY_COOLDOWN,
    TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY,
    TARGET_BLOCK_DORMANCY_PATIENCE,
    TARGET_BLOCK_DORMANCY_PROGRESS_RATIO,
    TARGET_BLOCK_DORMANCY_STATE_VERSION,
    TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO,
)
from env.optimizer.target_block_direction_shadow import (
    TARGET_BLOCK_DIRECTION_SHADOW_SOURCES,
    TARGET_BLOCK_DIRECTION_SHADOW_STATE_VERSION,
    build_target_block_direction_shadow_plan,
    summarize_target_block_direction_shadow,
)
from env.optimizer.target_block_challenge_response import (
    TARGET_BLOCK_CHALLENGE_RESPONSE_MODES,
    TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES,
    TARGET_BLOCK_CHALLENGE_RESPONSE_STATE_VERSION,
    apply_target_block_challenge_actuator,
    build_target_block_challenge_plan,
    summarize_target_block_challenge_response,
)


def _empty_event_slot_commit_audit(slot_count: int, n_agents: int) -> Dict:
    shape = (int(slot_count), int(n_agents))
    return {
        field: np.zeros(shape, dtype=np.float64)
        for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS
    }


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


def _write_native_crash_trace(event: str, context: Dict = None, **fields) -> None:
    """Append one best-effort JSONL heartbeat for opt-in native-crash diagnosis."""
    trace_dir = str(
        os.environ.get("OBJECTIVE_SPLIT_NATIVE_TRACE_DIR", "")
    ).strip()
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        payload = {
            "event": str(event),
            "time_unix": float(time.time()),
            "pid": int(os.getpid()),
        }
        if isinstance(context, dict):
            payload.update(context)
        payload.update(fields)
        line = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8", errors="replace")
        path = os.path.join(trace_dir, f"worker_{os.getpid()}.jsonl")
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        # Diagnostics must never alter optimizer behavior or mask the real error.
        return


def _native_trace_option_summary(options: Dict) -> Dict:
    keys = (
        "max_function_evaluations",
        "seed_rng",
        "sigma",
        "n_individuals",
        "m",
        "ms",
        "k_init",
        "kmax",
        "cs",
        "c_s",
        "cov_lr_scale",
        "c_s_scale",
        "k_inc_cond",
        "k_dec_cond",
        "optimizer_guide_strength",
        "optimizer_guide_mix_strength",
        "optimizer_guide_numeric_guard",
        "optimizer_numeric_fail_soft",
        "optimizer_guide_sigma_exp_clip",
        "optimizer_guide_sigma_clip_ratio",
        "optimizer_guide_sample_clip_ratio",
    )
    summary = {}
    for key in keys:
        value = options.get(key, None)
        if isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating)):
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                value = value.item()
            summary[key] = value
    mean = np.asarray(options.get("mean", []), dtype=np.float64).reshape(-1)
    if mean.size:
        summary.update(
            {
                "mean_finite": bool(np.all(np.isfinite(mean))),
                "mean_norm": float(np.linalg.norm(mean)),
                "mean_min": float(np.nanmin(mean)),
                "mean_max": float(np.nanmax(mean)),
            }
        )
    guide = np.asarray(
        options.get("optimizer_guide_direction", []),
        dtype=np.float64,
    ).reshape(-1)
    if guide.size:
        summary.update(
            {
                "guide_finite": bool(np.all(np.isfinite(guide))),
                "guide_norm": float(np.linalg.norm(guide)),
            }
        )
    return summary


def _objective_split_agent_optimize_worker(
    agent_id: int,
    optimizer_name: str,
    fun,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    x_base: np.ndarray,
    options: Dict,
    persistent_state: Dict = None,
    persistent_signature: Dict = None,
    persistent_recenter_max_shift: float = np.inf,
):
    aid = int(agent_id)
    d = int(dimension)
    worker_options = dict(options)
    trace_context = worker_options.pop(
        "_objective_split_native_trace_context",
        {},
    )
    trace_context = (
        dict(trace_context) if isinstance(trace_context, dict) else {}
    )
    trace_context.update(
        {
            "agent_id": aid,
            "optimizer": str(optimizer_name),
            "dimension": d,
        }
    )
    _write_native_crash_trace(
        "optimizer_start",
        trace_context,
        options=_native_trace_option_summary(worker_options),
    )

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
    try:
        if (
            persistent_state is not None
            or persistent_signature is not None
            or np.isfinite(float(persistent_recenter_max_shift))
        ) and str(optimizer_name).lower() == "sepcmaes":
            (
                res,
                next_persistent_state,
                next_persistent_signature,
                persistent_telemetry,
            ) = run_persistent_sepcmaes(
                problem=problem,
                options=worker_options,
                x_base=x_base,
                previous_state=persistent_state,
                previous_signature=persistent_signature,
                recenter_max_shift=float(persistent_recenter_max_shift),
            )
            _write_native_crash_trace(
                "persistent_optimizer_created",
                trace_context,
                fresh=bool(persistent_telemetry["fresh"]),
                config_reset=bool(
                    persistent_telemetry["config_reset"]
                ),
            )
        else:
            optimizer = create_optimizer(
                str(optimizer_name),
                problem,
                worker_options,
            )
            _write_native_crash_trace("optimizer_created", trace_context)
            res = optimizer.optimize()
    except BaseException as exc:
        _write_native_crash_trace(
            "optimizer_python_exception",
            trace_context,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        raise
    _write_native_crash_trace(
        "optimizer_completed",
        trace_context,
        n_function_evaluations=int(
            res.get("n_function_evaluations", -1)
        ),
        best_so_far_y=float(res.get("best_so_far_y", np.nan)),
    )
    legacy_result = (
        int(aid),
        str(optimizer_name),
        np.asarray(x_base, dtype=np.float64),
        worker_options,
        res,
    )
    if str(optimizer_name).lower() != "sepcmaes" or not (
        persistent_state is not None
        or persistent_signature is not None
        or np.isfinite(float(persistent_recenter_max_shift))
    ):
        return legacy_result
    return legacy_result + (
        next_persistent_state,
        next_persistent_signature,
        persistent_telemetry,
    )


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
        self.information_mode = str(
            getattr(self.opts, "objective_split_information_mode", "legacy_global")
        ).lower()
        if self.information_mode not in {"legacy_global", "local_only"}:
            raise ValueError(
                "Unsupported objective_split_information_mode: "
                f"{self.information_mode}."
            )
        self.local_only_information = self.information_mode == "local_only"
        self.global_monitor_enable = bool(
            int(getattr(self.opts, "objective_split_global_monitor_enable", 1))
        )
        if not self.local_only_information and not self.global_monitor_enable:
            raise ValueError(
                "legacy_global cannot disable the exact-global evaluation path; "
                "use local_only when objective_split_global_monitor_enable=0."
            )
        raw_state_comm_cost_mode = str(
            getattr(self.opts, "objective_split_state_comm_cost_mode", "auto")
        ).lower()
        if raw_state_comm_cost_mode == "auto":
            self.state_comm_cost_mode = (
                "independent" if self.local_only_information else "legacy_untracked"
            )
        else:
            self.state_comm_cost_mode = raw_state_comm_cost_mode
        if self.state_comm_cost_mode not in {
            "legacy_untracked",
            "piggyback",
            "independent",
        }:
            raise ValueError(
                "Unsupported objective_split_state_comm_cost_mode: "
                f"{self.state_comm_cost_mode}."
            )
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
        if not hasattr(self.fun, "local_eval_batch"):
            raise ValueError(
                f"{self.problem_family} function does not provide local_eval_batch."
            )
        raw_committee_mode = str(
            getattr(self.opts, "objective_split_committee_mode", "off")
        ).lower()
        raw_acceptance_mode = str(
            getattr(
                self.opts,
                "objective_split_committee_acceptance_mode",
                "off",
            )
        ).lower()
        raw_shadow_global_eval = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_committee_shadow_global_eval",
                    1,
                )
            )
        )
        needs_global_evaluator = bool(
            not self.local_only_information
            or self.global_monitor_enable
            or raw_acceptance_mode == "report_improve"
            or (raw_committee_mode == "shadow" and raw_shadow_global_eval)
        )
        self.global_evaluator_available = hasattr(self.fun, "local_eval_all_batch")
        if needs_global_evaluator and not self.global_evaluator_available:
            raise ValueError(
                f"{self.problem_family} requires local_eval_all_batch for the "
                "selected legacy/global-monitor diagnostics, but the benchmark "
                "does not provide it."
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
        self.ccsa_momentum_update_mode = str(
            getattr(self.opts, "objective_split_ccsa_momentum_update_mode", "lite_only")
        ).lower()
        if self.ccsa_momentum_update_mode not in {"off", "lite_only", "always"}:
            raise ValueError(
                "Unsupported objective_split_ccsa_momentum_update_mode: "
                f"{self.ccsa_momentum_update_mode}."
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
        self.rvcpd_arm = str(
            getattr(self.opts, "objective_split_rvcpd_arm", "off")
        ).lower()
        if self.rvcpd_arm not in {"off", "p0", "p1", "p2"}:
            raise ValueError(
                f"Unsupported objective_split_rvcpd_arm: {self.rvcpd_arm}."
            )
        self.rvcpd_enabled = self.rvcpd_arm != "off"
        self.rvcpd_integration_mode = str(
            getattr(
                self.opts,
                "objective_split_rvcpd_integration_mode",
                "isolated",
            )
        ).lower()
        if self.rvcpd_integration_mode not in {
            "isolated",
            "d5_post_commit",
        }:
            raise ValueError(
                "Unsupported objective_split_rvcpd_integration_mode: "
                f"{self.rvcpd_integration_mode}."
            )
        self.rvcpd_d5_post_commit = bool(
            self.rvcpd_enabled
            and self.rvcpd_integration_mode == "d5_post_commit"
        )
        self.rvcpd_path_decay = float(
            getattr(self.opts, "objective_split_rvcpd_path_decay", 0.8)
        )
        self.rvcpd_direction_lr = float(
            getattr(self.opts, "objective_split_rvcpd_direction_lr", 0.5)
        )
        self.rvcpd_initial_scale = float(
            getattr(self.opts, "objective_split_rvcpd_initial_scale", 0.25)
        )
        self.rvcpd_scale_min = float(
            getattr(self.opts, "objective_split_rvcpd_scale_min", 0.05)
        )
        self.rvcpd_scale_max = float(
            getattr(self.opts, "objective_split_rvcpd_scale_max", 0.5)
        )
        self.rvcpd_scale_growth = float(
            getattr(self.opts, "objective_split_rvcpd_scale_growth", 1.05)
        )
        self.rvcpd_scale_decay = float(
            getattr(self.opts, "objective_split_rvcpd_scale_decay", 0.5)
        )
        self.rvcpd_noop_path_decay = float(
            getattr(self.opts, "objective_split_rvcpd_noop_path_decay", 0.25)
        )
        self.rvcpd_ema_decay = float(
            getattr(self.opts, "objective_split_rvcpd_ema_decay", 0.8)
        )
        self.rvcpd_probe_min_ratio = float(
            getattr(self.opts, "objective_split_rvcpd_probe_min_ratio", 0.0005)
        )
        self.rvcpd_probe_max_ratio = float(
            getattr(self.opts, "objective_split_rvcpd_probe_max_ratio", 0.05)
        )
        self.rvcpd_min_log_improve = float(
            getattr(self.opts, "objective_split_rvcpd_min_log_improve", 0.0)
        )
        self.record_comm_cost = bool(
            int(getattr(self.opts, "objective_split_record_comm_cost", 1))
        )
        self.agent_parallel_workers = int(
            max(1, getattr(self.opts, "objective_split_agent_parallel_workers", 1))
        )
        self.event_slot_interleaving_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_event_slot_interleaving_enable",
                    0,
                )
            )
        )
        if self.event_slot_interleaving_enable and not self.local_only_information:
            raise ValueError(
                "Event-slot interleaving is restricted to strict local_only "
                "control."
            )
        if self.event_slot_interleaving_enable and (
            self.consensus_mode not in GRAPH_CONSENSUS_MODES
        ):
            raise ValueError(
                "The first event-slot contract supports only graph_mean or "
                "relaxed_graph_mean consensus."
            )
        self.persistent_sepcmaes_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_persistent_sepcmaes_enable",
                    0,
                )
            )
        )
        self.persistent_sepcmaes_recenter_max_ratio = float(
            getattr(
                self.opts,
                "objective_split_persistent_sepcmaes_recenter_max_ratio",
                0.05,
            )
        )
        if (
            not np.isfinite(self.persistent_sepcmaes_recenter_max_ratio)
            or self.persistent_sepcmaes_recenter_max_ratio < 0.0
        ):
            raise ValueError(
                "Persistent SepCMAES recenter ratio must be finite and "
                "non-negative."
            )
        if self.persistent_sepcmaes_enable and not self.local_only_information:
            raise ValueError(
                "Persistent SepCMAES is restricted to "
                "objective_split_information_mode=local_only."
            )
        self.persistent_sepcmaes_recenter_max_shift = float(
            self.persistent_sepcmaes_recenter_max_ratio
            * np.mean(
                np.asarray(self.ub, dtype=np.float64)
                - np.asarray(self.lb, dtype=np.float64)
            )
        )
        self.target_block_field_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_target_block_field_enable",
                    0,
                )
            )
        )
        if self.event_slot_interleaving_enable and (
            self.persistent_sepcmaes_enable or self.target_block_field_enable
        ):
            raise ValueError(
                "Event-slot interleaving cannot be combined with persistent "
                "SepCMAES or the target-block field before work package D."
            )
        self.target_block_field_path_decay = float(
            CCSA_WSN_PATH_DECAY
        )
        self.target_block_field_step_rate = float(
            CCSA_OUTER_STEP_RATE
        )
        self.target_block_field_radius_min_ratio = float(
            TARGET_BLOCK_RADIUS_MIN_RATIO
        )
        self.target_block_field_radius_max_ratio = float(
            TARGET_BLOCK_RADIUS_MAX_RATIO
        )
        self.target_block_dual_clock_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_target_block_dual_clock_enable",
                    0,
                )
            )
        )
        self.target_block_dual_clock_commit_lock_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_target_block_dual_clock_commit_lock_enable",
                    0,
                )
            )
        )
        self.target_block_commit_credit_mode = str(
            getattr(
                self.opts,
                "objective_split_target_block_commit_credit_mode",
                "off",
            )
        ).lower()
        if (
            self.target_block_commit_credit_mode
            not in PERSISTENT_SEPCMAES_COMMIT_CREDIT_MODES
        ):
            raise ValueError(
                "Unsupported target-block commit-credit mode: "
                f"{self.target_block_commit_credit_mode}."
            )
        self.target_block_dormancy_recovery_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_target_block_dormancy_recovery_enable",
                    0,
                )
            )
        )
        self.target_block_direction_shadow_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_target_block_direction_shadow_enable",
                    0,
                )
            )
        )
        self.target_block_challenge_response_mode = str(
            getattr(
                self.opts,
                "objective_split_target_block_challenge_response_mode",
                "off",
            )
        ).lower()
        if (
            self.target_block_challenge_response_mode
            not in TARGET_BLOCK_CHALLENGE_RESPONSE_MODES
        ):
            raise ValueError(
                "Unsupported target-block challenge-response mode: "
                f"{self.target_block_challenge_response_mode}."
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
        self.optimizer_guide_enable = bool(
            int(getattr(self.opts, "objective_split_optimizer_guide_enable", 0))
        )
        self.optimizer_guide_source = str(
            getattr(
                self.opts,
                "objective_split_optimizer_guide_source",
                "neighbor_improve_direction",
            )
        ).lower()
        self.optimizer_guide_strength = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_strength", 0.5))
        )
        self.optimizer_guide_strength_schedule = str(
            getattr(self.opts, "objective_split_optimizer_guide_strength_schedule", "fixed")
        ).lower()
        if self.optimizer_guide_strength_schedule not in {"fixed", "disagreement", "budget"}:
            self.optimizer_guide_strength_schedule = "fixed"
        # σ 跨事件继承（14 卡）：开关默认关闭，OFF 时与旧实现逐位一致。
        self.sigma_inherit_enable = bool(
            int(getattr(self.opts, "objective_split_sigma_inherit_enable", 0))
        )
        self.sigma_validity_gate_enable = bool(
            int(getattr(self.opts, "objective_split_sigma_validity_gate_enable", 0))
        )
        self.sigma_validity_gate_commit_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_sigma_validity_commit_ratio", 3.0))
        )
        self.sigma_state_obs_enable = bool(
            int(getattr(self.opts, "objective_split_sigma_state_obs_enable", 0))
        ) and self.sigma_inherit_enable
        self.mmes_state_transition_mode = str(
            getattr(self.opts, "objective_split_mmes_state_transition_mode", "native")
        ).lower()
        if self.mmes_state_transition_mode not in {"native", "neutralize_credit"}:
            raise ValueError("Invalid MMES state transition mode")
        self.mmes_ratio_success_mode = str(getattr(self.opts, "objective_split_mmes_ratio_success_mode", "native")).lower()
        if self.mmes_ratio_success_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid MMES ratio success mode")
        if self.mmes_ratio_success_mode != "native" and not self.event_slot_interleaving_enable:
            raise ValueError("MMES ratio success mode requires event-slot interleaving")
        if self.mmes_ratio_success_mode != "native" and self.mmes_state_transition_mode != "native":
            raise ValueError("MMES ratio success and diagnostic credit modes cannot be combined")
        self.mmes_ratio_success_metric = str(getattr(self.opts, "objective_split_mmes_ratio_success_metric", "sigma")).lower()
        if self.mmes_ratio_success_metric not in {"sigma", "full"}:
            raise ValueError("Invalid MMES ratio success metric")
        self.mmes_ratio_success_strength = float(getattr(self.opts, "objective_split_mmes_ratio_success_strength", 0.1))
        if not np.isfinite(self.mmes_ratio_success_strength) or not 0.0 <= self.mmes_ratio_success_strength <= 1.0:
            raise ValueError("Invalid MMES ratio success strength")
        self.mmes_ratio_direction_mode = str(getattr(self.opts, "objective_split_mmes_ratio_direction_mode", "native")).lower()
        if self.mmes_ratio_direction_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid MMES ratio direction mode")
        if self.mmes_ratio_direction_mode != "native" and (not self.event_slot_interleaving_enable
                or self.mmes_state_transition_mode != "native"):
            raise ValueError("MMES ratio direction trial requires event slots and native credit mode")
        if (self.mmes_ratio_direction_mode != "native" and self.mmes_ratio_success_mode != "native"
                and self.mmes_ratio_success_metric != "full"):
            raise ValueError("Combined MMES w/p trial requires the same full sampling metric")
        self.mmes_ratio_direction_strength = float(getattr(self.opts, "objective_split_mmes_ratio_direction_strength", 0.1))
        if not np.isfinite(self.mmes_ratio_direction_strength) or not 0.0 <= self.mmes_ratio_direction_strength <= 1.0:
            raise ValueError("Invalid MMES ratio direction strength")
        self.vkd_ps_outlet_mode = str(
            getattr(self.opts, "objective_split_vkd_ps_outlet_mode", "native")
        ).lower()
        if self.vkd_ps_outlet_mode not in {"native", "sigma", "shape", "both"}:
            raise ValueError("Invalid VKD ps outlet mode")
        if self.vkd_ps_outlet_mode != "native" and not self.event_slot_interleaving_enable:
            raise ValueError("VKD ps outlet modes require event-slot interleaving")
        self.vkd_boundary_update_mode = str(
            getattr(self.opts, "objective_split_vkd_boundary_update_mode", "native")
        ).lower()
        if self.vkd_boundary_update_mode not in {"native", "candidate_a"}:
            raise ValueError("Invalid VKD boundary update mode")
        self.vkd_ratio_ps_mode = str(getattr(self.opts, "objective_split_vkd_ratio_ps_mode", "native")).lower()
        if self.vkd_ratio_ps_mode not in {"native", "attenuate"}:
            raise ValueError("Invalid VKD ratio ps mode")
        if self.vkd_ratio_ps_mode != "native" and not self.event_slot_interleaving_enable:
            raise ValueError("VKD ratio ps mode requires event-slot interleaving")
        if self.vkd_ratio_ps_mode != "native" and self.vkd_ps_outlet_mode != "native":
            raise ValueError("VKD ratio ps and diagnostic outlet modes cannot be combined")
        self.vkd_ratio_ps_strength = float(getattr(self.opts, "objective_split_vkd_ratio_ps_strength", 0.1))
        if not np.isfinite(self.vkd_ratio_ps_strength) or not 0.0 <= self.vkd_ratio_ps_strength <= 1.0:
            raise ValueError("Invalid VKD ratio ps strength")
        self.cma_sep_ratio_path_mode = str(
            getattr(self.opts, "objective_split_cma_sep_ratio_path_mode", "native")
        ).lower()
        if self.cma_sep_ratio_path_mode not in {
            "native", "step_path", "shape_path", "both_paths"
        }:
            raise ValueError("Invalid CMAES/SepCMAES ratio path mode")
        if self.cma_sep_ratio_path_mode != "native" and not self.event_slot_interleaving_enable:
            raise ValueError("CMAES/SepCMAES ratio path modes require event-slot interleaving")
        self.cmaes_ratio_path_metric = str(
            getattr(self.opts, "objective_split_cmaes_ratio_path_metric", "rms")
        ).lower()
        if self.cmaes_ratio_path_metric not in {"rms", "directional"}:
            raise ValueError("Invalid CMAES ratio path metric")
        if self.cmaes_ratio_path_metric != "rms" and self.cma_sep_ratio_path_mode == "native":
            raise ValueError("CMAES directional metric requires an active path candidate")
        self.sepcmaes_ratio_path_metric = str(
            getattr(self.opts, "objective_split_sepcmaes_ratio_path_metric", "rms")
        ).lower()
        if self.sepcmaes_ratio_path_metric not in {"rms", "directional"}:
            raise ValueError("Invalid SepCMAES ratio path metric")
        if self.sepcmaes_ratio_path_metric != "rms" and self.cma_sep_ratio_path_mode == "native":
            raise ValueError("SepCMAES directional metric requires an active path candidate")
        self.cma_sep_ratio_path_strength = float(
            getattr(self.opts, "objective_split_cma_sep_ratio_path_strength", 1.0)
        )
        if not np.isfinite(self.cma_sep_ratio_path_strength) or not 0.0 <= self.cma_sep_ratio_path_strength <= 1.0:
            raise ValueError("Invalid CMAES/SepCMAES ratio path strength")
        if int(getattr(self.opts, "eval_save_vkd_state_trace", 0)) and not (
            self.event_slot_interleaving_enable and any(int(getattr(self.opts, flag, 0)) for flag in (
                "eval_save_event_slot_diagnostics", "eval_save_optimizer_forensic_trace",
            ))
        ):
            raise ValueError("VKD state trace requires event-slot and an enabled diagnostic writer")
        self.optimizer_guide_strength_scale_min = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_strength_scale_min", 1.0))
        )
        self.optimizer_guide_strength_scale_max = float(
            max(
                self.optimizer_guide_strength_scale_min,
                getattr(self.opts, "objective_split_optimizer_guide_strength_scale_max", 1.0),
            )
        )
        self.optimizer_guide_disagreement_low = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_disagreement_low", 0.0))
        )
        self.optimizer_guide_disagreement_high = float(
            max(
                self.optimizer_guide_disagreement_low,
                getattr(self.opts, "objective_split_optimizer_guide_disagreement_high", 1.0),
            )
        )
        self.last_optimizer_guide_strength_effective = float(self.optimizer_guide_strength)
        self.last_optimizer_guide_strength_scale = 1.0
        self.last_optimizer_guide_schedule_metric = 0.0
        self.optimizer_guide_mix_strength = float(
            max(
                0.0,
                getattr(
                    self.opts,
                    "objective_split_optimizer_guide_mix_strength",
                    self.optimizer_guide_strength,
                ),
            )
        )
        self.guide_replacement_mode = str(
            getattr(
                self.opts,
                "objective_split_guide_replacement_mode",
                "off",
            )
        ).lower()
        if self.guide_replacement_mode not in {"off", "shadow", "p0", "p1"}:
            raise ValueError(
                "Unsupported objective_split_guide_replacement_mode: "
                f"{self.guide_replacement_mode}."
            )
        self.guide_replacement_scope = str(
            getattr(
                self.opts,
                "objective_split_guide_replacement_scope",
                "guide_optimizers",
            )
        ).lower()
        if self.guide_replacement_scope not in {"guide_optimizers", "all"}:
            raise ValueError(
                "Unsupported objective_split_guide_replacement_scope: "
                f"{self.guide_replacement_scope}."
            )
        self.guide_replacement_enabled = self.guide_replacement_mode != "off"
        self.collective_guide_mode = str(
            getattr(
                self.opts,
                "objective_split_collective_guide_mode",
                "off",
            )
        ).lower()
        if self.collective_guide_mode not in {
            "off",
            "shadow",
            "rate_matched_null",
            "collective_veto",
        }:
            raise ValueError(
                "Unsupported objective_split_collective_guide_mode: "
                f"{self.collective_guide_mode}."
            )
        self.collective_guide_null_veto_rate = float(
            getattr(
                self.opts,
                "objective_split_collective_guide_null_veto_rate",
                0.0,
            )
        )
        if not 0.0 <= self.collective_guide_null_veto_rate <= 1.0:
            raise ValueError(
                "objective_split_collective_guide_null_veto_rate must be in [0,1]."
            )
        self.collective_guide_enabled = self.collective_guide_mode != "off"
        self.optimizer_guide_injection_pairs = int(
            max(0, getattr(self.opts, "objective_split_optimizer_guide_injection_pairs", 1))
        )
        self.optimizer_guide_use_negative_pair = bool(
            int(getattr(self.opts, "objective_split_optimizer_guide_use_negative_pair", 1))
        )
        self.optimizer_guide_min_improve = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_min_improve", 0.0))
        )
        self.optimizer_guide_gate_enable = bool(
            int(getattr(self.opts, "objective_split_optimizer_guide_gate_enable", 0))
        )
        self.optimizer_guide_min_norm = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_min_norm", 0.0))
        )
        self.optimizer_guide_min_alignment = float(
            np.clip(
                getattr(self.opts, "objective_split_optimizer_guide_min_alignment", -1.0),
                -1.0,
                1.0,
            )
        )
        self.optimizer_guide_numeric_guard = bool(
            int(getattr(self.opts, "objective_split_optimizer_guide_numeric_guard", 0))
        )
        self.cmaes_numeric_fail_soft = bool(
            int(getattr(self.opts, "objective_split_cmaes_numeric_fail_soft", 0))
        )
        self.optimizer_numeric_telemetry_enable = bool(
            int(getattr(self.opts, "objective_split_optimizer_numeric_telemetry", 0))
        )
        self.optimizer_numeric_counter_enable = bool(
            int(getattr(self.opts, "objective_split_optimizer_numeric_counter", 0))
        )
        self.optimizer_numeric_counter_dir = str(
            getattr(self.opts, "objective_split_optimizer_numeric_counter_dir", "")
        ).strip()
        self.optimizer_numeric_telemetry_dir = str(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_telemetry_dir",
                "",
            )
        ).strip()
        self.optimizer_numeric_forensics_enable = bool(
            int(getattr(self.opts, "objective_split_optimizer_numeric_forensics", 0))
        )
        self.optimizer_numeric_forensics_dir = str(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_dir",
                "",
            )
        ).strip()
        self.vkd_origin_trace_enable = bool(int(getattr(
            self.opts, "objective_split_vkd_origin_trace", 0
        )))
        self.vkd_origin_trace_dir = str(getattr(
            self.opts, "objective_split_vkd_origin_trace_dir", ""
        )).strip()
        if self.vkd_origin_trace_enable and (
            self.problem_family != "WSNLocation" or not self.vkd_origin_trace_dir
        ):
            raise ValueError("VKD origin trace requires WSNLocation and an output directory.")
        # Bounded jump-window controls (diagnostic only; neutral defaults).
        self.optimizer_numeric_forensics_jump_log10 = float(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_log10",
                2.0,
            )
        )
        self.optimizer_numeric_forensics_jump_max_windows = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_max_windows",
                4,
            )
        )
        self.optimizer_numeric_forensics_jump_alpha_gate = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_alpha_gate",
                0,
            )
        )
        self.optimizer_numeric_forensics_jump_early_write = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_early_write",
                0,
            )
        )
        self.optimizer_numeric_forensics_jump_early_milestone_log10 = float(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_early_milestone_log10",
                0.5,
            )
        )
        self.optimizer_numeric_forensics_jump_mid_milestone_log10 = float(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_jump_mid_milestone_log10",
                2.0,
            )
        )
        self.optimizer_numeric_forensics_target_function = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_target_function",
                -1,
            )
        )
        self.optimizer_numeric_forensics_target_seed = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_target_seed",
                -1,
            )
        )
        self.optimizer_numeric_forensics_target_agent = int(
            getattr(
                self.opts,
                "objective_split_optimizer_numeric_forensics_target_agent",
                -1,
            )
        )
        self.optimizer_guide_sigma_exp_clip = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_sigma_exp_clip", 20.0))
        )
        self.optimizer_guide_sigma_clip_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_sigma_clip_ratio", 0.5))
        )
        self.optimizer_guide_sample_clip_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_sample_clip_ratio", 0.0))
        )
        self.optimizer_guide_internal_mode = str(
            getattr(self.opts, "objective_split_optimizer_guide_internal_mode", "off")
        ).lower()
        if self.optimizer_guide_internal_mode not in {"off", "mean", "mean_path", "mean_path_covdiag"}:
            self.optimizer_guide_internal_mode = "off"
        raw_internal_optimizers = getattr(
            self.opts,
            "objective_split_optimizer_guide_internal_apply_optimizers",
            ["cmaes", "sepcmaes"],
        )
        if isinstance(raw_internal_optimizers, str):
            internal_opts = [
                x.strip().lower()
                for x in raw_internal_optimizers.split(",")
                if x.strip()
            ]
        else:
            internal_opts = [
                str(x).strip().lower()
                for x in raw_internal_optimizers
                if str(x).strip()
            ]
        self.optimizer_guide_internal_apply_all = (
            ("all" in internal_opts) or ("*" in internal_opts)
        )
        self.optimizer_guide_internal_apply_optimizers = set(
            [] if self.optimizer_guide_internal_apply_all else internal_opts
        )
        self.optimizer_guide_internal_mean_lr = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_mean_lr", 0.0))
        )
        self.optimizer_guide_internal_path_lr = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_path_lr", 0.0))
        )
        self.optimizer_guide_internal_cov_lr = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_cov_lr", 0.0))
        )
        self.optimizer_guide_internal_agree_cos_min = float(
            np.clip(
                getattr(self.opts, "objective_split_optimizer_guide_internal_agree_cos_min", -0.25),
                -1.0,
                1.0,
            )
        )
        self.optimizer_guide_internal_max_step_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_max_step_ratio", 0.05))
        )
        self.optimizer_guide_internal_max_rel_step = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_max_rel_step", 0.5))
        )
        self.optimizer_guide_internal_path_max_rel_norm = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_path_max_rel_norm", 0.5))
        )
        self.optimizer_guide_internal_cov_rank1_clip = float(
            max(0.0, getattr(self.opts, "objective_split_optimizer_guide_internal_cov_rank1_clip", 0.02))
        )
        self.optimizer_guide_internal_disable_sample_injection = bool(
            int(getattr(self.opts, "objective_split_optimizer_guide_internal_disable_sample_injection", 0))
        )
        self.collab_action_enable = bool(
            int(getattr(self.opts, "objective_split_collab_action_enable", 0))
        )
        raw_collab_modes = getattr(
            self.opts,
            "objective_split_collab_modes",
            ["consensus", "self", "leader", "soft_diversify"],
        )
        if isinstance(raw_collab_modes, str):
            collab_modes = [
                x.strip().lower() for x in raw_collab_modes.split(",") if x.strip()
            ]
        else:
            collab_modes = [str(x).strip().lower() for x in raw_collab_modes if str(x).strip()]
        if len(collab_modes) == 0:
            collab_modes = ["consensus"]
        self.collab_modes = collab_modes
        self.collab_soft_diversify_scale = float(
            max(0.0, getattr(self.opts, "objective_split_collab_soft_diversify_scale", 0.25))
        )
        self.collab_leader_fallback = str(
            getattr(self.opts, "objective_split_collab_leader_fallback", "consensus")
        ).lower()
        if self.collab_leader_fallback not in {"consensus", "off"}:
            self.collab_leader_fallback = "consensus"
        self.guide_scale_action_enable = bool(
            int(getattr(self.opts, "objective_split_guide_scale_action_enable", 0))
        )
        raw_guide_scales = getattr(
            self.opts,
            "objective_split_guide_scale_candidates",
            [1.0, 0.5, 0.75, 1.25],
        )
        if isinstance(raw_guide_scales, str):
            guide_scales = [
                float(x.strip()) for x in raw_guide_scales.split(",") if x.strip()
            ]
        else:
            guide_scales = [float(x) for x in raw_guide_scales]
        self.guide_scale_candidates = [
            float(max(0.0, x)) for x in (guide_scales if guide_scales else [1.0])
        ]
        raw_guide_optimizers = getattr(
            self.opts, "objective_split_optimizer_guide_apply_optimizers", ["all"]
        )
        if isinstance(raw_guide_optimizers, str):
            guide_opts = [
                x.strip().lower()
                for x in raw_guide_optimizers.split(",")
                if x.strip()
            ]
        else:
            guide_opts = [str(x).strip().lower() for x in raw_guide_optimizers if str(x).strip()]
        self.optimizer_guide_apply_all = ("all" in guide_opts) or ("*" in guide_opts)
        self.optimizer_guide_apply_optimizers = set(
            [] if self.optimizer_guide_apply_all else guide_opts
        )
        self.anchor_enable = bool(
            int(getattr(self.opts, "objective_split_anchor_enable", 0))
        )
        self.anchor_source = str(
            getattr(self.opts, "objective_split_anchor_source", "consensus")
        ).lower()
        if self.anchor_source not in {"consensus", "mean", "self"}:
            self.anchor_source = "consensus"
        raw_anchor_optimizers = getattr(
            self.opts, "objective_split_anchor_apply_optimizers", ["all"]
        )
        if isinstance(raw_anchor_optimizers, str):
            anchor_opts = [
                x.strip().lower()
                for x in raw_anchor_optimizers.split(",")
                if x.strip()
            ]
        else:
            anchor_opts = [
                str(x).strip().lower()
                for x in raw_anchor_optimizers
                if str(x).strip()
            ]
        self.anchor_apply_all = ("all" in anchor_opts) or ("*" in anchor_opts)
        self.anchor_apply_optimizers = set(
            [] if self.anchor_apply_all else anchor_opts
        )
        self.anchor_strength = float(
            max(0.0, getattr(self.opts, "objective_split_anchor_strength", 0.05))
        )
        raw_anchor_mix = float(
            getattr(self.opts, "objective_split_anchor_mix_strength", self.anchor_strength)
        )
        self.anchor_mix_strength = float(
            self.anchor_strength if raw_anchor_mix < 0.0 else max(0.0, raw_anchor_mix)
        )
        self.anchor_sample_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_anchor_sample_ratio", 0.25))
        )
        self.anchor_sample_clip_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_anchor_sample_clip_ratio", 0.25))
        )
        self.anchor_mean_pull = bool(
            int(getattr(self.opts, "objective_split_anchor_mean_pull", 1))
        )
        self.anchor_sample_injection = bool(
            int(getattr(self.opts, "objective_split_anchor_sample_injection", 1))
        )
        self.committee_mode = str(
            getattr(self.opts, "objective_split_committee_mode", "off")
        ).lower()
        if self.committee_mode not in {"off", "shadow", "guide"}:
            self.committee_mode = "off"
        self.committee_selection = str(
            getattr(self.opts, "objective_split_committee_selection", "target_block")
        ).lower()
        if self.committee_selection not in {"whole", "target_block", "both"}:
            self.committee_selection = "target_block"
        if self.committee_mode == "guide" and self.committee_selection == "both":
            raise ValueError(
                "objective_split_committee_selection=both is shadow-only."
            )
        self.committee_mix_strength = float(
            min(
                1.0,
                max(
                    0.0,
                    getattr(
                        self.opts,
                        "objective_split_committee_mix_strength",
                        1.0,
                    ),
                ),
            )
        )
        self.committee_acceptance_mode = str(
            getattr(
                self.opts,
                "objective_split_committee_acceptance_mode",
                "off",
            )
        ).lower()
        if self.committee_acceptance_mode not in {"off", "report_improve"}:
            self.committee_acceptance_mode = "off"
        self.committee_acceptance_min_log_improve = float(
            max(
                0.0,
                getattr(
                    self.opts,
                    "objective_split_committee_acceptance_min_log_improve",
                    0.0,
                ),
            )
        )
        if (
            self.committee_acceptance_mode == "report_improve"
            and self.committee_mode != "guide"
        ):
            raise ValueError(
                "objective_split_committee_acceptance_mode=report_improve "
                "requires objective_split_committee_mode=guide."
            )
        if (
            self.local_only_information
            and self.committee_acceptance_mode == "report_improve"
        ):
            raise ValueError(
                "objective_split_information_mode=local_only forbids "
                "objective_split_committee_acceptance_mode=report_improve."
            )
        self.committee_shadow_global_eval = bool(
            int(getattr(self.opts, "objective_split_committee_shadow_global_eval", 1))
        )
        if (
            self.local_only_information
            and not self.global_monitor_enable
            and self.committee_mode == "shadow"
            and self.committee_shadow_global_eval
        ):
            raise ValueError(
                "local_only with global monitor disabled cannot run committee "
                "shadow global evaluation."
            )
        if self.committee_mode == "guide" and not self.optimizer_guide_enable:
            raise ValueError(
                "objective_split_committee_mode=guide requires "
                "objective_split_optimizer_guide_enable=1."
            )
        self.committee_target_num = int(max(0, getattr(self.fun, "target_num", 0)))
        self.committee_coordinate_dim = int(
            max(0, getattr(self.fun, "coordinate_dim", 0))
        )
        self.committee_supported = bool(
            hasattr(self.fun, "local_target_residual_batch")
            and self.committee_target_num > 0
            and self.committee_coordinate_dim > 0
            and self.D
            == self.committee_target_num * self.committee_coordinate_dim
        )
        self.candidate_response_mode = str(
            getattr(self.opts, "objective_split_candidate_response_mode", "off")
        ).lower()
        if self.candidate_response_mode not in {"off", "shadow", "actuate"}:
            self.candidate_response_mode = "off"
        self.candidate_generator = str(
            getattr(self.opts, "objective_split_candidate_generator", "spsa_target")
        ).lower()
        if self.candidate_generator not in {
            "spsa_target",
            "multisecant_target",
            "multisecant_hybrid",
        }:
            raise ValueError(
                f"Unsupported objective-split candidate generator: {self.candidate_generator}."
            )
        self.candidate_history_size = int(
            max(
                1,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_history_size",
                    5,
                ),
            )
        )
        self.candidate_multisecant_min_rank = int(
            np.clip(
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_min_rank",
                    self.committee_coordinate_dim,
                ),
                1,
                max(1, self.committee_coordinate_dim),
            )
        )
        self.candidate_multisecant_rank_tolerance = float(
            max(
                0.0,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_rank_tolerance",
                    1e-6,
                ),
            )
        )
        self.candidate_multisecant_condition_max = float(
            max(
                1.0,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_condition_max",
                    1e4,
                ),
            )
        )
        self.candidate_multisecant_center_distance_max_ratio = float(
            max(
                0.0,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_center_distance_max_ratio",
                    0.10,
                ),
            )
        )
        self.candidate_multisecant_max_age = int(
            max(
                0,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_max_age",
                    8,
                ),
            )
        )
        self.candidate_multisecant_gradient_min_norm = float(
            max(
                0.0,
                getattr(
                    self.opts,
                    "objective_split_candidate_multisecant_gradient_min_norm",
                    1e-8,
                ),
            )
        )
        if (
            self.candidate_response_mode != "off"
            and not self.local_only_information
            and not bool(
                int(
                    getattr(
                        self.opts,
                        "objective_split_d6_pre_generator_selector_enable",
                        0,
                    )
                )
            )
        ):
            raise ValueError(
                "Candidate-response control outside local_only is reserved "
                "for the explicit D6 frozen-primary route."
            )
        if self.candidate_response_mode != "off" and self.committee_mode != "off":
            raise ValueError(
                "D3-P candidate-response and the C13 committee cannot be active together."
            )
        self.candidate_response_supported = bool(self.committee_supported)
        self.target_block_field_supported = bool(
            self.problem_family in {"WSNLocation", "WSNLocationMASOIE"}
            and self.committee_supported
            and self.committee_coordinate_dim == 3
        )
        self.candidate_probe_scale = float(
            max(0.0, getattr(self.opts, "objective_split_candidate_probe_scale", 0.25))
        )
        self.candidate_trust_scale = float(
            max(0.0, getattr(self.opts, "objective_split_candidate_trust_scale", 0.25))
        )
        self.candidate_trust_min_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_candidate_trust_min_ratio", 0.002))
        )
        self.candidate_trust_max_ratio = float(
            max(
                self.candidate_trust_min_ratio,
                getattr(self.opts, "objective_split_candidate_trust_max_ratio", 0.05),
            )
        )
        self.candidate_probe_min_ratio = float(
            max(0.0, getattr(self.opts, "objective_split_candidate_probe_min_ratio", 0.0005))
        )
        self.candidate_probe_max_ratio = float(
            max(
                self.candidate_probe_min_ratio,
                getattr(self.opts, "objective_split_candidate_probe_max_ratio", 0.01),
            )
        )
        self.candidate_confidence_min = float(
            np.clip(
                getattr(self.opts, "objective_split_candidate_confidence_min", 1e-4),
                0.0,
                1.0,
            )
        )
        self.candidate_support_min = float(
            np.clip(
                getattr(self.opts, "objective_split_candidate_support_min", 0.60),
                0.0,
                1.0,
            )
        )
        self.candidate_actuator_beta = float(
            np.clip(
                getattr(self.opts, "objective_split_candidate_actuator_beta", 0.50),
                0.0,
                1.0,
            )
        )
        self.candidate_history_obs_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_candidate_history_obs_enable",
                    0,
                )
            )
        )
        self.candidate_history_obs_dim = 5 if self.candidate_history_obs_enable else 0
        self.candidate_actuator_action_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_candidate_actuator_action_enable",
                    0,
                )
            )
        )
        raw_actuator_candidates = getattr(
            self.opts,
            "objective_split_candidate_actuator_candidates",
            [0.0, 0.25, 0.5],
        )
        if isinstance(raw_actuator_candidates, str):
            actuator_candidates = [
                float(x.strip())
                for x in raw_actuator_candidates.split(",")
                if x.strip()
            ]
        else:
            actuator_candidates = [float(x) for x in raw_actuator_candidates]
        self.candidate_actuator_candidates = [
            float(np.clip(x, 0.0, 1.0))
            for x in (actuator_candidates if actuator_candidates else [0.0])
        ]
        raw_actuator_prior = getattr(
            self.opts,
            "objective_split_candidate_actuator_initial_probs",
            [0.2, 0.7, 0.1],
        )
        if isinstance(raw_actuator_prior, str):
            actuator_prior = [
                float(x.strip())
                for x in raw_actuator_prior.split(",")
                if x.strip()
            ]
        else:
            actuator_prior = [float(x) for x in raw_actuator_prior]
        if (
            len(actuator_prior) != len(self.candidate_actuator_candidates)
            or not np.all(np.isfinite(actuator_prior))
            or float(np.sum(np.maximum(actuator_prior, 0.0))) <= 0.0
        ):
            actuator_prior = [1.0] * len(self.candidate_actuator_candidates)
        self.candidate_actuator_initial_action = int(
            np.argmax(np.maximum(np.asarray(actuator_prior, dtype=np.float64), 0.0))
        )
        self.candidate_actuator_initial_beta = float(
            self.candidate_actuator_candidates[
                self.candidate_actuator_initial_action
            ]
            if self.candidate_actuator_action_enable
            else self.candidate_actuator_beta
        )
        self.candidate_actuator_forced_action = int(
            getattr(
                self.opts,
                "objective_split_candidate_actuator_forced_action",
                -1,
            )
        )
        self.d5_two_stage_actuator_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_d5_two_stage_actuator_enable",
                    0,
                )
            )
        )
        self.d5_actuator_reward_mode = str(
            getattr(
                self.opts,
                "objective_split_d5_actuator_reward_mode",
                "mixed",
            )
        ).lower()
        self.d5_actuator_obs_schema = (
            "event_active",
            "opportunity",
            "accepted_target_ratio",
            "selection_confidence",
            "accepted_support_mean",
            "accepted_support_min",
            "accepted_generator_confidence_mean",
            "requested_shift_norm",
            "accepted_trust_radius_mean",
            "accepted_probe_radius_mean",
            "accepted_source_self_ratio",
            "progress",
        )
        self.d5_actuator_obs_dim = len(self.d5_actuator_obs_schema)
        self.d6_pre_generator_selector_enable = bool(
            int(
                getattr(
                    self.opts,
                    "objective_split_d6_pre_generator_selector_enable",
                    0,
                )
            )
        )
        self.d6_generator_obs_schema = (
            "event_active",
            "progress",
            "optimizer_local_improve",
            "proposal_shift_norm",
            "consensus_shift_norm",
            "last_committed_local_improve",
            "recent_committed_local_improve",
            "previous_accept_ratio",
            "previous_selection_confidence",
            "previous_support_ratio",
            "previous_requested_shift_norm",
            "previous_actuator_beta",
            "previous_multisecant_active_ratio",
            "history_fill_ratio",
            "multisecant_usable_ratio",
            "effective_rank_ratio",
            "condition_quality",
            "predicted_response_strength",
        )
        self.d6_generator_obs_dim = len(self.d6_generator_obs_schema)
        self.d6_post_actuator_forced_action = int(
            getattr(self.opts, "d6_post_actuator_forced_action", 2)
        )
        self._pending_step_generator = None
        self._pending_prepare_payload = None
        self._pending_transition_phase = "READY"
        self._d5_transition_broken = False
        if self.d5_two_stage_actuator_enable:
            if not self.candidate_actuator_action_enable:
                raise ValueError(
                    "D5 two-stage actuator requires the D4-B actuator head "
                    "to remain enabled for primary checkpoint compatibility."
                )
            if self.candidate_response_mode != "actuate":
                raise ValueError(
                    "D5 two-stage actuator requires candidate_response_mode=actuate."
                )
            if not self.local_only_information:
                raise ValueError(
                    "D5 two-stage actuator requires local_only information mode."
                )
        if self.d6_pre_generator_selector_enable:
            if not self.candidate_actuator_action_enable:
                raise ValueError(
                    "D6 pre-generator selector requires the separate "
                    "candidate actuator action column."
                )
            if self.candidate_response_mode != "actuate":
                raise ValueError(
                    "D6 pre-generator selector requires "
                    "candidate_response_mode=actuate."
                )
            if not self.candidate_history_obs_enable:
                raise ValueError(
                    "D6 pre-generator selector requires candidate history "
                    "fields in the runtime observation."
                )
            if self.candidate_generator not in {
                "spsa_target",
                "multisecant_hybrid",
            }:
                raise ValueError(
                    "D6 selector supports only SPSA/Hybrid candidate semantics."
                )
        if self.candidate_history_obs_enable and self.candidate_response_mode == "off":
            raise ValueError(
                "Candidate history observations require candidate-response mode."
            )
        if (
            self.candidate_actuator_action_enable
            and self.candidate_response_mode != "actuate"
        ):
            raise ValueError(
                "Candidate actuator actions require candidate_response_mode=actuate."
            )
        if self.rvcpd_enabled:
            rvcpd_conflicts = []
            if not self.local_only_information:
                rvcpd_conflicts.append("information_mode must be local_only")
            if self.neighbor_obs_enabled:
                rvcpd_conflicts.append("neighbor observations must be disabled")
            if self.consensus_reward_weight != 0.0:
                rvcpd_conflicts.append("consensus reward must be zero")
            if self.anchor_enable:
                rvcpd_conflicts.append("anchor must be disabled")
            if self.committee_mode != "off":
                rvcpd_conflicts.append("committee must be disabled")
            if self.collab_action_enable:
                rvcpd_conflicts.append("collaboration action head must be disabled")
            if self.guide_scale_action_enable:
                rvcpd_conflicts.append("guide-scale action head must be disabled")
            if self.d6_pre_generator_selector_enable:
                rvcpd_conflicts.append("D6 selector must be disabled")
            if self.rvcpd_integration_mode == "isolated":
                if self.consensus_mode != "none":
                    rvcpd_conflicts.append("consensus must be none")
                if self.state_comm_enabled:
                    rvcpd_conflicts.append(
                        "state communication must be disabled"
                    )
                if self.comm_action_enable:
                    rvcpd_conflicts.append(
                        "communication action head must be disabled"
                    )
                if self.optimizer_guide_enable:
                    rvcpd_conflicts.append("optimizer guide must be disabled")
                if self.candidate_response_mode != "off":
                    rvcpd_conflicts.append(
                        "candidate response must be disabled"
                    )
                if self.candidate_history_obs_enable:
                    rvcpd_conflicts.append(
                        "candidate history observations must be disabled"
                    )
                if self.candidate_actuator_action_enable:
                    rvcpd_conflicts.append(
                        "candidate actuator head must be disabled"
                    )
                if self.d5_two_stage_actuator_enable:
                    rvcpd_conflicts.append("D5 actuator must be disabled")
            else:
                if self.rvcpd_arm not in {"p0", "p1"}:
                    rvcpd_conflicts.append(
                        "d5_post_commit supports only P0/P1"
                    )
                if not self.d5_two_stage_actuator_enable:
                    rvcpd_conflicts.append(
                        "d5_post_commit requires the D5 actuator"
                    )
                if self.consensus_mode != "graph_mean":
                    rvcpd_conflicts.append(
                        "d5_post_commit requires graph_mean consensus"
                    )
                if self.state_comm_mode != "graph_mean":
                    rvcpd_conflicts.append(
                        "d5_post_commit requires graph_mean state communication"
                    )
                if not self.comm_action_enable:
                    rvcpd_conflicts.append(
                        "d5_post_commit requires the communication action head"
                    )
                if not self.optimizer_guide_enable:
                    rvcpd_conflicts.append(
                        "d5_post_commit requires the optimizer guide"
                    )
                if self.candidate_response_mode != "actuate":
                    rvcpd_conflicts.append(
                        "d5_post_commit requires candidate actuation"
                    )
                if not self.candidate_history_obs_enable:
                    rvcpd_conflicts.append(
                        "d5_post_commit requires candidate history observations"
                    )
                if not self.candidate_actuator_action_enable:
                    rvcpd_conflicts.append(
                        "d5_post_commit requires the candidate actuator head"
                    )
            if rvcpd_conflicts:
                raise ValueError(
                    "RVCPD environment integration contract failed "
                    f"({self.rvcpd_integration_mode}): "
                    + "; ".join(rvcpd_conflicts)
                    + "."
                )
        self.candidate_shadow_global_eval_interval = int(
            max(
                0,
                getattr(
                    self.opts,
                    "objective_split_candidate_shadow_global_eval_interval",
                    20,
                ),
            )
        )
        if (
            self.candidate_response_mode == "shadow"
            and self.candidate_shadow_global_eval_interval > 0
            and not self.global_monitor_enable
        ):
            raise ValueError(
                "D3-P shadow global sampling requires the detached global monitor."
            )
        if self.target_block_field_enable:
            field_conflicts = []
            if not self.local_only_information:
                field_conflicts.append("information_mode must be local_only")
            if not self.target_block_field_supported:
                field_conflicts.append(
                    "benchmark must expose WSN 3D target-block residuals"
                )
            if self.consensus_mode != "graph_mean":
                field_conflicts.append("consensus must be graph_mean")
            if self.comm_interval != 1:
                field_conflicts.append("comm_interval must be 1")
            if self.comm_rounds_per_event != 1:
                field_conflicts.append("comm_rounds must be 1")
            if self.comm_force_rounds != 0:
                field_conflicts.append("forced communication rounds must be 0")
            if self.comm_action_enable:
                field_conflicts.append(
                    "communication action head must be disabled"
                )
            if self.state_comm_enabled:
                field_conflicts.append("state communication must be disabled")
            if self.neighbor_obs_enabled:
                field_conflicts.append(
                    "neighbor observations must be disabled"
                )
            if self.consensus_reward_weight != 0.0:
                field_conflicts.append("consensus reward must be zero")
            if self.optimizer_guide_enable:
                field_conflicts.append("optimizer guide must be disabled")
            if self.anchor_enable:
                field_conflicts.append("anchor must be disabled")
            if self.committee_mode != "off":
                field_conflicts.append("committee must be disabled")
            if self.candidate_response_mode != "off":
                field_conflicts.append("candidate response must be disabled")
            if self.rvcpd_enabled:
                field_conflicts.append("RVCPD must be disabled")
            if self.guide_replacement_enabled:
                field_conflicts.append("guide replacement must be disabled")
            if self.collective_guide_enabled:
                field_conflicts.append("collective guide must be disabled")
            if self.collab_action_enable:
                field_conflicts.append(
                    "collaboration action head must be disabled"
                )
            if self.guide_scale_action_enable:
                field_conflicts.append(
                    "guide-scale action head must be disabled"
                )
            if self.d5_two_stage_actuator_enable:
                field_conflicts.append("D5 actuator must be disabled")
            if self.d6_pre_generator_selector_enable:
                field_conflicts.append("D6 selector must be disabled")
            if not self.record_comm_cost:
                field_conflicts.append(
                    "communication cost recording must be enabled"
                )
            if field_conflicts:
                raise ValueError(
                    "Target-block cooperative field contract failed: "
                    + "; ".join(field_conflicts)
                    + "."
                )
        if self.target_block_dual_clock_enable:
            dual_clock_conflicts = []
            if not self.target_block_field_enable:
                dual_clock_conflicts.append(
                    "target-block cooperative field must be enabled"
                )
            if not self.persistent_sepcmaes_enable:
                dual_clock_conflicts.append(
                    "persistent SepCMAES bank must be enabled"
                )
            if set(
                str(name).lower()
                for name in self.optimizer_candidates
            ) != {"sepcmaes"}:
                dual_clock_conflicts.append(
                    "optimizer_candidates must contain only sepcmaes"
                )
            if self.d5_two_stage_actuator_enable:
                dual_clock_conflicts.append(
                    "staged D5 protocol must be disabled"
                )
            if self.d6_pre_generator_selector_enable:
                dual_clock_conflicts.append(
                    "staged D6 protocol must be disabled"
                )
            if dual_clock_conflicts:
                raise ValueError(
                    "Target-block dual-clock contract failed: "
                    + "; ".join(dual_clock_conflicts)
                    + "."
                )
        elif self.target_block_dual_clock_commit_lock_enable:
            raise ValueError(
                "Target-block dual-clock commit lock requires target-block "
                "dual-clock mode."
            )
        if (
            self.target_block_commit_credit_mode != "off"
            and not self.target_block_dual_clock_commit_lock_enable
        ):
            raise ValueError(
                "Target-block commit credit requires dual-clock commit lock."
            )
        if (
            self.target_block_dormancy_recovery_enable
            and self.target_block_commit_credit_mode != "scale"
        ):
            raise ValueError(
                "Target-block dormancy recovery requires the fixed "
                "NT078 scale-credit kernel."
            )
        if (
            self.target_block_direction_shadow_enable
            and not self.target_block_dormancy_recovery_enable
        ):
            raise ValueError(
                "Target-block direction shadow requires NT079 dormancy "
                "evidence."
            )
        if (
            self.target_block_challenge_response_mode != "off"
            and not self.target_block_dormancy_recovery_enable
        ):
            raise ValueError(
                "Target-block challenge response requires NT079 dormancy "
                "evidence."
            )
        if (
            self.target_block_challenge_response_mode != "off"
            and self.target_block_direction_shadow_enable
        ):
            raise ValueError(
                "NT080 direction shadow and NT081 challenge response are "
                "mutually exclusive."
            )
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
        self.obs_dim += self.candidate_history_obs_dim
        if self.sigma_inherit_enable and self.local_only_information:
            self.obs_dim += 5
            if self.sigma_state_obs_enable:
                self.obs_dim += 3
        self.observation_space = self.observation_space.__class__(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_agents, self.obs_dim),
            dtype=np.float32,
        )
        if self.candidate_actuator_action_enable:
            primary_nvec = np.asarray(
                self.action_space.nvec[:, :-1],
                dtype=np.int64,
            )
            actuator_nvec = np.full(
                (self.n_agents,),
                len(self.candidate_actuator_candidates),
                dtype=np.int64,
            )
            self.primary_action_space = self.action_space.__class__(primary_nvec)
            self.actuator_action_space = self.action_space.__class__(
                actuator_nvec
            )
        else:
            self.primary_action_space = self.action_space
            self.actuator_action_space = None

        self.graph = None
        self.consensus_weight = None
        self.consensus_adjacency = None
        if (
            self._needs_consensus_graph()
            or self._needs_state_comm_graph()
            or self.optimizer_guide_enable
            or (self.anchor_enable and self.anchor_source == "consensus")
            or self.committee_mode != "off"
            or self.candidate_response_mode != "off"
            or self.rvcpd_enabled
            or self.collective_guide_enabled
            or self.target_block_field_enable
        ):
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
        self.last_committed_local_improve = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.committed_local_improve_hist: List[List[float]] = [
            [] for _ in range(self.n_agents)
        ]
        self.last_consensus_local_effect = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_local_commit_success = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_sigma_value = np.full(
            (self.n_agents,), float(max(1e-12, getattr(self.opts, "sigma", 0.3))),
            dtype=np.float64,
        )
        self.sigma_state_age = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_sigma_state_obs = np.zeros(
            (self.n_agents, 5 + (3 if self.sigma_state_obs_enable else 0)),
            dtype=np.float32,
        )
        self.last_actual_fes_per_agent = np.zeros((self.n_agents,), dtype=np.float64)
        self.cumulative_actual_fes_per_agent = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_cmaes_numeric_fail_soft = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_cmaes_numeric_fail_soft_generation = np.full(
            (self.n_agents,), -1, dtype=np.int64
        )
        self.cmaes_numeric_fail_soft_events = 0
        self.persistent_sepcmaes_states = [
            None for _ in range(self.n_agents)
        ]
        self.persistent_sepcmaes_signatures = [
            None for _ in range(self.n_agents)
        ]
        self.last_persistent_sepcmaes_active = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_persistent_sepcmaes_fresh = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_persistent_sepcmaes_config_reset = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_persistent_sepcmaes_recenter_requested = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_persistent_sepcmaes_recenter_applied = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_persistent_sepcmaes_recenter_remaining = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.persistent_sepcmaes_events = 0
        self.persistent_sepcmaes_fresh_events = 0
        self.persistent_sepcmaes_resume_events = 0
        self.persistent_sepcmaes_config_resets = 0
        field_shape = (
            self.n_agents,
            self.committee_target_num,
            self.committee_coordinate_dim,
        )
        field_block_shape = (
            self.n_agents,
            self.committee_target_num,
        )
        self.target_block_field_path = np.zeros(
            field_shape, dtype=np.float64
        )
        self.target_block_field_radius = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.target_block_field_initialized = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_secant = np.zeros(
            field_shape, dtype=np.float64
        )
        self.last_target_block_field_secant_valid = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_secant_alignment = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_field_direction = np.zeros(
            field_shape, dtype=np.float64
        )
        self.last_target_block_field_source_diversity = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_field_path_norm = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_field_path_alignment = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_field_conflict = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_radius_expand = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_radius_shrink = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_radius_clip_min = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_radius_clip_max = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_requested_commit_norm = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_field_applied_commit_norm = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_field_boundary_clipped = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_field_path_injection = 0.0
        self.last_target_block_field_path_angle_degrees = 90.0
        self.target_block_field_events = 0
        self.target_block_field_local_evals = 0
        self.target_block_field_comm_rounds = 0
        self.target_block_field_messages = 0
        self.target_block_field_transmitted_floats = 0
        self.target_block_dormancy_residual_reference = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.target_block_dormancy_reserve_radius = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.target_block_dormancy_initialized = np.zeros(
            field_block_shape, dtype=bool
        )
        self.target_block_dormancy_floor_age = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.target_block_dormancy_stagnation_age = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.target_block_dormancy_cooldown = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.target_block_dormancy_activation_count = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_dormancy_active = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_dormancy_unresolved = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_dormancy_material_progress = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_dormancy_reliable_scale = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_dormancy_residual_ratio = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_dormancy_progress_ratio = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_dormancy_disagreement = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_dormancy_disagreement_ratio = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_dormancy_restore_radius = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.target_block_dormancy_events = 0
        self.target_block_dormancy_activations = 0
        direction_source_num = len(TARGET_BLOCK_DIRECTION_SHADOW_SOURCES)
        self.last_target_block_direction_shadow_eligible = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_direction_shadow_selected = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_direction_shadow_candidate_valid = np.zeros(
            field_block_shape + (direction_source_num,), dtype=bool
        )
        self.last_target_block_direction_shadow_candidate_angle = np.zeros(
            field_block_shape + (direction_source_num,), dtype=np.float64
        )
        self.last_target_block_direction_shadow_candidate_response = np.zeros(
            field_block_shape + (direction_source_num,), dtype=np.float64
        )
        self.last_target_block_direction_shadow_candidate_positive = np.zeros(
            field_block_shape + (direction_source_num,), dtype=bool
        )
        self.last_target_block_direction_shadow_best_response = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_direction_shadow_best_source = np.full(
            field_block_shape, -1, dtype=np.int64
        )
        self.last_target_block_direction_shadow_best_sign = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_direction_shadow_best_direction = np.zeros(
            field_shape, dtype=np.float64
        )
        self.last_target_block_direction_shadow_support = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_direction_shadow_conflict = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.target_block_direction_shadow_events = 0
        self.target_block_direction_shadow_eligible_blocks = 0
        self.target_block_direction_shadow_probed_blocks = 0
        self.target_block_direction_shadow_local_evals = 0
        self.target_block_direction_shadow_comm_rounds = 0
        self.target_block_direction_shadow_messages = 0
        self.target_block_direction_shadow_transmitted_floats = 0
        challenge_source_num = len(
            TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES
        )
        self.last_target_block_challenge_selected = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_challenge_candidate_valid = np.zeros(
            field_block_shape + (challenge_source_num,), dtype=bool
        )
        self.last_target_block_challenge_candidate_angle = np.zeros(
            field_block_shape + (challenge_source_num,), dtype=np.float64
        )
        self.last_target_block_challenge_candidate_score = np.zeros(
            field_block_shape + (challenge_source_num,), dtype=np.float64
        )
        self.last_target_block_challenge_candidate_sign = np.zeros(
            field_block_shape + (challenge_source_num,), dtype=np.int64
        )
        self.last_target_block_challenge_path_score = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_alternative_score = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_margin = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_best_source = np.full(
            field_block_shape, -1, dtype=np.int64
        )
        self.last_target_block_challenge_best_sign = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_challenge_best_direction = np.zeros(
            field_shape, dtype=np.float64
        )
        self.last_target_block_challenge_coverage = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_positive_sources = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_challenge_neighbor_positive_sources = np.zeros(
            field_block_shape, dtype=np.int64
        )
        self.last_target_block_challenge_support = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_conflict = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_actuation_eligible = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_challenge_applied = np.zeros(
            field_block_shape, dtype=bool
        )
        self.last_target_block_challenge_applied_norm = np.zeros(
            field_block_shape, dtype=np.float64
        )
        self.last_target_block_challenge_boundary_clipped = np.zeros(
            field_block_shape, dtype=bool
        )
        self.target_block_challenge_events = 0
        self.target_block_challenge_challenges = 0
        self.target_block_challenge_directed_responses = 0
        self.target_block_challenge_local_evals = 0
        self.target_block_challenge_reported_evals = 0
        self.target_block_challenge_applied_commits = 0
        self.target_block_challenge_comm_rounds = 0
        self.target_block_challenge_messages = 0
        self.target_block_challenge_transmitted_floats = 0
        self.target_block_dual_clock_outer_events = 0
        self.target_block_dual_clock_local_generation_ticks = 0
        self.target_block_dual_clock_communication_ticks = 0
        self.target_block_dual_clock_commit_ticks = 0
        self.last_target_block_dual_clock_microcycles = 0
        self.last_target_block_dual_clock_generation_fes = np.zeros(
            (self.n_agents,),
            dtype=np.int64,
        )
        self.target_block_commit_credit_events = 0
        self.last_target_block_commit_credit_active = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_cosine = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_proposal_norm = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_commit_norm = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_correction_norm = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_path_retention = np.ones(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_scale_retention = np.ones(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_axis_rms_before = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_axis_rms_after = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_sigma_path_before = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_sigma_path_after = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_cov_path_before = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_target_block_commit_credit_cov_path_after = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_consensus_shift_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_state_neighbor_message = np.zeros(
            (self.n_agents, self.state_msg_dim),
            dtype=np.float64,
        )
        self.last_state_delta_message = np.zeros(
            (self.n_agents, self.state_msg_dim),
            dtype=np.float64,
        )
        self.ccsa_direction_momentum = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.masoie_velocity = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.rvcpd_path = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.rvcpd_support_ema = np.zeros((self.n_agents,), dtype=np.float64)
        self.rvcpd_conflict_ema = np.zeros((self.n_agents,), dtype=np.float64)
        self.rvcpd_uncertainty_ema = np.zeros((self.n_agents,), dtype=np.float64)
        self.rvcpd_trust = np.ones((self.n_agents,), dtype=np.float64)
        self.rvcpd_scale = np.full(
            (self.n_agents,), self.rvcpd_initial_scale, dtype=np.float64
        )
        self.rvcpd_age = np.zeros((self.n_agents,), dtype=np.int64)
        self.last_rvcpd_direction = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.last_rvcpd_sign = np.zeros((self.n_agents,), dtype=np.int64)
        self.last_rvcpd_active = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_rvcpd_agreement = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_rvcpd_requested_radius = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_rvcpd_applied_radius = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_rvcpd_plus_gain = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_rvcpd_minus_gain = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.optimizer_guide_direction = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.last_optimizer_guide_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_guide_applied = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_guide_alignment = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_guide_source_active_ratio = 0.0
        self.guide_replacement_direction = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.last_guide_replacement_eligible = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_valid = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_old_guide_suppressed = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_sign = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_guide_replacement_strength = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_sigma = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_requested_radius = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_applied_radius = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_plus_gain = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_guide_replacement_minus_gain = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_collective_guide_shared_base = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.last_collective_guide_direction = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.last_collective_guide_radius = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_collective_guide_vote = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_collective_guide_source_valid = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_collective_guide_vote_valid = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_collective_guide_suppressed = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_collective_guide_valid = False
        self.last_collective_guide_vote_sum = 0
        self.last_collective_guide_hypothetical_veto = False
        self.last_collective_guide_actual_veto = False
        self.last_collective_guide_null_veto = False
        self.last_optimizer_guide_internal_active = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_guide_internal_mean_step_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_guide_internal_alignment = np.zeros((self.n_agents,), dtype=np.float64)
        self.anchor_points = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.last_anchor_applied = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_anchor_dist = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_anchor_direction_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_anchor_applied = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_anchor_mean_step_norm = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_optimizer_anchor_sample_applied = np.zeros((self.n_agents,), dtype=np.float64)
        self.committee_direction = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.committee_confidence = np.zeros((self.n_agents,), dtype=np.float64)
        self.committee_verified_x = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.committee_selected_source = np.zeros(
            (self.n_agents, self.committee_target_num), dtype=np.int64
        )
        self.committee_candidate_score = np.zeros(
            (self.n_agents, self.committee_target_num), dtype=np.float64
        )
        self.committee_acceptance_active = False
        self.committee_candidate_global_f = np.full(
            (self.n_agents,), np.inf, dtype=np.float64
        )
        self.committee_candidate_report_f = float("inf")
        self.committee_candidate_vs_report_log_improve = float("nan")
        self.committee_effective_beta = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.committee_base_alignment = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_committee_metrics = self._empty_committee_metrics()
        self.committee_events = 0
        self.committee_decision_local_evals = 0
        self.committee_shadow_local_evals = 0
        self.committee_shadow_global_local_evals = 0
        self.committee_messages = 0
        self.committee_transmitted_floats = 0
        self.committee_acceptance_events = 0
        self.committee_acceptance_accepted_events = 0
        self.committee_acceptance_rejected_events = 0
        self.committee_acceptance_local_evals = 0
        self.committee_acceptance_messages = 0
        self.committee_acceptance_transmitted_floats = 0
        self.committee_acceptance_stage_events = np.zeros((3,), dtype=np.int64)
        self.committee_acceptance_stage_accepted_events = np.zeros(
            (3,), dtype=np.int64
        )
        self.candidate_response_verified_x = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.candidate_response_selected_source = np.full(
            (self.n_agents, self.committee_target_num), -1, dtype=np.int64
        )
        self.candidate_response_accepted_mask = np.zeros(
            (self.n_agents, self.committee_target_num), dtype=bool
        )
        self.last_candidate_accept_ratio = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_candidate_selection_confidence = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_candidate_support_ratio = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_candidate_requested_shift_norm = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        self.last_candidate_actuator_action = np.full(
            (self.n_agents,),
            (
                self.candidate_actuator_initial_action
                if self.candidate_actuator_action_enable
                else 0
            ),
            dtype=np.int64,
        )
        self.last_candidate_actuator_beta = np.full(
            (self.n_agents,),
            self.candidate_actuator_initial_beta,
            dtype=np.float64,
        )
        self.candidate_response_shadow_base_x = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.candidate_response_shadow_candidate_x = np.zeros(
            (self.n_agents, self.D), dtype=np.float64
        )
        self.candidate_response_shadow_due = False
        self.last_candidate_response_metrics = self._empty_candidate_response_metrics()
        self.candidate_response_events = 0
        self.candidate_response_actuated_events = 0
        self.candidate_response_probe_local_evals = 0
        self.candidate_response_verification_local_evals = 0
        self.candidate_response_shadow_global_local_evals = 0
        self.candidate_response_rounds = 0
        self.candidate_response_messages = 0
        self.candidate_response_transmitted_floats = 0
        history_shape = (
            self.n_agents,
            self.committee_target_num,
            self.candidate_history_size,
        )
        self.candidate_multisecant_history_step = np.zeros(
            history_shape + (self.committee_coordinate_dim,),
            dtype=np.float64,
        )
        self.candidate_multisecant_history_response = np.zeros(
            history_shape,
            dtype=np.float64,
        )
        self.candidate_multisecant_history_center = np.zeros(
            history_shape + (self.committee_coordinate_dim,),
            dtype=np.float64,
        )
        self.candidate_multisecant_history_event_id = np.full(
            history_shape,
            -1,
            dtype=np.int64,
        )
        self.candidate_multisecant_history_valid = np.zeros(
            history_shape,
            dtype=bool,
        )
        self.candidate_multisecant_history_cursor = np.zeros(
            (self.n_agents, self.committee_target_num),
            dtype=np.int64,
        )
        self.candidate_multisecant_history_count = np.zeros(
            (self.n_agents, self.committee_target_num),
            dtype=np.int64,
        )
        self.last_candidate_multisecant_active = np.zeros(
            (self.n_agents, self.committee_target_num),
            dtype=bool,
        )
        self.last_candidate_multisecant_fallback_reason = np.ones(
            (self.n_agents, self.committee_target_num),
            dtype=np.int64,
        )
        self.last_collab_mode_idx = np.zeros((self.n_agents,), dtype=np.int64)
        self.last_guide_scale_idx = np.zeros((self.n_agents,), dtype=np.int64)
        self.last_guide_scale_value = np.ones((self.n_agents,), dtype=np.float64)
        self.last_collab_leader_active = np.zeros((self.n_agents,), dtype=np.float64)
        self.last_collab_strength_multiplier = np.ones((self.n_agents,), dtype=np.float64)
        self.last_ccsa_scale = np.ones((self.n_agents,), dtype=np.float64)
        self.last_masoie_neighbor_pull = np.zeros((self.n_agents, self.D), dtype=np.float64)
        self.last_consensus_metrics = self._empty_consensus_metrics()
        self.local_search_fes = 0
        self.candidate_validation_local_evals = 0
        self.agent_state_local_evals = 0
        self.global_monitor_local_evals = 0
        self.global_monitor_rounds = 0
        self.state_comm_rounds = 0
        self.state_comm_messages = 0
        self.state_comm_transmitted_floats = 0
        self.graph_comm_rounds = 0
        self.last_comm_rounds_applied = 0
        self.last_comm_rounds_requested = int(self.comm_rounds_per_event)
        self.last_comm_round_idx = np.zeros((self.n_agents,), dtype=np.int64)
        self.ccsa_lite_rounds = 0
        self.masoie_lite_rounds = 0
        self.rvcpd_events = 0
        self.rvcpd_valid_event_count = 0
        self.rvcpd_commit_count = 0
        self.rvcpd_reverse_count = 0
        self.rvcpd_noop_count = 0
        self.rvcpd_probe_local_evals = 0
        self.rvcpd_messages = 0
        self.rvcpd_transmitted_floats = 0
        self.guide_replacement_events = 0
        self.guide_replacement_eligible_count = 0
        self.guide_replacement_valid_event_count = 0
        self.guide_replacement_forward_count = 0
        self.guide_replacement_reverse_count = 0
        self.guide_replacement_noop_count = 0
        self.guide_replacement_commit_count = 0
        self.guide_replacement_probe_local_evals = 0
        self.collective_guide_events = 0
        self.collective_guide_valid_events = 0
        self.collective_guide_hypothetical_veto_events = 0
        self.collective_guide_actual_veto_events = 0
        self.collective_guide_probe_local_evals = 0
        self.collective_guide_comm_rounds = 0
        self.collective_guide_messages = 0
        self.collective_guide_transmitted_floats = 0
        self.centralized_full_mean_rounds = 0
        self.total_comm_rounds_applied = 0
        self.total_comm_events = 0
        self.step_comm_rounds_applied = 0
        self.step_comm_events = 0
        self.graph_messages = 0
        self.graph_transmitted_floats = 0
        self.event_slot_interleaving_events = 0
        self.event_slot_interleaving_slots = 0
        self.event_slot_native_local_evals = 0
        self.event_slot_reported_local_evals = 0
        self.event_slot_physical_local_evals = 0
        self.last_event_slot_count = 0
        self.last_event_slot_packet_units = np.zeros(
            (0, self.n_agents), dtype=np.int64
        )
        self.last_event_slot_distribution_updates = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_event_slot_native_evals = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_event_slot_physical_evals = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_event_slot_center_shift_norms = np.zeros(
            (0, self.n_agents), dtype=np.float64
        )
        self.last_event_slot_packet_improvements = np.zeros(
            (0, self.n_agents), dtype=np.float64
        )
        self.last_event_slot_commit_audit = (
            _empty_event_slot_commit_audit(0, self.n_agents)
        )
        self.last_event_slot_mmes_neutral_success = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        self.last_sigma_diagnostics = {
            "requested_profile": np.full(self.n_agents, "unknown", dtype=object),
            "resolved_profile": np.full(self.n_agents, "unknown", dtype=object),
            "previous_optimizer": np.full(self.n_agents, "", dtype=object),
            "current_optimizer": np.full(self.n_agents, "", dtype=object),
            "sigma_preview": np.zeros(self.n_agents, dtype=np.float64),
            "sigma_used": np.zeros(self.n_agents, dtype=np.float64),
            "evolved_sigma": np.zeros(self.n_agents, dtype=np.float64),
            "inherit_applied": np.zeros(self.n_agents, dtype=np.int8),
            "reset_reason": np.full(self.n_agents, "", dtype=object),
            "signature_match": np.zeros(self.n_agents, dtype=np.int8),
            "stale_gate_shift": np.zeros(self.n_agents, dtype=np.float64),
            "stale_gate_scale": np.zeros(self.n_agents, dtype=np.float64),
        }
        self._reset_sigma_diagnostics()
        self.report_current_x = self.current_x.copy()
        self.report_current_f = float(self.current_f)
        self.report_best_x = self.gbest_x.copy()
        self.report_best_f = float(self.gbest_f)

    def _reset_sigma_diagnostics(self):
        d = self.last_sigma_diagnostics
        for key in ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "reset_reason"):
            d[key].fill("" if key not in ("requested_profile", "resolved_profile") else "unknown")
        for key in ("sigma_preview", "sigma_used", "evolved_sigma"):
            d[key].fill(0.0)
        for key in ("stale_gate_shift", "stale_gate_scale"):
            d[key].fill(0.0)
        for key in ("inherit_applied", "signature_match"):
            d[key].fill(0)

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
        if not self.global_evaluator_available:
            raise RuntimeError(
                "local_eval_all_batch is unavailable; exact-global evaluation "
                "is allowed only when the detached monitor interface exists."
            )
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

    def _run_objective_split_optimizer_batch(
        self,
        *,
        base_states: np.ndarray,
        local_reference: np.ndarray,
        opt_actions: np.ndarray,
        cfg_actions_block: np.ndarray,
        res_actions: np.ndarray,
        collab_actions: np.ndarray,
        guide_scale_actions: np.ndarray,
        subfes_overrides: np.ndarray = None,
        persistent_recenter_max_shift_override: float = None,
    ) -> Dict:
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents,
            self.D,
        )
        references = np.asarray(
            local_reference,
            dtype=np.float64,
        ).reshape(self.n_agents)
        override = None
        if subfes_overrides is not None:
            override = np.asarray(
                subfes_overrides,
                dtype=np.int64,
            ).reshape(self.n_agents)
            if np.any(override <= 0):
                raise ValueError(
                    "Objective-split per-agent subFEs overrides must be "
                    "positive."
                )
        persistent_recenter_max_shift = float(
            self.persistent_sepcmaes_recenter_max_shift
            if persistent_recenter_max_shift_override is None
            else persistent_recenter_max_shift_override
        )
        if (
            np.isnan(persistent_recenter_max_shift)
            or persistent_recenter_max_shift < 0.0
        ):
            raise ValueError(
                "Persistent SepCMAES recenter override must be "
                "non-negative."
            )

        agent_tasks = []
        for i in range(self.n_agents):
            optimizer_name = self.optimizer_candidates[int(opt_actions[i])]
            subfes_i = (
                int(override[i])
                if override is not None
                else max(
                    1,
                    int(
                        round(
                            self.subfes_per_agent
                            * float(
                                self.resource_factors[
                                    int(res_actions[i])
                                ]
                            )
                        )
                    ),
                )
            )
            seed = int(self.opts.seed + self.step_count * 1000 + i)
            x_base_i = bases[i].copy()
            options = self._build_optimizer_options(
                agent_id=i,
                optimizer_name=optimizer_name,
                cfg_levels=cfg_actions_block[i].tolist(),
                dims=self.full_dims,
                x_base=x_base_i,
                subfes_i=subfes_i,
                seed=seed,
            )
            options.update(
                self._optimizer_guide_options(
                    i,
                    optimizer_name,
                    collab_mode_idx=int(collab_actions[i]),
                    guide_scale_idx=int(guide_scale_actions[i]),
                    sigma_value=float(
                        options.get(
                            "sigma",
                            self._sigma_ref(i, optimizer_name),
                        )
                    ),
                )
            )
            options.update(
                self._anchor_options(i, optimizer_name, x_base_i)
            )
            if str(optimizer_name).lower() == "cmaes":
                options["optimizer_numeric_fail_soft"] = bool(
                    self.cmaes_numeric_fail_soft
                )
            optimizer_key = str(optimizer_name).lower()
            if optimizer_key in {"cmaes", "sepcmaes"} and (
                self.optimizer_numeric_telemetry_enable
                or (optimizer_key == "cmaes" and self.optimizer_numeric_counter_enable)
                or (
                    optimizer_key == "cmaes"
                    and self.cmaes_numeric_fail_soft
                )
            ):
                options["optimizer_numeric_telemetry_context"] = {
                    "problem_family": str(self.problem_family),
                    "function_id": int(self.question),
                    "env_step": int(self.step_count),
                    "agent_id": int(i),
                    "optimizer_action": int(opt_actions[i]),
                    "cfg_levels": [
                        int(x) for x in cfg_actions_block[i].tolist()
                    ],
                    "resource_action": int(res_actions[i]),
                    "subfes": int(subfes_i),
                    "seed": int(seed),
                }
                if self.optimizer_numeric_telemetry_enable:
                    options["optimizer_numeric_telemetry_enable"] = True
                    options["optimizer_numeric_telemetry_dir"] = str(
                        self.optimizer_numeric_telemetry_dir
                    )
                if optimizer_key == "cmaes" and self.optimizer_numeric_counter_enable:
                    options["optimizer_numeric_counter_enable"] = True
                    options["optimizer_numeric_counter_dir"] = self.optimizer_numeric_counter_dir
            if str(
                os.environ.get("OBJECTIVE_SPLIT_NATIVE_TRACE_DIR", "")
            ).strip():
                options["_objective_split_native_trace_context"] = {
                    "trace_label": str(
                        os.environ.get(
                            "OBJECTIVE_SPLIT_NATIVE_TRACE_LABEL",
                            "",
                        )
                    ),
                    "problem_family": str(self.problem_family),
                    "function_id": int(self.question),
                    "env_step": int(self.step_count),
                    "reported_sum_fes_before": int(self.sum_fes),
                    "optimizer_action": int(opt_actions[i]),
                    "cfg_levels": [
                        int(x) for x in cfg_actions_block[i].tolist()
                    ],
                    "resource_action": int(res_actions[i]),
                    "resource_factor": float(
                        self.resource_factors[int(res_actions[i])]
                    ),
                    "subfes": int(subfes_i),
                    "seed": int(seed),
                }
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
                    (
                        copy.deepcopy(
                            self.persistent_sepcmaes_states[i]
                        )
                        if self.persistent_sepcmaes_enable
                        and str(optimizer_name).lower() == "sepcmaes"
                        else None
                    ),
                    (
                        copy.deepcopy(
                            self.persistent_sepcmaes_signatures[i]
                        )
                        if self.persistent_sepcmaes_enable
                        and str(optimizer_name).lower() == "sepcmaes"
                        else None
                    ),
                    (
                        float(
                            persistent_recenter_max_shift
                        )
                        if self.persistent_sepcmaes_enable
                        and str(optimizer_name).lower() == "sepcmaes"
                        else float("inf")
                    ),
                )
            )

        if self.agent_parallel_workers <= 1 or len(agent_tasks) <= 1:
            agent_results = [
                _objective_split_agent_optimize_worker(*task)
                for task in agent_tasks
            ]
        else:
            worker_num = int(
                min(self.agent_parallel_workers, len(agent_tasks))
            )
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
                        [task[8] for task in agent_tasks],
                        [task[9] for task in agent_tasks],
                        [task[10] for task in agent_tasks],
                    )
                )
        agent_results = sorted(
            agent_results,
            key=lambda x: int(x[0]),
        )

        proposal_xs: List[np.ndarray] = []
        proposal_local_values: List[float] = []
        local_improvements: List[float] = []
        evals_per_agent = np.zeros(
            (self.n_agents,),
            dtype=np.int64,
        )
        cmaes_fail_soft_mask = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        cmaes_fail_soft_generations = np.full(
            (self.n_agents,), -1, dtype=np.int64
        )
        cmaes_fail_soft_evaluations = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        cmaes_fail_soft_reasons = ["" for _ in range(self.n_agents)]
        candidate_validation_evals = 0
        candidate_shape_fallbacks = 0
        for agent_result in agent_results:
            (
                i,
                optimizer_name,
                x_base_i,
                options,
                res,
            ) = agent_result[:5]
            if len(agent_result) == 8:
                (
                    next_persistent_state,
                    next_persistent_signature,
                    persistent_telemetry,
                ) = agent_result[5:]
                self.persistent_sepcmaes_states[int(i)] = copy.deepcopy(
                    next_persistent_state
                )
                self.persistent_sepcmaes_signatures[int(i)] = (
                    copy.deepcopy(next_persistent_signature)
                )
                self.last_persistent_sepcmaes_active[int(i)] = 1.0
                self.last_persistent_sepcmaes_fresh[int(i)] = float(
                    bool(persistent_telemetry["fresh"])
                )
                self.last_persistent_sepcmaes_config_reset[int(i)] = (
                    float(bool(persistent_telemetry["config_reset"]))
                )
                self.last_persistent_sepcmaes_recenter_requested[int(i)] = (
                    float(
                        persistent_telemetry[
                            "recenter_requested_norm"
                        ]
                    )
                )
                self.last_persistent_sepcmaes_recenter_applied[int(i)] = (
                    float(
                        persistent_telemetry["recenter_applied_norm"]
                    )
                )
                self.last_persistent_sepcmaes_recenter_remaining[int(i)] = (
                    float(
                        persistent_telemetry[
                            "recenter_remaining_norm"
                        ]
                    )
                )
                self.persistent_sepcmaes_events += 1
                self.persistent_sepcmaes_fresh_events += int(
                    bool(persistent_telemetry["fresh"])
                )
                self.persistent_sepcmaes_resume_events += int(
                    bool(persistent_telemetry["resumed"])
                )
                self.persistent_sepcmaes_config_resets += int(
                    bool(persistent_telemetry["config_reset"])
                )
            elif len(agent_result) != 5:
                raise RuntimeError(
                    "Unexpected objective-split worker result shape: "
                    f"{len(agent_result)}."
                )
            (
                best_x,
                best_local,
                used_shape_fallback,
                validation_eval_count,
            ) = self._normalize_candidate_x(
                res.get("best_so_far_x", x_base_i),
                agent_id=i,
                x_fallback=x_base_i,
            )
            n_eval = int(res["n_function_evaluations"])
            if n_eval < 0:
                raise RuntimeError(
                    "Objective-split optimizer reported negative FEs."
                )
            evals_per_agent[int(i)] = n_eval
            if bool(
                res.get("optimizer_numeric_fail_soft_triggered", False)
            ):
                cmaes_fail_soft_mask[int(i)] = 1.0
                cmaes_fail_soft_generations[int(i)] = int(
                    res.get("optimizer_numeric_fail_soft_generation", -1)
                )
                cmaes_fail_soft_evaluations[int(i)] = int(
                    res.get("optimizer_numeric_fail_soft_evaluations", n_eval)
                )
                cmaes_fail_soft_reasons[int(i)] = str(
                    res.get("optimizer_numeric_fail_soft_reason", "unknown")
                )
            sigma_used = float(
                res.get(
                    "sigma",
                    options.get(
                        "sigma",
                        self._sigma_ref(i, optimizer_name),
                    ),
                )
            )
            if np.isfinite(sigma_used) and sigma_used > 0:
                self.last_sigma_value[i] = sigma_used
            self.last_optimizer_guide_internal_active[i] = float(
                res.get("optimizer_guide_internal_applied", 0.0)
            )
            self.last_optimizer_guide_internal_mean_step_norm[i] = (
                float(
                    res.get(
                        "optimizer_guide_internal_mean_step_norm",
                        0.0,
                    )
                )
            )
            self.last_optimizer_guide_internal_alignment[i] = float(
                res.get("optimizer_guide_internal_alignment", 0.0)
            )
            self.last_optimizer_anchor_applied[i] = float(
                res.get("optimizer_anchor_applied", 0.0)
            )
            self.last_optimizer_anchor_mean_step_norm[i] = float(
                res.get("optimizer_anchor_mean_step_norm", 0.0)
            )
            self.last_optimizer_anchor_sample_applied[i] = float(
                res.get("optimizer_anchor_sample_applied", 0.0)
            )
            self.cumulative_actual_fes_per_agent[i] += float(
                n_eval
            )
            if not np.isfinite(best_local):
                best_local = float(
                    res.get("best_so_far_y", np.inf)
                )
            if used_shape_fallback:
                candidate_shape_fallbacks += 1
            local_improve = _safe_log_improvement(
                float(references[i]),
                best_local,
            )

            int_keys = {
                "n_individuals",
                "m",
                "ms",
                "k_init",
                "kmax",
                "lam",
                "distance",
            }
            float_keys = {
                "sigma",
                "c_s",
                "a_z",
                "c_a",
                "gamma",
                "cs",
                "ds",
                "k_inc_cond",
                "k_dec_cond",
            }
            if not self._uses_pre_caf4a62_arch():
                float_keys = set(float_keys) | {
                    "cov_lr_scale",
                    "c_s_scale",
                }
            rec = {}
            for key, value in options.items():
                if key in int_keys:
                    rec[key] = int(value)
                elif key in float_keys:
                    rec[key] = float(value)
            self.param_state_cache[int(i)][
                str(optimizer_name).lower()
            ] = rec

            proposal_xs.append(best_x)
            proposal_local_values.append(float(best_local))
            local_improvements.append(float(local_improve))
            candidate_validation_evals += int(
                validation_eval_count
            )

        return {
            "proposal_states": np.stack(proposal_xs, axis=0),
            "proposal_local_values": np.asarray(
                proposal_local_values,
                dtype=np.float64,
            ),
            "local_improvements": np.asarray(
                local_improvements,
                dtype=np.float64,
            ),
            "evals_per_agent": evals_per_agent,
            "total_evals": int(np.sum(evals_per_agent)),
            "candidate_validation_evals": int(
                candidate_validation_evals
            ),
            "candidate_shape_fallbacks": int(
                candidate_shape_fallbacks
            ),
            "cmaes_numeric_fail_soft_mask": cmaes_fail_soft_mask,
            "cmaes_numeric_fail_soft_generations": (
                cmaes_fail_soft_generations
            ),
            "cmaes_numeric_fail_soft_evaluations": (
                cmaes_fail_soft_evaluations
            ),
            "cmaes_numeric_fail_soft_reasons": list(
                cmaes_fail_soft_reasons
            ),
        }

    def _run_event_slot_interleaving_event(
        self,
        *,
        base_states: np.ndarray,
        local_reference: np.ndarray,
        opt_actions: np.ndarray,
        cfg_actions_block: np.ndarray,
        res_actions: np.ndarray,
        collab_actions: np.ndarray,
        guide_scale_actions: np.ndarray,
        comm_rounds: int,
    ) -> Dict:
        """Run one slow Actor event as K cost-matched native-work slots."""
        if not self.event_slot_interleaving_enable:
            raise RuntimeError("Event-slot interleaving is not enabled.")
        slot_count = int(comm_rounds)
        if slot_count <= 0:
            raise ValueError("Event-slot interleaving requires positive K.")
        self._reset_sigma_diagnostics()
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        original_reference = np.asarray(
            local_reference, dtype=np.float64
        ).reshape(self.n_agents)
        packet_reference = original_reference.copy()

        session_specs = []
        plans = []
        option_records = []
        optimizer_names = []
        # Event-level forensic accumulators (docs 28 card).  Defined here, before
        # the session-setup loop, because the fitness closure below captures them.
        # When the switch is off they stay None and the closure does no extra work.
        _forensic_capture = bool(
            int(getattr(self.opts, "eval_save_optimizer_forensic_trace", 0))
        )
        _forensic_f_min = (
            np.full((self.n_agents,), np.inf, dtype=np.float64)
            if _forensic_capture
            else None
        )
        _forensic_f_max = (
            np.full((self.n_agents,), -np.inf, dtype=np.float64)
            if _forensic_capture
            else None
        )
        for agent_id in range(self.n_agents):
            optimizer_name = self.optimizer_candidates[
                int(opt_actions[agent_id])
            ]
            optimizer_key = str(optimizer_name).lower()
            subfes_i = max(
                1,
                int(
                    round(
                        self.subfes_per_agent
                        * float(
                            self.resource_factors[int(res_actions[agent_id])]
                        )
                    )
                ),
            )
            seed = int(self.opts.seed + self.step_count * 1000 + agent_id)
            options = self._build_optimizer_options(
                agent_id=agent_id,
                optimizer_name=optimizer_name,
                cfg_levels=cfg_actions_block[agent_id].tolist(),
                dims=self.full_dims,
                x_base=bases[agent_id],
                subfes_i=subfes_i,
                seed=seed,
            )
            guide_options = self._optimizer_guide_options(
                agent_id,
                optimizer_name,
                collab_mode_idx=int(collab_actions[agent_id]),
                guide_scale_idx=int(guide_scale_actions[agent_id]),
                sigma_value=float(
                    options.get(
                        "sigma", self._sigma_ref(agent_id, optimizer_name)
                    )
                ),
            )
            options.update(guide_options)
            options.update(
                self._anchor_options(
                    agent_id, optimizer_name, bases[agent_id]
                )
            )
            if int(getattr(self.opts, "eval_save_boundary_compare_trace", 0)):
                if optimizer_key in {"cmaes", "sepcmaes"}:
                    options["optimizer_numeric_counter_enable"] = True
                options["boundary_compare_trace_dir"] = os.path.join(
                    str(self.opts.test_dir), "boundary_compare_trace"
                )
                options["boundary_compare_identity"] = {
                    "optimizer": optimizer_key,
                    "function_id": int(self.question),
                    "run_seed": int(getattr(self.opts, "seed", -1)),
                    "env_step": int(self.step_count),
                    "agent_id": int(agent_id),
                    "optimizer_seed": int(seed),
                    "requested_fes": int(subfes_i),
                    "population": int(options.get("n_individuals", 0)),
                    "sigma_used": float(options.get("sigma", np.nan)),
                    "inherit_applied": int(self.last_sigma_diagnostics["inherit_applied"][agent_id]),
                }
            if optimizer_key == "vkd" and self.vkd_origin_trace_enable:
                options["vkd_origin_trace_dir"] = self.vkd_origin_trace_dir
                options["vkd_origin_trace_identity"] = {
                    "problem_family": str(self.problem_family),
                    "function_id": int(self.question),
                    "run_seed": int(getattr(self.opts, "seed", -1)),
                    "env_step": int(self.step_count),
                    "agent_id": int(agent_id),
                    "optimizer_seed": int(seed),
                    "sigma_used": float(options.get("sigma", np.nan)),
                    "inherit_applied": int(self.last_sigma_diagnostics["inherit_applied"][agent_id]),
                    "previous_optimizer": str(self.last_sigma_diagnostics["previous_optimizer"][agent_id]),
                }
                options["vkd_origin_objective"] = {
                    "source": np.asarray(self.fun.source[agent_id], dtype=np.float64).tolist(),
                    "observed": np.asarray(self.fun.observed[agent_id], dtype=np.float64).tolist(),
                    "metric_mode": str(self.fun.metric_mode),
                    "target_num": int(self.fun.target_num),
                    "coordinate_dim": int(self.fun.coordinate_dim),
                    "bounds": [float(self.lb), float(self.ub)],
                }
            if optimizer_key in {"cmaes", "sepcmaes"} and int(
                getattr(self.opts, "eval_save_event_slot_diagnostics", 0)
            ):
                options["record_native_first_generation"] = True
            if optimizer_key == "cmaes":
                options["optimizer_numeric_fail_soft"] = bool(
                    self.cmaes_numeric_fail_soft
                )
            if optimizer_key in {"cmaes", "sepcmaes"} and (
                self.optimizer_numeric_telemetry_enable
                or (optimizer_key == "cmaes" and self.optimizer_numeric_counter_enable)
                or (
                    optimizer_key == "cmaes"
                    and self.cmaes_numeric_fail_soft
                )
            ):
                options["optimizer_numeric_telemetry_context"] = {
                    "problem_family": str(self.problem_family),
                    "function_id": int(self.question),
                    "env_step": int(self.step_count),
                    "agent_id": int(agent_id),
                    "optimizer_action": int(opt_actions[agent_id]),
                    "cfg_levels": [
                        int(x)
                        for x in cfg_actions_block[agent_id].tolist()
                    ],
                    "resource_action": int(res_actions[agent_id]),
                    "subfes": int(subfes_i),
                    "seed": int(seed),
                    "event_slot_interleaving": True,
                    "event_slot_count": int(slot_count),
                }
                if self.optimizer_numeric_telemetry_enable:
                    options["optimizer_numeric_telemetry_enable"] = True
                    options["optimizer_numeric_telemetry_dir"] = str(
                        self.optimizer_numeric_telemetry_dir
                    )
                if optimizer_key == "cmaes" and self.optimizer_numeric_counter_enable:
                    options["optimizer_numeric_counter_enable"] = True
                    options["optimizer_numeric_counter_dir"] = self.optimizer_numeric_counter_dir
            population = int(
                options.get(
                    "n_individuals",
                    4 + int(3 * np.log(max(2, self.D))),
                )
            )
            plan = build_event_slot_plan(
                optimizer_key,
                subfes_i,
                population,
                slot_count,
                dimension=self.D,
            )
            if "boundary_compare_identity" in options:
                options["boundary_compare_identity"]["slot_targets"] = list(
                    plan.cumulative_reported_targets
                )

            aid = int(agent_id)

            def fitness_local(z_batch: np.ndarray, aid=aid):
                z_batch = np.asarray(z_batch, dtype=np.float64)
                if z_batch.ndim == 1:
                    z_batch = z_batch[None, :]
                elif z_batch.ndim == 3 and z_batch.shape[1] == 1:
                    z_batch = np.squeeze(z_batch, axis=1)
                if z_batch.ndim != 2 or z_batch.shape[1] != self.D:
                    raise ValueError(
                        f"fitness_local expected [N,{self.D}], got "
                        f"{z_batch.shape}."
                    )
                _values = np.asarray(
                    self.fun.local_eval_batch(aid, z_batch),
                    dtype=np.float64,
                ).reshape(-1)
                # Forensic spread accumulator (docs 28 card).  Pure bookkeeping
                # on values we already have: no extra objective call, no RNG, no
                # state write.  Fully skipped when the switch is off.
                if _forensic_capture and _values.size:
                    _finite_values = _values[np.isfinite(_values)]
                    if _finite_values.size:
                        _lo = float(np.min(_finite_values))
                        _hi = float(np.max(_finite_values))
                        if _lo < _forensic_f_min[aid]:
                            _forensic_f_min[aid] = _lo
                        if _hi > _forensic_f_max[aid]:
                            _forensic_f_max[aid] = _hi
                return _values

            problem = {
                "fitness_function": fitness_local,
                "ndim_problem": int(self.D),
                "lower_boundary": float(self.lb)
                * np.ones((self.D,), dtype=np.float64),
                "upper_boundary": float(self.ub)
                * np.ones((self.D,), dtype=np.float64),
            }
            session_specs.append((optimizer_key, problem, options))
            plans.append(plan)
            option_records.append(options)
            optimizer_names.append(optimizer_key)

        # Build every heterogeneous session only after all plans validate.
        # A repeated target is an explicit communication-only slot: it keeps
        # the Actor-selected K without adding objective calls.
        sessions = [
            create_event_slot_session(name, problem, options)
            for name, problem, options in session_specs
        ]

        packet_units = np.zeros(
            (slot_count, self.n_agents), dtype=np.int64
        )
        packet_improvements = np.zeros(
            (slot_count, self.n_agents), dtype=np.float64
        )
        center_shift_norms = np.zeros(
            (slot_count, self.n_agents), dtype=np.float64
        )
        commit_audit = _empty_event_slot_commit_audit(
            slot_count, self.n_agents
        )
        mmes_neutral_success = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        # Best-effort forensic capture (docs 28 card).  Gated by the eval-side
        # switch: when off, `forensic` stays None and every capture below is
        # skipped, so the code path is unchanged.  NaN means "not captured" and
        # must stay distinguishable from a real zero.
        forensic_enabled = bool(
            int(getattr(self.opts, "eval_save_optimizer_forensic_trace", 0))
        )
        forensic = (
            {
                field: np.full(
                    (slot_count, self.n_agents), np.nan, dtype=np.float64
                )
                for field in EVENT_SLOT_FORENSIC_FIELDS
            }
            if forensic_enabled
            else None
        )
        forensic_details = (
            [[{} for _ in range(self.n_agents)] for _ in range(slot_count)]
            if forensic_enabled
            else None
        )
        def _fnum(value):
            """Total conversion for the forensic capture: never raises."""
            if isinstance(value, (bool, int, float, np.integer, np.floating)):
                try:
                    return float(value)
                except (OverflowError, ValueError, TypeError):
                    return float("nan")
            return float("nan")

        def _vector_summary(value):
            """Stable finite-state summary without squaring unscaled values."""
            try:
                array = np.asarray(value, dtype=np.float64).reshape(-1)
            except Exception:
                return (0.0, 0.0, 0.0, np.nan, np.nan, 0.0)
            has_nan = bool(np.any(np.isnan(array)))
            has_inf = bool(np.any(np.isinf(array)))
            finite = bool(np.all(np.isfinite(array)))
            finite_values = np.abs(array[np.isfinite(array)])
            max_abs = float(np.max(finite_values)) if finite_values.size else np.nan
            if finite_values.size and max_abs > 0.0:
                stable_norm = float(max_abs * np.linalg.norm(finite_values / max_abs))
            elif finite_values.size:
                stable_norm = 0.0
            else:
                stable_norm = np.nan
            ordinary_norm = float(np.linalg.norm(array))
            norm_overflow = bool(finite and not np.isfinite(ordinary_norm))
            return (
                float(finite), float(has_nan), float(has_inf), max_abs,
                stable_norm, float(norm_overflow),
            )

        def _store_vector_summary(prefix, slot_id, agent_id, value):
            names = (
                "finite", "is_nan", "is_inf", "max_abs",
                "stable_norm", "norm_overflow",
            )
            for name, result in zip(names, _vector_summary(value)):
                forensic[f"{prefix}_{name}"][slot_id, agent_id] = result
            try:
                array = np.asarray(value, dtype=np.float64)
                flat_finite = np.isfinite(array).reshape(-1)
                forensic_details[slot_id][agent_id].update(
                    {
                        f"{prefix}_shape": list(array.shape),
                        f"{prefix}_finite_mask": flat_finite.tolist(),
                        f"{prefix}_nonfinite_indices": np.flatnonzero(
                            ~flat_finite
                        ).astype(int).tolist(),
                    }
                )
            except Exception:
                forensic_details[slot_id][agent_id].update(
                    {
                        f"{prefix}_shape": [],
                        f"{prefix}_finite_mask": [],
                        f"{prefix}_nonfinite_indices": [],
                    }
                )
        vkd_capture = any(int(getattr(self.opts, flag, 0)) for flag in (
            "eval_save_event_slot_diagnostics", "eval_save_optimizer_forensic_trace",
        ))
        vkd_diagnostics = {
            field: np.full((slot_count, self.n_agents), np.nan, dtype=np.float64)
            for field in EVENT_SLOT_VKD_FIELDS
        } if vkd_capture else None
        mmes_diagnostics = {
            field: np.full((slot_count, self.n_agents), None, dtype=object)
            for field in MMES_FIELDS
        } if vkd_capture else None
        native_state_capture = bool(int(getattr(self.opts, "eval_save_event_slot_diagnostics", 0)))
        native_state_trace = (
            [[None for _ in range(self.n_agents)] for _ in range(slot_count)]
            if native_state_capture else None
        )
        detailed_vkd_trace = bool(int(getattr(self.opts, "eval_save_vkd_state_trace", 0)))
        vkd_generation_traces = [[[] for _ in range(self.n_agents)] for _ in range(slot_count)] if detailed_vkd_trace else None
        vkd_slot_start_centers = np.full((slot_count, self.n_agents, self.D), np.nan, dtype=np.float64) if vkd_capture else None
        vkd_slot_proposal_centers = np.full((slot_count, self.n_agents, self.D), np.nan, dtype=np.float64) if vkd_capture else None
        vkd_slot_committed_centers = np.full((slot_count, self.n_agents, self.D), np.nan, dtype=np.float64) if vkd_capture else None
        vkd_slot_shift_vectors = np.full((slot_count, self.n_agents, self.D), np.nan, dtype=np.float64) if vkd_capture else None
        current_bases = bases.copy()
        slot_start_centers = current_bases.copy()
        final_proposals = bases.copy()
        final_proposal_values = original_reference.copy()
        final_results = [None for _ in range(self.n_agents)]
        communication_applied = False
        applied_slot_count = 0
        candidate_validation_evals = 0
        candidate_shape_fallbacks = 0

        for slot_id in range(slot_count):
            slot_start_centers = current_bases.copy()
            if vkd_capture:
                vkd_slot_start_centers[slot_id] = current_bases
            slot_proposals = np.empty_like(current_bases)
            slot_values = np.empty((self.n_agents,), dtype=np.float64)
            for agent_id, (session, plan) in enumerate(
                zip(sessions, plans)
            ):
                session._boundary_slot_id = int(slot_id)
                if self.vkd_origin_trace_enable and session.optimizer_name == "vkd":
                    session._origin_slot_id = int(slot_id)
                target = int(plan.cumulative_reported_targets[slot_id])
                slot_units = int(plan.units_per_slot[slot_id])
                if session.terminated or (
                    target <= session.reported_evaluations
                    and (slot_units <= 0 or plan.optimizer_name != "mmes")
                ):
                    # 旧语义：非 MMES 会话（如 VKD 收敛重启 _lam 翻倍导致
                    # reported 越过后续槽目标）一律回到 communication-only，
                    # 不进入 force 通道。
                    packet = session.communication_only_packet(
                        current_bases[agent_id],
                        float(packet_reference[agent_id]),
                    )
                elif target <= session.reported_evaluations:
                    # 保底最少评估（F5 死锁修复）：B<=D 时计划器给 MMES 分配的
                    # 唯一强制单元在此真实执行一次，即使 reported 已达预算；
                    # 经首分支过滤，此分支只有存活且 units>0 的 MMES 可达，
                    # 而只有 MMES 会话支持 force 参数。
                    packet = session.advance_to_reported_target(
                        target, force=True
                    )
                else:
                    packet = session.advance_to_reported_target(target)
                if self.sigma_inherit_enable and slot_id == slot_count - 1:
                    # σ 继承写入点（14 卡）：事件末（最后一个 slot）采集每个
                    # agent 会话的演化后 σ——四会话统一新增的 evolved_sigma()
                    # 暴露（CMAES/MMES/Sep = core.sigma、VKD = core.sigma 或
                    # _last_sigma），不动 result() 既有字段语义——连同 optimizer
                    # 名与签名（维度/bounds/population）写入 σ 槽；prev_optimizer
                    # 供观测 one-hot 使用。换车由读取端比对 optimizer 名实现。
                    packet_sigma = session.evolved_sigma()
                    if packet_sigma is not None:
                        aid_slot = int(agent_id)
                        self.last_sigma_diagnostics["evolved_sigma"][aid_slot] = float(packet_sigma)
                        self._sigma_inherit_slots[aid_slot] = (
                            str(plan.optimizer_name).lower(),
                            float(packet_sigma),
                            (
                                int(self.D),
                                float(self.lb),
                                float(self.ub),
                                int(plan.population),
                            ),
                        )
                        self._sigma_inherit_prev_optimizer[aid_slot] = str(
                            plan.optimizer_name
                        ).lower()
                packet_units[slot_id, agent_id] = int(
                    packet["packet_population_units"]
                )
                raw_x = np.asarray(
                    packet["packet_best_x"], dtype=np.float64
                ).reshape(-1)
                raw_y = float(packet["packet_best_y"])
                if forensic_enabled:
                    _store_vector_summary(
                        "packet_candidate", slot_id, agent_id, raw_x
                    )
                if raw_x.shape != (self.D,) or not np.all(
                    np.isfinite(raw_x)
                ):
                    raise FloatingPointError(
                        "Event-slot packet candidate is malformed or non-finite."
                    )
                if slot_id == slot_count - 1:
                    (
                        candidate_x,
                        candidate_y,
                        used_shape_fallback,
                        validation_eval_count,
                    ) = self._normalize_candidate_x(
                        raw_x,
                        agent_id=agent_id,
                        x_fallback=current_bases[agent_id],
                    )
                    candidate_validation_evals += int(
                        validation_eval_count
                    )
                    candidate_shape_fallbacks += int(
                        bool(used_shape_fallback)
                    )
                else:
                    candidate_x = np.clip(raw_x, self.lb, self.ub)
                    candidate_y = raw_y
                slot_proposals[agent_id] = candidate_x
                slot_values[agent_id] = float(candidate_y)
                final_results[agent_id] = packet

            slot_local_improvements = np.asarray(
                [
                    _safe_log_improvement(float(before), float(after))
                    for before, after in zip(packet_reference, slot_values)
                ],
                dtype=np.float64,
            )
            packet_improvements[slot_id] = slot_local_improvements
            if vkd_capture:
                vkd_slot_proposal_centers[slot_id] = slot_proposals
            next_states, slot_communication = self._apply_consensus(
                base_states=current_bases,
                proposal_states=slot_proposals,
                local_improvements=slot_local_improvements,
                comm_rounds=1,
            )
            communication_applied = bool(
                communication_applied or slot_communication
            )
            if vkd_capture:
                vkd_slot_committed_centers[slot_id] = next_states
                vkd_slot_shift_vectors[slot_id] = next_states - slot_proposals
            applied_slot_count += int(bool(slot_communication))

            proposal_steps = slot_proposals - current_bases
            commit_steps = next_states - current_bases
            corrections = next_states - slot_proposals
            for agent_id, session in enumerate(sessions):
                proposal_step = proposal_steps[agent_id]
                commit_step = commit_steps[agent_id]
                proposal_norm = float(np.linalg.norm(proposal_step))
                commit_norm = float(np.linalg.norm(commit_step))
                correction_norm = float(
                    np.linalg.norm(corrections[agent_id])
                )
                cosine = 0.0
                if proposal_norm > 1e-12 and commit_norm > 1e-12:
                    cosine = float(
                        np.clip(
                            np.dot(proposal_step, commit_step)
                            / (proposal_norm * commit_norm),
                            -1.0,
                            1.0,
                        )
                    )
                credit_proxy = (
                    float(
                        max(0.0, cosine)
                        * min(1.0, commit_norm / proposal_norm)
                    )
                    if proposal_norm > 1e-12 and commit_norm > 1e-12
                    else 0.0
                )
                if detailed_vkd_trace and session.optimizer_name == "vkd":
                    trace = getattr(session, "_vkd_generations", [])
                    vkd_generation_traces[slot_id][agent_id] = copy.deepcopy(trace)
                memory = session.diagnostic_snapshot(commit_step)
                if native_state_trace is not None and session.optimizer_name in {"cmaes", "sepcmaes"}:
                    native_state_trace[slot_id][agent_id] = {
                        "before_commit": session.native_state_snapshot(),
                        "first_generation": session.packet_native_first_generation_snapshot(),
                    }
                if mmes_diagnostics is not None and session.optimizer_name == "mmes":
                    for field in MMES_FIELDS:
                        mmes_diagnostics[field][slot_id, agent_id] = memory.get(field)
                if vkd_capture and session.optimizer_name == "vkd":
                    for field in EVENT_SLOT_VKD_FIELDS:
                        vkd_diagnostics[field][slot_id, agent_id] = _fnum(
                            memory.get(field)
                        )
                _search_center = np.asarray(
                    memory["search_center"], dtype=np.float64
                ).reshape(self.D)
                internal_step = _search_center - current_bases[agent_id]
                if forensic_enabled:
                    _synchronization_shift = (
                        next_states[agent_id] - _search_center
                    )
                    _store_vector_summary(
                        "internal_mean", slot_id, agent_id, _search_center
                    )
                    _store_vector_summary(
                        "internal_shift", slot_id, agent_id,
                        _synchronization_shift,
                    )
                internal_norm = float(np.linalg.norm(internal_step))
                internal_cosine = 0.0
                if internal_norm > 1e-12 and commit_norm > 1e-12:
                    internal_cosine = float(
                        np.clip(
                            np.dot(internal_step, commit_step)
                            / (internal_norm * commit_norm),
                            -1.0,
                            1.0,
                        )
                    )
                commit_rms = float(
                    commit_norm / np.sqrt(float(self.D))
                )
                scale_ratio = (
                    float(memory["effective_scale_rms"]) / commit_rms
                    if commit_rms > 1e-12
                    else -1.0
                )
                if not np.isfinite(scale_ratio):
                    scale_ratio = float(np.finfo(np.float64).max)
                commit_audit["proposal_step_norm"][
                    slot_id, agent_id
                ] = proposal_norm
                commit_audit["commit_step_norm"][
                    slot_id, agent_id
                ] = commit_norm
                commit_audit["correction_norm"][
                    slot_id, agent_id
                ] = correction_norm
                commit_audit["proposal_commit_cosine"][
                    slot_id, agent_id
                ] = cosine
                commit_audit["commit_credit_proxy"][
                    slot_id, agent_id
                ] = credit_proxy
                commit_audit["internal_center_step_norm"][
                    slot_id, agent_id
                ] = internal_norm
                commit_audit["internal_center_commit_cosine"][
                    slot_id, agent_id
                ] = internal_cosine
                for field in (
                    "primary_memory_norm",
                    "primary_memory_commit_cosine",
                    "secondary_memory_norm",
                    "secondary_memory_commit_cosine",
                    "effective_scale_rms",
                    "guide_norm",
                    "guide_strength",
                    "guide_mix_strength",
                    "guide_commit_cosine",
                ):
                    commit_audit[field][slot_id, agent_id] = float(
                        memory[field]
                    )
                commit_audit["effective_scale_commit_ratio"][
                    slot_id, agent_id
                ] = scale_ratio

                # --- best-effort forensic capture (docs 28 card) ----------
                # Read-only and total: never inside a branch condition, never
                # raises, never touches optimizer state, RNG or the objective.
                # Skipped entirely when the switch is off.
                if forensic_enabled:
                    forensic["internal_step_norm"][slot_id, agent_id] = _fnum(
                        internal_norm
                    )
                    _center = (
                        memory.get("search_center")
                        if isinstance(memory, dict)
                        else None
                    )
                    if _center is not None:
                        try:
                            forensic["internal_mean_norm"][
                                slot_id, agent_id
                            ] = float(
                                np.linalg.norm(
                                    np.asarray(_center, dtype=np.float64)
                                )
                            )
                        except Exception:
                            pass
                    forensic["optimizer_lambda"][slot_id, agent_id] = _fnum(
                        getattr(session, "population", None)
                    )
                    forensic["optimizer_terminated"][
                        slot_id, agent_id
                    ] = _fnum(getattr(session, "terminated", None))
                    _generations = _fnum(
                        getattr(session, "distribution_updates", None)
                    )
                    forensic["optimizer_generation_index"][
                        slot_id, agent_id
                    ] = _generations
                    _requested = getattr(session, "requested_fes", None)
                    _reported = getattr(session, "reported_evaluations", None)
                    if _requested is not None and _reported is not None:
                        forensic["remaining_budget_fes"][
                            slot_id, agent_id
                        ] = _fnum(_requested) - _fnum(_reported)

            if self.ccsa_momentum_update_mode == "always":
                self._update_ccsa_direction_momentum(
                    base_states=current_bases,
                    proposals=slot_proposals,
                    local_improvements=slot_local_improvements,
                )
            self._update_optimizer_guide_direction(
                base_states=current_bases,
                proposal_states=slot_proposals,
                local_improvements=slot_local_improvements,
            )

            for agent_id, session in enumerate(sessions):
                origin_before = None
                if self.vkd_origin_trace_enable and session.optimizer_name == "vkd":
                    origin_core = session._core
                    origin_before = np.asarray(
                        session._last_mean if origin_core is None else origin_core.xmean,
                        dtype=np.float64,
                    ).copy()
                sync = session.synchronize_center(next_states[agent_id])
                if int(getattr(self.opts, "eval_save_boundary_compare_trace", 0)):
                    session.boundary_commit(
                        slot_id, final_results[agent_id]["search_center"],
                        slot_proposals[agent_id], next_states[agent_id],
                        sync.get("transition"),
                    )
                if origin_before is not None:
                    try:
                        from optimizers.unified_opt.vkd_origin_trace import get_trace
                        trace = get_trace(self.vkd_origin_trace_dir)
                        if trace is not None:
                            identity = session.options["vkd_origin_trace_identity"]
                            trace.note_commit(identity, slot_id, origin_before,
                                              next_states[agent_id], session)
                            if not np.isfinite(float(sync["applied_norm"])):
                                trace.failure(identity, "event_slot_center_shift_norms",
                                              slot_id, agent_id, [slot_id, agent_id], session)
                    except Exception as exc:
                        print(f"[vkd-origin-trace] commit capture failed: {exc}", flush=True)
                if native_state_trace is not None and native_state_trace[slot_id][agent_id] is not None:
                    native_state_trace[slot_id][agent_id]["after_commit"] = (
                        session.native_state_snapshot()
                    )
                    if "transition" in sync:
                        native_state_trace[slot_id][agent_id]["transition"] = sync["transition"]
                center_shift_norms[slot_id, agent_id] = float(
                    sync["applied_norm"]
                )
                if (
                    optimizer_names[agent_id] == "mmes"
                    and sync.get("next_success_credit") == "decay_only"
                    and slot_id < slot_count - 1
                ):
                    mmes_neutral_success[agent_id] += 1
                if slot_id < slot_count - 1:
                    guide_options = self._optimizer_guide_options(
                        agent_id,
                        optimizer_names[agent_id],
                        collab_mode_idx=int(collab_actions[agent_id]),
                        guide_scale_idx=int(
                            guide_scale_actions[agent_id]
                        ),
                        sigma_value=float(
                            final_results[agent_id].get(
                                "sigma",
                                option_records[agent_id].get(
                                    "sigma",
                                    self._sigma_ref(
                                        agent_id,
                                        optimizer_names[agent_id],
                                    ),
                                ),
                            )
                        ),
                    )
                    session.configure_guide(guide_options)

            current_bases = np.asarray(
                next_states, dtype=np.float64
            ).reshape(self.n_agents, self.D)
            packet_reference = slot_values.copy()
            final_proposals = slot_proposals.copy()
            final_proposal_values = slot_values.copy()

        if communication_applied:
            self.last_comm_rounds_applied = int(applied_slot_count)
            self.step_comm_rounds_applied = int(applied_slot_count)
            self.step_comm_events = int(applied_slot_count)

        evals_per_agent = np.asarray(
            [session.reported_evaluations for session in sessions],
            dtype=np.int64,
        )
        native_evals_per_agent = np.asarray(
            [session.native_evaluations for session in sessions],
            dtype=np.int64,
        )
        physical_evals_per_agent = np.asarray(
            [session.physical_evaluations for session in sessions],
            dtype=np.int64,
        )
        distribution_updates = np.asarray(
            [session.distribution_updates for session in sessions],
            dtype=np.int64,
        )
        cmaes_fail_soft_mask = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        cmaes_fail_soft_generations = np.full(
            (self.n_agents,), -1, dtype=np.int64
        )
        cmaes_fail_soft_evaluations = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        cmaes_fail_soft_reasons = ["" for _ in range(self.n_agents)]

        int_keys = {
            "n_individuals",
            "m",
            "ms",
            "k_init",
            "kmax",
            "lam",
            "distance",
        }
        float_keys = {
            "sigma",
            "c_s",
            "a_z",
            "c_a",
            "gamma",
            "cs",
            "ds",
            "k_inc_cond",
            "k_dec_cond",
        }
        if not self._uses_pre_caf4a62_arch():
            float_keys |= {"cov_lr_scale", "c_s_scale"}

        for agent_id, (name, options, result) in enumerate(
            zip(optimizer_names, option_records, final_results)
        ):
            sigma_used = float(
                result.get(
                    "sigma",
                    options.get(
                        "sigma", self._sigma_ref(agent_id, name)
                    ),
                )
            )
            if np.isfinite(sigma_used) and sigma_used > 0.0:
                self.last_sigma_value[agent_id] = sigma_used
            self.last_optimizer_guide_internal_active[agent_id] = float(
                result.get("optimizer_guide_internal_applied", 0.0)
            )
            self.last_optimizer_guide_internal_mean_step_norm[
                agent_id
            ] = float(
                result.get(
                    "optimizer_guide_internal_mean_step_norm", 0.0
                )
            )
            self.last_optimizer_guide_internal_alignment[agent_id] = float(
                result.get("optimizer_guide_internal_alignment", 0.0)
            )
            self.last_optimizer_anchor_applied[agent_id] = float(
                result.get("optimizer_anchor_applied", 0.0)
            )
            self.last_optimizer_anchor_mean_step_norm[agent_id] = float(
                result.get("optimizer_anchor_mean_step_norm", 0.0)
            )
            self.last_optimizer_anchor_sample_applied[agent_id] = float(
                result.get("optimizer_anchor_sample_applied", 0.0)
            )
            self.cumulative_actual_fes_per_agent[agent_id] += float(
                physical_evals_per_agent[agent_id]
            )
            if bool(
                result.get(
                    "optimizer_numeric_fail_soft_triggered", False
                )
            ):
                cmaes_fail_soft_mask[agent_id] = 1.0
                cmaes_fail_soft_generations[agent_id] = int(
                    result.get(
                        "optimizer_numeric_fail_soft_generation", -1
                    )
                )
                cmaes_fail_soft_evaluations[agent_id] = int(
                    result.get(
                        "optimizer_numeric_fail_soft_evaluations",
                        evals_per_agent[agent_id],
                    )
                )
                cmaes_fail_soft_reasons[agent_id] = str(
                    result.get(
                        "optimizer_numeric_fail_soft_reason", "unknown"
                    )
                )
            rec = {}
            for key, value in options.items():
                if key in int_keys:
                    rec[key] = int(value)
                elif key in float_keys:
                    rec[key] = float(value)
            self.param_state_cache[agent_id][name] = rec

        event_local_improvements = np.asarray(
            [
                _safe_log_improvement(float(before), float(after))
                for before, after in zip(
                    original_reference, final_proposal_values
                )
            ],
            dtype=np.float64,
        )
        self.event_slot_interleaving_events += 1
        self.event_slot_interleaving_slots += int(slot_count)
        self.event_slot_native_local_evals += int(
            np.sum(native_evals_per_agent)
        )
        self.event_slot_reported_local_evals += int(
            np.sum(evals_per_agent)
        )
        self.event_slot_physical_local_evals += int(
            np.sum(physical_evals_per_agent)
        )
        self.last_event_slot_count = int(slot_count)
        self.last_event_slot_packet_units = packet_units.copy()
        self.last_event_slot_distribution_updates = (
            distribution_updates.copy()
        )
        self.last_event_slot_native_evals = native_evals_per_agent.copy()
        self.last_event_slot_physical_evals = (
            physical_evals_per_agent.copy()
        )
        self.last_event_slot_center_shift_norms = (
            center_shift_norms.copy()
        )
        self.last_event_slot_packet_improvements = (
            packet_improvements.copy()
        )
        self.last_event_slot_commit_audit = {
            key: value.copy() for key, value in commit_audit.items()
        }
        self.last_event_slot_mmes_neutral_success = (
            mmes_neutral_success.copy()
        )
        # Event-level forensic values: identical for every slot of this event.
        if forensic_enabled:
            for _aid in range(self.n_agents):
                forensic["objective_local_f_before"][:, _aid] = _fnum(
                    original_reference[_aid]
                )
                forensic["objective_local_f_after"][:, _aid] = _fnum(
                    final_proposal_values[_aid]
                )
                _hi = _fnum(_forensic_f_max[_aid])
                _lo = _fnum(_forensic_f_min[_aid])
                forensic["event_f_spread"][:, _aid] = (
                    _hi - _lo
                    if np.isfinite(_hi) and np.isfinite(_lo)
                    else np.nan
                )
        self.last_event_slot_vkd_diagnostics = vkd_diagnostics
        self.last_event_slot_vkd_generation_traces = vkd_generation_traces
        self.last_event_slot_centers = {
            "slot_start": slot_start_centers.copy(),
            "proposal": final_proposals.copy(),
            "committed": current_bases.copy(),
            "shift": current_bases.copy() - final_proposals.copy(),
        }
        self.last_event_slot_vector_centers = {
            "slot_start": vkd_slot_start_centers.copy() if vkd_slot_start_centers is not None else None,
            "proposal": vkd_slot_proposal_centers.copy() if vkd_slot_proposal_centers is not None else None,
            "committed": vkd_slot_committed_centers.copy() if vkd_slot_committed_centers is not None else None,
            "shift": vkd_slot_shift_vectors.copy() if vkd_slot_shift_vectors is not None else None,
        }
        self.last_event_slot_mmes_diagnostics = mmes_diagnostics
        self.last_event_slot_native_state_trace = native_state_trace
        self.last_event_slot_forensic = forensic
        self.last_event_slot_forensic_details = forensic_details
        return {
            "proposal_states": final_proposals,
            "proposal_local_values": final_proposal_values,
            "local_improvements": event_local_improvements,
            "evals_per_agent": evals_per_agent,
            "total_evals": int(np.sum(evals_per_agent)),
            "candidate_validation_evals": int(
                candidate_validation_evals
            ),
            "candidate_shape_fallbacks": int(
                candidate_shape_fallbacks
            ),
            "cmaes_numeric_fail_soft_mask": cmaes_fail_soft_mask,
            "cmaes_numeric_fail_soft_generations": (
                cmaes_fail_soft_generations
            ),
            "cmaes_numeric_fail_soft_evaluations": (
                cmaes_fail_soft_evaluations
            ),
            "cmaes_numeric_fail_soft_reasons": list(
                cmaes_fail_soft_reasons
            ),
            "next_agent_x": current_bases,
            "communication_applied": bool(communication_applied),
            "event_slot_count": int(slot_count),
            "event_slot_packet_units": packet_units,
            "event_slot_packet_improvements": packet_improvements,
            "event_slot_distribution_updates": distribution_updates,
            "event_slot_native_evals_per_agent": native_evals_per_agent,
            "event_slot_physical_evals_per_agent": (
                physical_evals_per_agent
            ),
            "event_slot_center_shift_norms": center_shift_norms,
            **({"vkd_origin_trace_dir": self.vkd_origin_trace_dir,
                "vkd_origin_identity": {
                    "problem_family": str(self.problem_family),
                    "function_id": int(self.question),
                    "run_seed": int(getattr(self.opts, "seed", -1)),
                    "env_step": int(self.step_count),
                }} if self.vkd_origin_trace_enable else {}),
            **({
                "event_slot_vkd_slot_start_centers": vkd_slot_start_centers.copy(),
                "event_slot_vkd_slot_proposal_centers": vkd_slot_proposal_centers.copy(),
                "event_slot_vkd_slot_committed_centers": vkd_slot_committed_centers.copy(),
                "event_slot_vkd_slot_shift_vectors": vkd_slot_shift_vectors.copy(),
                "event_slot_vkd_generation_traces": copy.deepcopy(vkd_generation_traces),
            } if vkd_capture else {}),
            "event_slot_commit_audit": {
                key: value.copy() for key, value in commit_audit.items()
            },
            "event_slot_mmes_neutral_success": mmes_neutral_success,
            **({"event_slot_mmes_diagnostics": {
                field: values.copy() for field, values in mmes_diagnostics.items()
            }} if mmes_diagnostics is not None else {}),
            **({"event_slot_native_state_trace": copy.deepcopy(native_state_trace)}
               if native_state_trace is not None else {}),
            "event_slot_sigma_diagnostics": {
                key: value.copy() for key, value in self.last_sigma_diagnostics.items()
            },
            **({
                "event_slot_vkd_diagnostics": {
                    key: value.copy()
                    for key, value in self.last_event_slot_vkd_diagnostics.items()
                },
                "vkd_ps_outlet_mode": str(self.vkd_ps_outlet_mode),
            } if getattr(self, "last_event_slot_vkd_diagnostics", None) is not None
               and self.event_slot_interleaving_enable else {}),
        }

    def _persistent_sepcmaes_diagnostic_summary(self) -> Dict[str, float]:
        """Reduce detached bank state into scalar, evaluator-safe metrics."""
        sigmas = []
        generations = []
        effective_axis_stds = []
        effective_axis_cap_ratios = []
        counter_names = (
            "sigma_max_clip",
            "exp_arg_clip",
            "covariance_nonfinite_repair",
            "nonpositive_covariance_repair",
            "covariance_axis_std_clip",
        )
        counter_totals = {name: 0 for name in counter_names}

        for state in self.persistent_sepcmaes_states:
            if state is None:
                continue
            progress = state["progress"]
            arrays = state["arrays"]
            metadata = state["metadata"]
            sigma = float(progress["sigma"])
            d = np.asarray(arrays["d"], dtype=np.float64)
            lower = np.asarray(
                metadata["lower_boundary"],
                dtype=np.float64,
            )
            upper = np.asarray(
                metadata["upper_boundary"],
                dtype=np.float64,
            )
            clip_ratio = float(
                metadata["optimizer_guide_sigma_clip_ratio"]
            )
            axis_std = sigma * d
            axis_cap = clip_ratio * (upper - lower)

            sigmas.append(sigma)
            generations.append(int(progress["n_generations"]))
            effective_axis_stds.extend(axis_std.tolist())
            effective_axis_cap_ratios.extend(
                np.divide(
                    axis_std,
                    axis_cap,
                    out=np.zeros_like(axis_std),
                    where=axis_cap > 0.0,
                ).tolist()
            )
            counters = state.get("numeric_telemetry", {}).get(
                "counters",
                {},
            )
            for name in counter_names:
                counter_totals[name] += int(counters.get(name, 0))

        state_count = len(sigmas)
        if state_count == 0:
            sigmas = [0.0]
            generations = [0]
            effective_axis_stds = [0.0]
            effective_axis_cap_ratios = [0.0]

        summary = {
            "persistent_sepcmaes_state_count": float(state_count),
            "persistent_sepcmaes_generation_mean": float(
                np.mean(generations)
            ),
            "persistent_sepcmaes_generation_max": float(
                np.max(generations)
            ),
            "persistent_sepcmaes_sigma_mean": float(np.mean(sigmas)),
            "persistent_sepcmaes_sigma_max": float(np.max(sigmas)),
            "persistent_sepcmaes_effective_axis_std_mean": float(
                np.mean(effective_axis_stds)
            ),
            "persistent_sepcmaes_effective_axis_std_max": float(
                np.max(effective_axis_stds)
            ),
            "persistent_sepcmaes_axis_cap_ratio_mean": float(
                np.mean(effective_axis_cap_ratios)
            ),
            "persistent_sepcmaes_axis_cap_ratio_max": float(
                np.max(effective_axis_cap_ratios)
            ),
        }
        for name, value in counter_totals.items():
            summary[
                f"persistent_sepcmaes_numeric_{name}"
            ] = float(value)
        return summary

    def _build_target_block_dual_clock_plan(
        self,
        *,
        base_states: np.ndarray,
        opt_actions: np.ndarray,
        cfg_actions_block: np.ndarray,
        res_actions: np.ndarray,
    ):
        requested = np.zeros((self.n_agents,), dtype=np.int64)
        populations = np.zeros((self.n_agents,), dtype=np.int64)
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents,
            self.D,
        )
        for i in range(self.n_agents):
            optimizer_name = str(
                self.optimizer_candidates[int(opt_actions[i])]
            ).lower()
            if optimizer_name != "sepcmaes":
                raise ValueError(
                    "Target-block dual-clock runtime actions must select "
                    "SepCMAES for every agent."
                )
            requested[i] = max(
                1,
                int(
                    round(
                        self.subfes_per_agent
                        * float(
                            self.resource_factors[int(res_actions[i])]
                        )
                    )
                ),
            )
            options = self._build_optimizer_options(
                agent_id=i,
                optimizer_name=optimizer_name,
                cfg_levels=cfg_actions_block[i].tolist(),
                dims=self.full_dims,
                x_base=bases[i],
                subfes_i=int(requested[i]),
                seed=int(
                    self.opts.seed + self.step_count * 1000 + i
                ),
            )
            populations[i] = int(
                options.get("n_individuals", 0)
            )
        return build_target_block_dual_clock_plan(
            requested_fes_per_agent=requested,
            population_per_agent=populations,
        )

    def _apply_target_block_commit_credit(
        self,
        *,
        previous_states: np.ndarray,
        proposal_states: np.ndarray,
        committed_states: np.ndarray,
    ) -> None:
        """Reconcile each detached local optimizer with its actual commit."""
        if self.target_block_commit_credit_mode == "off":
            return
        previous = np.asarray(
            previous_states, dtype=np.float64
        ).reshape(self.n_agents, self.D)
        proposals = np.asarray(
            proposal_states, dtype=np.float64
        ).reshape(self.n_agents, self.D)
        committed = np.asarray(
            committed_states, dtype=np.float64
        ).reshape(self.n_agents, self.D)

        for agent_id in range(self.n_agents):
            state = self.persistent_sepcmaes_states[agent_id]
            signature = self.persistent_sepcmaes_signatures[agent_id]
            if state is None or signature is None:
                raise RuntimeError(
                    "Target-block commit credit requires an active persistent "
                    f"SepCMAES state for agent {agent_id}."
                )
            reconciled, telemetry = (
                reconcile_persistent_sepcmaes_commit(
                    state,
                    previous_center=previous[agent_id],
                    proposal=proposals[agent_id],
                    committed=committed[agent_id],
                    mode=self.target_block_commit_credit_mode,
                    # The main process owns this detached post-worker state.
                    # Avoid copying its growing lifetime histories again at
                    # every fast commit.
                    copy_state=False,
                )
            )
            self.persistent_sepcmaes_states[agent_id] = reconciled
            self.last_target_block_commit_credit_active[agent_id] = float(
                bool(telemetry["active"])
            )
            self.last_target_block_commit_credit[agent_id] = float(
                telemetry["credit"]
            )
            self.last_target_block_commit_credit_cosine[agent_id] = float(
                telemetry["cosine"]
            )
            self.last_target_block_commit_credit_proposal_norm[
                agent_id
            ] = float(telemetry["proposal_norm"])
            self.last_target_block_commit_credit_commit_norm[
                agent_id
            ] = float(telemetry["commit_norm"])
            self.last_target_block_commit_credit_correction_norm[
                agent_id
            ] = float(telemetry["correction_norm"])
            self.last_target_block_commit_credit_path_retention[
                agent_id
            ] = float(telemetry["path_retention"])
            self.last_target_block_commit_credit_scale_retention[
                agent_id
            ] = float(telemetry["scale_retention"])
            self.last_target_block_commit_credit_axis_rms_before[
                agent_id
            ] = float(telemetry["effective_axis_rms_before"])
            self.last_target_block_commit_credit_axis_rms_after[
                agent_id
            ] = float(telemetry["effective_axis_rms_after"])
            self.last_target_block_commit_credit_sigma_path_before[
                agent_id
            ] = float(telemetry["sigma_path_norm_before"])
            self.last_target_block_commit_credit_sigma_path_after[
                agent_id
            ] = float(telemetry["sigma_path_norm_after"])
            self.last_target_block_commit_credit_cov_path_before[
                agent_id
            ] = float(telemetry["covariance_path_norm_before"])
            self.last_target_block_commit_credit_cov_path_after[
                agent_id
            ] = float(telemetry["covariance_path_norm_after"])
        self.target_block_commit_credit_events += int(self.n_agents)

    def _run_target_block_dual_clock_event(
        self,
        *,
        base_states: np.ndarray,
        local_reference: np.ndarray,
        opt_actions: np.ndarray,
        cfg_actions_block: np.ndarray,
        res_actions: np.ndarray,
        collab_actions: np.ndarray,
        guide_scale_actions: np.ndarray,
    ) -> Dict:
        if not self.target_block_dual_clock_enable:
            raise RuntimeError(
                "Target-block dual-clock event called while disabled."
            )
        plan = self._build_target_block_dual_clock_plan(
            base_states=base_states,
            opt_actions=opt_actions,
            cfg_actions_block=cfg_actions_block,
            res_actions=res_actions,
        )
        current_bases = np.asarray(
            base_states,
            dtype=np.float64,
        ).reshape(self.n_agents, self.D).copy()
        reference = np.asarray(
            local_reference,
            dtype=np.float64,
        ).reshape(self.n_agents)
        total_evals_per_agent = np.zeros(
            (self.n_agents,),
            dtype=np.int64,
        )
        total_validation_evals = 0
        total_shape_fallbacks = 0
        total_cmaes_fail_soft_mask = np.zeros(
            (self.n_agents,), dtype=np.float64
        )
        last_cmaes_fail_soft_generations = np.full(
            (self.n_agents,), -1, dtype=np.int64
        )
        last_cmaes_fail_soft_evaluations = np.zeros(
            (self.n_agents,), dtype=np.int64
        )
        last_cmaes_fail_soft_reasons = ["" for _ in range(self.n_agents)]
        field_reported_evals = 0
        final_batch = None
        for _ in range(plan.microcycles):
            batch = self._run_objective_split_optimizer_batch(
                base_states=current_bases,
                local_reference=reference,
                opt_actions=opt_actions,
                cfg_actions_block=cfg_actions_block,
                res_actions=res_actions,
                collab_actions=collab_actions,
                guide_scale_actions=guide_scale_actions,
                subfes_overrides=plan.generation_fes_per_agent,
                persistent_recenter_max_shift_override=(
                    # Under the opt-in commit-lock mechanism, a cooperative
                    # commit is the next optimizer generation's state
                    # transition rather than a weak external hint. Move the
                    # persistent mean to that state exactly while preserving
                    # covariance, sigma, and evolution paths.
                    float(
                        np.linalg.norm(
                            np.broadcast_to(
                                np.asarray(
                                    self.ub,
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    self.lb,
                                    dtype=np.float64,
                                ),
                                (self.D,),
                            )
                        )
                    )
                    if self.target_block_dual_clock_commit_lock_enable
                    else None
                ),
            )
            if not np.array_equal(
                batch["evals_per_agent"],
                plan.generation_fes_per_agent,
            ):
                raise RuntimeError(
                    "Dual-clock optimizer tick did not consume exactly one "
                    "complete generation per agent."
                )
            total_evals_per_agent += batch["evals_per_agent"]
            total_validation_evals += int(
                batch["candidate_validation_evals"]
            )
            total_shape_fallbacks += int(
                batch["candidate_shape_fallbacks"]
            )
            batch_fail_mask = np.asarray(
                batch["cmaes_numeric_fail_soft_mask"],
                dtype=np.float64,
            ).reshape(self.n_agents)
            triggered_ids = np.flatnonzero(batch_fail_mask > 0.5)
            total_cmaes_fail_soft_mask = np.maximum(
                total_cmaes_fail_soft_mask, batch_fail_mask
            )
            for agent_id in triggered_ids:
                last_cmaes_fail_soft_generations[agent_id] = int(
                    batch["cmaes_numeric_fail_soft_generations"][agent_id]
                )
                last_cmaes_fail_soft_evaluations[agent_id] = int(
                    batch["cmaes_numeric_fail_soft_evaluations"][agent_id]
                )
                last_cmaes_fail_soft_reasons[agent_id] = str(
                    batch["cmaes_numeric_fail_soft_reasons"][agent_id]
                )
            progress_fes = int(
                self.sum_fes
                + np.sum(total_evals_per_agent)
                + field_reported_evals
            )
            previous_bases = current_bases.copy()
            current_bases, field_evals = (
                self._run_target_block_field_event(
                    base_states=previous_bases,
                    proposal_states=batch["proposal_states"],
                    reported_fes_before_event=progress_fes,
                )
            )
            self._apply_target_block_commit_credit(
                previous_states=previous_bases,
                proposal_states=batch["proposal_states"],
                committed_states=current_bases,
            )
            field_reported_evals += int(field_evals)
            final_batch = batch

        if final_batch is None:
            raise RuntimeError(
                "Dual-clock plan produced no fast microcycles."
            )
        if not np.array_equal(
            total_evals_per_agent,
            plan.requested_fes_per_agent,
        ):
            raise RuntimeError(
                "Dual-clock total local-search FEs differ from the held slow "
                "action budget."
            )
        final_local_improvements = np.asarray(
            [
                _safe_log_improvement(
                    float(reference[i]),
                    float(final_batch["proposal_local_values"][i]),
                )
                for i in range(self.n_agents)
            ],
            dtype=np.float64,
        )
        self.target_block_dual_clock_outer_events += 1
        self.target_block_dual_clock_local_generation_ticks += int(
            plan.microcycles
        )
        self.target_block_dual_clock_communication_ticks += int(
            plan.microcycles
        )
        self.target_block_dual_clock_commit_ticks += int(
            plan.microcycles
        )
        self.last_target_block_dual_clock_microcycles = int(
            plan.microcycles
        )
        self.last_target_block_dual_clock_generation_fes = (
            plan.generation_fes_per_agent.copy()
        )
        return {
            "proposal_states": final_batch["proposal_states"],
            "proposal_local_values": final_batch[
                "proposal_local_values"
            ],
            "local_improvements": final_local_improvements,
            "evals_per_agent": total_evals_per_agent,
            "total_evals": int(np.sum(total_evals_per_agent)),
            "candidate_validation_evals": int(
                total_validation_evals
            ),
            "candidate_shape_fallbacks": int(
                total_shape_fallbacks
            ),
            "cmaes_numeric_fail_soft_mask": total_cmaes_fail_soft_mask,
            "cmaes_numeric_fail_soft_generations": (
                last_cmaes_fail_soft_generations
            ),
            "cmaes_numeric_fail_soft_evaluations": (
                last_cmaes_fail_soft_evaluations
            ),
            "cmaes_numeric_fail_soft_reasons": list(
                last_cmaes_fail_soft_reasons
            ),
            "next_agent_x": current_bases,
            "target_block_field_reported_evals": int(
                field_reported_evals
            ),
            "microcycles": int(plan.microcycles),
        }

    def _run_target_block_field_event(
        self,
        *,
        base_states: np.ndarray,
        proposal_states: np.ndarray,
        reported_fes_before_event: int = None,
    ) -> Tuple[np.ndarray, int]:
        if not self.target_block_field_enable:
            raise RuntimeError(
                "Target-block field event called while disabled."
            )
        if self.consensus_weight is None or self.graph is None:
            raise RuntimeError(
                "Target-block field requires a resolved communication graph."
            )
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        proposals = np.asarray(
            proposal_states, dtype=np.float64
        ).reshape(self.n_agents, self.D)
        base_residuals = np.empty(
            (self.n_agents, self.committee_target_num),
            dtype=np.float64,
        )
        proposal_residuals = np.empty_like(base_residuals)
        for agent_id in range(self.n_agents):
            residual_pair = np.asarray(
                self.fun.local_target_residual_batch(
                    agent_id,
                    np.stack(
                        [bases[agent_id], proposals[agent_id]],
                        axis=0,
                    ),
                ),
                dtype=np.float64,
            ).reshape(2, self.committee_target_num)
            base_residuals[agent_id] = residual_pair[0]
            proposal_residuals[agent_id] = residual_pair[1]

        # This is one CCSA-like graph message, not a second consensus round:
        # x, own-local secant, and previous cooperative path are transmitted
        # together.  Coordinate mixing and path diffusion therefore use the
        # same one-hop W event.
        commit_bases = self.consensus_weight @ proposals
        progress = float(
            np.clip(
                (
                    self.sum_fes
                    if reported_fes_before_event is None
                    else int(reported_fes_before_event)
                )
                / float(max(1, self.max_fes)),
                0.0,
                1.0,
            )
        )
        result = build_target_block_cooperative_field(
            base_states=bases,
            proposal_states=proposals,
            commit_base_states=commit_bases,
            base_residuals=base_residuals,
            proposal_residuals=proposal_residuals,
            previous_path=self.target_block_field_path,
            previous_radius=self.target_block_field_radius,
            previous_initialized=self.target_block_field_initialized,
            weight=self.consensus_weight,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            budget_progress=progress,
            path_decay=self.target_block_field_path_decay,
            step_rate=self.target_block_field_step_rate,
            radius_min_ratio=self.target_block_field_radius_min_ratio,
            radius_max_ratio=self.target_block_field_radius_max_ratio,
            dormancy_recovery_enable=(
                self.target_block_dormancy_recovery_enable
            ),
            previous_recovery_residual_reference=(
                self.target_block_dormancy_residual_reference
            ),
            previous_recovery_reserve_radius=(
                self.target_block_dormancy_reserve_radius
            ),
            previous_recovery_initialized=(
                self.target_block_dormancy_initialized
            ),
            previous_recovery_floor_age=(
                self.target_block_dormancy_floor_age
            ),
            previous_recovery_stagnation_age=(
                self.target_block_dormancy_stagnation_age
            ),
            previous_recovery_cooldown=(
                self.target_block_dormancy_cooldown
            ),
            previous_recovery_activation_count=(
                self.target_block_dormancy_activation_count
            ),
        )
        self.target_block_field_path = result.path.copy()
        self.target_block_field_radius = result.radius.copy()
        self.target_block_field_initialized = (
            result.path_initialized.copy()
        )
        self.last_target_block_field_secant = result.secant.copy()
        self.last_target_block_field_secant_valid = (
            result.secant_valid.copy()
        )
        self.last_target_block_field_secant_alignment = (
            result.secant_field_alignment.copy()
        )
        self.last_target_block_field_direction = (
            result.field_direction.copy()
        )
        self.last_target_block_field_source_diversity = (
            result.source_diversity.copy()
        )
        self.last_target_block_field_path_norm = result.path_norm.copy()
        self.last_target_block_field_path_alignment = (
            result.path_alignment.copy()
        )
        self.last_target_block_field_conflict = result.conflict.copy()
        self.last_target_block_field_radius_expand = (
            result.radius_expand.copy()
        )
        self.last_target_block_field_radius_shrink = (
            result.radius_shrink.copy()
        )
        self.last_target_block_field_radius_clip_min = (
            result.radius_clip_min.copy()
        )
        self.last_target_block_field_radius_clip_max = (
            result.radius_clip_max.copy()
        )
        self.last_target_block_field_requested_commit_norm = (
            result.requested_commit_norm.copy()
        )
        self.last_target_block_field_applied_commit_norm = (
            result.applied_commit_norm.copy()
        )
        self.last_target_block_field_boundary_clipped = (
            result.boundary_clipped.copy()
        )
        self.last_target_block_field_path_injection = float(
            result.path_injection
        )
        self.last_target_block_field_path_angle_degrees = float(
            result.path_angle_degrees
        )
        recovery = result.dormancy_recovery
        if self.target_block_dormancy_recovery_enable:
            if recovery is None:
                raise RuntimeError(
                    "Enabled target-block dormancy recovery returned no "
                    "state."
                )
            self.target_block_dormancy_residual_reference = (
                recovery.residual_reference.copy()
            )
            self.target_block_dormancy_reserve_radius = (
                recovery.reserve_radius.copy()
            )
            self.target_block_dormancy_initialized = (
                recovery.initialized.copy()
            )
            self.target_block_dormancy_floor_age = (
                recovery.floor_age.copy()
            )
            self.target_block_dormancy_stagnation_age = (
                recovery.stagnation_age.copy()
            )
            self.target_block_dormancy_cooldown = (
                recovery.cooldown.copy()
            )
            self.target_block_dormancy_activation_count = (
                recovery.activation_count.copy()
            )
            self.last_target_block_dormancy_active = (
                recovery.active.copy()
            )
            self.last_target_block_dormancy_unresolved = (
                recovery.unresolved.copy()
            )
            self.last_target_block_dormancy_material_progress = (
                recovery.material_progress.copy()
            )
            self.last_target_block_dormancy_reliable_scale = (
                recovery.reliable_scale.copy()
            )
            self.last_target_block_dormancy_residual_ratio = (
                recovery.residual_ratio.copy()
            )
            self.last_target_block_dormancy_progress_ratio = (
                recovery.progress_ratio.copy()
            )
            self.last_target_block_dormancy_disagreement = (
                recovery.disagreement.copy()
            )
            self.last_target_block_dormancy_disagreement_ratio = (
                recovery.disagreement_ratio.copy()
            )
            self.last_target_block_dormancy_restore_radius = (
                recovery.restore_radius.copy()
            )
            self.target_block_dormancy_events += 1
            self.target_block_dormancy_activations += int(
                np.sum(recovery.active)
            )
        elif recovery is not None:
            raise RuntimeError(
                "Disabled target-block dormancy recovery returned active "
                "state."
            )

        shadow_communicated = False
        if self.target_block_direction_shadow_enable:
            if recovery is None:
                raise RuntimeError(
                    "Direction shadow requires current dormancy evidence."
                )
            _, shadow_communicated = (
                self._run_target_block_direction_shadow(
                    commit_base_states=commit_bases,
                    proposal_states=proposals,
                    path=result.path,
                    probe_radius=recovery.restore_radius,
                    eligible=recovery.active,
                    residual_ratio=recovery.residual_ratio,
                )
            )

        committed_states = result.committed_states.copy()
        challenge_reported_evals = 0
        challenge_communicated = False
        if self.target_block_challenge_response_mode != "off":
            if recovery is None:
                raise RuntimeError(
                    "Challenge-response requires current dormancy evidence."
                )
            (
                committed_states,
                _,
                challenge_reported_evals,
                challenge_communicated,
            ) = self._run_target_block_challenge_response(
                commit_base_states=commit_bases,
                fallback_committed_states=committed_states,
                proposal_states=proposals,
                path=result.path,
                probe_radius=recovery.restore_radius,
                eligible=recovery.active,
                residual_ratio=recovery.residual_ratio,
            )

        local_evals = int(2 * self.n_agents)
        self.target_block_field_events += 1
        self.target_block_field_local_evals += local_evals
        self.target_block_field_comm_rounds += 1
        message_floats = int(
            3 * self.D + self.committee_target_num + 2
        )
        step_messages = int(self.graph.directed_edge_count)
        self.target_block_field_messages += step_messages
        self.target_block_field_transmitted_floats += int(
            step_messages * message_floats
        )
        self.last_comm_rounds_applied = 1
        self.last_comm_rounds_requested = 1
        self._record_graph_communication(
            floats_per_message=message_floats,
            rounds=1,
        )
        if shadow_communicated:
            shadow_message_floats = int(
                self.committee_target_num
                * self.committee_coordinate_dim
            )
            self._record_graph_communication(
                floats_per_message=shadow_message_floats,
                rounds=1,
            )
            self.last_comm_rounds_applied = 2
            self.last_comm_rounds_requested = 2
            self.step_comm_rounds_applied = 2
            self.step_comm_events = 2
        if challenge_communicated:
            source_num = len(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES)
            challenge_message_floats = int(
                2 + source_num + source_num * self.committee_coordinate_dim
            )
            response_message_floats = int(2 * source_num)
            self._record_graph_communication(
                floats_per_message=challenge_message_floats,
                rounds=1,
            )
            self._record_graph_communication(
                floats_per_message=response_message_floats,
                rounds=1,
            )
            self.last_comm_rounds_applied = 3
            self.last_comm_rounds_requested = 3
            self.step_comm_rounds_applied = 3
            self.step_comm_events = 3
        return (
            committed_states,
            int(local_evals + challenge_reported_evals),
        )

    def _run_target_block_direction_shadow(
        self,
        *,
        commit_base_states: np.ndarray,
        proposal_states: np.ndarray,
        path: np.ndarray,
        probe_radius: np.ndarray,
        eligible: np.ndarray,
        residual_ratio: np.ndarray,
    ) -> Tuple[int, bool]:
        """Probe alternative dormant-block directions without actuation."""

        plan = build_target_block_direction_shadow_plan(
            commit_base_states=commit_base_states,
            proposal_states=proposal_states,
            persistent_states=self.persistent_sepcmaes_states,
            path=path,
            probe_radius=probe_radius,
            eligible=eligible,
            residual_ratio=residual_ratio,
            weight=self.consensus_weight,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        residual_batches = []
        for agent_id, probe_states in enumerate(plan.probe_states):
            if probe_states.shape[0] == 0:
                residual_batches.append(
                    np.empty(
                        (0, self.committee_target_num),
                        dtype=np.float64,
                    )
                )
                continue
            residual_batches.append(
                np.asarray(
                    self.fun.local_target_residual_batch(
                        agent_id,
                        probe_states,
                    ),
                    dtype=np.float64,
                ).reshape(
                    probe_states.shape[0],
                    self.committee_target_num,
                )
            )
        shadow = summarize_target_block_direction_shadow(
            plan=plan,
            probe_residuals=residual_batches,
            weight=self.consensus_weight,
        )
        selected = np.zeros_like(plan.eligible, dtype=bool)
        for agent_id, block_id in enumerate(plan.selected_block):
            if int(block_id) >= 0:
                selected[agent_id, int(block_id)] = True
        self.last_target_block_direction_shadow_eligible = (
            plan.eligible.copy()
        )
        self.last_target_block_direction_shadow_selected = selected
        self.last_target_block_direction_shadow_candidate_valid = (
            plan.candidate_valid.copy()
        )
        self.last_target_block_direction_shadow_candidate_angle = (
            plan.candidate_angle_degrees.copy()
        )
        self.last_target_block_direction_shadow_candidate_response = (
            shadow.candidate_response.copy()
        )
        self.last_target_block_direction_shadow_candidate_positive = (
            shadow.candidate_positive.copy()
        )
        self.last_target_block_direction_shadow_best_response = (
            shadow.best_response.copy()
        )
        self.last_target_block_direction_shadow_best_source = (
            shadow.best_source.copy()
        )
        self.last_target_block_direction_shadow_best_sign = (
            shadow.best_sign.copy()
        )
        self.last_target_block_direction_shadow_best_direction = (
            shadow.best_direction.copy()
        )
        self.last_target_block_direction_shadow_support = (
            shadow.support.copy()
        )
        self.last_target_block_direction_shadow_conflict = (
            shadow.conflict.copy()
        )
        eligible_blocks = int(np.sum(plan.eligible))
        probed_blocks = int(np.sum(selected))
        communicated = bool(probed_blocks > 0)
        self.target_block_direction_shadow_events += 1
        self.target_block_direction_shadow_eligible_blocks += (
            eligible_blocks
        )
        self.target_block_direction_shadow_probed_blocks += probed_blocks
        self.target_block_direction_shadow_local_evals += int(
            plan.local_evals
        )
        if communicated:
            step_messages = int(self.graph.directed_edge_count)
            message_floats = int(
                self.committee_target_num
                * self.committee_coordinate_dim
            )
            self.target_block_direction_shadow_comm_rounds += 1
            self.target_block_direction_shadow_messages += step_messages
            self.target_block_direction_shadow_transmitted_floats += int(
                step_messages * message_floats
            )
        return int(plan.local_evals), communicated

    def _run_target_block_challenge_response(
        self,
        *,
        commit_base_states: np.ndarray,
        fallback_committed_states: np.ndarray,
        proposal_states: np.ndarray,
        path: np.ndarray,
        probe_radius: np.ndarray,
        eligible: np.ndarray,
        residual_ratio: np.ndarray,
    ) -> Tuple[np.ndarray, int, int, bool]:
        """Run the two one-hop block-addressed challenge/response rounds."""

        plan = build_target_block_challenge_plan(
            commit_base_states=commit_base_states,
            proposal_states=proposal_states,
            persistent_states=self.persistent_sepcmaes_states,
            path=path,
            probe_radius=probe_radius,
            eligible=eligible,
            residual_ratio=residual_ratio,
            weight=self.consensus_weight,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        residual_batches = []
        for receiver_id, probe_states in enumerate(
            plan.receiver_probe_states
        ):
            if probe_states.shape[0] == 0:
                residual_batches.append(
                    np.empty(
                        (0, self.committee_target_num),
                        dtype=np.float64,
                    )
                )
                continue
            residual_batches.append(
                np.asarray(
                    self.fun.local_target_residual_batch(
                        receiver_id, probe_states
                    ),
                    dtype=np.float64,
                ).reshape(
                    probe_states.shape[0], self.committee_target_num
                )
            )
        challenge = summarize_target_block_challenge_response(
            plan=plan,
            probe_residuals=residual_batches,
            weight=self.consensus_weight,
        )
        selected = np.zeros_like(plan.eligible, dtype=bool)
        for agent_id, block_id in enumerate(plan.selected_block):
            if int(block_id) >= 0:
                selected[agent_id, int(block_id)] = True

        self.last_target_block_challenge_selected = selected
        self.last_target_block_challenge_candidate_valid = (
            plan.candidate_valid.copy()
        )
        self.last_target_block_challenge_candidate_angle = (
            plan.candidate_angle_degrees.copy()
        )
        self.last_target_block_challenge_candidate_score = (
            challenge.candidate_score.copy()
        )
        self.last_target_block_challenge_candidate_sign = (
            challenge.candidate_sign.copy()
        )
        self.last_target_block_challenge_path_score = (
            challenge.path_score.copy()
        )
        self.last_target_block_challenge_alternative_score = (
            challenge.alternative_score.copy()
        )
        self.last_target_block_challenge_margin = (
            challenge.alternative_margin.copy()
        )
        self.last_target_block_challenge_best_source = (
            challenge.best_source.copy()
        )
        self.last_target_block_challenge_best_sign = (
            challenge.best_sign.copy()
        )
        self.last_target_block_challenge_best_direction = (
            challenge.best_direction.copy()
        )
        self.last_target_block_challenge_coverage = (
            challenge.response_coverage.copy()
        )
        self.last_target_block_challenge_positive_sources = (
            challenge.positive_sources.copy()
        )
        self.last_target_block_challenge_neighbor_positive_sources = (
            challenge.neighbor_positive_sources.copy()
        )
        self.last_target_block_challenge_support = challenge.support.copy()
        self.last_target_block_challenge_conflict = challenge.conflict.copy()
        self.last_target_block_challenge_actuation_eligible = (
            challenge.actuation_eligible.copy()
        )

        applied = np.zeros_like(selected)
        applied_norm = np.zeros_like(probe_radius, dtype=np.float64)
        boundary_clipped = np.zeros_like(selected)
        committed = np.asarray(
            fallback_committed_states, dtype=np.float64
        ).copy()
        if self.target_block_challenge_response_mode == "actuate":
            actuation = apply_target_block_challenge_actuator(
                commit_base_states=commit_base_states,
                fallback_committed_states=committed,
                probe_radius=probe_radius,
                result=challenge,
                lower_bound=self.lb,
                upper_bound=self.ub,
                target_num=self.committee_target_num,
                coordinate_dim=self.committee_coordinate_dim,
            )
            committed = actuation.committed_states.copy()
            applied = actuation.applied.copy()
            applied_norm = actuation.applied_norm.copy()
            boundary_clipped = actuation.boundary_clipped.copy()
        self.last_target_block_challenge_applied = applied
        self.last_target_block_challenge_applied_norm = applied_norm
        self.last_target_block_challenge_boundary_clipped = boundary_clipped

        communicated = bool(plan.challenges > 0)
        reported_evals = (
            int(plan.local_evals)
            if self.target_block_challenge_response_mode == "actuate"
            else 0
        )
        self.target_block_challenge_events += 1
        self.target_block_challenge_challenges += int(plan.challenges)
        self.target_block_challenge_directed_responses += int(
            plan.directed_responses
        )
        self.target_block_challenge_local_evals += int(plan.local_evals)
        self.target_block_challenge_reported_evals += reported_evals
        applied_count = int(np.sum(applied))
        self.target_block_challenge_applied_commits += applied_count
        if communicated:
            source_num = len(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES)
            step_messages = int(self.graph.directed_edge_count)
            message_floats = int(
                2
                + source_num
                + source_num * self.committee_coordinate_dim
                + 2 * source_num
            )
            self.target_block_challenge_comm_rounds += 2
            self.target_block_challenge_messages += 2 * step_messages
            self.target_block_challenge_transmitted_floats += int(
                step_messages * message_floats
            )
        return committed, int(plan.local_evals), reported_evals, communicated

    @staticmethod
    def _rvcpd_clip_row(vector: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        value = np.nan_to_num(
            np.asarray(vector, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        norm = float(np.linalg.norm(value))
        if norm <= 1e-15:
            return np.zeros_like(value)
        if norm > float(max_norm):
            value = value * (float(max_norm) / norm)
        return value

    def _run_rvcpd_event(
        self,
        base_states: np.ndarray,
        proposal_states: np.ndarray,
        proposal_local_f: np.ndarray,
        *,
        commit_base_states: np.ndarray = None,
        commit_base_local_f: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        proposals = np.asarray(
            proposal_states, dtype=np.float64
        ).reshape(self.n_agents, self.D)
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        commit_bases = (
            proposals
            if commit_base_states is None
            else np.asarray(
                commit_base_states,
                dtype=np.float64,
            ).reshape(self.n_agents, self.D)
        )
        base_local = np.asarray(
            proposal_local_f
            if commit_base_local_f is None
            else commit_base_local_f,
            dtype=np.float64,
        ).reshape(self.n_agents)
        if not self.rvcpd_enabled:
            return proposals.copy(), base_local.copy(), 0
        if not self.local_only_information:
            raise RuntimeError("RVCPD crossed the strict local-only boundary.")
        if self.consensus_adjacency is None or self.consensus_weight is None:
            raise RuntimeError("RVCPD requires a resolved communication graph.")

        moves = proposals - bases
        move_norms = np.linalg.norm(moves, axis=1)
        valid_sender = np.isfinite(move_norms) & (move_norms > 1e-15)
        sender_units = np.zeros_like(moves)
        sender_units[valid_sender] = (
            moves[valid_sender] / move_norms[valid_sender, None]
        )
        adjacency = np.asarray(self.consensus_adjacency, dtype=bool).copy()
        np.fill_diagonal(adjacency, False)
        weights = np.asarray(self.consensus_weight, dtype=np.float64).copy()
        np.fill_diagonal(weights, 0.0)

        next_states = commit_bases.copy()
        next_local = base_local.copy()
        probe_evals = 0
        step_span = float(
            max(1e-12, np.sqrt(self.D) * max(1e-12, self.ub - self.lb))
        )
        min_radius = float(self.rvcpd_probe_min_ratio * step_span)
        max_radius = float(self.rvcpd_probe_max_ratio * step_span)

        self.last_rvcpd_direction.fill(0.0)
        self.last_rvcpd_sign.fill(0)
        self.last_rvcpd_active.fill(0.0)
        self.last_rvcpd_agreement.fill(0.0)
        self.last_rvcpd_requested_radius.fill(0.0)
        self.last_rvcpd_applied_radius.fill(0.0)
        self.last_rvcpd_plus_gain.fill(0.0)
        self.last_rvcpd_minus_gain.fill(0.0)
        self.rvcpd_events += 1
        if self.rvcpd_arm == "p1":
            # P1 is the memoryless causal arm: every event starts from a
            # fixed scale and carries no path, trust, evidence, or age across
            # rollout steps. Current-event evidence is still exposed below
            # for diagnostics, but it cannot affect a later event.
            self.rvcpd_path.fill(0.0)
            self.rvcpd_support_ema.fill(0.0)
            self.rvcpd_conflict_ema.fill(0.0)
            self.rvcpd_uncertainty_ema.fill(0.0)
            self.rvcpd_trust.fill(1.0)
            self.rvcpd_scale.fill(self.rvcpd_initial_scale)
            self.rvcpd_age.fill(0)

        def _pay_reserved_noop_probes(receiver_id: int) -> None:
            nonlocal probe_evals
            # The three arms must keep identical reported/physical probe
            # budgets even after their trajectories diverge. When topology
            # offers a communication opportunity but the current directions
            # are invalid or cancel, reserve the two receiver-local calls at
            # the proposal itself and keep the event an explicit no-op.
            for _ in range(2):
                self.fun.local_eval_batch(
                    int(receiver_id),
                    commit_bases[int(receiver_id)],
                )
            probe_evals += 2
            self.rvcpd_noop_count += 1

        for receiver in range(self.n_agents):
            topology_neighbors = np.flatnonzero(adjacency[receiver])
            neighbor_mask = adjacency[receiver] & valid_sender
            neighbor_ids = np.flatnonzero(neighbor_mask)
            if neighbor_ids.size == 0:
                if topology_neighbors.size > 0:
                    _pay_reserved_noop_probes(receiver)
                if self.rvcpd_arm == "p1":
                    self.rvcpd_age[receiver] = 1
                    self.rvcpd_path[receiver].fill(0.0)
                    self.rvcpd_trust[receiver] = 1.0
                    self.rvcpd_scale[receiver] = self.rvcpd_initial_scale
                else:
                    self.rvcpd_age[receiver] += 1
                    self.rvcpd_path[receiver] *= self.rvcpd_noop_path_decay
                    self.rvcpd_scale[receiver] = max(
                        self.rvcpd_scale_min,
                        self.rvcpd_scale[receiver] * self.rvcpd_scale_decay,
                    )
                continue

            row_weights = np.maximum(0.0, weights[receiver, neighbor_ids])
            if float(np.sum(row_weights)) <= 1e-15:
                row_weights = np.ones(
                    (neighbor_ids.size,), dtype=np.float64
                )
            row_weights = row_weights / float(np.sum(row_weights))
            merged = np.sum(
                row_weights[:, None] * sender_units[neighbor_ids], axis=0
            )
            agreement = float(np.clip(np.linalg.norm(merged), 0.0, 1.0))
            if not np.isfinite(agreement) or agreement <= 1e-12:
                _pay_reserved_noop_probes(receiver)
                if self.rvcpd_arm == "p1":
                    self.rvcpd_age[receiver] = 1
                    self.rvcpd_conflict_ema[receiver] = 1.0
                    self.rvcpd_uncertainty_ema[receiver] = 1.0
                    self.rvcpd_path[receiver].fill(0.0)
                    self.rvcpd_trust[receiver] = 1.0
                    self.rvcpd_scale[receiver] = self.rvcpd_initial_scale
                else:
                    self.rvcpd_age[receiver] += 1
                    self.rvcpd_conflict_ema[receiver] = (
                        self.rvcpd_ema_decay
                        * self.rvcpd_conflict_ema[receiver]
                        + (1.0 - self.rvcpd_ema_decay)
                    )
                    self.rvcpd_uncertainty_ema[receiver] = (
                        self.rvcpd_ema_decay
                        * self.rvcpd_uncertainty_ema[receiver]
                        + (1.0 - self.rvcpd_ema_decay)
                    )
                    self.rvcpd_path[receiver] *= self.rvcpd_noop_path_decay
                    self.rvcpd_scale[receiver] = max(
                        self.rvcpd_scale_min,
                        self.rvcpd_scale[receiver] * self.rvcpd_scale_decay,
                    )
                continue

            message_direction = merged / agreement
            if self.rvcpd_arm == "p1":
                tentative = message_direction
            else:
                tentative = (
                    self.rvcpd_path_decay * self.rvcpd_path[receiver]
                    + self.rvcpd_direction_lr * message_direction
                )
            tentative_norm = float(np.linalg.norm(tentative))
            if not np.isfinite(tentative_norm) or tentative_norm <= 1e-12:
                _pay_reserved_noop_probes(receiver)
                if self.rvcpd_arm == "p1":
                    self.rvcpd_age[receiver] = 1
                    self.rvcpd_path[receiver].fill(0.0)
                    self.rvcpd_trust[receiver] = 1.0
                    self.rvcpd_scale[receiver] = self.rvcpd_initial_scale
                else:
                    self.rvcpd_age[receiver] += 1
                    self.rvcpd_path[receiver] *= self.rvcpd_noop_path_decay
                    self.rvcpd_scale[receiver] = max(
                        self.rvcpd_scale_min,
                        self.rvcpd_scale[receiver] * self.rvcpd_scale_decay,
                    )
                continue

            direction = tentative / tentative_norm
            neighbor_step = float(
                np.sum(row_weights * move_norms[neighbor_ids])
            )
            geometric_step = max(
                float(move_norms[receiver]),
                neighbor_step,
                min_radius,
            )
            requested_radius = float(
                np.clip(
                    self.rvcpd_scale[receiver]
                    * self.rvcpd_trust[receiver]
                    * geometric_step,
                    min_radius,
                    max_radius,
                )
            )
            plus_x = np.clip(
                commit_bases[receiver] + requested_radius * direction,
                self.lb,
                self.ub,
            )
            minus_x = np.clip(
                commit_bases[receiver] - requested_radius * direction,
                self.lb,
                self.ub,
            )
            plus_y = float(
                np.asarray(
                    self.fun.local_eval_batch(receiver, plus_x),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            minus_y = float(
                np.asarray(
                    self.fun.local_eval_batch(receiver, minus_x),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            probe_evals += 2
            plus_gain = _safe_log_improvement(
                float(base_local[receiver]), plus_y
            )
            minus_gain = _safe_log_improvement(
                float(base_local[receiver]), minus_y
            )
            sign = 0
            tie_tolerance = 1e-12
            if (
                plus_gain > self.rvcpd_min_log_improve
                and plus_gain > minus_gain + tie_tolerance
            ):
                sign = 1
            elif (
                minus_gain > self.rvcpd_min_log_improve
                and minus_gain > plus_gain + tie_tolerance
            ):
                sign = -1

            plus_shift = float(
                np.linalg.norm(plus_x - commit_bases[receiver])
            )
            minus_shift = float(
                np.linalg.norm(minus_x - commit_bases[receiver])
            )
            selected_shift = plus_shift if sign > 0 else (
                minus_shift if sign < 0 else 0.0
            )
            clipping_uncertainty = float(
                np.clip(
                    1.0
                    - max(plus_shift, minus_shift)
                    / max(requested_radius, 1e-15),
                    0.0,
                    1.0,
                )
            )
            ambiguous = float(
                abs(plus_y - minus_y)
                <= 1e-12 * max(1.0, abs(base_local[receiver]))
            )
            receiver_conflict = 1.0 if sign == 0 else (
                0.5 if sign < 0 else 0.0
            )
            conflict = float(max(1.0 - agreement, receiver_conflict))
            uncertainty = float(
                max(1.0 - agreement, clipping_uncertainty, ambiguous)
            )
            support = float(max(0.0, plus_gain if sign > 0 else minus_gain))
            if self.rvcpd_arm == "p1":
                self.rvcpd_support_ema[receiver] = min(1.0, support)
                self.rvcpd_conflict_ema[receiver] = conflict
                self.rvcpd_uncertainty_ema[receiver] = uncertainty
                self.rvcpd_trust[receiver] = 1.0
            else:
                ema = self.rvcpd_ema_decay
                self.rvcpd_support_ema[receiver] = (
                    ema * self.rvcpd_support_ema[receiver]
                    + (1.0 - ema) * min(1.0, support)
                )
                self.rvcpd_conflict_ema[receiver] = (
                    ema * self.rvcpd_conflict_ema[receiver]
                    + (1.0 - ema) * conflict
                )
                self.rvcpd_uncertainty_ema[receiver] = (
                    ema * self.rvcpd_uncertainty_ema[receiver]
                    + (1.0 - ema) * uncertainty
                )
                trust_logit = float(
                    2.0 * self.rvcpd_support_ema[receiver]
                    - 2.0 * self.rvcpd_conflict_ema[receiver]
                    - self.rvcpd_uncertainty_ema[receiver]
                )
                self.rvcpd_trust[receiver] = float(
                    1.0
                    / (1.0 + np.exp(-np.clip(trust_logit, -20.0, 20.0)))
                )

            if self.rvcpd_arm == "p1":
                self.rvcpd_path[receiver].fill(0.0)
                self.rvcpd_scale[receiver] = self.rvcpd_initial_scale
                self.rvcpd_age[receiver] = 1 if sign == 0 else 0
            else:
                if sign > 0:
                    self.rvcpd_path[receiver] = self._rvcpd_clip_row(tentative)
                elif sign < 0:
                    self.rvcpd_path[receiver] = self._rvcpd_clip_row(-tentative)
                else:
                    self.rvcpd_path[receiver] *= self.rvcpd_noop_path_decay

                if sign > 0:
                    self.rvcpd_scale[receiver] = min(
                        self.rvcpd_scale_max,
                        self.rvcpd_scale[receiver] * self.rvcpd_scale_growth,
                    )
                else:
                    self.rvcpd_scale[receiver] = max(
                        self.rvcpd_scale_min,
                        self.rvcpd_scale[receiver] * self.rvcpd_scale_decay,
                    )
                if sign == 0:
                    self.rvcpd_age[receiver] += 1
                else:
                    self.rvcpd_age[receiver] = 0

            self.last_rvcpd_direction[receiver] = direction
            self.last_rvcpd_sign[receiver] = sign
            self.last_rvcpd_active[receiver] = 1.0
            self.last_rvcpd_agreement[receiver] = agreement
            self.last_rvcpd_requested_radius[receiver] = requested_radius
            self.last_rvcpd_applied_radius[receiver] = selected_shift
            self.last_rvcpd_plus_gain[receiver] = plus_gain
            self.last_rvcpd_minus_gain[receiver] = minus_gain
            self.rvcpd_valid_event_count += 1
            if self.rvcpd_arm != "p0" and sign != 0:
                if sign > 0:
                    next_states[receiver] = plus_x
                    next_local[receiver] = plus_y
                else:
                    next_states[receiver] = minus_x
                    next_local[receiver] = minus_y
                    self.rvcpd_reverse_count += 1
                self.rvcpd_commit_count += 1
            else:
                self.rvcpd_noop_count += 1

        # Every directed topology edge carries the fixed-schema message,
        # including valid=0 when a sender produced no usable direction.
        # Counting only valid directions would make communication cost depend
        # on arm-diverged trajectories and break the P0/P1/P2 cost contract.
        step_messages = int(np.count_nonzero(adjacency))
        self.rvcpd_probe_local_evals += int(probe_evals)
        self.rvcpd_messages += step_messages
        self.rvcpd_transmitted_floats += int(
            step_messages * (self.D + 3)
        )
        return next_states, next_local, int(probe_evals)

    def _set_report_monitor_state(self, x: np.ndarray, value: float) -> None:
        report_x = np.asarray(x, dtype=np.float64).reshape(self.D)
        report_f = float(value)
        self.report_current_x = report_x.copy()
        self.report_current_f = report_f
        if np.isfinite(report_f) and (
            not np.isfinite(self.report_best_f) or report_f <= self.report_best_f
        ):
            self.report_best_f = report_f
            self.report_best_x = report_x.copy()

        # Compatibility aliases consumed by existing evaluation/report code.
        # local_only control code must not read these fields.
        self.current_x = self.report_current_x.copy()
        self.current_f = float(self.report_current_f)
        self.gbest_x = self.report_best_x.copy()
        self.gbest_f = float(self.report_best_f)

    def _run_detached_global_monitor(
        self,
        x: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        if not self.global_monitor_enable:
            self._set_report_monitor_state(x, float("nan"))
            return float("nan"), np.empty((0,), dtype=np.float64)
        value, local_vals = self._eval_global_and_all_local(x)
        self.global_monitor_local_evals += int(local_vals.size)
        self.global_monitor_rounds += 1
        self._set_report_monitor_state(x, value)
        return float(value), local_vals

    def _own_stagnation_ratio(self) -> np.ndarray:
        cur = np.asarray(self.agent_local_f, dtype=np.float64).reshape(self.n_agents)
        init = np.asarray(
            self.agent_initial_local_f,
            dtype=np.float64,
        ).reshape(self.n_agents)
        best = np.asarray(
            self.agent_best_local_f,
            dtype=np.float64,
        ).reshape(self.n_agents)
        denom_best = np.abs(init - best)
        denom_init = np.maximum(1.0, np.abs(init))
        denom = np.where(denom_best > 1e-12, denom_best, denom_init)
        raw = np.maximum(0.0, (cur - best) / (denom + 1e-12))
        bounded = raw / (1.0 + raw)
        return np.clip(
            np.nan_to_num(bounded, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

    def _own_best_gain(self) -> np.ndarray:
        values = np.asarray(
            [
                _safe_log_improvement(float(init), float(best))
                for init, best in zip(
                    np.asarray(self.agent_initial_local_f, dtype=np.float64),
                    np.asarray(self.agent_best_local_f, dtype=np.float64),
                )
            ],
            dtype=np.float64,
        )
        return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)

    def _record_state_communication_event(
        self,
        communication_applied: bool = False,
    ) -> None:
        if not self.state_comm_enabled:
            return
        if self.state_comm_cost_mode == "legacy_untracked":
            return
        # Keep the algorithmic state-message round visible even when detailed
        # communication-cost recording is disabled, matching graph_comm_rounds.
        self.state_comm_rounds += 1
        if not self.record_comm_cost:
            return
        if self.state_comm_mode == "graph_mean" and self.graph is not None:
            messages = int(self.graph.directed_edge_count)
        else:
            messages = int(self.n_agents * max(0, self.n_agents - 1))
        floats = int(messages * max(1, self.state_msg_dim))
        self.state_comm_messages += messages
        self.state_comm_transmitted_floats += floats
        if (
            self.state_comm_cost_mode == "piggyback"
            and bool(communication_applied)
            and self.graph is not None
        ):
            self.graph_transmitted_floats += floats

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
            "optimizer_guide_norm_mean": 0.0,
            "optimizer_guide_norm_max": 0.0,
            "optimizer_guide_applied_ratio": 0.0,
            "optimizer_guide_alignment_mean": 0.0,
            "optimizer_guide_source_active_ratio": 0.0,
            "optimizer_guide_strength_effective": 0.0,
            "optimizer_guide_strength_scale": 1.0,
            "optimizer_guide_schedule_metric": 0.0,
            "optimizer_guide_internal_active_ratio": 0.0,
            "optimizer_guide_internal_mean_step_norm_mean": 0.0,
            "optimizer_guide_internal_mean_step_norm_max": 0.0,
            "optimizer_guide_internal_alignment_mean": 0.0,
            "anchor_enabled": 0.0,
            "anchor_applied_ratio": 0.0,
            "anchor_dist_mean": 0.0,
            "anchor_dist_max": 0.0,
            "anchor_direction_norm_mean": 0.0,
            "anchor_direction_norm_max": 0.0,
            "optimizer_anchor_applied_ratio": 0.0,
            "optimizer_anchor_mean_step_norm_mean": 0.0,
            "optimizer_anchor_mean_step_norm_max": 0.0,
            "optimizer_anchor_sample_applied_ratio": 0.0,
            "collab_mode_idx_mean": 0.0,
            "guide_scale_idx_mean": 0.0,
            "guide_scale_value_mean": 1.0,
            "collab_leader_active_ratio": 0.0,
            "collab_strength_multiplier_mean": 1.0,
        }

    @staticmethod
    def _empty_candidate_response_metrics() -> Dict[str, float]:
        return {
            "candidate_response_supported": 0.0,
            "candidate_response_active": 0.0,
            "candidate_response_actuated": 0.0,
            "candidate_response_probe_active_ratio": 0.0,
            "candidate_response_generator_active_ratio": 0.0,
            "candidate_response_accept_ratio": 0.0,
            "candidate_response_selection_confidence_mean": 0.0,
            "candidate_response_support_mean": 0.0,
            "candidate_response_confidence_mean": 0.0,
            "candidate_response_trust_radius_mean": 0.0,
            "candidate_response_probe_radius_mean": 0.0,
            "candidate_response_requested_shift_mean": 0.0,
            "candidate_response_applied_shift_mean": 0.0,
            "candidate_response_noop_ratio": 1.0,
            "candidate_response_rollback_ratio": 0.0,
            "candidate_response_agent_opportunity_ratio": 0.0,
            "candidate_response_agent_requested_ratio": 0.0,
            "candidate_response_agent_effective_ratio": 0.0,
            "candidate_response_actuator_beta_mean": 0.0,
            "candidate_response_actuator_beta_min": 0.0,
            "candidate_response_actuator_beta_max": 0.0,
            "candidate_response_shadow_base_report_f": 0.0,
            "candidate_response_shadow_candidate_report_f": 0.0,
            "candidate_response_shadow_report_win": 0.0,
            "candidate_response_shadow_report_log_improve": 0.0,
            "candidate_response_multisecant_attempt_ratio": 0.0,
            "candidate_response_multisecant_active_ratio": 0.0,
            "candidate_response_spsa_fallback_ratio": 0.0,
            "candidate_response_calibration_only_ratio": 0.0,
            "candidate_response_history_size_mean": 0.0,
            "candidate_response_effective_rank_mean": 0.0,
            "candidate_response_condition_proxy_mean": 0.0,
            "candidate_response_gradient_norm_mean": 0.0,
            "candidate_response_multisecant_predicted_response_mean": 0.0,
            "candidate_response_multisecant_cold_ratio": 0.0,
            "candidate_response_multisecant_stale_ratio": 0.0,
            "candidate_response_multisecant_rank_fail_ratio": 0.0,
            "candidate_response_multisecant_condition_fail_ratio": 0.0,
            "candidate_response_multisecant_zero_gradient_ratio": 0.0,
            "candidate_response_multisecant_nonfinite_ratio": 0.0,
        }

    def _candidate_response_event_enabled(self, communication_applied: bool) -> bool:
        return bool(
            communication_applied
            and self.last_comm_rounds_applied > 0
            and self.candidate_response_mode != "off"
            and self.candidate_response_supported
            and self.consensus_adjacency is not None
            and self.consensus_weight is not None
        )

    def _candidate_response_direction_contract(
        self,
        base_states: np.ndarray,
        generator_actions: np.ndarray = None,
        multisecant_preview=None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, np.ndarray]]:
        """Resolve R/M/H probe directions from only environment-local history."""
        legacy = deterministic_rademacher(
            seed=int(getattr(self.opts, "seed", 0)),
            function_id=int(getattr(self, "question", 0)),
            event_id=int(self.candidate_response_events),
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
        )
        full_shape = (
            self.n_agents,
            self.committee_target_num,
            self.committee_coordinate_dim,
        )
        legacy_per_agent = np.broadcast_to(
            legacy[None, :, :],
            full_shape,
        ).copy()
        all_blocks = np.ones(
            (self.n_agents, self.committee_target_num),
            dtype=bool,
        )
        empty_metrics = {
            "candidate_response_multisecant_attempt_ratio": 0.0,
            "candidate_response_multisecant_active_ratio": 0.0,
            "candidate_response_spsa_fallback_ratio": 0.0,
            "candidate_response_calibration_only_ratio": 0.0,
            "candidate_response_history_size_mean": 0.0,
            "candidate_response_effective_rank_mean": 0.0,
            "candidate_response_condition_proxy_mean": 0.0,
            "candidate_response_gradient_norm_mean": 0.0,
            "candidate_response_multisecant_predicted_response_mean": 0.0,
            "candidate_response_multisecant_cold_ratio": 0.0,
            "candidate_response_multisecant_stale_ratio": 0.0,
            "candidate_response_multisecant_rank_fail_ratio": 0.0,
            "candidate_response_multisecant_condition_fail_ratio": 0.0,
            "candidate_response_multisecant_zero_gradient_ratio": 0.0,
            "candidate_response_multisecant_nonfinite_ratio": 0.0,
        }
        diagnostics = {
            "multisecant_active": np.zeros_like(all_blocks),
            "fallback_reason": np.ones_like(all_blocks, dtype=np.int64),
        }
        if generator_actions is None and self.candidate_generator == "spsa_target":
            return legacy, all_blocks, empty_metrics, diagnostics

        multisecant = (
            multisecant_preview
            if multisecant_preview is not None
            else self._candidate_response_multisecant_preview(base_states)
        )
        use_multisecant = np.asarray(
            multisecant.active_mask,
            dtype=bool,
        )
        if generator_actions is not None:
            selected = np.clip(
                np.asarray(generator_actions, dtype=np.int64).reshape(
                    self.n_agents
                ),
                0,
                1,
            )
            hybrid_requested = selected[:, None] == 1
            applied_multisecant = hybrid_requested & use_multisecant
            direction = np.where(
                applied_multisecant[:, :, None],
                multisecant.direction,
                legacy_per_agent,
            )
            generator_enabled = all_blocks
            calibration_only = np.zeros_like(use_multisecant)
            spsa_fallback = hybrid_requested & ~use_multisecant
            attempt_ratio = float(np.mean(hybrid_requested))
        else:
            applied_multisecant = use_multisecant
            direction = np.where(
                use_multisecant[:, :, None],
                multisecant.direction,
                legacy_per_agent,
            )
            if self.candidate_generator == "multisecant_target":
                generator_enabled = use_multisecant.copy()
                calibration_only = ~use_multisecant
                spsa_fallback = np.zeros_like(use_multisecant)
            else:
                generator_enabled = all_blocks
                calibration_only = np.zeros_like(use_multisecant)
                spsa_fallback = ~use_multisecant
            attempt_ratio = 1.0
        reasons = np.asarray(
            multisecant.fallback_reason,
            dtype=np.int64,
        )
        finite_conditions = np.asarray(
            multisecant.condition_proxy,
            dtype=np.float64,
        )
        finite_conditions = finite_conditions[np.isfinite(finite_conditions)]
        empty_metrics.update(
            {
                "candidate_response_multisecant_attempt_ratio": attempt_ratio,
                "candidate_response_multisecant_active_ratio": float(
                    np.mean(applied_multisecant)
                ),
                "candidate_response_spsa_fallback_ratio": float(
                    np.mean(spsa_fallback)
                ),
                "candidate_response_calibration_only_ratio": float(
                    np.mean(calibration_only)
                ),
                "candidate_response_history_size_mean": float(
                    np.mean(multisecant.history_size)
                ),
                "candidate_response_effective_rank_mean": float(
                    np.mean(multisecant.effective_rank)
                ),
                "candidate_response_condition_proxy_mean": float(
                    np.mean(finite_conditions)
                    if finite_conditions.size
                    else 0.0
                ),
                "candidate_response_gradient_norm_mean": float(
                    np.mean(multisecant.gradient_norm)
                ),
                "candidate_response_multisecant_predicted_response_mean": float(
                    np.mean(
                        multisecant.predicted_response[applied_multisecant]
                    )
                    if np.any(applied_multisecant)
                    else 0.0
                ),
                "candidate_response_multisecant_cold_ratio": float(
                    np.mean(reasons == 1)
                ),
                "candidate_response_multisecant_stale_ratio": float(
                    np.mean(reasons == 2)
                ),
                "candidate_response_multisecant_rank_fail_ratio": float(
                    np.mean(reasons == 3)
                ),
                "candidate_response_multisecant_condition_fail_ratio": float(
                    np.mean(reasons == 4)
                ),
                "candidate_response_multisecant_zero_gradient_ratio": float(
                    np.mean(reasons == 5)
                ),
                "candidate_response_multisecant_nonfinite_ratio": float(
                    np.mean(reasons == 6)
                ),
            }
        )
        diagnostics = {
            "multisecant_active": applied_multisecant.copy(),
            "multisecant_usable": use_multisecant.copy(),
            "fallback_reason": reasons.copy(),
        }
        return direction, generator_enabled, empty_metrics, diagnostics

    def _candidate_response_multisecant_preview(
        self,
        base_states: np.ndarray,
    ):
        """Compute current history geometry without objective calls or mutation."""
        return build_multisecant_directions(
            history_step=self.candidate_multisecant_history_step,
            history_response=self.candidate_multisecant_history_response,
            history_center=self.candidate_multisecant_history_center,
            history_event_id=self.candidate_multisecant_history_event_id,
            history_valid=self.candidate_multisecant_history_valid,
            base_states=base_states,
            current_event_id=int(self.candidate_response_events),
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            min_rank=self.candidate_multisecant_min_rank,
            rank_tolerance=self.candidate_multisecant_rank_tolerance,
            condition_max=self.candidate_multisecant_condition_max,
            center_distance_max_ratio=(
                self.candidate_multisecant_center_distance_max_ratio
            ),
            max_age=self.candidate_multisecant_max_age,
            gradient_min_norm=self.candidate_multisecant_gradient_min_norm,
        )

    def _build_d6_generator_context(
        self,
        *,
        primary_actions: np.ndarray,
        optimizer_base_states: np.ndarray,
        proposal_states: np.ndarray,
        base_states: np.ndarray,
        local_improvements: np.ndarray,
        communication_applied: bool,
        multisecant_preview,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Build the legal pre-probe D6 context and split actor/critic masks."""
        del primary_actions  # carried separately for learned embeddings
        obs = np.zeros(
            (self.n_agents, self.d6_generator_obs_dim),
            dtype=np.float64,
        )
        event_active = self._candidate_response_event_enabled(
            communication_applied
        )
        span_norm = max(
            1e-12,
            float(np.sqrt(max(1, self.D)) * max(1e-12, self.ub - self.lb)),
        )
        proposal_shift = np.linalg.norm(
            np.asarray(proposal_states) - np.asarray(optimizer_base_states),
            axis=1,
        ) / span_norm
        consensus_shift = np.linalg.norm(
            np.asarray(base_states) - np.asarray(proposal_states),
            axis=1,
        ) / span_norm
        recent_committed = np.asarray(
            [
                (
                    float(np.mean(history[-5:]))
                    if history
                    else 0.0
                )
                for history in self.committed_local_improve_hist
            ],
            dtype=np.float64,
        )
        active = np.asarray(multisecant_preview.active_mask, dtype=bool)
        history_fill = np.mean(
            np.asarray(
                self.candidate_multisecant_history_count,
                dtype=np.float64,
            )
            / float(max(1, self.candidate_history_size)),
            axis=1,
        )
        usable_ratio = np.mean(active, axis=1)
        rank_ratio = np.mean(
            np.asarray(multisecant_preview.effective_rank, dtype=np.float64)
            / float(max(1, self.committee_coordinate_dim)),
            axis=1,
        )
        conditions = np.asarray(
            multisecant_preview.condition_proxy,
            dtype=np.float64,
        )
        finite_condition = np.where(np.isfinite(conditions), conditions, np.inf)
        condition_quality = np.mean(
            np.where(
                active,
                1.0 / (1.0 + np.log1p(np.maximum(finite_condition, 0.0))),
                0.0,
            ),
            axis=1,
        )
        predicted = np.asarray(
            multisecant_preview.predicted_response,
            dtype=np.float64,
        )
        predicted_strength = np.mean(
            np.where(
                active,
                np.tanh(np.log1p(np.abs(predicted))),
                0.0,
            ),
            axis=1,
        )
        obs[:, 0] = float(event_active)
        obs[:, 1] = float(
            np.clip(self.sum_fes / max(1.0, float(self.max_fes)), 0.0, 1.0)
        )
        obs[:, 2] = np.asarray(local_improvements, dtype=np.float64)
        obs[:, 3] = proposal_shift
        obs[:, 4] = consensus_shift
        obs[:, 5] = self.last_committed_local_improve
        obs[:, 6] = recent_committed
        obs[:, 7] = self.last_candidate_accept_ratio
        obs[:, 8] = self.last_candidate_selection_confidence
        obs[:, 9] = self.last_candidate_support_ratio
        obs[:, 10] = self.last_candidate_requested_shift_norm
        obs[:, 11] = self.last_candidate_actuator_beta
        obs[:, 12] = np.mean(
            self.last_candidate_multisecant_active,
            axis=1,
        )
        obs[:, 13] = history_fill
        obs[:, 14] = usable_ratio
        obs[:, 15] = rank_ratio
        obs[:, 16] = condition_quality
        obs[:, 17] = predicted_strength
        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        actor_mask = (
            np.full((self.n_agents,), event_active, dtype=bool)
            & np.any(active, axis=1)
        )
        critic_mask = np.ones((self.n_agents,), dtype=bool)
        info = {
            "d6_three_stage": True,
            "generator_obs_schema": list(self.d6_generator_obs_schema),
            "actor_mask": actor_mask.astype(np.float32),
            "critic_mask": critic_mask.astype(np.float32),
            "multisecant_usable_mask": active.copy(),
            "multisecant_history_size": np.asarray(
                multisecant_preview.history_size
            ).copy(),
            "multisecant_effective_rank": np.asarray(
                multisecant_preview.effective_rank
            ).copy(),
            "multisecant_condition_proxy": conditions.copy(),
            "multisecant_predicted_response": predicted.copy(),
            "candidate_probe_local_evals": 0,
            "candidate_verification_local_evals": 0,
            "reported_sum_fes_before_probe": int(self.sum_fes),
            "step_before_probe": int(self.step_count),
        }
        return obs, actor_mask, critic_mask, info

    def _candidate_response_stage_secants(
        self,
        geometry,
        residual_plus: np.ndarray,
        residual_minus: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        samples = build_probe_secant_samples(
            geometry=geometry,
            residual_plus=residual_plus,
            residual_minus=residual_minus,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        return {
            "step": np.asarray(samples.step, dtype=np.float64).copy(),
            "response": np.asarray(
                samples.response,
                dtype=np.float64,
            ).copy(),
            "center": np.asarray(samples.center, dtype=np.float64).copy(),
            "valid_mask": np.asarray(
                samples.valid_mask,
                dtype=bool,
            ).copy(),
            "event_id": int(self.candidate_response_events),
        }

    def _candidate_response_commit_secants(self, staged: Dict) -> None:
        if not staged:
            return
        step = np.asarray(staged["step"], dtype=np.float64).reshape(
            self.n_agents,
            self.committee_target_num,
            self.committee_coordinate_dim,
        )
        response = np.asarray(
            staged["response"],
            dtype=np.float64,
        ).reshape(self.n_agents, self.committee_target_num)
        center = np.asarray(staged["center"], dtype=np.float64).reshape(
            self.n_agents,
            self.committee_target_num,
            self.committee_coordinate_dim,
        )
        valid = np.asarray(
            staged["valid_mask"],
            dtype=bool,
        ).reshape(self.n_agents, self.committee_target_num)
        event_id = int(staged["event_id"])
        for agent_id, target_id in np.argwhere(valid):
            cursor = int(
                self.candidate_multisecant_history_cursor[
                    int(agent_id),
                    int(target_id),
                ]
            )
            index = cursor % self.candidate_history_size
            self.candidate_multisecant_history_step[
                int(agent_id),
                int(target_id),
                index,
            ] = step[int(agent_id), int(target_id)]
            self.candidate_multisecant_history_response[
                int(agent_id),
                int(target_id),
                index,
            ] = response[int(agent_id), int(target_id)]
            self.candidate_multisecant_history_center[
                int(agent_id),
                int(target_id),
                index,
            ] = center[int(agent_id), int(target_id)]
            self.candidate_multisecant_history_event_id[
                int(agent_id),
                int(target_id),
                index,
            ] = event_id
            self.candidate_multisecant_history_valid[
                int(agent_id),
                int(target_id),
                index,
            ] = True
            self.candidate_multisecant_history_cursor[
                int(agent_id),
                int(target_id),
            ] = (index + 1) % self.candidate_history_size
            self.candidate_multisecant_history_count[
                int(agent_id),
                int(target_id),
            ] = min(
                self.candidate_history_size,
                int(
                    self.candidate_multisecant_history_count[
                        int(agent_id),
                        int(target_id),
                    ]
                )
                + 1,
            )

    def _candidate_response_probe_residuals(
        self,
        plus_states: np.ndarray,
        minus_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        plus = np.asarray(plus_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        minus = np.asarray(minus_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        r_plus = np.empty(
            (self.n_agents, self.committee_target_num), dtype=np.float64
        )
        r_minus = np.empty_like(r_plus)
        for agent_id in range(self.n_agents):
            probe_pair = np.stack([plus[agent_id], minus[agent_id]], axis=0)
            values = np.asarray(
                self.fun.local_target_residual_batch(agent_id, probe_pair),
                dtype=np.float64,
            )
            expected = (2, self.committee_target_num)
            if values.shape != expected:
                raise ValueError(
                    "D3-P local probe residual shape mismatch: "
                    f"expected {expected}, got {values.shape} for agent={agent_id}."
                )
            r_plus[agent_id] = values[0]
            r_minus[agent_id] = values[1]
        return r_plus, r_minus, int(2 * self.n_agents)

    def _record_candidate_response_communication(self) -> None:
        if not self.record_comm_cost or self.graph is None:
            return
        directed_edges = int(self.graph.directed_edge_count)
        self.candidate_response_rounds += 2
        self.candidate_response_messages += int(2 * directed_edges)
        self.candidate_response_transmitted_floats += int(
            directed_edges * (self.D + self.committee_target_num)
        )

    def _candidate_response_shadow_global_metrics(
        self,
        base_states: np.ndarray,
        candidate_states: np.ndarray,
    ) -> Tuple[Dict[str, float], int]:
        reports = np.stack(
            [
                np.mean(np.asarray(base_states, dtype=np.float64), axis=0),
                np.mean(np.asarray(candidate_states, dtype=np.float64), axis=0),
            ],
            axis=0,
        )
        local_values = np.asarray(
            self.fun.local_eval_all_batch(reports), dtype=np.float64
        )
        expected = (2, self.n_agents)
        if local_values.shape != expected:
            raise ValueError(
                "D3-P detached global monitor shape mismatch: "
                f"expected {expected}, got {local_values.shape}."
            )
        values = np.mean(local_values, axis=1)
        return (
            {
                "candidate_response_shadow_base_report_f": float(values[0]),
                "candidate_response_shadow_candidate_report_f": float(values[1]),
                "candidate_response_shadow_report_win": float(values[1] < values[0]),
                "candidate_response_shadow_report_log_improve": float(
                    _safe_log_improvement(float(values[0]), float(values[1]))
                ),
            },
            int(local_values.size),
        )

    def _run_candidate_response_detached_shadow_monitor(self) -> None:
        if not self.candidate_response_shadow_due:
            return
        shadow_metrics, shadow_evals = self._candidate_response_shadow_global_metrics(
            self.candidate_response_shadow_base_x,
            self.candidate_response_shadow_candidate_x,
        )
        self.last_candidate_response_metrics.update(shadow_metrics)
        self.candidate_response_shadow_global_local_evals += int(shadow_evals)
        self.candidate_response_shadow_due = False

    def _run_candidate_response_event(
        self,
        *,
        optimizer_base_states: np.ndarray,
        proposal_states: np.ndarray,
        base_states: np.ndarray,
        communication_applied: bool,
        actuator_beta_per_agent: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        beta_per_agent = np.clip(
            np.asarray(actuator_beta_per_agent, dtype=np.float64).reshape(
                self.n_agents
            ),
            0.0,
            1.0,
        )
        metrics = self._empty_candidate_response_metrics()
        metrics["candidate_response_supported"] = float(
            self.candidate_response_supported
        )
        self.last_candidate_response_metrics = metrics
        if not self._candidate_response_event_enabled(communication_applied):
            return bases.copy(), 0

        (
            direction,
            generator_enabled_mask,
            generator_metrics,
            generator_diagnostics,
        ) = self._candidate_response_direction_contract(
            bases,
        )
        geometry = build_probe_geometry(
            base_states=bases,
            optimizer_base_states=optimizer_base_states,
            proposal_states=proposal_states,
            direction=direction,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            trust_scale=self.candidate_trust_scale,
            trust_min_ratio=self.candidate_trust_min_ratio,
            trust_max_ratio=self.candidate_trust_max_ratio,
            probe_scale=self.candidate_probe_scale,
            probe_min_ratio=self.candidate_probe_min_ratio,
            probe_max_ratio=self.candidate_probe_max_ratio,
        )
        r_plus, r_minus, probe_evals = self._candidate_response_probe_residuals(
            geometry.plus_x, geometry.minus_x
        )
        staged_secants = self._candidate_response_stage_secants(
            geometry,
            r_plus,
            r_minus,
        )
        generation = build_spsa_candidates(
            base_states=bases,
            residual_plus=r_plus,
            residual_minus=r_minus,
            geometry=geometry,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            confidence_min=self.candidate_confidence_min,
            generator_enabled_mask=generator_enabled_mask,
        )
        base_residuals, closed, base_evals = self._committee_residual_tensor(bases)
        candidate_residuals, candidate_closed, candidate_evals = (
            self._committee_residual_tensor(generation.candidate_x)
        )
        if not np.array_equal(closed, candidate_closed):
            raise RuntimeError("D3-P verifier graph changed within one event.")
        selection = select_supported_target_blocks(
            base_states=bases,
            candidate_states=generation.candidate_x,
            base_residuals=base_residuals,
            candidate_residuals=candidate_residuals,
            generator_active_mask=generation.active_mask,
            closed_mask=closed,
            weight=self.consensus_weight,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            support_min=self.candidate_support_min,
        )
        actuator = apply_bounded_actuator(
            base_states=bases,
            verified_states=selection.verified_x,
            accepted_mask=selection.accepted_mask,
            trust_radius=geometry.trust_radius,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            beta=beta_per_agent,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        verification_evals = int(base_evals + candidate_evals)
        self._candidate_response_commit_secants(staged_secants)
        self.candidate_response_events += 1
        self.candidate_response_probe_local_evals += int(probe_evals)
        self.candidate_response_verification_local_evals += verification_evals
        self._record_candidate_response_communication()
        self.candidate_response_verified_x = np.asarray(
            selection.verified_x, dtype=np.float64
        ).copy()
        self.candidate_response_selected_source = np.asarray(
            selection.selected_source, dtype=np.int64
        ).copy()
        self.candidate_response_accepted_mask = np.asarray(
            selection.accepted_mask, dtype=bool
        ).copy()
        shift_denom = max(
            1e-12,
            float(
                np.sqrt(max(1, self.committee_coordinate_dim))
                * max(1e-12, self.ub - self.lb)
            ),
        )
        self.last_candidate_accept_ratio = np.mean(
            selection.accepted_mask, axis=1
        ).astype(np.float64, copy=False)
        self.last_candidate_selection_confidence = np.asarray(
            selection.selection_confidence, dtype=np.float64
        ).reshape(self.n_agents)
        self.last_candidate_support_ratio = np.mean(
            selection.support_ratio, axis=1
        ).astype(np.float64, copy=False)
        self.last_candidate_requested_shift_norm = np.clip(
            np.mean(actuator.requested_shift_norm, axis=1) / shift_denom,
            0.0,
            np.inf,
        )
        self.last_candidate_actuator_beta = beta_per_agent.copy()
        self.last_candidate_multisecant_active = np.asarray(
            generator_diagnostics["multisecant_active"],
            dtype=bool,
        ).copy()
        self.last_candidate_multisecant_fallback_reason = np.asarray(
            generator_diagnostics["fallback_reason"],
            dtype=np.int64,
        ).copy()
        agent_opportunity = np.any(selection.accepted_mask, axis=1)
        agent_requested = (
            beta_per_agent > 0.0
        ) & np.any(actuator.requested_shift_norm > 1e-15, axis=1)
        agent_effective = np.any(actuator.applied_mask, axis=1)
        metrics.update(
            {
                "candidate_response_active": 1.0,
                "candidate_response_actuated": float(
                    self.candidate_response_mode == "actuate"
                ),
                "candidate_response_probe_active_ratio": float(
                    np.mean(geometry.probe_active)
                ),
                "candidate_response_generator_active_ratio": float(
                    np.mean(generation.active_mask)
                ),
                "candidate_response_accept_ratio": float(
                    np.mean(selection.accepted_mask)
                ),
                "candidate_response_selection_confidence_mean": float(
                    np.mean(selection.selection_confidence)
                ),
                "candidate_response_support_mean": float(
                    np.mean(selection.support_ratio)
                ),
                "candidate_response_confidence_mean": float(
                    np.mean(generation.confidence)
                ),
                "candidate_response_trust_radius_mean": float(
                    np.mean(geometry.trust_radius)
                ),
                "candidate_response_probe_radius_mean": float(
                    np.mean(geometry.probe_radius)
                ),
                "candidate_response_requested_shift_mean": float(
                    np.mean(actuator.requested_shift_norm)
                ),
                "candidate_response_applied_shift_mean": float(
                    np.mean(actuator.applied_shift_norm)
                ),
                "candidate_response_noop_ratio": float(
                    1.0 - np.mean(actuator.applied_mask)
                ),
                "candidate_response_rollback_ratio": float(
                    np.mean(actuator.rollback_mask)
                ),
                "candidate_response_agent_opportunity_ratio": float(
                    np.mean(agent_opportunity)
                ),
                "candidate_response_agent_requested_ratio": float(
                    np.mean(agent_requested)
                ),
                "candidate_response_agent_effective_ratio": float(
                    np.mean(agent_effective)
                ),
                "candidate_response_actuator_beta_mean": float(
                    np.mean(beta_per_agent)
                ),
                "candidate_response_actuator_beta_min": float(
                    np.min(beta_per_agent)
                ),
                "candidate_response_actuator_beta_max": float(
                    np.max(beta_per_agent)
                ),
            }
        )
        metrics.update(generator_metrics)
        self.candidate_response_shadow_base_x = bases.copy()
        self.candidate_response_shadow_candidate_x = np.asarray(
            actuator.actuated_x, dtype=np.float64
        ).copy()
        self.candidate_response_shadow_due = bool(
            self.candidate_response_mode == "shadow"
            and self.candidate_shadow_global_eval_interval > 0
            and self.candidate_response_events
            % self.candidate_shadow_global_eval_interval
            == 0
        )
        self.last_candidate_response_metrics = metrics
        if self.candidate_response_mode == "actuate":
            self.candidate_response_actuated_events += 1
            return (
                np.asarray(actuator.actuated_x, dtype=np.float64).copy(),
                int(probe_evals + verification_evals),
            )
        return bases.copy(), 0

    def _prepare_d5_candidate_response_event(
        self,
        *,
        optimizer_base_states: np.ndarray,
        proposal_states: np.ndarray,
        base_states: np.ndarray,
        communication_applied: bool,
        generator_actions: np.ndarray = None,
        multisecant_preview=None,
    ) -> Dict:
        """Build one local-only verifier contract without committing a shift."""
        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        progress = float(
            np.clip(
                self.sum_fes / max(1.0, float(self.max_fes)),
                0.0,
                1.0,
            )
        )
        actuator_obs = np.zeros(
            (self.n_agents, self.d5_actuator_obs_dim),
            dtype=np.float64,
        )
        actuator_obs[:, 11] = progress
        opportunity = np.zeros((self.n_agents,), dtype=bool)
        pending = {
            "active": False,
            "base_states": bases.copy(),
            "actuator_obs": actuator_obs,
            "opportunity_mask": opportunity,
            "probe_evals": 0,
            "verification_evals": 0,
            "base_own_local_f": np.zeros(
                (self.n_agents,), dtype=np.float64
            ),
        }
        if not self._candidate_response_event_enabled(communication_applied):
            return pending

        (
            direction,
            generator_enabled_mask,
            generator_metrics,
            generator_diagnostics,
        ) = self._candidate_response_direction_contract(
            bases,
            generator_actions=generator_actions,
            multisecant_preview=multisecant_preview,
        )
        geometry = build_probe_geometry(
            base_states=bases,
            optimizer_base_states=optimizer_base_states,
            proposal_states=proposal_states,
            direction=direction,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            trust_scale=self.candidate_trust_scale,
            trust_min_ratio=self.candidate_trust_min_ratio,
            trust_max_ratio=self.candidate_trust_max_ratio,
            probe_scale=self.candidate_probe_scale,
            probe_min_ratio=self.candidate_probe_min_ratio,
            probe_max_ratio=self.candidate_probe_max_ratio,
        )
        r_plus, r_minus, probe_evals = self._candidate_response_probe_residuals(
            geometry.plus_x,
            geometry.minus_x,
        )
        staged_secants = self._candidate_response_stage_secants(
            geometry,
            r_plus,
            r_minus,
        )
        generation = build_spsa_candidates(
            base_states=bases,
            residual_plus=r_plus,
            residual_minus=r_minus,
            geometry=geometry,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            lower_bound=self.lb,
            upper_bound=self.ub,
            confidence_min=self.candidate_confidence_min,
            generator_enabled_mask=generator_enabled_mask,
        )
        base_residuals, closed, base_evals = self._committee_residual_tensor(
            bases
        )
        candidate_residuals, candidate_closed, candidate_evals = (
            self._committee_residual_tensor(generation.candidate_x)
        )
        if not np.array_equal(closed, candidate_closed):
            raise RuntimeError("D5 verifier graph changed within one event.")
        selection = select_supported_target_blocks(
            base_states=bases,
            candidate_states=generation.candidate_x,
            base_residuals=base_residuals,
            candidate_residuals=candidate_residuals,
            generator_active_mask=generation.active_mask,
            closed_mask=closed,
            weight=self.consensus_weight,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            support_min=self.candidate_support_min,
        )
        unit_actuator = apply_bounded_actuator(
            base_states=bases,
            verified_states=selection.verified_x,
            accepted_mask=selection.accepted_mask,
            trust_radius=geometry.trust_radius,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            beta=1.0,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        opportunity = np.any(selection.accepted_mask, axis=1)
        span = max(1e-12, float(self.ub - self.lb))
        shift_denom = max(
            1e-12,
            float(
                np.sqrt(max(1, self.committee_coordinate_dim))
                * span
            ),
        )
        for agent_id in range(self.n_agents):
            accepted = np.asarray(
                selection.accepted_mask[agent_id],
                dtype=bool,
            )
            actuator_obs[agent_id, 0] = 1.0
            actuator_obs[agent_id, 1] = float(opportunity[agent_id])
            if not opportunity[agent_id]:
                continue
            target_ids = np.flatnonzero(accepted)
            support_values = np.asarray(
                selection.support_ratio[agent_id, target_ids],
                dtype=np.float64,
            )
            source_ids = np.asarray(
                selection.selected_source[agent_id, target_ids],
                dtype=np.int64,
            )
            generator_values = []
            for source_id, target_id in zip(source_ids, target_ids):
                if 0 <= int(source_id) < self.n_agents:
                    generator_values.append(
                        float(
                            generation.confidence[
                                int(source_id),
                                int(target_id),
                            ]
                        )
                    )
            actuator_obs[agent_id, 2] = float(np.mean(accepted))
            actuator_obs[agent_id, 3] = float(
                selection.selection_confidence[agent_id]
            )
            actuator_obs[agent_id, 4] = float(np.mean(support_values))
            actuator_obs[agent_id, 5] = float(np.min(support_values))
            actuator_obs[agent_id, 6] = float(
                np.mean(generator_values) if generator_values else 0.0
            )
            actuator_obs[agent_id, 7] = float(
                np.mean(unit_actuator.requested_shift_norm[agent_id])
                / shift_denom
            )
            actuator_obs[agent_id, 8] = float(
                np.mean(geometry.trust_radius[agent_id, target_ids])
                / span
            )
            actuator_obs[agent_id, 9] = float(
                np.mean(geometry.probe_radius[agent_id, target_ids])
                / span
            )
            actuator_obs[agent_id, 10] = float(
                np.mean(source_ids == agent_id)
            )
        actuator_obs = np.nan_to_num(
            actuator_obs,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        base_own_local_f = np.asarray(
            [
                np.sum(base_residuals[i, i, :], dtype=np.float64)
                for i in range(self.n_agents)
            ],
            dtype=np.float64,
        )
        return {
            "active": True,
            "base_states": bases.copy(),
            "geometry": geometry,
            "generation": generation,
            "selection": selection,
            "actuator_obs": actuator_obs,
            "opportunity_mask": opportunity.astype(bool, copy=True),
            "probe_evals": int(probe_evals),
            "verification_evals": int(base_evals + candidate_evals),
            "base_own_local_f": base_own_local_f,
            "staged_secants": staged_secants,
            "generator_metrics": dict(generator_metrics),
            "generator_diagnostics": {
                key: np.asarray(value).copy()
                for key, value in generator_diagnostics.items()
            },
        }

    def _commit_d5_candidate_response_event(
        self,
        pending: Dict,
        actuator_beta_per_agent: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Commit exactly one prepared verifier contract."""
        bases = np.asarray(
            pending["base_states"],
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        beta_per_agent = np.clip(
            np.asarray(
                actuator_beta_per_agent,
                dtype=np.float64,
            ).reshape(self.n_agents),
            0.0,
            1.0,
        )
        metrics = self._empty_candidate_response_metrics()
        metrics["candidate_response_supported"] = float(
            self.candidate_response_supported
        )
        self.last_candidate_response_metrics = metrics
        if not bool(pending.get("active", False)):
            return (
                bases.copy(),
                0,
                np.asarray(
                    pending.get(
                        "base_own_local_f",
                        np.zeros((self.n_agents,), dtype=np.float64),
                    ),
                    dtype=np.float64,
                ).reshape(self.n_agents),
            )

        geometry = pending["geometry"]
        generation = pending["generation"]
        selection = pending["selection"]
        actuator = apply_bounded_actuator(
            base_states=bases,
            verified_states=selection.verified_x,
            accepted_mask=selection.accepted_mask,
            trust_radius=geometry.trust_radius,
            target_num=self.committee_target_num,
            coordinate_dim=self.committee_coordinate_dim,
            beta=beta_per_agent,
            lower_bound=self.lb,
            upper_bound=self.ub,
        )
        probe_evals = int(pending["probe_evals"])
        verification_evals = int(pending["verification_evals"])
        self._candidate_response_commit_secants(
            pending.get("staged_secants", {})
        )
        self.candidate_response_events += 1
        self.candidate_response_probe_local_evals += probe_evals
        self.candidate_response_verification_local_evals += verification_evals
        self._record_candidate_response_communication()
        self.candidate_response_verified_x = np.asarray(
            selection.verified_x,
            dtype=np.float64,
        ).copy()
        self.candidate_response_selected_source = np.asarray(
            selection.selected_source,
            dtype=np.int64,
        ).copy()
        self.candidate_response_accepted_mask = np.asarray(
            selection.accepted_mask,
            dtype=bool,
        ).copy()
        shift_denom = max(
            1e-12,
            float(
                np.sqrt(max(1, self.committee_coordinate_dim))
                * max(1e-12, self.ub - self.lb)
            ),
        )
        self.last_candidate_accept_ratio = np.mean(
            selection.accepted_mask,
            axis=1,
        ).astype(np.float64, copy=False)
        self.last_candidate_selection_confidence = np.asarray(
            selection.selection_confidence,
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_candidate_support_ratio = np.mean(
            selection.support_ratio,
            axis=1,
        ).astype(np.float64, copy=False)
        self.last_candidate_requested_shift_norm = np.clip(
            np.mean(actuator.requested_shift_norm, axis=1) / shift_denom,
            0.0,
            np.inf,
        )
        self.last_candidate_actuator_beta = beta_per_agent.copy()
        generator_diagnostics = pending.get(
            "generator_diagnostics",
            {
                "multisecant_active": np.zeros(
                    (
                        self.n_agents,
                        self.committee_target_num,
                    ),
                    dtype=bool,
                ),
                "fallback_reason": np.ones(
                    (
                        self.n_agents,
                        self.committee_target_num,
                    ),
                    dtype=np.int64,
                ),
            },
        )
        self.last_candidate_multisecant_active = np.asarray(
            generator_diagnostics["multisecant_active"],
            dtype=bool,
        ).reshape(
            self.n_agents,
            self.committee_target_num,
        ).copy()
        self.last_candidate_multisecant_fallback_reason = np.asarray(
            generator_diagnostics["fallback_reason"],
            dtype=np.int64,
        ).reshape(
            self.n_agents,
            self.committee_target_num,
        ).copy()
        agent_opportunity = np.any(selection.accepted_mask, axis=1)
        agent_requested = (
            beta_per_agent > 0.0
        ) & np.any(actuator.requested_shift_norm > 1e-15, axis=1)
        agent_effective = np.any(actuator.applied_mask, axis=1)
        metrics.update(
            {
                "candidate_response_active": 1.0,
                "candidate_response_actuated": float(
                    self.candidate_response_mode == "actuate"
                ),
                "candidate_response_probe_active_ratio": float(
                    np.mean(geometry.probe_active)
                ),
                "candidate_response_generator_active_ratio": float(
                    np.mean(generation.active_mask)
                ),
                "candidate_response_accept_ratio": float(
                    np.mean(selection.accepted_mask)
                ),
                "candidate_response_selection_confidence_mean": float(
                    np.mean(selection.selection_confidence)
                ),
                "candidate_response_support_mean": float(
                    np.mean(selection.support_ratio)
                ),
                "candidate_response_confidence_mean": float(
                    np.mean(generation.confidence)
                ),
                "candidate_response_trust_radius_mean": float(
                    np.mean(geometry.trust_radius)
                ),
                "candidate_response_probe_radius_mean": float(
                    np.mean(geometry.probe_radius)
                ),
                "candidate_response_requested_shift_mean": float(
                    np.mean(actuator.requested_shift_norm)
                ),
                "candidate_response_applied_shift_mean": float(
                    np.mean(actuator.applied_shift_norm)
                ),
                "candidate_response_noop_ratio": float(
                    1.0 - np.mean(actuator.applied_mask)
                ),
                "candidate_response_rollback_ratio": float(
                    np.mean(actuator.rollback_mask)
                ),
                "candidate_response_agent_opportunity_ratio": float(
                    np.mean(agent_opportunity)
                ),
                "candidate_response_agent_requested_ratio": float(
                    np.mean(agent_requested)
                ),
                "candidate_response_agent_effective_ratio": float(
                    np.mean(agent_effective)
                ),
                "candidate_response_actuator_beta_mean": float(
                    np.mean(beta_per_agent)
                ),
                "candidate_response_actuator_beta_min": float(
                    np.min(beta_per_agent)
                ),
                "candidate_response_actuator_beta_max": float(
                    np.max(beta_per_agent)
                ),
            }
        )
        metrics.update(
            {
                key: float(value)
                for key, value in pending.get(
                    "generator_metrics",
                    {},
                ).items()
            }
        )
        self.candidate_response_shadow_base_x = bases.copy()
        self.candidate_response_shadow_candidate_x = np.asarray(
            actuator.actuated_x,
            dtype=np.float64,
        ).copy()
        self.candidate_response_shadow_due = bool(
            self.candidate_response_mode == "shadow"
            and self.candidate_shadow_global_eval_interval > 0
            and (self.candidate_response_events
                 % self.candidate_shadow_global_eval_interval == 0)
        )
        self.last_candidate_response_metrics = metrics
        if self.candidate_response_mode == "actuate":
            self.candidate_response_actuated_events += 1
            next_states = np.asarray(
                actuator.actuated_x,
                dtype=np.float64,
            ).copy()
            reported_evals = int(probe_evals + verification_evals)
        else:
            next_states = bases.copy()
            reported_evals = 0
        return (
            next_states,
            reported_evals,
            np.asarray(
                pending["base_own_local_f"],
                dtype=np.float64,
            ).reshape(self.n_agents),
        )

    @staticmethod
    def _empty_committee_metrics() -> Dict[str, float]:
        return {
            "committee_supported": 0.0,
            "committee_active": 0.0,
            "committee_selection_target_block": 0.0,
            "committee_best_score_mean": 0.0,
            "committee_best_score_max": 0.0,
            "committee_confidence_mean": 0.0,
            "committee_confidence_max": 0.0,
            "committee_guide_norm_mean": 0.0,
            "committee_guide_norm_max": 0.0,
            "committee_self_source_ratio": 0.0,
            "committee_source_diversity_mean": 0.0,
            "committee_source_diversity_max": 0.0,
            "committee_shadow_whole_report_f": 0.0,
            "committee_shadow_target_block_report_f": 0.0,
            "committee_shadow_whole_report_win": 0.0,
            "committee_shadow_target_block_report_win": 0.0,
            "committee_shadow_whole_report_log_improve": 0.0,
            "committee_shadow_target_block_report_log_improve": 0.0,
            "committee_shadow_whole_agent_f_mean": 0.0,
            "committee_shadow_whole_agent_f_min": 0.0,
            "committee_shadow_target_block_agent_f_mean": 0.0,
            "committee_shadow_target_block_agent_f_min": 0.0,
            "committee_shadow_whole_agent_win_ratio": 0.0,
            "committee_shadow_target_block_agent_win_ratio": 0.0,
            "committee_shadow_target_block_win_vs_whole": 0.0,
            "committee_acceptance_supported": 0.0,
            "committee_acceptance_ratio": 0.0,
            "committee_rejected_ratio": 0.0,
            "committee_candidate_report_f": 0.0,
            "committee_candidate_global_f_mean": 0.0,
            "committee_candidate_global_f_min": 0.0,
            "committee_candidate_vs_report_log_improve": 0.0,
            "committee_effective_beta_mean": 0.0,
            "committee_effective_beta_max": 0.0,
            "committee_base_alignment_mean": 0.0,
            "committee_noop_ratio": 0.0,
        }

    def _committee_event_enabled(self, communication_applied: bool) -> bool:
        return bool(
            communication_applied
            and self.last_comm_rounds_applied > 0
            and self.committee_mode != "off"
            and self.committee_supported
            and self.consensus_adjacency is not None
            and self.consensus_weight is not None
        )

    def _committee_residual_tensor(
        self,
        proposal_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        proposals = np.asarray(proposal_states, dtype=np.float64).reshape(
            self.n_agents,
            self.D,
        )
        closed = build_closed_neighborhood_mask(self.consensus_adjacency)
        residuals = np.full(
            (self.n_agents, self.n_agents, self.committee_target_num),
            np.nan,
            dtype=np.float64,
        )
        eval_count = 0
        for verifier in range(self.n_agents):
            candidate_ids = np.flatnonzero(closed[verifier])
            values = np.asarray(
                self.fun.local_target_residual_batch(
                    verifier,
                    proposals[candidate_ids],
                ),
                dtype=np.float64,
            )
            expected_shape = (candidate_ids.size, self.committee_target_num)
            if values.shape != expected_shape:
                raise ValueError(
                    "Committee target-residual shape mismatch: "
                    f"expected {expected_shape}, got {values.shape} "
                    f"for verifier={verifier}."
                )
            residuals[verifier, candidate_ids, :] = values
            eval_count += int(candidate_ids.size)
        return residuals, closed, int(eval_count)

    def _committee_selections(
        self,
        proposal_states: np.ndarray,
    ) -> Tuple[Dict[str, CommitteeSelection], np.ndarray, int]:
        residuals, closed, eval_count = self._committee_residual_tensor(
            proposal_states
        )
        ranks = build_rank_tensor(residuals, closed)
        scores = aggregate_candidate_scores(
            ranks,
            self.consensus_weight,
            closed,
        )
        requested = (
            ["whole", "target_block"]
            if self.committee_selection == "both"
            else [self.committee_selection]
        )
        selections = {
            mode: select_verified_candidates(
                proposals=proposal_states,
                candidate_scores=scores,
                closed_mask=closed,
                coordinate_dim=self.committee_coordinate_dim,
                selection=mode,
            )
            for mode in requested
        }
        return selections, scores, int(eval_count)

    def _committee_direction_from_selection(
        self,
        selection: CommitteeSelection,
        next_agent_x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        direction = np.asarray(
            selection.verified_x,
            dtype=np.float64,
        ) - np.asarray(next_agent_x, dtype=np.float64)
        units, norms = self._safe_unit_vectors(direction)
        return (
            np.nan_to_num(units, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _record_committee_communication(self) -> None:
        if not self.record_comm_cost or self.graph is None:
            return
        directed_edges = int(self.graph.directed_edge_count)
        messages = int(2 * directed_edges)
        self.committee_messages += messages
        self.committee_transmitted_floats += int(
            messages * self.committee_target_num
        )

    def _record_committee_acceptance_communication(self) -> None:
        if not self.record_comm_cost:
            return
        candidate_objective_pairs = int(self.n_agents * (self.n_agents + 1))
        messages = int(2 * candidate_objective_pairs)
        transmitted_floats = int(
            candidate_objective_pairs * (self.D + 1)
        )
        self.committee_acceptance_messages += messages
        self.committee_acceptance_transmitted_floats += transmitted_floats
        self.committee_messages += messages
        self.committee_transmitted_floats += transmitted_floats

    def _committee_global_selection_values(
        self,
        selection: CommitteeSelection,
    ) -> Tuple[np.ndarray, float, int]:
        verified = np.asarray(selection.verified_x, dtype=np.float64).reshape(
            self.n_agents,
            self.D,
        )
        verified_report = np.mean(verified, axis=0)
        candidates = np.concatenate([verified, verified_report[None, :]], axis=0)
        local_vals = np.asarray(
            self.fun.local_eval_all_batch(candidates),
            dtype=np.float64,
        )
        expected_shape = (self.n_agents + 1, self.n_agents)
        if local_vals.shape != expected_shape:
            raise ValueError(
                "Committee global-eval shape mismatch: "
                f"expected {expected_shape}, got {local_vals.shape}."
            )
        global_vals = np.mean(local_vals, axis=1)
        return (
            np.asarray(global_vals[: self.n_agents], dtype=np.float64),
            float(global_vals[-1]),
            int(local_vals.size),
        )

    def _committee_shadow_selection_metrics(
        self,
        name: str,
        selection: CommitteeSelection,
        f_report: float,
    ) -> Tuple[Dict[str, float], int]:
        agent_vals, report_value, local_eval_count = (
            self._committee_global_selection_values(selection)
        )
        prefix = f"committee_shadow_{name}"
        metrics = {
            f"{prefix}_report_f": report_value,
            f"{prefix}_report_win": float(report_value < float(f_report)),
            f"{prefix}_report_log_improve": float(
                _safe_log_improvement(float(f_report), report_value)
            ),
            f"{prefix}_agent_f_mean": float(np.mean(agent_vals)),
            f"{prefix}_agent_f_min": float(np.min(agent_vals)),
            f"{prefix}_agent_win_ratio": float(
                np.mean(agent_vals < float(f_report))
            ),
        }
        return metrics, int(local_eval_count)

    def _committee_acceptance_stage_index(self) -> int:
        progress = float(self.sum_fes) / max(1.0, float(self.max_fes))
        if progress < (1.0 / 3.0):
            return 0
        if progress < (2.0 / 3.0):
            return 1
        return 2

    def _run_committee_event(
        self,
        proposal_states: np.ndarray,
        next_agent_x: np.ndarray,
        f_report: float,
        communication_applied: bool,
    ) -> int:
        """Update the retained next-step committee guide and diagnostics."""
        metrics = dict(self.last_committee_metrics)
        metrics["committee_supported"] = float(self.committee_supported)
        metrics["committee_acceptance_supported"] = float(
            self.committee_supported
            and self.committee_mode == "guide"
            and self.committee_acceptance_mode == "report_improve"
        )
        metrics["committee_active"] = 0.0
        self.last_committee_metrics = metrics
        if not self._committee_event_enabled(communication_applied):
            return 0

        selections, scores, eval_count = self._committee_selections(proposal_states)
        primary_name = (
            "target_block" if self.committee_selection == "both" else self.committee_selection
        )
        primary = selections[primary_name]
        direction, direction_norms = self._committee_direction_from_selection(
            primary,
            next_agent_x,
        )
        self.committee_direction = direction
        self.committee_confidence = np.asarray(
            primary.confidence,
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.committee_verified_x = np.asarray(
            primary.verified_x,
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.committee_selected_source = np.asarray(
            primary.source_ids,
            dtype=np.int64,
        ).reshape(self.n_agents, self.committee_target_num)
        self.committee_candidate_score = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(self.n_agents, self.committee_target_num)

        summary = summarize_selection(primary)
        metrics = self._empty_committee_metrics()
        metrics.update(
            {
                "committee_supported": 1.0,
                "committee_active": 1.0,
                "committee_selection_target_block": float(
                    primary_name == "target_block"
                ),
                "committee_best_score_mean": float(summary["best_score_mean"]),
                "committee_best_score_max": float(summary["best_score_max"]),
                "committee_confidence_mean": float(summary["confidence_mean"]),
                "committee_confidence_max": float(summary["confidence_max"]),
                "committee_guide_norm_mean": float(np.mean(direction_norms)),
                "committee_guide_norm_max": float(np.max(direction_norms)),
                "committee_self_source_ratio": float(summary["self_source_ratio"]),
                "committee_source_diversity_mean": float(
                    summary["source_diversity_mean"]
                ),
                "committee_source_diversity_max": float(
                    summary["source_diversity_max"]
                ),
            }
        )

        shadow_global_evals = 0
        acceptance_evals = 0
        if self.committee_mode == "shadow" and self.committee_shadow_global_eval:
            for name, selection in selections.items():
                selection_metrics, selection_evals = (
                    self._committee_shadow_selection_metrics(
                        name,
                        selection,
                        f_report,
                    )
                )
                metrics.update(selection_metrics)
                shadow_global_evals += int(selection_evals)
            if "whole" in selections and "target_block" in selections:
                metrics["committee_shadow_target_block_win_vs_whole"] = float(
                    metrics["committee_shadow_target_block_report_f"]
                    < metrics["committee_shadow_whole_report_f"]
                )

        if (
            self.committee_mode == "guide"
            and self.committee_acceptance_mode == "report_improve"
        ):
            agent_vals, report_value, acceptance_evals = (
                self._committee_global_selection_values(primary)
            )
            acceptance = evaluate_report_improve_acceptance(
                candidate_report_f=report_value,
                report_f=float(f_report),
                confidence=self.committee_confidence,
                mix_strength=self.committee_mix_strength,
                min_log_improve=self.committee_acceptance_min_log_improve,
            )
            self.committee_acceptance_active = bool(acceptance.accepted)
            self.committee_candidate_global_f = agent_vals.copy()
            self.committee_candidate_report_f = float(report_value)
            self.committee_candidate_vs_report_log_improve = float(
                acceptance.log_improvement
            )
            self.committee_effective_beta = np.asarray(
                acceptance.effective_beta,
                dtype=np.float64,
            ).reshape(self.n_agents)
            noop = float(np.all(self.committee_effective_beta <= 1e-12))
            metrics.update(
                {
                    "committee_acceptance_supported": 1.0,
                    "committee_acceptance_ratio": float(acceptance.accepted),
                    "committee_rejected_ratio": float(not acceptance.accepted),
                    "committee_candidate_report_f": float(report_value),
                    "committee_candidate_global_f_mean": float(np.mean(agent_vals)),
                    "committee_candidate_global_f_min": float(np.min(agent_vals)),
                    "committee_candidate_vs_report_log_improve": float(
                        np.nan_to_num(
                            acceptance.log_improvement,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    ),
                    "committee_effective_beta_mean": float(
                        np.mean(self.committee_effective_beta)
                    ),
                    "committee_effective_beta_max": float(
                        np.max(self.committee_effective_beta)
                    ),
                    "committee_noop_ratio": noop,
                }
            )
            self.committee_acceptance_events += 1
            if acceptance.accepted:
                self.committee_acceptance_accepted_events += 1
            else:
                self.committee_acceptance_rejected_events += 1
            stage = self._committee_acceptance_stage_index()
            self.committee_acceptance_stage_events[stage] += 1
            if acceptance.accepted:
                self.committee_acceptance_stage_accepted_events[stage] += 1
            self.committee_acceptance_local_evals += int(acceptance_evals)
            self._record_committee_acceptance_communication()

        self.last_committee_metrics = metrics
        self.committee_events += 1
        self._record_committee_communication()
        if self.committee_mode == "guide":
            self.committee_decision_local_evals += int(eval_count)
            return int(eval_count + acceptance_evals)
        self.committee_shadow_local_evals += int(eval_count)
        self.committee_shadow_global_local_evals += int(shadow_global_evals)
        return 0

    def _fuse_committee_with_optimizer_guide(self) -> None:
        if self.committee_mode != "guide" or not self.committee_supported:
            return
        base = np.asarray(
            self.optimizer_guide_direction,
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        committee = np.asarray(
            self.committee_direction,
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        base_units, base_norms = self._safe_unit_vectors(base)
        committee_units, committee_norms = self._safe_unit_vectors(committee)
        active_pair = (base_norms > 1e-12) & (committee_norms > 1e-12)
        alignment = np.zeros((self.n_agents,), dtype=np.float64)
        alignment[active_pair] = np.sum(
            base_units[active_pair] * committee_units[active_pair], axis=1
        )
        self.committee_base_alignment = np.clip(alignment, -1.0, 1.0)
        if self.committee_acceptance_mode == "report_improve":
            beta = np.asarray(
                self.committee_effective_beta,
                dtype=np.float64,
            ).reshape(self.n_agents)
        else:
            beta = np.clip(
                float(self.committee_mix_strength)
                * np.asarray(self.committee_confidence, dtype=np.float64),
                0.0,
                1.0,
            )
            self.committee_effective_beta = beta.copy()
        self.last_committee_metrics["committee_base_alignment_mean"] = float(
            np.mean(self.committee_base_alignment)
        )
        self.last_committee_metrics["committee_effective_beta_mean"] = float(
            np.mean(beta)
        )
        self.last_committee_metrics["committee_effective_beta_max"] = float(
            np.max(beta)
        )
        self.last_committee_metrics["committee_noop_ratio"] = float(
            np.all(beta <= 1e-12)
        )
        if (
            self.committee_acceptance_mode == "report_improve"
            and np.all(beta <= 1e-12)
        ):
            return
        mixed = (1.0 - beta[:, None]) * base + beta[:, None] * committee
        mixed_units, mixed_norms = self._safe_unit_vectors(mixed)
        active = mixed_norms > 1e-12
        self.optimizer_guide_direction = np.nan_to_num(
            mixed_units,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.last_optimizer_guide_norm = np.asarray(
            mixed_norms,
            dtype=np.float64,
        )
        self.last_optimizer_guide_source_active_ratio = float(np.mean(active))

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

    def _optimizer_guide_enabled_for(self, optimizer_name: str) -> bool:
        if not self.optimizer_guide_enable:
            return False
        if self.optimizer_guide_apply_all:
            return True
        return str(optimizer_name).lower() in self.optimizer_guide_apply_optimizers

    def _optimizer_guide_internal_enabled_for(self, optimizer_name: str) -> bool:
        if self.optimizer_guide_internal_mode == "off":
            return False
        if self.optimizer_guide_internal_apply_all:
            return True
        return str(optimizer_name).lower() in self.optimizer_guide_internal_apply_optimizers

    def _anchor_enabled_for(self, optimizer_name: str) -> bool:
        if not self.anchor_enable:
            return False
        if self.anchor_apply_all:
            return True
        return str(optimizer_name).lower() in self.anchor_apply_optimizers

    def _anchor_normalizer(self) -> float:
        return max(
            1e-12,
            float(np.sqrt(self.D) * max(1e-12, self.ub - self.lb)),
        )

    def _compute_anchor_points(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64).reshape(self.n_agents, self.D)
        if not self.anchor_enable or self.anchor_source == "self":
            return x.copy()
        if self.anchor_source == "mean":
            center = np.mean(x, axis=0)
            return np.repeat(center[None, :], self.n_agents, axis=0)
        if self.anchor_source == "consensus":
            if self.consensus_weight is not None:
                return np.asarray(self.consensus_weight @ x, dtype=np.float64)
            center = np.mean(x, axis=0)
            return np.repeat(center[None, :], self.n_agents, axis=0)
        return x.copy()

    def _refresh_anchor_points(self, states: np.ndarray) -> None:
        anchors = self._compute_anchor_points(states)
        anchors = np.clip(
            np.nan_to_num(anchors, nan=0.0, posinf=self.ub, neginf=self.lb),
            self.lb,
            self.ub,
        )
        self.anchor_points = anchors.reshape(self.n_agents, self.D)

    def _anchor_options(
        self,
        agent_id: int,
        optimizer_name: str,
        x_base: np.ndarray,
    ) -> Dict:
        aid = int(agent_id)
        self.last_anchor_applied[aid] = 0.0
        self.last_anchor_dist[aid] = 0.0
        self.last_anchor_direction_norm[aid] = 0.0
        if (
            not self._anchor_enabled_for(optimizer_name)
            or self.anchor_source == "self"
            or self.anchor_strength <= 0.0
        ):
            return {"optimizer_anchor_enable": False}
        base = np.asarray(x_base, dtype=np.float64).reshape(self.D)
        anchor = np.asarray(self.anchor_points[aid], dtype=np.float64).reshape(self.D)
        direction = anchor - base
        dist = float(np.linalg.norm(direction))
        if (not np.isfinite(dist)) or dist <= 1e-12:
            return {"optimizer_anchor_enable": False}
        normed = float(dist / self._anchor_normalizer())
        self.last_anchor_applied[aid] = 1.0
        self.last_anchor_dist[aid] = normed
        self.last_anchor_direction_norm[aid] = dist
        return {
            "optimizer_anchor_enable": True,
            "optimizer_anchor_point": anchor.copy(),
            "optimizer_anchor_strength": float(self.anchor_strength),
            "optimizer_anchor_mix_strength": float(self.anchor_mix_strength),
            "optimizer_anchor_sample_ratio": float(self.anchor_sample_ratio),
            "optimizer_anchor_sample_clip_ratio": float(self.anchor_sample_clip_ratio),
            "optimizer_anchor_mean_pull": bool(self.anchor_mean_pull),
            "optimizer_anchor_sample_injection": bool(self.anchor_sample_injection),
        }

    def _optimizer_guide_effective_strength(self) -> Tuple[float, float, float]:
        metric = float(getattr(self, "last_optimizer_guide_schedule_metric", 0.0))
        scale = 1.0
        anneal = 1.0
        if self.optimizer_guide_strength_schedule == "disagreement":
            low = float(self.optimizer_guide_disagreement_low)
            high = float(self.optimizer_guide_disagreement_high)
            if high <= low + 1e-12:
                t = 1.0 if metric > low else 0.0
            else:
                t = (metric - low) / (high - low)
            t = float(np.clip(t, 0.0, 1.0))
            scale = float(
                self.optimizer_guide_strength_scale_min
                + t
                * (
                    self.optimizer_guide_strength_scale_max
                    - self.optimizer_guide_strength_scale_min
                )
            )
        elif self.optimizer_guide_strength_schedule == "budget":
            # 预算退火（F5 停滞修复）：前 50% 预算内强度从满值线性退到 0，
            # 之后恒 0，让优化器后期自主收缩 sigma。用真实记账字段
            # sum_fes/max_fes（与既有 progress/rem_ratio 同源，不发明新预算）。
            elapsed_ratio = float(self.sum_fes) / max(1.0, float(self.max_fes))
            anneal = float(max(0.0, 1.0 - elapsed_ratio / 0.5))
        strength = float(max(0.0, self.optimizer_guide_strength * scale * anneal))
        return strength, scale, metric

    def _collab_mode_name(self, idx: int) -> str:
        if not self.collab_action_enable:
            return "consensus"
        j = int(np.clip(int(idx), 0, len(self.collab_modes) - 1))
        return str(self.collab_modes[j]).lower()

    def _guide_scale_value(self, idx: int) -> float:
        if not self.guide_scale_action_enable:
            return 1.0
        j = int(np.clip(int(idx), 0, len(self.guide_scale_candidates) - 1))
        return float(max(0.0, self.guide_scale_candidates[j]))

    def _leader_guide_direction(self, agent_id: int) -> Tuple[np.ndarray, bool]:
        if self.consensus_adjacency is None:
            return np.zeros((self.D,), dtype=np.float64), False
        aid = int(agent_id)
        neigh = np.flatnonzero(np.asarray(self.consensus_adjacency[aid], dtype=bool))
        neigh = np.asarray([j for j in neigh if int(j) != aid], dtype=np.int64)
        if neigh.size <= 0:
            return np.zeros((self.D,), dtype=np.float64), False
        local = np.asarray(self.agent_local_f, dtype=np.float64)
        valid = neigh[np.isfinite(local[neigh]) & (local[neigh] < local[aid])]
        if valid.size <= 0:
            return np.zeros((self.D,), dtype=np.float64), False
        best = int(valid[np.argmin(local[valid])])
        direction = np.asarray(self.agent_x[best] - self.agent_x[aid], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            return np.zeros((self.D,), dtype=np.float64), False
        return direction / norm, True

    def _guide_replacement_eligible_for(self, optimizer_name: str) -> bool:
        if not self.guide_replacement_enabled:
            return False
        if self.guide_replacement_scope == "all":
            return True
        return self._optimizer_guide_enabled_for(optimizer_name)

    def _reset_collective_guide_event(self) -> None:
        self.last_collective_guide_shared_base.fill(0.0)
        self.last_collective_guide_direction.fill(0.0)
        self.last_collective_guide_radius.fill(0.0)
        self.last_collective_guide_vote.fill(0)
        self.last_collective_guide_source_valid.fill(0.0)
        self.last_collective_guide_vote_valid.fill(0.0)
        self.last_collective_guide_suppressed.fill(0.0)
        self.last_collective_guide_valid = False
        self.last_collective_guide_vote_sum = 0
        self.last_collective_guide_hypothetical_veto = False
        self.last_collective_guide_actual_veto = False
        self.last_collective_guide_null_veto = False

    def _collective_all_gather(self, payloads: np.ndarray) -> np.ndarray:
        """Flood fixed-schema source records over the explicit consensus graph."""
        if self.consensus_adjacency is None or self.graph is None:
            raise RuntimeError("Collective guide all-gather requires a resolved graph.")
        values = np.asarray(payloads, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self.n_agents:
            raise ValueError(
                "Collective guide payload must have shape [n_agents,payload_dim]."
            )
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Collective guide payload contains NaN or infinity.")

        event_id = int(self.step_count)
        known = [
            {int(i): (event_id, values[i].copy())}
            for i in range(self.n_agents)
        ]
        adjacency = np.asarray(self.consensus_adjacency, dtype=bool)
        rounds = 0
        messages = 0
        transmitted_floats = 0
        for _ in range(max(1, self.n_agents - 1)):
            snapshot = [dict(records) for records in known]
            updated = [dict(records) for records in known]
            for receiver in range(self.n_agents):
                for sender in np.flatnonzero(adjacency[receiver]):
                    sender = int(sender)
                    messages += 1
                    transmitted_floats += len(snapshot[sender]) * (
                        int(values.shape[1]) + 2
                    )
                    for source_id, (record_event, record) in snapshot[sender].items():
                        if int(record_event) != event_id:
                            raise RuntimeError(
                                "Collective guide all-gather received a stale event record."
                            )
                        source_id = int(source_id)
                        if source_id in updated[receiver]:
                            existing_event, existing = updated[receiver][source_id]
                            if int(existing_event) != event_id or not np.array_equal(
                                existing, record
                            ):
                                raise RuntimeError(
                                    "Collective guide all-gather received conflicting duplicate sources."
                                )
                        else:
                            updated[receiver][source_id] = (
                                event_id,
                                record.copy(),
                            )
            known = updated
            rounds += 1
            if all(len(records) == self.n_agents for records in known):
                break
        if not all(len(records) == self.n_agents for records in known):
            raise RuntimeError(
                "Collective guide all-gather did not receive every source record."
            )

        gathered = np.stack(
            [
                np.stack(
                    [
                        known[receiver][source][1]
                        for source in range(self.n_agents)
                    ],
                    axis=0,
                )
                for receiver in range(self.n_agents)
            ],
            axis=0,
        )
        if not np.allclose(
            gathered,
            gathered[0:1],
            rtol=0.0,
            atol=0.0,
            equal_nan=False,
        ):
            raise RuntimeError(
                "Collective guide all-gather produced receiver-dependent records."
            )

        self.collective_guide_comm_rounds += int(rounds)
        self.collective_guide_messages += int(messages)
        self.collective_guide_transmitted_floats += int(transmitted_floats)
        self.graph_comm_rounds += int(rounds)
        self.graph_messages += int(messages)
        self.graph_transmitted_floats += int(transmitted_floats)
        self.total_comm_rounds_applied += int(rounds)
        self.total_comm_events += 1
        return gathered

    def _collective_null_veto(self) -> bool:
        modulus = 1 << 64
        raw = (
            (int(self.opts.seed) + 1) * 6364136223846793005
            + (int(self.step_count) + 1) * 1442695040888963407
        ) % modulus
        return (float(raw) / float(modulus)) < float(
            self.collective_guide_null_veto_rate
        )

    def _run_collective_guide_validation_event(
        self,
        base_states: np.ndarray,
        optimizer_actions: np.ndarray,
        config_actions: np.ndarray = None,
    ) -> int:
        """Validate one shared historical direction before worker creation."""
        self._reset_collective_guide_event()
        if not self.collective_guide_enabled:
            return 0
        if not self.local_only_information:
            raise RuntimeError(
                "Collective guide validation crossed the strict local-only boundary."
            )

        bases = np.asarray(base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        actions = np.asarray(optimizer_actions, dtype=np.int64).reshape(
            self.n_agents
        )
        if config_actions is None:
            cfg = np.zeros(
                (self.n_agents, self.cfg_param_num), dtype=np.int64
            )
        else:
            cfg = np.asarray(config_actions, dtype=np.int64).reshape(
                self.n_agents, self.cfg_param_num
            )
        guide_units, guide_norms = self._safe_unit_vectors(
            self.optimizer_guide_direction
        )
        strength, _, _ = self._optimizer_guide_effective_strength()
        source_valid = np.zeros((self.n_agents,), dtype=bool)
        radii = np.zeros((self.n_agents,), dtype=np.float64)
        for agent_id in range(self.n_agents):
            optimizer_name = self.optimizer_candidates[int(actions[agent_id])]
            eligible = self._optimizer_guide_enabled_for(optimizer_name)
            sigma_options = self._build_optimizer_options(
                agent_id=agent_id,
                optimizer_name=optimizer_name,
                cfg_levels=cfg[agent_id].tolist(),
                dims=self.full_dims,
                x_base=bases[agent_id],
                subfes_i=1,
                seed=int(self.opts.seed + self.step_count * 1000 + agent_id),
            )
            sigma_value = float(
                sigma_options.get(
                    "sigma",
                    self._sigma_ref(agent_id, optimizer_name),
                )
            )
            radius = float(strength * sigma_value)
            if self.optimizer_guide_sample_clip_ratio > 0.0:
                radius = min(
                    radius,
                    float(
                        self.optimizer_guide_sample_clip_ratio
                        * max(1e-12, self.ub - self.lb)
                    ),
                )
            source_valid[agent_id] = bool(
                eligible
                and guide_norms[agent_id] > 1e-12
                and np.isfinite(radius)
                and radius > 0.0
            )
            if source_valid[agent_id]:
                radii[agent_id] = radius

        geometry_payload = np.concatenate(
            [
                bases,
                guide_units,
                radii[:, None],
                source_valid.astype(np.float64)[:, None],
            ],
            axis=1,
        )
        geometry_views = self._collective_all_gather(geometry_payload)
        canonical = geometry_views[0]
        shared_base = np.mean(canonical[:, : self.D], axis=0)
        gathered_units = canonical[:, self.D : 2 * self.D]
        gathered_radius = canonical[:, 2 * self.D]
        gathered_valid = canonical[:, 2 * self.D + 1] > 0.5
        direction_raw = np.mean(gathered_units[gathered_valid], axis=0) if np.any(
            gathered_valid
        ) else np.zeros((self.D,), dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction_raw))
        shared_direction = (
            direction_raw / direction_norm
            if np.isfinite(direction_norm) and direction_norm > 1e-12
            else np.zeros((self.D,), dtype=np.float64)
        )
        shared_radius = float(np.median(gathered_radius[gathered_valid])) if np.any(
            gathered_valid
        ) else 0.0
        plus_x = np.clip(
            shared_base + shared_radius * shared_direction,
            self.lb,
            self.ub,
        )
        minus_x = np.clip(
            shared_base - shared_radius * shared_direction,
            self.lb,
            self.ub,
        )
        geometry_valid = bool(
            np.any(gathered_valid)
            and direction_norm > 1e-12
            and np.isfinite(shared_radius)
            and shared_radius > 0.0
            and max(
                float(np.linalg.norm(plus_x - shared_base)),
                float(np.linalg.norm(minus_x - shared_base)),
            )
            > 1e-15
        )

        votes = np.zeros((self.n_agents,), dtype=np.int64)
        vote_valid = np.zeros((self.n_agents,), dtype=bool)
        probe_evals = 0
        if geometry_valid:
            points = np.stack([shared_base, plus_x, minus_x], axis=0)
            for agent_id in range(self.n_agents):
                values = np.asarray(
                    self.fun.local_eval_batch(agent_id, points),
                    dtype=np.float64,
                ).reshape(-1)
                probe_evals += 3
                if values.shape != (3,) or not np.all(np.isfinite(values)):
                    continue
                base_y, plus_y, minus_y = [float(value) for value in values]
                vote_valid[agent_id] = True
                if plus_y < base_y and plus_y < minus_y:
                    votes[agent_id] = 1
                elif minus_y < base_y and minus_y < plus_y:
                    votes[agent_id] = -1

        vote_payload = np.stack(
            [votes.astype(np.float64), vote_valid.astype(np.float64)],
            axis=1,
        )
        vote_views = self._collective_all_gather(vote_payload)
        canonical_votes = vote_views[0]
        final_votes = canonical_votes[:, 0].astype(np.int64)
        final_vote_valid = canonical_votes[:, 1] > 0.5
        vote_sum = int(np.sum(final_votes[final_vote_valid]))
        hypothetical_veto = bool(geometry_valid and vote_sum < 0)
        null_veto = bool(geometry_valid and self._collective_null_veto())
        actual_veto = bool(
            (self.collective_guide_mode == "collective_veto" and hypothetical_veto)
            or (
                self.collective_guide_mode == "rate_matched_null"
                and null_veto
            )
        )

        self.last_collective_guide_shared_base[:] = shared_base[None, :]
        self.last_collective_guide_direction[:] = shared_direction[None, :]
        self.last_collective_guide_radius.fill(shared_radius if geometry_valid else 0.0)
        self.last_collective_guide_vote[:] = final_votes
        self.last_collective_guide_source_valid[:] = gathered_valid.astype(np.float64)
        self.last_collective_guide_vote_valid[:] = final_vote_valid.astype(np.float64)
        self.last_collective_guide_valid = bool(geometry_valid)
        self.last_collective_guide_vote_sum = int(vote_sum)
        self.last_collective_guide_hypothetical_veto = bool(hypothetical_veto)
        self.last_collective_guide_null_veto = bool(null_veto)
        self.last_collective_guide_actual_veto = bool(actual_veto)
        self.collective_guide_events += 1
        self.collective_guide_valid_events += int(geometry_valid)
        self.collective_guide_hypothetical_veto_events += int(hypothetical_veto)
        self.collective_guide_actual_veto_events += int(actual_veto)
        self.collective_guide_probe_local_evals += int(probe_evals)
        return int(probe_evals)

    def _reset_guide_replacement_event(self) -> None:
        self.guide_replacement_direction.fill(0.0)
        self.last_guide_replacement_eligible.fill(0.0)
        self.last_guide_replacement_valid.fill(0.0)
        self.last_guide_replacement_old_guide_suppressed.fill(0.0)
        self.last_guide_replacement_sign.fill(0)
        self.last_guide_replacement_strength.fill(0.0)
        self.last_guide_replacement_sigma.fill(0.0)
        self.last_guide_replacement_requested_radius.fill(0.0)
        self.last_guide_replacement_applied_radius.fill(0.0)
        self.last_guide_replacement_plus_gain.fill(0.0)
        self.last_guide_replacement_minus_gain.fill(0.0)

    def _run_guide_replacement_event(
        self,
        commit_base_states: np.ndarray,
        commit_base_local_f: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        bases = np.asarray(commit_base_states, dtype=np.float64).reshape(
            self.n_agents, self.D
        )
        base_local = np.asarray(
            commit_base_local_f, dtype=np.float64
        ).reshape(self.n_agents)
        if not self.guide_replacement_enabled:
            return bases.copy(), base_local.copy(), 0
        if not self.local_only_information:
            raise RuntimeError(
                "Guide replacement crossed the strict local-only boundary."
            )

        next_states = bases.copy()
        next_local = base_local.copy()
        probe_evals = 0
        self.guide_replacement_events += 1
        step_max = None
        if self.optimizer_guide_sample_clip_ratio > 0.0:
            step_max = float(
                self.optimizer_guide_sample_clip_ratio
                * max(1e-12, self.ub - self.lb)
            )

        for receiver in range(self.n_agents):
            if self.last_guide_replacement_eligible[receiver] <= 0.0:
                continue
            self.guide_replacement_eligible_count += 1
            direction = np.asarray(
                self.guide_replacement_direction[receiver], dtype=np.float64
            )
            norm = float(np.linalg.norm(direction))
            strength = float(self.last_guide_replacement_strength[receiver])
            sigma = float(self.last_guide_replacement_sigma[receiver])
            requested_radius = float(strength * sigma)
            if step_max is not None:
                requested_radius = float(min(requested_radius, step_max))
            valid = bool(
                np.isfinite(norm)
                and norm > 1e-12
                and np.isfinite(requested_radius)
                and requested_radius > 0.0
            )
            if valid:
                direction = direction / norm
                plus_x = np.clip(
                    bases[receiver] + requested_radius * direction,
                    self.lb,
                    self.ub,
                )
                minus_x = np.clip(
                    bases[receiver] - requested_radius * direction,
                    self.lb,
                    self.ub,
                )
                plus_shift = float(np.linalg.norm(plus_x - bases[receiver]))
                minus_shift = float(np.linalg.norm(minus_x - bases[receiver]))
                valid = max(plus_shift, minus_shift) > 1e-15
            else:
                plus_x = bases[receiver].copy()
                minus_x = bases[receiver].copy()
                plus_shift = 0.0
                minus_shift = 0.0

            plus_y = float(
                np.asarray(
                    self.fun.local_eval_batch(receiver, plus_x),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            minus_y = float(
                np.asarray(
                    self.fun.local_eval_batch(receiver, minus_x),
                    dtype=np.float64,
                ).reshape(-1)[0]
            )
            probe_evals += 2
            plus_gain = _safe_log_improvement(
                float(base_local[receiver]), plus_y
            )
            minus_gain = _safe_log_improvement(
                float(base_local[receiver]), minus_y
            )
            sign = 0
            if valid:
                if plus_gain > 0.0 and plus_gain > minus_gain + 1e-12:
                    sign = 1
                elif minus_gain > 0.0 and minus_gain > plus_gain + 1e-12:
                    sign = -1

            self.last_guide_replacement_valid[receiver] = 1.0 if valid else 0.0
            self.last_guide_replacement_sign[receiver] = int(sign)
            self.last_guide_replacement_requested_radius[receiver] = (
                requested_radius if valid else 0.0
            )
            self.last_guide_replacement_plus_gain[receiver] = plus_gain
            self.last_guide_replacement_minus_gain[receiver] = minus_gain
            if valid:
                self.guide_replacement_valid_event_count += 1
            if sign > 0:
                self.guide_replacement_forward_count += 1
            elif sign < 0:
                self.guide_replacement_reverse_count += 1
            else:
                self.guide_replacement_noop_count += 1

            if self.guide_replacement_mode == "p1" and sign != 0:
                if sign > 0:
                    next_states[receiver] = plus_x
                    next_local[receiver] = plus_y
                    applied_radius = plus_shift
                else:
                    next_states[receiver] = minus_x
                    next_local[receiver] = minus_y
                    applied_radius = minus_shift
                self.last_guide_replacement_applied_radius[receiver] = float(
                    applied_radius
                )
                self.guide_replacement_commit_count += 1

        self.guide_replacement_probe_local_evals += int(probe_evals)
        return next_states, next_local, int(probe_evals)

    def _optimizer_guide_options(
        self,
        agent_id: int,
        optimizer_name: str,
        collab_mode_idx: int = 0,
        guide_scale_idx: int = 0,
        sigma_value: float = None,
    ) -> Dict:
        enabled = self._optimizer_guide_enabled_for(optimizer_name)
        guide = np.asarray(self.optimizer_guide_direction[int(agent_id)], dtype=np.float64)
        mode_name = self._collab_mode_name(collab_mode_idx)
        actor_scale = self._guide_scale_value(guide_scale_idx)
        collab_multiplier = 1.0
        leader_active = False
        if mode_name == "self":
            enabled = False
            collab_multiplier = 0.0
        elif mode_name == "leader":
            leader_guide, leader_active = self._leader_guide_direction(int(agent_id))
            if leader_active:
                guide = leader_guide
            elif self.collab_leader_fallback == "off":
                enabled = False
                collab_multiplier = 0.0
        elif mode_name == "soft_diversify":
            collab_multiplier = float(self.collab_soft_diversify_scale)
        norm = float(np.linalg.norm(guide))
        strength, scale, metric = self._optimizer_guide_effective_strength()
        total_scale = float(scale * actor_scale * collab_multiplier)
        strength = float(max(0.0, strength * actor_scale * collab_multiplier))
        mix_strength = float(
            max(0.0, self.optimizer_guide_mix_strength * scale * actor_scale * collab_multiplier)
        )
        aid = int(agent_id)
        self.last_optimizer_guide_strength_effective = strength
        self.last_optimizer_guide_strength_scale = total_scale
        self.last_guide_scale_value[aid] = float(actor_scale)
        self.last_collab_leader_active[aid] = 1.0 if leader_active else 0.0
        self.last_collab_strength_multiplier[aid] = float(collab_multiplier)
        if self.guide_replacement_enabled:
            eligible = self._guide_replacement_eligible_for(optimizer_name)
            self.last_guide_replacement_eligible[aid] = 1.0 if eligible else 0.0
            self.guide_replacement_direction[aid] = np.nan_to_num(
                guide,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            self.last_guide_replacement_strength[aid] = float(strength)
            sigma = self.last_sigma_value[aid] if sigma_value is None else sigma_value
            self.last_guide_replacement_sigma[aid] = float(
                sigma if np.isfinite(sigma) and sigma > 0.0 else 0.0
            )
            if eligible and self.guide_replacement_mode in {"p0", "p1"}:
                self.last_guide_replacement_old_guide_suppressed[aid] = 1.0
                enabled = False
        if (
            self.collective_guide_enabled
            and self.last_collective_guide_actual_veto
            and enabled
        ):
            self.last_collective_guide_suppressed[aid] = 1.0
            enabled = False
        if not enabled or norm <= 1e-12 or strength <= 0.0:
            self.last_optimizer_guide_applied[aid] = 0.0
            return {"optimizer_guide_enable": False}
        self.last_optimizer_guide_applied[aid] = 1.0
        guide_options = {
            "optimizer_guide_enable": True,
            "optimizer_guide_direction": guide.copy(),
            "optimizer_guide_strength": float(strength),
            "optimizer_guide_mix_strength": float(mix_strength),
            "optimizer_guide_strength_scale": float(total_scale),
            "optimizer_guide_schedule_metric": float(metric),
            "optimizer_guide_injection_pairs": int(self.optimizer_guide_injection_pairs),
            "optimizer_guide_use_negative_pair": bool(
                self.optimizer_guide_use_negative_pair
            ),
            "optimizer_guide_numeric_guard": bool(self.optimizer_guide_numeric_guard),
            "optimizer_guide_sigma_exp_clip": float(self.optimizer_guide_sigma_exp_clip),
            "optimizer_guide_sigma_clip_ratio": float(
                self.optimizer_guide_sigma_clip_ratio
            ),
            "optimizer_guide_sample_clip_ratio": float(
                self.optimizer_guide_sample_clip_ratio
            ),
        }
        if self._optimizer_guide_internal_enabled_for(optimizer_name):
            guide_options.update(
                {
                    "optimizer_guide_internal_mode": str(self.optimizer_guide_internal_mode),
                    "optimizer_guide_internal_mean_lr": float(
                        self.optimizer_guide_internal_mean_lr
                    ),
                    "optimizer_guide_internal_path_lr": float(
                        self.optimizer_guide_internal_path_lr
                    ),
                    "optimizer_guide_internal_cov_lr": float(
                        self.optimizer_guide_internal_cov_lr
                    ),
                    "optimizer_guide_internal_agree_cos_min": float(
                        self.optimizer_guide_internal_agree_cos_min
                    ),
                    "optimizer_guide_internal_max_step_ratio": float(
                        self.optimizer_guide_internal_max_step_ratio
                    ),
                    "optimizer_guide_internal_max_rel_step": float(
                        self.optimizer_guide_internal_max_rel_step
                    ),
                    "optimizer_guide_internal_path_max_rel_norm": float(
                        self.optimizer_guide_internal_path_max_rel_norm
                    ),
                    "optimizer_guide_internal_cov_rank1_clip": float(
                        self.optimizer_guide_internal_cov_rank1_clip
                    ),
                    "optimizer_guide_internal_disable_sample_injection": bool(
                        self.optimizer_guide_internal_disable_sample_injection
                    ),
                }
            )
        return guide_options

    def _apply_optimizer_guide_gate(
        self,
        guide_units: np.ndarray,
        source_norms: np.ndarray,
        proposal_directions: np.ndarray,
    ) -> np.ndarray:
        guide_units = np.asarray(guide_units, dtype=np.float64).reshape(self.n_agents, self.D)
        source_norms = np.asarray(source_norms, dtype=np.float64).reshape(self.n_agents)
        prop_units, prop_norms = self._safe_unit_vectors(proposal_directions)
        align = np.sum(prop_units * guide_units, axis=1)
        align[prop_norms <= 1e-12] = 0.0
        align = np.nan_to_num(align, nan=0.0, posinf=0.0, neginf=0.0)
        self.last_optimizer_guide_alignment = align
        if not self.optimizer_guide_gate_enable:
            self.last_optimizer_guide_source_active_ratio = float(np.mean(source_norms > 1e-12))
            return guide_units
        active = source_norms > max(float(self.optimizer_guide_min_norm), 1e-12)
        active &= align >= float(self.optimizer_guide_min_alignment)
        gated = guide_units.copy()
        gated[~active] = 0.0
        self.last_optimizer_guide_source_active_ratio = float(np.mean(active))
        return gated

    def _update_optimizer_guide_direction(
        self,
        base_states: np.ndarray,
        proposal_states: np.ndarray,
        local_improvements: np.ndarray,
    ) -> None:
        if not self.optimizer_guide_enable:
            self.optimizer_guide_direction.fill(0.0)
            self.last_optimizer_guide_norm.fill(0.0)
            self.last_optimizer_guide_alignment.fill(0.0)
            self.last_optimizer_guide_source_active_ratio = 0.0
            return
        if self.optimizer_guide_source == "ccsa_direction_momentum":
            guide, norms = self._safe_unit_vectors(self.ccsa_direction_momentum)
            directions = (
                np.asarray(proposal_states, dtype=np.float64).reshape(self.n_agents, self.D)
                - np.asarray(base_states, dtype=np.float64).reshape(self.n_agents, self.D)
            )
            guide = self._apply_optimizer_guide_gate(guide, norms, directions)
            self.optimizer_guide_direction = guide
            self.last_optimizer_guide_norm = norms
            return

        if self.consensus_weight is None:
            raise ValueError("optimizer guide requires a resolved consensus graph.")
        base = np.asarray(base_states, dtype=np.float64).reshape(self.n_agents, self.D)
        props = np.asarray(proposal_states, dtype=np.float64).reshape(self.n_agents, self.D)
        directions = props - base
        direction_units, _ = self._safe_unit_vectors(directions)
        improvements = np.nan_to_num(
            np.asarray(local_improvements, dtype=np.float64).reshape(self.n_agents),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        strengths = np.maximum(improvements - float(self.optimizer_guide_min_improve), 0.0)
        weighted_signal = direction_units * strengths[:, None]
        guide_raw = self.consensus_weight @ weighted_signal
        guide_units, guide_norms = self._safe_unit_vectors(guide_raw)
        fallback_units, _ = self._safe_unit_vectors(self.consensus_weight @ direction_units)
        empty_rows = guide_norms <= 1e-12
        if np.any(empty_rows):
            guide_units[empty_rows] = fallback_units[empty_rows]
        guide_units = np.nan_to_num(guide_units, nan=0.0, posinf=0.0, neginf=0.0)
        guide_units = self._apply_optimizer_guide_gate(
            guide_units,
            np.linalg.norm(guide_raw, axis=1),
            directions,
        )
        self.optimizer_guide_direction = guide_units
        self.last_optimizer_guide_norm = np.linalg.norm(guide_raw, axis=1)

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
        _, step_norms = self._safe_unit_vectors(directions)
        self._update_ccsa_direction_momentum(
            base_states=base,
            proposals=props,
            local_improvements=local_improvements,
        )

        # Keep zero local movement at zero, even when the cooperative scale is non-one.
        adjusted = base + self.last_ccsa_scale[:, None] * directions
        adjusted[step_norms <= 1e-12] = props[step_norms <= 1e-12]
        self.ccsa_lite_rounds += 1
        return adjusted

    def _update_ccsa_direction_momentum(
        self,
        base_states: np.ndarray,
        proposals: np.ndarray,
        local_improvements: np.ndarray,
    ) -> None:
        if self.ccsa_momentum_update_mode == "off":
            self.ccsa_direction_momentum.fill(0.0)
            self.last_ccsa_scale.fill(1.0)
            return
        base = np.asarray(base_states, dtype=np.float64).reshape(self.n_agents, self.D)
        props = np.asarray(proposals, dtype=np.float64).reshape(self.n_agents, self.D)
        directions = props - base
        direction_units, _ = self._safe_unit_vectors(directions)
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

    def _normalize_candidate_x(
        self,
        raw_x,
        agent_id: int,
        x_fallback: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, int]:
        """
        Normalize optimizer outputs to one full decision vector of length D.
        """
        arr = np.asarray(raw_x, dtype=np.float64)
        if arr.size == self.D:
            x = arr.reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, False, 1

        if arr.ndim >= 2 and arr.shape[-1] == self.D:
            candidates = arr.reshape(-1, self.D)
        elif arr.size > 0 and arr.size % self.D == 0:
            candidates = arr.reshape(-1, self.D)
        else:
            x = np.asarray(x_fallback, dtype=np.float64).reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, True, 1

        candidates = np.clip(candidates, self.lb, self.ub)
        vals = np.asarray(self.fun.local_eval_batch(agent_id, candidates), dtype=np.float64).reshape(-1)
        if vals.size != candidates.shape[0] or not np.any(np.isfinite(vals)):
            x = np.asarray(x_fallback, dtype=np.float64).reshape(self.D)
            y = float(np.asarray(self.fun.local_eval_batch(agent_id, x), dtype=np.float64).reshape(-1)[0])
            return np.clip(x, self.lb, self.ub), y, True, int(candidates.shape[0] + 1)

        vals = np.nan_to_num(vals, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        idx = int(np.argmin(vals))
        return candidates[idx].copy(), float(vals[idx]), True, int(candidates.shape[0])

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

    def _build_local_only_base_obs(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.base_obs_dim), dtype=np.float32)
        rem_ratio = max(
            0.0,
            (self.max_fes - self.sum_fes) / max(1, self.max_fes),
        )
        sigma_norm = self._sigma_log_norm_values()
        actual_fes_norm = self._last_actual_fes_step_norm()
        comm_or_fes_norm = self._comm_round_value_norm()
        resource_norm = self._last_resource_factor_norm()
        stagnation = self._own_stagnation_ratio()
        best_gain = self._own_best_gain()

        for i in range(self.n_agents):
            proposal_last = float(self.last_local_improve[i])
            proposal_mean = (
                float(np.mean(self.local_improve_hist[i][-5:]))
                if self.local_improve_hist[i]
                else 0.0
            )
            committed_last = float(self.last_committed_local_improve[i])
            committed_mean = (
                float(np.mean(self.committed_local_improve_hist[i][-5:]))
                if self.committed_local_improve_hist[i]
                else 0.0
            )
            opt_ratio = self.last_optimizer_idx[i] / max(
                1,
                len(self.optimizer_candidates) - 1,
            )

            obs[i, 0] = float(stagnation[i])
            obs[i, 1] = float(sigma_norm[i])
            obs[i, 2] = proposal_last
            obs[i, 3] = float(actual_fes_norm[i])
            obs[i, 4] = float(rem_ratio)
            obs[i, 5] = float(comm_or_fes_norm[i])
            obs[i, 6] = proposal_mean
            obs[i, 7] = committed_last
            obs[i, 8] = float(best_gain[i])
            obs[i, 9] = float(opt_ratio)
            obs[i, 10] = float(resource_norm[i])
            obs[i, 11] = float(self.last_local_commit_success[i])
            obs[i, 12] = float(self.last_consensus_local_effect[i])
            obs[i, 13] = float(committed_last - committed_mean)
            obs[i, 14] = float(self.last_consensus_shift_norm[i])
            obs[i, 15] = float(i) / float(max(1, self.n_agents - 1))

        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _build_objective_split_base_obs(self) -> np.ndarray:
        if self.local_only_information:
            return self._build_local_only_base_obs()
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

    def _update_sigma_state_obs(
        self, agent_id: int, applied: bool, cur_sigma: float, reset_reason: str
    ) -> None:
        if not self.sigma_state_obs_enable:
            return
        aid = int(agent_id)
        # E8-SSO 重设计（19/21 卡）：本函数只维护 σ 继承链长度（age）——
        # 即当前 slot 中的 σ 已经连续被继承了多少个事件，是唯一在“σ 每事件
        # 被重写”的前提下仍有意义的“年龄”。原 last_applied 与 age>0 冗余、
        # 原 log_delta 自比恒零、原 reset_code 描述上一事件请求结果，均已移除；
        # 其余状态维度改在 _sigma_inherit_obs_block 内由当前 slot 与 preview
        # 同一时间基准计算。
        if applied:
            self.sigma_state_age[aid] += 1.0
        else:
            self.sigma_state_age[aid] = 0.0

    def _build_optimizer_options(
        self,
        agent_id: int,
        optimizer_name: str,
        cfg_levels: List[int],
        dims: List[int],
        x_base: np.ndarray,
        subfes_i: int,
        seed: int,
    ) -> Dict:
        # σ 继承读取点（14 卡合同，叠加在既有选项构建之上）：
        # 仅当开关 ON 且 resolve 后 profile == "inherit" 时，同车（上一事件与
        # 本事件同一 optimizer）且槽非空且签名（维度/bounds/population）相符
        # 且 σ 有限为正 → 以演化后 σ 替换 options["sigma"]；否则清槽、保留
        # super 的旧解析（配置档缓存旧语义原样）。OFF 时直接返回 super 结果，
        # 与旧实现逐位一致。
        options = super()._build_optimizer_options(
            agent_id, optimizer_name, cfg_levels, dims, x_base, subfes_i, seed
        )
        aid = int(agent_id)
        name = str(optimizer_name).lower()
        if name == "mmes":
            options["mmes_state_transition_mode"] = self.mmes_state_transition_mode
            if self.mmes_ratio_success_mode != "native":
                options["mmes_ratio_success_mode"] = self.mmes_ratio_success_mode
                options["mmes_ratio_success_metric"] = self.mmes_ratio_success_metric
                options["mmes_ratio_success_strength"] = self.mmes_ratio_success_strength
            if self.mmes_ratio_direction_mode != "native":
                options["mmes_ratio_direction_mode"] = self.mmes_ratio_direction_mode
                options["mmes_ratio_direction_strength"] = self.mmes_ratio_direction_strength
        if name in {"cmaes", "sepcmaes"} and self.cma_sep_ratio_path_mode != "native":
            options["cma_sep_ratio_path_mode"] = self.cma_sep_ratio_path_mode
            options["cma_sep_ratio_path_strength"] = self.cma_sep_ratio_path_strength
            if name == "cmaes" and self.cmaes_ratio_path_metric != "rms":
                options["cmaes_ratio_path_metric"] = self.cmaes_ratio_path_metric
            if name == "sepcmaes" and self.sepcmaes_ratio_path_metric != "rms":
                options["sepcmaes_ratio_path_metric"] = self.sepcmaes_ratio_path_metric
        if name == "vkd":
            options["vkd_boundary_update_mode"] = self.vkd_boundary_update_mode
            mode = getattr(self, "vkd_ps_outlet_mode", "native")
            if mode != "native":
                options["vkd_ps_outlet_mode"] = mode
            ratio_mode = getattr(self, "vkd_ratio_ps_mode", "native")
            if ratio_mode != "native":
                options["vkd_ratio_ps_mode"] = ratio_mode
                options["vkd_ratio_ps_strength"] = self.vkd_ratio_ps_strength
            if any(int(getattr(self.opts, flag, 0)) for flag in (
                "eval_save_event_slot_diagnostics", "eval_save_optimizer_forensic_trace",
            )):
                options["vkd_record_state"] = True
            if int(getattr(self.opts, "eval_save_vkd_state_trace", 0)):
                options["vkd_record_detail"] = True
        if "nonfinite_dump_path" not in options:
            dump_dir = getattr(self.opts, "data_save_dir", None) or getattr(
                self.opts, "test_dir", None
            )
            if dump_dir:
                options["nonfinite_dump_path"] = os.path.abspath(
                    os.path.join(str(dump_dir), "nonfinite_geometry_dump.jsonl")
                )
        if "optimizer_numeric_forensics_enable" not in options:
            # Diagnostic-only forensic recorder options; None of these keys change the
            # optimizer control path, they only tell the session where to flush windows.
            options["optimizer_numeric_forensics_enable"] = bool(
                self.optimizer_numeric_forensics_enable
            )
            forensics_dir = str(self.optimizer_numeric_forensics_dir).strip()
            if not forensics_dir:
                run_dir = getattr(self.opts, "data_save_dir", None) or getattr(
                    self.opts, "test_dir", None
                )
                forensics_dir = os.path.abspath(str(run_dir)) if run_dir else ""
            prev_optimizers = getattr(self, "_sigma_inherit_prev_optimizer", None)
            if not isinstance(prev_optimizers, (list, tuple)):
                prev_optimizers = []
            prev_optimizer = (
                str(prev_optimizers[aid])
                if 0 <= aid < len(prev_optimizers) and prev_optimizers[aid]
                else ""
            )
            options["optimizer_numeric_forensics_dir"] = forensics_dir
            options["optimizer_numeric_forensics_jump_log10"] = float(
                self.optimizer_numeric_forensics_jump_log10
            )
            options["optimizer_numeric_forensics_jump_max_windows"] = int(
                self.optimizer_numeric_forensics_jump_max_windows
            )
            options["optimizer_numeric_forensics_jump_alpha_gate"] = bool(
                self.optimizer_numeric_forensics_jump_alpha_gate
            )
            options["optimizer_numeric_forensics_jump_early_write"] = bool(
                self.optimizer_numeric_forensics_jump_early_write
            )
            options["optimizer_numeric_forensics_jump_early_milestone_log10"] = float(
                self.optimizer_numeric_forensics_jump_early_milestone_log10
            )
            options["optimizer_numeric_forensics_jump_mid_milestone_log10"] = float(
                self.optimizer_numeric_forensics_jump_mid_milestone_log10
            )
            if self.optimizer_numeric_forensics_target_function >= 0:
                options["optimizer_numeric_forensics_target_function"] = int(
                    self.optimizer_numeric_forensics_target_function
                )
            if self.optimizer_numeric_forensics_target_seed >= 0:
                options["optimizer_numeric_forensics_target_seed"] = int(
                    self.optimizer_numeric_forensics_target_seed
                )
            if self.optimizer_numeric_forensics_target_agent >= 0:
                options["optimizer_numeric_forensics_target_agent"] = int(
                    self.optimizer_numeric_forensics_target_agent
                )
            options["optimizer_numeric_forensics_context"] = {
                "problem_family": str(self.problem_family),
                "function_id": int(self.question),
                "env_step": int(self.step_count),
                "agent_id": int(aid),
                "optimizer": str(name),
                "subfes": int(subfes_i),
                "seed": int(seed),
                # Per-step optimizer RNG seed above; run_seed is the run-level seed
                # (training seed or evaluation seed) and is what target filters use.
                "run_seed": int(getattr(self.opts, "seed", -1)),
                "cfg_levels": [int(x) for x in np.asarray(cfg_levels).reshape(-1)],
                "sigma_inherit_enable": bool(
                    getattr(self, "sigma_inherit_enable", False)
                ),
                "sigma_inherit_previous_optimizer": prev_optimizer,
            }
        diag = self.last_sigma_diagnostics
        diag["current_optimizer"][aid] = name
        diag["previous_optimizer"][aid] = str(self._sigma_inherit_prev_optimizer[aid] or "")
        diag["sigma_used"][aid] = float(options.get("sigma", 0.0))
        if not self.sigma_inherit_enable:
            diag["reset_reason"][aid] = "disabled"
            return options
        raw_profile = self._profile_name(int(cfg_levels[0]))
        prof = self._resolve_profile_name(agent_id, optimizer_name, int(cfg_levels[0]))
        diag["resolved_profile"][aid] = str(raw_profile)
        diag["requested_profile"][aid] = str(raw_profile)
        if raw_profile != "numeric_inherit":
            reason = "legacy_inherit" if raw_profile == "inherit" else "not_numeric_inherit"
            diag["reset_reason"][aid] = reason
            self._update_sigma_state_obs(aid, False, 0.0, reason)
            return options
        slot = self._sigma_inherit_slots[aid]
        if slot is not None:
            prev_name, s, sig = slot
            lam = int(
                options.get(
                    "n_individuals",
                    options.get(
                        "lam", 4 + int(3 * np.log(max(2, self.D)))
                    ),
                )
            )
            if (
                prev_name == name
                and sig is not None
                and int(sig[0]) == int(self.D)
                and float(sig[1]) == float(self.lb)
                and float(sig[2]) == float(self.ub)
                and int(sig[3]) == int(lam)
                and np.isfinite(float(s))
                and float(s) > 0
            ):
                if self.sigma_validity_gate_enable:
                    shift_m = self.last_event_slot_center_shift_norms
                    scale_m = self.last_event_slot_commit_audit.get(
                        "effective_scale_rms"
                    )
                    shift_col = (
                        np.asarray(shift_m[:, aid], dtype=np.float64)
                        if shift_m.size and shift_m.shape[1] > aid
                        else np.zeros((0,), dtype=np.float64)
                    )
                    scale_col = np.zeros((0,), dtype=np.float64)
                    if scale_m is not None:
                        scale_arr = np.asarray(scale_m, dtype=np.float64)
                        if scale_arr.size and scale_arr.shape[1] > aid:
                            scale_col = np.asarray(
                                scale_arr[:, aid], dtype=np.float64
                            )
                    n_slot = int(min(shift_col.size, scale_col.size))
                    max_shift = (
                        float(np.max(shift_col)) if shift_col.size else 0.0
                    )
                    eff_scale = (
                        float(np.max(scale_col)) if scale_col.size else 0.0
                    )
                    # 量纲一致比较：逐 slot 用 commit 中心位移的 L2 范数除以该 slot
                    # 的 optimizer 坐标空间有效尺度（sigma × covariance/对角轴 RMS），
                    # 取最大比值判 stale。位移与尺度必须来自同一 slot，否则比值不
                    # 对应任何真实事件。旧实现拿 D 维 L2 范数直接比标量 sigma，量纲
                    # 不一致，收敛后 sigma 很小时会把几乎全部继承判成 stale。
                    dim_scale = float(np.sqrt(max(1, int(self.D))))
                    if n_slot > 0:
                        usable = (
                            np.isfinite(shift_col[:n_slot])
                            & np.isfinite(scale_col[:n_slot])
                            & (scale_col[:n_slot] > 0.0)
                        )
                    else:
                        usable = np.zeros((0,), dtype=bool)
                    if n_slot > 0 and bool(np.any(usable)):
                        # 只让“尺度真实为正且有限”的 slot 参与判定：零尺度槽若
                        # 用 1e-12 兜底会把比值放大成天文数字而误判 stale，非有限
                        # 值则会让 NaN > ratio 恒为 False 而静默放行。两者都属
                        # “无证据”，应从比值中剔除，而不是当作极端证据。
                        # 量纲：位移列是 D 维 L2 范数，尺度列是逐坐标 RMS，必须
                        # 先把位移除以 sqrt(D) 换成 RMS 再比；否则比值被放大
                        # sqrt(D) 倍，阈值会退化成“几乎全拒”。首轮修订正是在真实
                        # 冒烟（E6e/F5，21120 行）上暴露：stale 94.8%、继承 0 次。
                        shift_rms = shift_col[:n_slot][usable] / dim_scale
                        gate_ratio = float(
                            np.max(shift_rms / scale_col[:n_slot][usable])
                        )
                        stale = bool(
                            gate_ratio
                            > float(self.sigma_validity_gate_commit_ratio)
                        )
                    else:
                        # 无有效尺度时退回 sigma：commit_rms > ratio × sigma
                        # 等价于 commit_norm > ratio × sigma × sqrt(D)。
                        stale = bool(
                            max_shift
                            > float(self.sigma_validity_gate_commit_ratio)
                            * float(s)
                            * dim_scale
                        )
                    diag["stale_gate_shift"][aid] = float(
                        max_shift / dim_scale
                    )
                    diag["stale_gate_scale"][aid] = float(eff_scale)
                    if stale:
                        self._sigma_inherit_slots[aid] = None
                        diag["reset_reason"][aid] = "stale_commit"
                        self._update_sigma_state_obs(aid, False, 0.0, "stale_commit")
                        return options
                diag["signature_match"][aid] = 1
                diag["sigma_preview"][aid] = float(s)
                options["sigma"] = float(max(1e-12, float(s)))
                diag["sigma_used"][aid] = float(options["sigma"])
                diag["inherit_applied"][aid] = 1
                diag["reset_reason"][aid] = ""
                self._update_sigma_state_obs(aid, True, float(s), "")
                return options
        self._sigma_inherit_slots[aid] = None
        diag["reset_reason"][aid] = "empty_or_mismatch"
        self._update_sigma_state_obs(aid, False, 0.0, "empty_or_mismatch")
        return options

    def _sigma_inherit_obs_block(self) -> np.ndarray:
        # 观测 +5 维：上一事件 optimizer one-hot ×4 + numeric_inherit 预览 σ ×1。
        # preview 是环境提示，不保证 numeric_inherit 最终通过 signature 检查。
        # E8-SSO 重设计（19/21 卡）：追加 sigma 链龄 / 相对默认尺度的 log-delta /
        # 当前候选有效性码 共 3 维；三者与 preview 在本次观测内、同一时间基准计算。
        state_dim = 3 if self.sigma_state_obs_enable else 0
        block = np.zeros((self.n_agents, 5 + state_dim), dtype=np.float32)
        for aid in range(self.n_agents):
            prev = self._sigma_inherit_prev_optimizer[aid]
            if prev is not None:
                try:
                    idx = int(self.optimizer_candidates.index(str(prev).lower()))
                    block[aid, idx] = 1.0
                except ValueError:
                    pass
            cur_name = None
            raw_idx = getattr(self, "last_optimizer_idx", None)
            if raw_idx is not None and len(np.asarray(raw_idx)) > aid:
                oi = int(
                    np.clip(
                        int(np.asarray(raw_idx)[aid]),
                        0,
                        len(self.optimizer_candidates) - 1,
                    )
                )
                cur_name = str(self.optimizer_candidates[oi]).lower()
            slot = self._sigma_inherit_slots[aid]
            valid = False
            if prev is None or cur_name is None:
                preview = 0.0
            elif prev == cur_name and slot is not None:
                _, s, sig = slot
                if (
                    sig is not None
                    and int(sig[0]) == int(self.D)
                    and float(sig[1]) == float(self.lb)
                    and float(sig[2]) == float(self.ub)
                    and np.isfinite(float(s))
                    and float(s) > 0
                ):
                    preview = float(s)
                    valid = True
                else:
                    preview = 0.0  # 空槽/失效
            elif prev == cur_name:
                preview = 0.0  # 同车但空槽
            else:
                # 换车 → 旧配置档规则值（super 的旧解析，默认 0.3）
                preview = float(super()._sigma_ref(aid, cur_name))
            block[aid, 4] = preview
            if self.sigma_state_obs_enable:
                ref = (
                    float(super()._sigma_ref(aid, cur_name))
                    if cur_name is not None
                    else 0.0
                )
                # 维度 5：σ 继承链龄（连续被继承的事件数），归一化到 [0,1]。
                block[aid, 5] = float(
                    np.clip(self.sigma_state_age[aid] / 20.0, 0.0, 1.0)
                )
                # 维度 6：当前候选 σ 相对该 optimizer 默认尺度的对数比。
                # 参考量是长期存在的默认档，而不是上一事件末的同一 σ，
                # 因此不再恒零，能表达“继承来的尺度偏大还是偏小”。
                if valid and ref > 0.0:
                    block[aid, 6] = float(
                        np.clip(
                            np.log(max(preview, 1e-12) / max(ref, 1e-12)),
                            -1.0,
                            1.0,
                        )
                    )
                else:
                    block[aid, 6] = 0.0
                # 维度 7：当前候选σ对“本次将要使用的 optimizer”的有效性码
                # （0 可直接继承 / 1 换车 / 2 同车空槽或失效 / 3 无候选）。
                # 与原实现的区别：原reset_code描述上一事件的请求结果，
                # 这里描述当前候选，与 preview 指向同一个 σ。
                if prev is None or cur_name is None:
                    validity_code = 3.0
                elif prev != cur_name:
                    validity_code = 1.0
                elif valid:
                    validity_code = 0.0
                else:
                    validity_code = 2.0
                block[aid, 7] = float(validity_code / 3.0)
        self.last_sigma_state_obs = block.copy()
        return block

    def _build_obs(self) -> np.ndarray:
        if self.local_only_information:
            base_obs = self._build_local_only_base_obs()
            parts = [base_obs]
            if self.sigma_inherit_enable:
                parts.append(self._sigma_inherit_obs_block())
            if self.neighbor_obs_enabled:
                parts.append(
                    self.last_neighbor_summary[
                        :,
                        : self.neighbor_obs_dim,
                    ].astype(np.float32, copy=False)
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
            if self.candidate_history_obs_enable:
                parts.append(
                    np.stack(
                        [
                            self.last_candidate_accept_ratio,
                            self.last_candidate_selection_confidence,
                            self.last_candidate_support_ratio,
                            self.last_candidate_requested_shift_norm,
                            self.last_candidate_actuator_beta,
                        ],
                        axis=1,
                    ).astype(np.float32, copy=False)
                )
            obs = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
            return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        if not self.state_comm_enabled:
            obs = super()._build_obs()
            obs[:, 0] = 1.0
            if self.neighbor_obs_enabled:
                obs[:, 16 : 16 + self.neighbor_obs_dim] = self.last_neighbor_summary[
                    :, : self.neighbor_obs_dim
                ].astype(np.float32, copy=False)
            if self.candidate_history_obs_enable:
                history = np.stack(
                    [
                        self.last_candidate_accept_ratio,
                        self.last_candidate_selection_confidence,
                        self.last_candidate_support_ratio,
                        self.last_candidate_requested_shift_norm,
                        self.last_candidate_actuator_beta,
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
                obs = np.concatenate([obs, history], axis=1)
            return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

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
        if self.candidate_history_obs_enable:
            parts.append(
                np.stack(
                    [
                        self.last_candidate_accept_ratio,
                        self.last_candidate_selection_confidence,
                        self.last_candidate_support_ratio,
                        self.last_candidate_requested_shift_norm,
                        self.last_candidate_actuator_beta,
                    ],
                    axis=1,
                ).astype(np.float32, copy=False)
            )
        obs = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        return np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _assert_no_pending_transition(self, operation: str) -> None:
        if bool(getattr(self, "_d5_transition_broken", False)):
            raise RuntimeError(
                f"Cannot {operation}: the previous staged transition failed and this "
                "environment must be reconstructed."
            )
        if getattr(self, "_pending_step_generator", None) is not None:
            raise RuntimeError(
                f"Cannot {operation} while a staged transition is pending "
                f"({getattr(self, '_pending_transition_phase', 'UNKNOWN')})."
            )

    def reset(self):
        self._assert_no_pending_transition("reset")
        self._pending_transition_phase = "READY"
        self.agent_x = np.zeros((self.n_agents, self.D), dtype=np.float64)
        initial_report_x = np.mean(self.agent_x, axis=0)
        if self.local_only_information:
            self.agent_local_f = self._eval_agent_local_states(self.agent_x)
            self.report_current_x = initial_report_x.copy()
            self.report_current_f = float("nan")
            self.report_best_x = initial_report_x.copy()
            self.report_best_f = float("inf")
            self._set_report_monitor_state(initial_report_x, float("nan"))
        else:
            self.current_x = initial_report_x.copy()
            self.current_f, self.agent_local_f = self._eval_global_and_all_local(
                self.current_x
            )
            self.gbest_x = self.current_x.copy()
            self.gbest_f = self.current_f
            self.report_current_x = self.current_x.copy()
            self.report_current_f = float(self.current_f)
            self.report_best_x = self.gbest_x.copy()
            self.report_best_f = float(self.gbest_f)
        self.agent_initial_local_f = self.agent_local_f.copy()
        self.agent_best_local_f = self.agent_local_f.copy()

        self.sum_fes = 0
        self.step_count = 0
        self.last_action_idx.fill(0)
        self.last_optimizer_idx.fill(0)
        self.last_resource_idx.fill(0)
        self.last_local_improve.fill(0.0)
        self.last_team_improve = 0.0
        self.local_improve_hist = [[] for _ in range(self.n_agents)]
        self.last_committed_local_improve.fill(0.0)
        self.committed_local_improve_hist = [
            [] for _ in range(self.n_agents)
        ]
        self.last_consensus_local_effect.fill(0.0)
        self.last_local_commit_success.fill(0.0)
        self.last_commit_success = 0.0
        self.last_joint_vs_mean_local_gap = 0.0
        self.last_local_vs_joint_gap.fill(0.0)
        self.last_local_centered.fill(0.0)
        del self.current_eval_fitness_record[:]
        del self.current_eval_individual_record[:]
        self.param_state_cache = [dict() for _ in range(self.n_agents)]
        # σ 跨事件继承（14 卡）：每 agent 一个"当前 σ"槽，槽存
        # (上一事件 optimizer 名, 演化后 σ, 签名(维度, bounds, population))；
        # 换车/签名不符/σ 无效即清槽。prev_optimizer 供观测 one-hot 使用，
        # 独立于槽的清理（"我上次选的谁"始终可读）。OFF 时两者闲置。
        self._sigma_inherit_slots = [None for _ in range(self.n_agents)]
        self._sigma_inherit_prev_optimizer = [None for _ in range(self.n_agents)]
        self.last_neighbor_summary.fill(0.0)
        self.last_sigma_value.fill(float(max(1e-12, getattr(self.opts, "sigma", 0.3))))
        self.sigma_state_age.fill(0.0)
        self.last_sigma_state_obs.fill(0.0)
        self.last_actual_fes_per_agent.fill(0.0)
        self.cumulative_actual_fes_per_agent.fill(0.0)
        self.last_cmaes_numeric_fail_soft.fill(0.0)
        self.last_cmaes_numeric_fail_soft_generation.fill(-1)
        self.cmaes_numeric_fail_soft_events = 0
        self.persistent_sepcmaes_states = [
            None for _ in range(self.n_agents)
        ]
        self.persistent_sepcmaes_signatures = [
            None for _ in range(self.n_agents)
        ]
        self.last_persistent_sepcmaes_active.fill(0.0)
        self.last_persistent_sepcmaes_fresh.fill(0.0)
        self.last_persistent_sepcmaes_config_reset.fill(0.0)
        self.last_persistent_sepcmaes_recenter_requested.fill(0.0)
        self.last_persistent_sepcmaes_recenter_applied.fill(0.0)
        self.last_persistent_sepcmaes_recenter_remaining.fill(0.0)
        self.persistent_sepcmaes_events = 0
        self.persistent_sepcmaes_fresh_events = 0
        self.persistent_sepcmaes_resume_events = 0
        self.persistent_sepcmaes_config_resets = 0
        self.target_block_field_path.fill(0.0)
        self.target_block_field_radius.fill(0.0)
        self.target_block_field_initialized.fill(False)
        self.last_target_block_field_secant.fill(0.0)
        self.last_target_block_field_secant_valid.fill(False)
        self.last_target_block_field_secant_alignment.fill(0.0)
        self.last_target_block_field_direction.fill(0.0)
        self.last_target_block_field_source_diversity.fill(0)
        self.last_target_block_field_path_norm.fill(0.0)
        self.last_target_block_field_path_alignment.fill(0.0)
        self.last_target_block_field_conflict.fill(False)
        self.last_target_block_field_radius_expand.fill(False)
        self.last_target_block_field_radius_shrink.fill(False)
        self.last_target_block_field_radius_clip_min.fill(False)
        self.last_target_block_field_radius_clip_max.fill(False)
        self.last_target_block_field_requested_commit_norm.fill(0.0)
        self.last_target_block_field_applied_commit_norm.fill(0.0)
        self.last_target_block_field_boundary_clipped.fill(False)
        self.last_target_block_field_path_injection = 0.0
        self.last_target_block_field_path_angle_degrees = 90.0
        self.target_block_field_events = 0
        self.target_block_field_local_evals = 0
        self.target_block_field_comm_rounds = 0
        self.target_block_field_messages = 0
        self.target_block_field_transmitted_floats = 0
        self.target_block_dormancy_residual_reference.fill(0.0)
        self.target_block_dormancy_reserve_radius.fill(0.0)
        self.target_block_dormancy_initialized.fill(False)
        self.target_block_dormancy_floor_age.fill(0)
        self.target_block_dormancy_stagnation_age.fill(0)
        self.target_block_dormancy_cooldown.fill(0)
        self.target_block_dormancy_activation_count.fill(0)
        self.last_target_block_dormancy_active.fill(False)
        self.last_target_block_dormancy_unresolved.fill(False)
        self.last_target_block_dormancy_material_progress.fill(False)
        self.last_target_block_dormancy_reliable_scale.fill(False)
        self.last_target_block_dormancy_residual_ratio.fill(0.0)
        self.last_target_block_dormancy_progress_ratio.fill(0.0)
        self.last_target_block_dormancy_disagreement.fill(0.0)
        self.last_target_block_dormancy_disagreement_ratio.fill(0.0)
        self.last_target_block_dormancy_restore_radius.fill(0.0)
        self.target_block_dormancy_events = 0
        self.target_block_dormancy_activations = 0
        self.last_target_block_direction_shadow_eligible.fill(False)
        self.last_target_block_direction_shadow_selected.fill(False)
        self.last_target_block_direction_shadow_candidate_valid.fill(False)
        self.last_target_block_direction_shadow_candidate_angle.fill(0.0)
        self.last_target_block_direction_shadow_candidate_response.fill(0.0)
        self.last_target_block_direction_shadow_candidate_positive.fill(False)
        self.last_target_block_direction_shadow_best_response.fill(0.0)
        self.last_target_block_direction_shadow_best_source.fill(-1)
        self.last_target_block_direction_shadow_best_sign.fill(0)
        self.last_target_block_direction_shadow_best_direction.fill(0.0)
        self.last_target_block_direction_shadow_support.fill(0.0)
        self.last_target_block_direction_shadow_conflict.fill(0.0)
        self.target_block_direction_shadow_events = 0
        self.target_block_direction_shadow_eligible_blocks = 0
        self.target_block_direction_shadow_probed_blocks = 0
        self.target_block_direction_shadow_local_evals = 0
        self.target_block_direction_shadow_comm_rounds = 0
        self.target_block_direction_shadow_messages = 0
        self.target_block_direction_shadow_transmitted_floats = 0
        self.last_target_block_challenge_selected.fill(False)
        self.last_target_block_challenge_candidate_valid.fill(False)
        self.last_target_block_challenge_candidate_angle.fill(0.0)
        self.last_target_block_challenge_candidate_score.fill(0.0)
        self.last_target_block_challenge_candidate_sign.fill(0)
        self.last_target_block_challenge_path_score.fill(0.0)
        self.last_target_block_challenge_alternative_score.fill(0.0)
        self.last_target_block_challenge_margin.fill(0.0)
        self.last_target_block_challenge_best_source.fill(-1)
        self.last_target_block_challenge_best_sign.fill(0)
        self.last_target_block_challenge_best_direction.fill(0.0)
        self.last_target_block_challenge_coverage.fill(0.0)
        self.last_target_block_challenge_positive_sources.fill(0)
        self.last_target_block_challenge_neighbor_positive_sources.fill(0)
        self.last_target_block_challenge_support.fill(0.0)
        self.last_target_block_challenge_conflict.fill(0.0)
        self.last_target_block_challenge_actuation_eligible.fill(False)
        self.last_target_block_challenge_applied.fill(False)
        self.last_target_block_challenge_applied_norm.fill(0.0)
        self.last_target_block_challenge_boundary_clipped.fill(False)
        self.target_block_challenge_events = 0
        self.target_block_challenge_challenges = 0
        self.target_block_challenge_directed_responses = 0
        self.target_block_challenge_local_evals = 0
        self.target_block_challenge_reported_evals = 0
        self.target_block_challenge_applied_commits = 0
        self.target_block_challenge_comm_rounds = 0
        self.target_block_challenge_messages = 0
        self.target_block_challenge_transmitted_floats = 0
        self.target_block_dual_clock_outer_events = 0
        self.target_block_dual_clock_local_generation_ticks = 0
        self.target_block_dual_clock_communication_ticks = 0
        self.target_block_dual_clock_commit_ticks = 0
        self.last_target_block_dual_clock_microcycles = 0
        self.last_target_block_dual_clock_generation_fes.fill(0)
        self.target_block_commit_credit_events = 0
        self.last_target_block_commit_credit_active.fill(0.0)
        self.last_target_block_commit_credit.fill(0.0)
        self.last_target_block_commit_credit_cosine.fill(0.0)
        self.last_target_block_commit_credit_proposal_norm.fill(0.0)
        self.last_target_block_commit_credit_commit_norm.fill(0.0)
        self.last_target_block_commit_credit_correction_norm.fill(0.0)
        self.last_target_block_commit_credit_path_retention.fill(1.0)
        self.last_target_block_commit_credit_scale_retention.fill(1.0)
        self.last_target_block_commit_credit_axis_rms_before.fill(0.0)
        self.last_target_block_commit_credit_axis_rms_after.fill(0.0)
        self.last_target_block_commit_credit_sigma_path_before.fill(0.0)
        self.last_target_block_commit_credit_sigma_path_after.fill(0.0)
        self.last_target_block_commit_credit_cov_path_before.fill(0.0)
        self.last_target_block_commit_credit_cov_path_after.fill(0.0)
        self.last_consensus_shift_norm.fill(0.0)
        self.last_state_neighbor_message.fill(0.0)
        self.last_state_delta_message.fill(0.0)
        self.ccsa_direction_momentum.fill(0.0)
        self.masoie_velocity.fill(0.0)
        self.rvcpd_path.fill(0.0)
        self.rvcpd_support_ema.fill(0.0)
        self.rvcpd_conflict_ema.fill(0.0)
        self.rvcpd_uncertainty_ema.fill(0.0)
        self.rvcpd_trust.fill(1.0)
        self.rvcpd_scale.fill(self.rvcpd_initial_scale)
        self.rvcpd_age.fill(0)
        self.last_rvcpd_direction.fill(0.0)
        self.last_rvcpd_sign.fill(0)
        self.last_rvcpd_active.fill(0.0)
        self.last_rvcpd_agreement.fill(0.0)
        self.last_rvcpd_requested_radius.fill(0.0)
        self.last_rvcpd_applied_radius.fill(0.0)
        self.last_rvcpd_plus_gain.fill(0.0)
        self.last_rvcpd_minus_gain.fill(0.0)
        self.last_ccsa_scale.fill(1.0)
        self.last_masoie_neighbor_pull.fill(0.0)
        self.optimizer_guide_direction.fill(0.0)
        self.last_optimizer_guide_norm.fill(0.0)
        self.last_optimizer_guide_applied.fill(0.0)
        self.last_optimizer_guide_alignment.fill(0.0)
        self.last_optimizer_guide_source_active_ratio = 0.0
        self.guide_replacement_direction.fill(0.0)
        self.last_guide_replacement_eligible.fill(0.0)
        self.last_guide_replacement_valid.fill(0.0)
        self.last_guide_replacement_old_guide_suppressed.fill(0.0)
        self.last_guide_replacement_sign.fill(0)
        self.last_guide_replacement_strength.fill(0.0)
        self.last_guide_replacement_sigma.fill(0.0)
        self.last_guide_replacement_requested_radius.fill(0.0)
        self.last_guide_replacement_applied_radius.fill(0.0)
        self.last_guide_replacement_plus_gain.fill(0.0)
        self.last_guide_replacement_minus_gain.fill(0.0)
        self._reset_collective_guide_event()
        self.last_optimizer_guide_internal_active.fill(0.0)
        self.last_optimizer_guide_internal_mean_step_norm.fill(0.0)
        self.last_optimizer_guide_internal_alignment.fill(0.0)
        self.anchor_points.fill(0.0)
        self.last_anchor_applied.fill(0.0)
        self.last_anchor_dist.fill(0.0)
        self.last_anchor_direction_norm.fill(0.0)
        self.last_optimizer_anchor_applied.fill(0.0)
        self.last_optimizer_anchor_mean_step_norm.fill(0.0)
        self.last_optimizer_anchor_sample_applied.fill(0.0)
        self.committee_direction.fill(0.0)
        self.committee_confidence.fill(0.0)
        self.committee_verified_x.fill(0.0)
        self.committee_selected_source.fill(0)
        self.committee_candidate_score.fill(0.0)
        self.committee_acceptance_active = False
        self.committee_candidate_global_f.fill(np.inf)
        self.committee_candidate_report_f = float("inf")
        self.committee_candidate_vs_report_log_improve = float("nan")
        self.committee_effective_beta.fill(0.0)
        self.committee_base_alignment.fill(0.0)
        self.last_collab_mode_idx.fill(0)
        self.last_guide_scale_idx.fill(0)
        self.last_guide_scale_value.fill(1.0)
        self.last_collab_leader_active.fill(0.0)
        self.last_collab_strength_multiplier.fill(1.0)
        self.last_consensus_metrics = self._empty_consensus_metrics()
        self.last_committee_metrics = self._empty_committee_metrics()
        self.candidate_response_verified_x.fill(0.0)
        self.candidate_response_selected_source.fill(-1)
        self.candidate_response_accepted_mask.fill(False)
        self.last_candidate_accept_ratio.fill(0.0)
        self.last_candidate_selection_confidence.fill(0.0)
        self.last_candidate_support_ratio.fill(0.0)
        self.last_candidate_requested_shift_norm.fill(0.0)
        self.last_candidate_actuator_action.fill(
            self.candidate_actuator_initial_action
            if self.candidate_actuator_action_enable
            else 0
        )
        self.last_candidate_actuator_beta.fill(
            self.candidate_actuator_initial_beta
        )
        self.candidate_response_shadow_base_x.fill(0.0)
        self.candidate_response_shadow_candidate_x.fill(0.0)
        self.candidate_response_shadow_due = False
        self.last_candidate_response_metrics = self._empty_candidate_response_metrics()
        self.candidate_multisecant_history_step.fill(0.0)
        self.candidate_multisecant_history_response.fill(0.0)
        self.candidate_multisecant_history_center.fill(0.0)
        self.candidate_multisecant_history_event_id.fill(-1)
        self.candidate_multisecant_history_valid.fill(False)
        self.candidate_multisecant_history_cursor.fill(0)
        self.candidate_multisecant_history_count.fill(0)
        self.last_candidate_multisecant_active.fill(False)
        self.last_candidate_multisecant_fallback_reason.fill(1)
        self.local_search_fes = 0
        self.candidate_validation_local_evals = 0
        self.agent_state_local_evals = 0
        self.global_monitor_local_evals = 0
        self.global_monitor_rounds = 0
        self.state_comm_rounds = 0
        self.state_comm_messages = 0
        self.state_comm_transmitted_floats = 0
        self.graph_comm_rounds = 0
        self.last_comm_rounds_applied = 0
        self.last_comm_rounds_requested = int(self.comm_rounds_per_event)
        self.last_comm_round_idx.fill(0)
        self.event_slot_interleaving_events = 0
        self.event_slot_interleaving_slots = 0
        self.event_slot_native_local_evals = 0
        self.event_slot_reported_local_evals = 0
        self.event_slot_physical_local_evals = 0
        self.last_event_slot_count = 0
        self.last_event_slot_packet_units = np.zeros(
            (0, self.n_agents), dtype=np.int64
        )
        self.last_event_slot_distribution_updates.fill(0)
        self.last_event_slot_native_evals.fill(0)
        self.last_event_slot_physical_evals.fill(0)
        self.last_event_slot_center_shift_norms = np.zeros(
            (0, self.n_agents), dtype=np.float64
        )
        self.last_event_slot_packet_improvements = np.zeros(
            (0, self.n_agents), dtype=np.float64
        )
        self.last_event_slot_commit_audit = (
            _empty_event_slot_commit_audit(0, self.n_agents)
        )
        self.last_event_slot_mmes_neutral_success.fill(0)
        self.ccsa_lite_rounds = 0
        self.masoie_lite_rounds = 0
        self.rvcpd_events = 0
        self.rvcpd_valid_event_count = 0
        self.rvcpd_commit_count = 0
        self.rvcpd_reverse_count = 0
        self.rvcpd_noop_count = 0
        self.rvcpd_probe_local_evals = 0
        self.rvcpd_messages = 0
        self.rvcpd_transmitted_floats = 0
        self.guide_replacement_events = 0
        self.guide_replacement_eligible_count = 0
        self.guide_replacement_valid_event_count = 0
        self.guide_replacement_forward_count = 0
        self.guide_replacement_reverse_count = 0
        self.guide_replacement_noop_count = 0
        self.guide_replacement_commit_count = 0
        self.guide_replacement_probe_local_evals = 0
        self.collective_guide_events = 0
        self.collective_guide_valid_events = 0
        self.collective_guide_hypothetical_veto_events = 0
        self.collective_guide_actual_veto_events = 0
        self.collective_guide_probe_local_evals = 0
        self.collective_guide_comm_rounds = 0
        self.collective_guide_messages = 0
        self.collective_guide_transmitted_floats = 0
        self.centralized_full_mean_rounds = 0
        self.total_comm_rounds_applied = 0
        self.total_comm_events = 0
        self.step_comm_rounds_applied = 0
        self.step_comm_events = 0
        self.graph_messages = 0
        self.graph_transmitted_floats = 0
        self.committee_events = 0
        self.committee_decision_local_evals = 0
        self.committee_shadow_local_evals = 0
        self.committee_shadow_global_local_evals = 0
        self.committee_messages = 0
        self.committee_transmitted_floats = 0
        self.committee_acceptance_events = 0
        self.committee_acceptance_accepted_events = 0
        self.committee_acceptance_rejected_events = 0
        self.committee_acceptance_local_evals = 0
        self.committee_acceptance_messages = 0
        self.committee_acceptance_transmitted_floats = 0
        self.committee_acceptance_stage_events.fill(0)
        self.committee_acceptance_stage_accepted_events.fill(0)
        self.candidate_response_events = 0
        self.candidate_response_actuated_events = 0
        self.candidate_response_probe_local_evals = 0
        self.candidate_response_verification_local_evals = 0
        self.candidate_response_shadow_global_local_evals = 0
        self.candidate_response_rounds = 0
        self.candidate_response_messages = 0
        self.candidate_response_transmitted_floats = 0
        self.early_stop_counter = 0
        self.last_early_stop_metric = float("inf")
        self.last_early_stop_triggered = False
        self.last_early_stop_reason = ""

        if self.local_only_information:
            self.agent_state_local_evals += int(self.n_agents)
            self._run_detached_global_monitor(initial_report_x)
        else:
            # The legacy reset already evaluated all local objectives through
            # _eval_global_and_all_local before the counters were cleared.
            self.global_monitor_local_evals += int(self.n_agents)
            self.global_monitor_rounds += 1
        self._record_state_communication_event(communication_applied=False)

        return self._build_obs()

    def export_state(self) -> Dict:
        self._assert_no_pending_transition("export state")
        state = super().export_state()
        state.update(
            {
                "information_mode": str(self.information_mode),
                "global_monitor_enable": bool(self.global_monitor_enable),
                "state_comm_cost_mode": str(self.state_comm_cost_mode),
                "persistent_sepcmaes_bank": {
                    "state_version": int(
                        PERSISTENT_SEPCMAES_BANK_STATE_VERSION
                    ),
                    "enabled": bool(self.persistent_sepcmaes_enable),
                    "recenter_max_ratio": float(
                        self.persistent_sepcmaes_recenter_max_ratio
                    ),
                    "dimension": int(self.D),
                    "n_agents": int(self.n_agents),
                    "states": copy.deepcopy(
                        self.persistent_sepcmaes_states
                    ),
                    "signatures": copy.deepcopy(
                        self.persistent_sepcmaes_signatures
                    ),
                    "events": int(self.persistent_sepcmaes_events),
                    "fresh_events": int(
                        self.persistent_sepcmaes_fresh_events
                    ),
                    "resume_events": int(
                        self.persistent_sepcmaes_resume_events
                    ),
                    "config_resets": int(
                        self.persistent_sepcmaes_config_resets
                    ),
                    "last_active": self.last_persistent_sepcmaes_active.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_fresh": self.last_persistent_sepcmaes_fresh.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_config_reset": self.last_persistent_sepcmaes_config_reset.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_recenter_requested": self.last_persistent_sepcmaes_recenter_requested.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_recenter_applied": self.last_persistent_sepcmaes_recenter_applied.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_recenter_remaining": self.last_persistent_sepcmaes_recenter_remaining.astype(
                        np.float64, copy=True
                    ).tolist(),
                },
                "target_block_cooperative_field": {
                    "state_version": int(
                        TARGET_BLOCK_FIELD_STATE_VERSION
                    ),
                    "enabled": bool(self.target_block_field_enable),
                    "dimension": int(self.D),
                    "n_agents": int(self.n_agents),
                    "target_num": int(self.committee_target_num),
                    "coordinate_dim": int(
                        self.committee_coordinate_dim
                    ),
                    "path_decay": float(
                        self.target_block_field_path_decay
                    ),
                    "step_rate": float(
                        self.target_block_field_step_rate
                    ),
                    "radius_min_ratio": float(
                        self.target_block_field_radius_min_ratio
                    ),
                    "radius_max_ratio": float(
                        self.target_block_field_radius_max_ratio
                    ),
                    "path": self.target_block_field_path.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "radius": self.target_block_field_radius.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "initialized": self.target_block_field_initialized.astype(
                        bool, copy=True
                    ).tolist(),
                    "events": int(self.target_block_field_events),
                    "local_evals": int(
                        self.target_block_field_local_evals
                    ),
                    "comm_rounds": int(
                        self.target_block_field_comm_rounds
                    ),
                    "messages": int(self.target_block_field_messages),
                    "transmitted_floats": int(
                        self.target_block_field_transmitted_floats
                    ),
                    "last_secant": self.last_target_block_field_secant.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_secant_valid": self.last_target_block_field_secant_valid.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_secant_alignment": self.last_target_block_field_secant_alignment.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_direction": self.last_target_block_field_direction.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_source_diversity": self.last_target_block_field_source_diversity.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "last_path_norm": self.last_target_block_field_path_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_path_alignment": self.last_target_block_field_path_alignment.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_conflict": self.last_target_block_field_conflict.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_radius_expand": self.last_target_block_field_radius_expand.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_radius_shrink": self.last_target_block_field_radius_shrink.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_radius_clip_min": self.last_target_block_field_radius_clip_min.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_radius_clip_max": self.last_target_block_field_radius_clip_max.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_requested_commit_norm": self.last_target_block_field_requested_commit_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_applied_commit_norm": self.last_target_block_field_applied_commit_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_boundary_clipped": self.last_target_block_field_boundary_clipped.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_path_injection": float(
                        self.last_target_block_field_path_injection
                    ),
                    "last_path_angle_degrees": float(
                        self.last_target_block_field_path_angle_degrees
                    ),
                },
                "target_block_dormancy_recovery": {
                    "state_version": int(
                        TARGET_BLOCK_DORMANCY_STATE_VERSION
                    ),
                    "enabled": bool(
                        self.target_block_dormancy_recovery_enable
                    ),
                    "n_agents": int(self.n_agents),
                    "target_num": int(self.committee_target_num),
                    "progress_ratio": float(
                        TARGET_BLOCK_DORMANCY_PROGRESS_RATIO
                    ),
                    "unresolved_ratio": float(
                        TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO
                    ),
                    "patience": int(TARGET_BLOCK_DORMANCY_PATIENCE),
                    "cooldown_events": int(
                        TARGET_BLOCK_DORMANCY_COOLDOWN
                    ),
                    "min_source_diversity": int(
                        TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY
                    ),
                    "residual_reference": self.target_block_dormancy_residual_reference.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "reserve_radius": self.target_block_dormancy_reserve_radius.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "initialized": self.target_block_dormancy_initialized.astype(
                        bool, copy=True
                    ).tolist(),
                    "floor_age": self.target_block_dormancy_floor_age.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "stagnation_age": self.target_block_dormancy_stagnation_age.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "cooldown": self.target_block_dormancy_cooldown.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "activation_count": self.target_block_dormancy_activation_count.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "events": int(self.target_block_dormancy_events),
                    "activations": int(
                        self.target_block_dormancy_activations
                    ),
                    "last_active": self.last_target_block_dormancy_active.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_unresolved": self.last_target_block_dormancy_unresolved.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_material_progress": self.last_target_block_dormancy_material_progress.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_reliable_scale": self.last_target_block_dormancy_reliable_scale.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_residual_ratio": self.last_target_block_dormancy_residual_ratio.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_progress_ratio": self.last_target_block_dormancy_progress_ratio.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_disagreement": self.last_target_block_dormancy_disagreement.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_disagreement_ratio": self.last_target_block_dormancy_disagreement_ratio.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_restore_radius": self.last_target_block_dormancy_restore_radius.astype(
                        np.float64, copy=True
                    ).tolist(),
                },
                "target_block_direction_shadow": {
                    "state_version": int(
                        TARGET_BLOCK_DIRECTION_SHADOW_STATE_VERSION
                    ),
                    "enabled": bool(
                        self.target_block_direction_shadow_enable
                    ),
                    "n_agents": int(self.n_agents),
                    "target_num": int(self.committee_target_num),
                    "coordinate_dim": int(
                        self.committee_coordinate_dim
                    ),
                    "sources": list(
                        TARGET_BLOCK_DIRECTION_SHADOW_SOURCES
                    ),
                    "events": int(
                        self.target_block_direction_shadow_events
                    ),
                    "eligible_blocks": int(
                        self.target_block_direction_shadow_eligible_blocks
                    ),
                    "probed_blocks": int(
                        self.target_block_direction_shadow_probed_blocks
                    ),
                    "local_evals": int(
                        self.target_block_direction_shadow_local_evals
                    ),
                    "comm_rounds": int(
                        self.target_block_direction_shadow_comm_rounds
                    ),
                    "messages": int(
                        self.target_block_direction_shadow_messages
                    ),
                    "transmitted_floats": int(
                        self.target_block_direction_shadow_transmitted_floats
                    ),
                    "last_eligible": self.last_target_block_direction_shadow_eligible.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_selected": self.last_target_block_direction_shadow_selected.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_candidate_valid": self.last_target_block_direction_shadow_candidate_valid.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_candidate_angle": self.last_target_block_direction_shadow_candidate_angle.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_candidate_response": self.last_target_block_direction_shadow_candidate_response.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_candidate_positive": self.last_target_block_direction_shadow_candidate_positive.astype(
                        bool, copy=True
                    ).tolist(),
                    "last_best_response": self.last_target_block_direction_shadow_best_response.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_best_source": self.last_target_block_direction_shadow_best_source.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "last_best_sign": self.last_target_block_direction_shadow_best_sign.astype(
                        np.int64, copy=True
                    ).tolist(),
                    "last_best_direction": self.last_target_block_direction_shadow_best_direction.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_support": self.last_target_block_direction_shadow_support.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_conflict": self.last_target_block_direction_shadow_conflict.astype(
                        np.float64, copy=True
                    ).tolist(),
                },
                "target_block_challenge_response": {
                    "state_version": int(
                        TARGET_BLOCK_CHALLENGE_RESPONSE_STATE_VERSION
                    ),
                    "mode": str(self.target_block_challenge_response_mode),
                    "n_agents": int(self.n_agents),
                    "target_num": int(self.committee_target_num),
                    "coordinate_dim": int(self.committee_coordinate_dim),
                    "sources": list(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES),
                    "events": int(self.target_block_challenge_events),
                    "challenges": int(self.target_block_challenge_challenges),
                    "directed_responses": int(
                        self.target_block_challenge_directed_responses
                    ),
                    "local_evals": int(
                        self.target_block_challenge_local_evals
                    ),
                    "reported_evals": int(
                        self.target_block_challenge_reported_evals
                    ),
                    "applied_commits": int(
                        self.target_block_challenge_applied_commits
                    ),
                    "comm_rounds": int(
                        self.target_block_challenge_comm_rounds
                    ),
                    "messages": int(self.target_block_challenge_messages),
                    "transmitted_floats": int(
                        self.target_block_challenge_transmitted_floats
                    ),
                    "last_selected": self.last_target_block_challenge_selected.astype(bool, copy=True).tolist(),
                    "last_candidate_valid": self.last_target_block_challenge_candidate_valid.astype(bool, copy=True).tolist(),
                    "last_candidate_angle": self.last_target_block_challenge_candidate_angle.astype(np.float64, copy=True).tolist(),
                    "last_candidate_score": self.last_target_block_challenge_candidate_score.astype(np.float64, copy=True).tolist(),
                    "last_candidate_sign": self.last_target_block_challenge_candidate_sign.astype(np.int64, copy=True).tolist(),
                    "last_path_score": self.last_target_block_challenge_path_score.astype(np.float64, copy=True).tolist(),
                    "last_alternative_score": self.last_target_block_challenge_alternative_score.astype(np.float64, copy=True).tolist(),
                    "last_margin": self.last_target_block_challenge_margin.astype(np.float64, copy=True).tolist(),
                    "last_best_source": self.last_target_block_challenge_best_source.astype(np.int64, copy=True).tolist(),
                    "last_best_sign": self.last_target_block_challenge_best_sign.astype(np.int64, copy=True).tolist(),
                    "last_best_direction": self.last_target_block_challenge_best_direction.astype(np.float64, copy=True).tolist(),
                    "last_coverage": self.last_target_block_challenge_coverage.astype(np.float64, copy=True).tolist(),
                    "last_positive_sources": self.last_target_block_challenge_positive_sources.astype(np.int64, copy=True).tolist(),
                    "last_neighbor_positive_sources": self.last_target_block_challenge_neighbor_positive_sources.astype(np.int64, copy=True).tolist(),
                    "last_support": self.last_target_block_challenge_support.astype(np.float64, copy=True).tolist(),
                    "last_conflict": self.last_target_block_challenge_conflict.astype(np.float64, copy=True).tolist(),
                    "last_actuation_eligible": self.last_target_block_challenge_actuation_eligible.astype(bool, copy=True).tolist(),
                    "last_applied": self.last_target_block_challenge_applied.astype(bool, copy=True).tolist(),
                    "last_applied_norm": self.last_target_block_challenge_applied_norm.astype(np.float64, copy=True).tolist(),
                    "last_boundary_clipped": self.last_target_block_challenge_boundary_clipped.astype(bool, copy=True).tolist(),
                },
                "target_block_dual_clock": {
                    "state_version": int(
                        TARGET_BLOCK_DUAL_CLOCK_STATE_VERSION
                    ),
                    "enabled": bool(
                        self.target_block_dual_clock_enable
                    ),
                    "commit_lock_enabled": bool(
                        self.target_block_dual_clock_commit_lock_enable
                    ),
                    "commit_credit_mode": str(
                        self.target_block_commit_credit_mode
                    ),
                    "commit_credit_events": int(
                        self.target_block_commit_credit_events
                    ),
                    "last_commit_credit_active": self.last_target_block_commit_credit_active.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit": self.last_target_block_commit_credit.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_cosine": self.last_target_block_commit_credit_cosine.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_proposal_norm": self.last_target_block_commit_credit_proposal_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_commit_norm": self.last_target_block_commit_credit_commit_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_correction_norm": self.last_target_block_commit_credit_correction_norm.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_path_retention": self.last_target_block_commit_credit_path_retention.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_scale_retention": self.last_target_block_commit_credit_scale_retention.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_axis_rms_before": self.last_target_block_commit_credit_axis_rms_before.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_axis_rms_after": self.last_target_block_commit_credit_axis_rms_after.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_sigma_path_before": self.last_target_block_commit_credit_sigma_path_before.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_sigma_path_after": self.last_target_block_commit_credit_sigma_path_after.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_cov_path_before": self.last_target_block_commit_credit_cov_path_before.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "last_commit_credit_cov_path_after": self.last_target_block_commit_credit_cov_path_after.astype(
                        np.float64, copy=True
                    ).tolist(),
                    "n_agents": int(self.n_agents),
                    "outer_events": int(
                        self.target_block_dual_clock_outer_events
                    ),
                    "local_generation_ticks": int(
                        self.target_block_dual_clock_local_generation_ticks
                    ),
                    "communication_ticks": int(
                        self.target_block_dual_clock_communication_ticks
                    ),
                    "commit_ticks": int(
                        self.target_block_dual_clock_commit_ticks
                    ),
                    "last_microcycles": int(
                        self.last_target_block_dual_clock_microcycles
                    ),
                    "last_generation_fes_per_agent": (
                        self.last_target_block_dual_clock_generation_fes.astype(
                            np.int64,
                            copy=True,
                        ).tolist()
                    ),
                },
                "agent_x": self.agent_x.astype(np.float64, copy=True).tolist(),
                "agent_local_f": self.agent_local_f.astype(np.float64, copy=True).tolist(),
                "agent_initial_local_f": self.agent_initial_local_f.astype(
                    np.float64, copy=True
                ).tolist(),
                "agent_best_local_f": self.agent_best_local_f.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_committed_local_improve": self.last_committed_local_improve.astype(
                    np.float64, copy=True
                ).tolist(),
                "committed_local_improve_hist": [
                    [float(value) for value in history]
                    for history in self.committed_local_improve_hist
                ],
                "last_consensus_local_effect": self.last_consensus_local_effect.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_local_commit_success": self.last_local_commit_success.astype(
                    np.float64, copy=True
                ).tolist(),
                "report_current_x": self.report_current_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "report_current_f": float(self.report_current_f),
                "report_best_x": self.report_best_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "report_best_f": float(self.report_best_f),
                "last_neighbor_summary": self.last_neighbor_summary.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_sigma_value": self.last_sigma_value.astype(
                    np.float64, copy=True
                ).tolist(),
                "sigma_state_age": self.sigma_state_age.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_actual_fes_per_agent": self.last_actual_fes_per_agent.astype(
                    np.float64, copy=True
                ).tolist(),
                "cumulative_actual_fes_per_agent": self.cumulative_actual_fes_per_agent.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_cmaes_numeric_fail_soft": self.last_cmaes_numeric_fail_soft.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_cmaes_numeric_fail_soft_generation": self.last_cmaes_numeric_fail_soft_generation.astype(
                    np.int64, copy=True
                ).tolist(),
                "cmaes_numeric_fail_soft_events": int(
                    self.cmaes_numeric_fail_soft_events
                ),
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
                "committee_direction": self.committee_direction.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_confidence": self.committee_confidence.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_verified_x": self.committee_verified_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_selected_source": self.committee_selected_source.astype(
                    np.int64, copy=True
                ).tolist(),
                "committee_candidate_score": self.committee_candidate_score.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_acceptance_active": bool(
                    self.committee_acceptance_active
                ),
                "committee_candidate_global_f": self.committee_candidate_global_f.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_candidate_report_f": float(
                    self.committee_candidate_report_f
                ),
                "committee_candidate_vs_report_log_improve": float(
                    self.committee_candidate_vs_report_log_improve
                ),
                "committee_effective_beta": self.committee_effective_beta.astype(
                    np.float64, copy=True
                ).tolist(),
                "committee_base_alignment": self.committee_base_alignment.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_committee_metrics": dict(self.last_committee_metrics),
                "candidate_response_verified_x": self.candidate_response_verified_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_response_selected_source": self.candidate_response_selected_source.astype(
                    np.int64, copy=True
                ).tolist(),
                "candidate_response_accepted_mask": self.candidate_response_accepted_mask.astype(
                    bool, copy=True
                ).tolist(),
                "last_candidate_accept_ratio": self.last_candidate_accept_ratio.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_candidate_selection_confidence": self.last_candidate_selection_confidence.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_candidate_support_ratio": self.last_candidate_support_ratio.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_candidate_requested_shift_norm": self.last_candidate_requested_shift_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_candidate_actuator_action": self.last_candidate_actuator_action.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_candidate_actuator_beta": self.last_candidate_actuator_beta.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_response_shadow_base_x": self.candidate_response_shadow_base_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_response_shadow_candidate_x": self.candidate_response_shadow_candidate_x.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_response_shadow_due": bool(
                    self.candidate_response_shadow_due
                ),
                "last_candidate_response_metrics": dict(
                    self.last_candidate_response_metrics
                ),
                "candidate_multisecant_state_version": 1,
                "candidate_multisecant_generator": str(
                    self.candidate_generator
                ),
                "candidate_multisecant_history_step": self.candidate_multisecant_history_step.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_multisecant_history_response": self.candidate_multisecant_history_response.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_multisecant_history_center": self.candidate_multisecant_history_center.astype(
                    np.float64, copy=True
                ).tolist(),
                "candidate_multisecant_history_event_id": self.candidate_multisecant_history_event_id.astype(
                    np.int64, copy=True
                ).tolist(),
                "candidate_multisecant_history_valid": self.candidate_multisecant_history_valid.astype(
                    bool, copy=True
                ).tolist(),
                "candidate_multisecant_history_cursor": self.candidate_multisecant_history_cursor.astype(
                    np.int64, copy=True
                ).tolist(),
                "candidate_multisecant_history_count": self.candidate_multisecant_history_count.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_candidate_multisecant_active": self.last_candidate_multisecant_active.astype(
                    bool, copy=True
                ).tolist(),
                "last_candidate_multisecant_fallback_reason": self.last_candidate_multisecant_fallback_reason.astype(
                    np.int64, copy=True
                ).tolist(),
                "rvcpd_state_version": 1,
                "rvcpd_arm": str(self.rvcpd_arm),
                "rvcpd_integration_mode": str(
                    self.rvcpd_integration_mode
                ),
                "rvcpd_dimension": int(self.D),
                "rvcpd_n_agents": int(self.n_agents),
                "rvcpd_message_schema": "unit_direction_valid_age_step_ratio_v1",
                "rvcpd_path": self.rvcpd_path.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_support_ema": self.rvcpd_support_ema.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_conflict_ema": self.rvcpd_conflict_ema.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_uncertainty_ema": self.rvcpd_uncertainty_ema.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_trust": self.rvcpd_trust.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_scale": self.rvcpd_scale.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_age": self.rvcpd_age.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_rvcpd_direction": self.last_rvcpd_direction.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_sign": self.last_rvcpd_sign.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_rvcpd_active": self.last_rvcpd_active.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_agreement": self.last_rvcpd_agreement.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_requested_radius": self.last_rvcpd_requested_radius.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_applied_radius": self.last_rvcpd_applied_radius.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_plus_gain": self.last_rvcpd_plus_gain.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_rvcpd_minus_gain": self.last_rvcpd_minus_gain.astype(
                    np.float64, copy=True
                ).tolist(),
                "rvcpd_events": int(self.rvcpd_events),
                "rvcpd_valid_event_count": int(
                    self.rvcpd_valid_event_count
                ),
                "rvcpd_commit_count": int(self.rvcpd_commit_count),
                "rvcpd_reverse_count": int(self.rvcpd_reverse_count),
                "rvcpd_noop_count": int(self.rvcpd_noop_count),
                "rvcpd_probe_local_evals": int(
                    self.rvcpd_probe_local_evals
                ),
                "rvcpd_messages": int(self.rvcpd_messages),
                "rvcpd_transmitted_floats": int(
                    self.rvcpd_transmitted_floats
                ),
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
                "optimizer_guide_direction": self.optimizer_guide_direction.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_norm": self.last_optimizer_guide_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_applied": self.last_optimizer_guide_applied.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_alignment": self.last_optimizer_guide_alignment.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_source_active_ratio": float(
                    self.last_optimizer_guide_source_active_ratio
                ),
                "guide_replacement_state_version": 1,
                "guide_replacement_mode": str(self.guide_replacement_mode),
                "guide_replacement_scope": str(self.guide_replacement_scope),
                "guide_replacement_dimension": int(self.D),
                "guide_replacement_n_agents": int(self.n_agents),
                "guide_replacement_direction": self.guide_replacement_direction.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_eligible": self.last_guide_replacement_eligible.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_valid": self.last_guide_replacement_valid.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_old_guide_suppressed": self.last_guide_replacement_old_guide_suppressed.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_sign": self.last_guide_replacement_sign.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_guide_replacement_strength": self.last_guide_replacement_strength.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_sigma": self.last_guide_replacement_sigma.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_requested_radius": self.last_guide_replacement_requested_radius.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_applied_radius": self.last_guide_replacement_applied_radius.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_plus_gain": self.last_guide_replacement_plus_gain.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_guide_replacement_minus_gain": self.last_guide_replacement_minus_gain.astype(
                    np.float64, copy=True
                ).tolist(),
                "guide_replacement_events": int(self.guide_replacement_events),
                "guide_replacement_eligible_count": int(
                    self.guide_replacement_eligible_count
                ),
                "guide_replacement_valid_event_count": int(
                    self.guide_replacement_valid_event_count
                ),
                "guide_replacement_forward_count": int(
                    self.guide_replacement_forward_count
                ),
                "guide_replacement_reverse_count": int(
                    self.guide_replacement_reverse_count
                ),
                "guide_replacement_noop_count": int(
                    self.guide_replacement_noop_count
                ),
                "guide_replacement_commit_count": int(
                    self.guide_replacement_commit_count
                ),
                "guide_replacement_probe_local_evals": int(
                    self.guide_replacement_probe_local_evals
                ),
                "collective_guide_state_version": 1,
                "collective_guide_mode": str(self.collective_guide_mode),
                "collective_guide_null_veto_rate": float(
                    self.collective_guide_null_veto_rate
                ),
                "collective_guide_dimension": int(self.D),
                "collective_guide_n_agents": int(self.n_agents),
                "last_collective_guide_shared_base": self.last_collective_guide_shared_base.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_direction": self.last_collective_guide_direction.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_radius": self.last_collective_guide_radius.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_vote": self.last_collective_guide_vote.astype(
                    np.int64, copy=True
                ).tolist(),
                "last_collective_guide_source_valid": self.last_collective_guide_source_valid.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_vote_valid": self.last_collective_guide_vote_valid.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_suppressed": self.last_collective_guide_suppressed.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collective_guide_valid": bool(
                    self.last_collective_guide_valid
                ),
                "last_collective_guide_vote_sum": int(
                    self.last_collective_guide_vote_sum
                ),
                "last_collective_guide_hypothetical_veto": bool(
                    self.last_collective_guide_hypothetical_veto
                ),
                "last_collective_guide_actual_veto": bool(
                    self.last_collective_guide_actual_veto
                ),
                "last_collective_guide_null_veto": bool(
                    self.last_collective_guide_null_veto
                ),
                "collective_guide_events": int(self.collective_guide_events),
                "collective_guide_valid_events": int(
                    self.collective_guide_valid_events
                ),
                "collective_guide_hypothetical_veto_events": int(
                    self.collective_guide_hypothetical_veto_events
                ),
                "collective_guide_actual_veto_events": int(
                    self.collective_guide_actual_veto_events
                ),
                "collective_guide_probe_local_evals": int(
                    self.collective_guide_probe_local_evals
                ),
                "collective_guide_comm_rounds": int(
                    self.collective_guide_comm_rounds
                ),
                "collective_guide_messages": int(
                    self.collective_guide_messages
                ),
                "collective_guide_transmitted_floats": int(
                    self.collective_guide_transmitted_floats
                ),
                "last_optimizer_guide_internal_active": self.last_optimizer_guide_internal_active.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_internal_mean_step_norm": self.last_optimizer_guide_internal_mean_step_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_guide_internal_alignment": self.last_optimizer_guide_internal_alignment.astype(
                    np.float64, copy=True
                ).tolist(),
                "anchor_points": self.anchor_points.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_anchor_applied": self.last_anchor_applied.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_anchor_dist": self.last_anchor_dist.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_anchor_direction_norm": self.last_anchor_direction_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_anchor_applied": self.last_optimizer_anchor_applied.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_anchor_mean_step_norm": self.last_optimizer_anchor_mean_step_norm.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_optimizer_anchor_sample_applied": self.last_optimizer_anchor_sample_applied.astype(
                    np.float64, copy=True
                ).tolist(),
                "last_collab_mode_idx": self.last_collab_mode_idx.astype(
                    np.int64, copy=True
                ),
                "last_guide_scale_idx": self.last_guide_scale_idx.astype(
                    np.int64, copy=True
                ),
                "last_guide_scale_value": self.last_guide_scale_value.astype(
                    np.float64, copy=True
                ),
                "last_collab_leader_active": self.last_collab_leader_active.astype(
                    np.float64, copy=True
                ),
                "last_collab_strength_multiplier": self.last_collab_strength_multiplier.astype(
                    np.float64, copy=True
                ),
                "last_optimizer_guide_strength_effective": float(
                    self.last_optimizer_guide_strength_effective
                ),
                "last_optimizer_guide_strength_scale": float(
                    self.last_optimizer_guide_strength_scale
                ),
                "last_optimizer_guide_schedule_metric": float(
                    self.last_optimizer_guide_schedule_metric
                ),
                "local_search_fes": int(self.local_search_fes),
                "candidate_validation_local_evals": int(
                    self.candidate_validation_local_evals
                ),
                "agent_state_local_evals": int(self.agent_state_local_evals),
                "global_monitor_local_evals": int(
                    self.global_monitor_local_evals
                ),
                "global_monitor_rounds": int(self.global_monitor_rounds),
                "state_comm_rounds": int(self.state_comm_rounds),
                "state_comm_messages": int(self.state_comm_messages),
                "state_comm_transmitted_floats": int(
                    self.state_comm_transmitted_floats
                ),
                "graph_comm_rounds": int(self.graph_comm_rounds),
                "event_slot_interleaving_enable": bool(
                    self.event_slot_interleaving_enable
                ),
                "event_slot_interleaving_events": int(
                    self.event_slot_interleaving_events
                ),
                "event_slot_interleaving_slots": int(
                    self.event_slot_interleaving_slots
                ),
                "event_slot_native_local_evals": int(
                    self.event_slot_native_local_evals
                ),
                "event_slot_reported_local_evals": int(
                    self.event_slot_reported_local_evals
                ),
                "event_slot_physical_local_evals": int(
                    self.event_slot_physical_local_evals
                ),
                "last_event_slot_count": int(self.last_event_slot_count),
                "last_event_slot_packet_units": (
                    self.last_event_slot_packet_units.astype(
                        np.int64, copy=True
                    ).tolist()
                ),
                "last_event_slot_distribution_updates": (
                    self.last_event_slot_distribution_updates.astype(
                        np.int64, copy=True
                    ).tolist()
                ),
                "last_event_slot_native_evals": (
                    self.last_event_slot_native_evals.astype(
                        np.int64, copy=True
                    ).tolist()
                ),
                "last_event_slot_physical_evals": (
                    self.last_event_slot_physical_evals.astype(
                        np.int64, copy=True
                    ).tolist()
                ),
                "last_event_slot_center_shift_norms": (
                    self.last_event_slot_center_shift_norms.astype(
                        np.float64, copy=True
                    ).tolist()
                ),
                "last_event_slot_packet_improvements": (
                    self.last_event_slot_packet_improvements.astype(
                        np.float64, copy=True
                    ).tolist()
                ),
                "last_event_slot_commit_audit": {
                    field: self.last_event_slot_commit_audit[field].astype(
                        np.float64, copy=True
                    ).tolist()
                    for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS
                },
                "last_event_slot_mmes_neutral_success": (
                    self.last_event_slot_mmes_neutral_success.astype(
                        np.int64, copy=True
                    ).tolist()
                ),
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
                "committee_events": int(self.committee_events),
                "committee_decision_local_evals": int(
                    self.committee_decision_local_evals
                ),
                "committee_shadow_local_evals": int(
                    self.committee_shadow_local_evals
                ),
                "committee_shadow_global_local_evals": int(
                    self.committee_shadow_global_local_evals
                ),
                "committee_messages": int(self.committee_messages),
                "committee_transmitted_floats": int(
                    self.committee_transmitted_floats
                ),
                "committee_acceptance_events": int(
                    self.committee_acceptance_events
                ),
                "committee_acceptance_accepted_events": int(
                    self.committee_acceptance_accepted_events
                ),
                "committee_acceptance_rejected_events": int(
                    self.committee_acceptance_rejected_events
                ),
                "committee_acceptance_local_evals": int(
                    self.committee_acceptance_local_evals
                ),
                "committee_acceptance_messages": int(
                    self.committee_acceptance_messages
                ),
                "committee_acceptance_transmitted_floats": int(
                    self.committee_acceptance_transmitted_floats
                ),
                "committee_acceptance_stage_events": self.committee_acceptance_stage_events.astype(
                    np.int64, copy=True
                ).tolist(),
                "committee_acceptance_stage_accepted_events": self.committee_acceptance_stage_accepted_events.astype(
                    np.int64, copy=True
                ).tolist(),
                "candidate_response_events": int(self.candidate_response_events),
                "candidate_response_actuated_events": int(
                    self.candidate_response_actuated_events
                ),
                "candidate_response_probe_local_evals": int(
                    self.candidate_response_probe_local_evals
                ),
                "candidate_response_verification_local_evals": int(
                    self.candidate_response_verification_local_evals
                ),
                "candidate_response_shadow_global_local_evals": int(
                    self.candidate_response_shadow_global_local_evals
                ),
                "candidate_response_rounds": int(self.candidate_response_rounds),
                "candidate_response_messages": int(self.candidate_response_messages),
                "candidate_response_transmitted_floats": int(
                    self.candidate_response_transmitted_floats
                ),
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
                    "information_mode": str(self.information_mode),
                    "global_monitor_enable": bool(self.global_monitor_enable),
                    "state_comm_cost_mode": str(self.state_comm_cost_mode),
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
                    "ccsa_momentum_update_mode": str(
                        self.ccsa_momentum_update_mode
                    ),
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
                    "optimizer_guide_enable": bool(self.optimizer_guide_enable),
                    "optimizer_guide_source": str(self.optimizer_guide_source),
                    "guide_replacement_mode": str(self.guide_replacement_mode),
                    "guide_replacement_scope": str(self.guide_replacement_scope),
                    "optimizer_guide_strength": float(self.optimizer_guide_strength),
                    "optimizer_guide_strength_schedule": str(
                        self.optimizer_guide_strength_schedule
                    ),
                    "optimizer_guide_strength_scale_min": float(
                        self.optimizer_guide_strength_scale_min
                    ),
                    "optimizer_guide_strength_scale_max": float(
                        self.optimizer_guide_strength_scale_max
                    ),
                    "optimizer_guide_disagreement_low": float(
                        self.optimizer_guide_disagreement_low
                    ),
                    "optimizer_guide_disagreement_high": float(
                        self.optimizer_guide_disagreement_high
                    ),
                    "optimizer_guide_mix_strength": float(
                        self.optimizer_guide_mix_strength
                    ),
                    "optimizer_guide_injection_pairs": int(
                        self.optimizer_guide_injection_pairs
                    ),
                    "optimizer_guide_use_negative_pair": bool(
                        self.optimizer_guide_use_negative_pair
                    ),
                    "optimizer_guide_min_improve": float(
                        self.optimizer_guide_min_improve
                    ),
                    "optimizer_guide_gate_enable": bool(
                        self.optimizer_guide_gate_enable
                    ),
                    "optimizer_guide_min_norm": float(
                        self.optimizer_guide_min_norm
                    ),
                    "optimizer_guide_min_alignment": float(
                        self.optimizer_guide_min_alignment
                    ),
                    "optimizer_guide_apply_optimizers": (
                        ["all"]
                        if self.optimizer_guide_apply_all
                        else sorted(self.optimizer_guide_apply_optimizers)
                    ),
                    "anchor_enable": bool(self.anchor_enable),
                    "anchor_source": str(self.anchor_source),
                    "anchor_apply_optimizers": (
                        ["all"] if self.anchor_apply_all else sorted(self.anchor_apply_optimizers)
                    ),
                    "anchor_strength": float(self.anchor_strength),
                    "anchor_mix_strength": float(self.anchor_mix_strength),
                    "anchor_sample_ratio": float(self.anchor_sample_ratio),
                    "anchor_sample_clip_ratio": float(self.anchor_sample_clip_ratio),
                    "anchor_mean_pull": bool(self.anchor_mean_pull),
                    "anchor_sample_injection": bool(self.anchor_sample_injection),
                    "committee_mode": str(self.committee_mode),
                    "committee_selection": str(self.committee_selection),
                    "committee_mix_strength": float(self.committee_mix_strength),
                    "committee_acceptance_mode": str(
                        self.committee_acceptance_mode
                    ),
                    "committee_acceptance_min_log_improve": float(
                        self.committee_acceptance_min_log_improve
                    ),
                    "committee_shadow_global_eval": bool(
                        self.committee_shadow_global_eval
                    ),
                    "committee_supported": bool(self.committee_supported),
                    "candidate_response_mode": str(self.candidate_response_mode),
                    "candidate_generator": str(self.candidate_generator),
                    "candidate_response_supported": bool(
                        self.candidate_response_supported
                    ),
                    "candidate_multisecant_history_size": int(
                        self.candidate_history_size
                    ),
                    "candidate_multisecant_min_rank": int(
                        self.candidate_multisecant_min_rank
                    ),
                    "candidate_multisecant_rank_tolerance": float(
                        self.candidate_multisecant_rank_tolerance
                    ),
                    "candidate_multisecant_condition_max": float(
                        self.candidate_multisecant_condition_max
                    ),
                    "candidate_multisecant_center_distance_max_ratio": float(
                        self.candidate_multisecant_center_distance_max_ratio
                    ),
                    "candidate_multisecant_max_age": int(
                        self.candidate_multisecant_max_age
                    ),
                    "candidate_multisecant_gradient_min_norm": float(
                        self.candidate_multisecant_gradient_min_norm
                    ),
                    "candidate_probe_scale": float(self.candidate_probe_scale),
                    "candidate_trust_scale": float(self.candidate_trust_scale),
                    "candidate_trust_min_ratio": float(
                        self.candidate_trust_min_ratio
                    ),
                    "candidate_trust_max_ratio": float(
                        self.candidate_trust_max_ratio
                    ),
                    "candidate_probe_min_ratio": float(
                        self.candidate_probe_min_ratio
                    ),
                    "candidate_probe_max_ratio": float(
                        self.candidate_probe_max_ratio
                    ),
                    "candidate_confidence_min": float(
                        self.candidate_confidence_min
                    ),
                    "candidate_support_min": float(self.candidate_support_min),
                    "candidate_actuator_beta": float(self.candidate_actuator_beta),
                    "candidate_shadow_global_eval_interval": int(
                        self.candidate_shadow_global_eval_interval
                    ),
                    "optimizer_guide_numeric_guard": bool(
                        self.optimizer_guide_numeric_guard
                    ),
                    "cmaes_numeric_fail_soft": bool(
                        self.cmaes_numeric_fail_soft
                    ),
                    "optimizer_numeric_telemetry_enable": bool(
                        self.optimizer_numeric_telemetry_enable
                    ),
                    "optimizer_numeric_telemetry_dir": str(
                        self.optimizer_numeric_telemetry_dir
                    ),
                    "collab_action_enable": bool(self.collab_action_enable),
                    "collab_modes": [str(x) for x in self.collab_modes],
                    "collab_soft_diversify_scale": float(
                        self.collab_soft_diversify_scale
                    ),
                    "collab_leader_fallback": str(self.collab_leader_fallback),
                    "guide_scale_action_enable": bool(self.guide_scale_action_enable),
                    "guide_scale_candidates": [
                        float(x) for x in self.guide_scale_candidates
                    ],
                    "optimizer_guide_sigma_exp_clip": float(
                        self.optimizer_guide_sigma_exp_clip
                    ),
                    "optimizer_guide_sigma_clip_ratio": float(
                        self.optimizer_guide_sigma_clip_ratio
                    ),
                    "optimizer_guide_sample_clip_ratio": float(
                        self.optimizer_guide_sample_clip_ratio
                    ),
                    "optimizer_guide_internal_mode": str(
                        self.optimizer_guide_internal_mode
                    ),
                    "optimizer_guide_internal_apply_optimizers": (
                        ["all"]
                        if self.optimizer_guide_internal_apply_all
                        else sorted(self.optimizer_guide_internal_apply_optimizers)
                    ),
                    "optimizer_guide_internal_mean_lr": float(
                        self.optimizer_guide_internal_mean_lr
                    ),
                    "optimizer_guide_internal_path_lr": float(
                        self.optimizer_guide_internal_path_lr
                    ),
                    "optimizer_guide_internal_cov_lr": float(
                        self.optimizer_guide_internal_cov_lr
                    ),
                    "optimizer_guide_internal_agree_cos_min": float(
                        self.optimizer_guide_internal_agree_cos_min
                    ),
                    "optimizer_guide_internal_max_step_ratio": float(
                        self.optimizer_guide_internal_max_step_ratio
                    ),
                    "optimizer_guide_internal_max_rel_step": float(
                        self.optimizer_guide_internal_max_rel_step
                    ),
                    "optimizer_guide_internal_path_max_rel_norm": float(
                        self.optimizer_guide_internal_path_max_rel_norm
                    ),
                    "optimizer_guide_internal_cov_rank1_clip": float(
                        self.optimizer_guide_internal_cov_rank1_clip
                    ),
                    "optimizer_guide_internal_disable_sample_injection": bool(
                        self.optimizer_guide_internal_disable_sample_injection
                    ),
                },
            }
        )
        return state

    def import_state(self, state: Dict):
        self._assert_no_pending_transition("import state")
        restored_event_slot_enable = bool(
            state.get("event_slot_interleaving_enable", False)
        )
        if restored_event_slot_enable != self.event_slot_interleaving_enable:
            raise ValueError(
                "Cannot import objective-split state across event-slot "
                "interleaving enable/disable semantics."
            )
        field_payload = state.get(
            "target_block_cooperative_field", None
        )
        restored_target_block_field = None
        if field_payload is None:
            if self.target_block_field_enable:
                raise ValueError(
                    "Cannot import a legacy objective-split state while "
                    "the target-block cooperative field is enabled."
                )
        else:
            if not isinstance(field_payload, dict):
                raise TypeError(
                    "target_block_cooperative_field must be a dictionary."
                )
            if int(
                field_payload.get("state_version", -1)
            ) != TARGET_BLOCK_FIELD_STATE_VERSION:
                raise ValueError(
                    "Unsupported target-block cooperative field state "
                    "version."
                )
            if bool(
                field_payload.get("enabled", False)
            ) != self.target_block_field_enable:
                raise ValueError(
                    "Cannot import objective-split state across "
                    "target-block field enable modes."
                )
            if (
                int(field_payload.get("dimension", -1)) != self.D
                or int(field_payload.get("n_agents", -1))
                != self.n_agents
                or int(field_payload.get("target_num", -1))
                != self.committee_target_num
                or int(field_payload.get("coordinate_dim", -1))
                != self.committee_coordinate_dim
            ):
                raise ValueError(
                    "Target-block field dimension/layout mismatch."
                )
            for key, expected in (
                ("path_decay", self.target_block_field_path_decay),
                ("step_rate", self.target_block_field_step_rate),
                (
                    "radius_min_ratio",
                    self.target_block_field_radius_min_ratio,
                ),
                (
                    "radius_max_ratio",
                    self.target_block_field_radius_max_ratio,
                ),
            ):
                observed = float(field_payload.get(key, np.nan))
                if not np.isclose(
                    observed, expected, rtol=0.0, atol=1e-15
                ):
                    raise ValueError(
                        f"Target-block field {key} mismatch."
                    )

            field_shape = (
                self.n_agents,
                self.committee_target_num,
                self.committee_coordinate_dim,
            )
            block_shape = (
                self.n_agents,
                self.committee_target_num,
            )

            def _field_float(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    field_payload.get(name, None),
                    dtype=np.float64,
                )
                if value.shape != shape or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        "Invalid target-block field tensor: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            def _field_bool(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    field_payload.get(name, None),
                    dtype=bool,
                )
                if value.shape != shape:
                    raise ValueError(
                        "Invalid target-block field mask: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            def _field_int(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    field_payload.get(name, None),
                    dtype=np.int64,
                )
                if value.shape != shape or np.any(value < 0):
                    raise ValueError(
                        "Invalid target-block field integer tensor: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            restored_target_block_field = {
                "path": _field_float("path", field_shape),
                "radius": _field_float("radius", block_shape),
                "initialized": _field_bool(
                    "initialized", block_shape
                ),
                "last_secant": _field_float(
                    "last_secant", field_shape
                ),
                "last_secant_valid": _field_bool(
                    "last_secant_valid", block_shape
                ),
                "last_secant_alignment": _field_float(
                    "last_secant_alignment", block_shape
                ),
                "last_direction": _field_float(
                    "last_direction", field_shape
                ),
                "last_source_diversity": _field_int(
                    "last_source_diversity", block_shape
                ),
                "last_path_norm": _field_float(
                    "last_path_norm", block_shape
                ),
                "last_path_alignment": _field_float(
                    "last_path_alignment", block_shape
                ),
                "last_conflict": _field_bool(
                    "last_conflict", block_shape
                ),
                "last_radius_expand": _field_bool(
                    "last_radius_expand", block_shape
                ),
                "last_radius_shrink": _field_bool(
                    "last_radius_shrink", block_shape
                ),
                "last_radius_clip_min": _field_bool(
                    "last_radius_clip_min", block_shape
                ),
                "last_radius_clip_max": _field_bool(
                    "last_radius_clip_max", block_shape
                ),
                "last_requested_commit_norm": _field_float(
                    "last_requested_commit_norm", block_shape
                ),
                "last_applied_commit_norm": _field_float(
                    "last_applied_commit_norm", block_shape
                ),
                "last_boundary_clipped": _field_bool(
                    "last_boundary_clipped", block_shape
                ),
            }
            if np.any(restored_target_block_field["radius"] < 0.0):
                raise ValueError(
                    "Target-block field radius must be non-negative."
                )
            counters = {
                key: int(field_payload.get(key, 0))
                for key in (
                    "events",
                    "local_evals",
                    "comm_rounds",
                    "messages",
                    "transmitted_floats",
                )
            }
            if any(value < 0 for value in counters.values()):
                raise ValueError(
                    "Target-block field counters must be non-negative."
                )
            message_floats = int(
                3 * self.D + self.committee_target_num + 2
            )
            expected_messages = int(
                counters["events"]
                * (
                    self.graph.directed_edge_count
                    if self.graph is not None
                    else 0
                )
            )
            if (
                counters["local_evals"]
                != 2 * self.n_agents * counters["events"]
                or counters["comm_rounds"] != counters["events"]
                or counters["messages"] != expected_messages
                or counters["transmitted_floats"]
                != expected_messages * message_floats
            ):
                raise ValueError(
                    "Target-block field accounting is inconsistent."
                )
            if (
                not self.target_block_field_enable
                and (
                    any(counters.values())
                    or np.any(
                        restored_target_block_field["initialized"]
                    )
                    or np.any(
                        restored_target_block_field["path"] != 0.0
                    )
                    or np.any(
                        restored_target_block_field["radius"] != 0.0
                    )
                )
            ):
                raise ValueError(
                    "Disabled target-block field contains active state."
                )
            path_injection = float(
                field_payload.get("last_path_injection", np.nan)
            )
            path_angle = float(
                field_payload.get(
                    "last_path_angle_degrees", np.nan
                )
            )
            if (
                not np.isfinite(path_injection)
                or path_injection < 0.0
                or not np.isfinite(path_angle)
                or path_angle < 0.0
                or path_angle > 90.0
            ):
                raise ValueError(
                    "Target-block field path coefficient telemetry is "
                    "invalid."
                )
            restored_target_block_field.update(counters)
            restored_target_block_field[
                "last_path_injection"
            ] = path_injection
            restored_target_block_field[
                "last_path_angle_degrees"
            ] = path_angle
        dormancy_payload = state.get(
            "target_block_dormancy_recovery",
            None,
        )
        restored_target_block_dormancy = None
        if dormancy_payload is None:
            if self.target_block_dormancy_recovery_enable:
                raise ValueError(
                    "Cannot import a legacy objective-split state while "
                    "target-block dormancy recovery is enabled."
                )
        else:
            if not isinstance(dormancy_payload, dict):
                raise TypeError(
                    "target_block_dormancy_recovery must be a dictionary."
                )
            if int(
                dormancy_payload.get("state_version", -1)
            ) != TARGET_BLOCK_DORMANCY_STATE_VERSION:
                raise ValueError(
                    "Unsupported target-block dormancy recovery state "
                    "version."
                )
            if bool(
                dormancy_payload.get("enabled", False)
            ) != self.target_block_dormancy_recovery_enable:
                raise ValueError(
                    "Cannot import objective-split state across target-block "
                    "dormancy recovery enable modes."
                )
            if (
                int(dormancy_payload.get("n_agents", -1))
                != self.n_agents
                or int(dormancy_payload.get("target_num", -1))
                != self.committee_target_num
            ):
                raise ValueError(
                    "Target-block dormancy recovery layout mismatch."
                )
            for key, expected in (
                (
                    "progress_ratio",
                    TARGET_BLOCK_DORMANCY_PROGRESS_RATIO,
                ),
                (
                    "unresolved_ratio",
                    TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO,
                ),
            ):
                observed = float(
                    dormancy_payload.get(key, np.nan)
                )
                if not np.isclose(
                    observed, expected, rtol=0.0, atol=1e-15
                ):
                    raise ValueError(
                        "Target-block dormancy recovery "
                        f"{key} mismatch."
                    )
            for key, expected in (
                ("patience", TARGET_BLOCK_DORMANCY_PATIENCE),
                (
                    "cooldown_events",
                    TARGET_BLOCK_DORMANCY_COOLDOWN,
                ),
                (
                    "min_source_diversity",
                    TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY,
                ),
            ):
                if int(dormancy_payload.get(key, -1)) != int(expected):
                    raise ValueError(
                        "Target-block dormancy recovery "
                        f"{key} mismatch."
                    )

            dormancy_shape = (
                self.n_agents,
                self.committee_target_num,
            )

            def _dormancy_float(name: str) -> np.ndarray:
                value = np.asarray(
                    dormancy_payload.get(name, None),
                    dtype=np.float64,
                )
                if value.shape != dormancy_shape or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        "Invalid target-block dormancy floating tensor: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            def _dormancy_bool(name: str) -> np.ndarray:
                value = np.asarray(
                    dormancy_payload.get(name, None),
                    dtype=bool,
                )
                if value.shape != dormancy_shape:
                    raise ValueError(
                        "Invalid target-block dormancy mask: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            def _dormancy_int(name: str) -> np.ndarray:
                value = np.asarray(
                    dormancy_payload.get(name, None),
                    dtype=np.int64,
                )
                if value.shape != dormancy_shape or np.any(value < 0):
                    raise ValueError(
                        "Invalid target-block dormancy integer tensor: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            restored_target_block_dormancy = {
                "residual_reference": _dormancy_float(
                    "residual_reference"
                ),
                "reserve_radius": _dormancy_float("reserve_radius"),
                "initialized": _dormancy_bool("initialized"),
                "floor_age": _dormancy_int("floor_age"),
                "stagnation_age": _dormancy_int(
                    "stagnation_age"
                ),
                "cooldown": _dormancy_int("cooldown"),
                "activation_count": _dormancy_int(
                    "activation_count"
                ),
                "last_active": _dormancy_bool("last_active"),
                "last_unresolved": _dormancy_bool(
                    "last_unresolved"
                ),
                "last_material_progress": _dormancy_bool(
                    "last_material_progress"
                ),
                "last_reliable_scale": _dormancy_bool(
                    "last_reliable_scale"
                ),
                "last_residual_ratio": _dormancy_float(
                    "last_residual_ratio"
                ),
                "last_progress_ratio": _dormancy_float(
                    "last_progress_ratio"
                ),
                "last_disagreement": _dormancy_float(
                    "last_disagreement"
                ),
                "last_disagreement_ratio": _dormancy_float(
                    "last_disagreement_ratio"
                ),
                "last_restore_radius": _dormancy_float(
                    "last_restore_radius"
                ),
            }
            initialized = restored_target_block_dormancy[
                "initialized"
            ]
            residual_reference = restored_target_block_dormancy[
                "residual_reference"
            ]
            reserve_radius = restored_target_block_dormancy[
                "reserve_radius"
            ]
            if (
                np.any(residual_reference < 0.0)
                or np.any(reserve_radius < 0.0)
                or np.any(initialized & (residual_reference <= 0.0))
                or np.any(
                    restored_target_block_dormancy["cooldown"]
                    > TARGET_BLOCK_DORMANCY_COOLDOWN
                )
            ):
                raise ValueError(
                    "Target-block dormancy recovery state is invalid."
                )
            if np.any(
                (~initialized)
                & (
                    (residual_reference != 0.0)
                    | (reserve_radius != 0.0)
                    | (
                        restored_target_block_dormancy["floor_age"]
                        != 0
                    )
                    | (
                        restored_target_block_dormancy[
                            "stagnation_age"
                        ]
                        != 0
                    )
                    | (
                        restored_target_block_dormancy["cooldown"]
                        != 0
                    )
                    | (
                        restored_target_block_dormancy[
                            "activation_count"
                        ]
                        != 0
                    )
                )
            ):
                raise ValueError(
                    "Uninitialized target-block dormancy entries contain "
                    "persistent state."
                )
            lower_vector = np.broadcast_to(
                np.asarray(self.lb, dtype=np.float64),
                (self.D,),
            )
            upper_vector = np.broadcast_to(
                np.asarray(self.ub, dtype=np.float64),
                (self.D,),
            )
            block_span = (upper_vector - lower_vector).reshape(
                self.committee_target_num,
                self.committee_coordinate_dim,
            )
            block_diagonal = np.linalg.norm(block_span, axis=1)
            reserve_min = (
                self.target_block_field_radius_min_ratio
                * block_diagonal
            )
            reserve_max = (
                self.target_block_field_radius_max_ratio
                * block_diagonal
            )
            if np.any(
                (reserve_radius > 0.0)
                & (
                    (reserve_radius < reserve_min[None, :] - 1e-12)
                    | (
                        reserve_radius
                        > reserve_max[None, :] + 1e-12
                    )
                )
            ):
                raise ValueError(
                    "Target-block dormancy reserve radius is outside "
                    "the field envelope."
                )
            for name in (
                "last_residual_ratio",
                "last_progress_ratio",
                "last_disagreement",
                "last_disagreement_ratio",
                "last_restore_radius",
            ):
                if np.any(restored_target_block_dormancy[name] < 0.0):
                    raise ValueError(
                        "Target-block dormancy non-negative telemetry is "
                        f"invalid: {name}."
                    )
            events = int(dormancy_payload.get("events", 0))
            activations = int(
                dormancy_payload.get("activations", 0)
            )
            if events < 0 or activations < 0:
                raise ValueError(
                    "Target-block dormancy counters must be non-negative."
                )
            if (
                np.any(
                    restored_target_block_dormancy["floor_age"]
                    > events
                )
                or np.any(
                    restored_target_block_dormancy[
                        "stagnation_age"
                    ]
                    > events
                )
                or np.any(
                    restored_target_block_dormancy[
                        "activation_count"
                    ]
                    > events
                )
            ):
                raise ValueError(
                    "Target-block dormancy per-block counters exceed the "
                    "event clock."
                )
            if activations != int(
                np.sum(
                    restored_target_block_dormancy[
                        "activation_count"
                    ]
                )
            ):
                raise ValueError(
                    "Target-block dormancy activation accounting is "
                    "inconsistent."
                )
            if (
                self.target_block_dormancy_recovery_enable
                and (
                    restored_target_block_field is None
                    or events
                    != restored_target_block_field["events"]
                )
            ):
                raise ValueError(
                    "Target-block dormancy and field clocks disagree."
                )
            if (
                not self.target_block_dormancy_recovery_enable
                and (
                    events != 0
                    or activations != 0
                    or np.any(initialized)
                    or np.any(residual_reference != 0.0)
                    or np.any(reserve_radius != 0.0)
                    or any(
                        np.any(
                            restored_target_block_dormancy[name]
                        )
                        for name in (
                            "floor_age",
                            "stagnation_age",
                            "cooldown",
                            "activation_count",
                            "last_active",
                            "last_unresolved",
                            "last_material_progress",
                            "last_reliable_scale",
                            "last_residual_ratio",
                            "last_progress_ratio",
                            "last_disagreement",
                            "last_disagreement_ratio",
                            "last_restore_radius",
                        )
                    )
                )
            ):
                raise ValueError(
                    "Disabled target-block dormancy payload contains "
                    "active state."
                )
            restored_target_block_dormancy["events"] = events
            restored_target_block_dormancy[
                "activations"
            ] = activations
        direction_shadow_payload = state.get(
            "target_block_direction_shadow",
            None,
        )
        restored_target_block_direction_shadow = None
        if direction_shadow_payload is None:
            if self.target_block_direction_shadow_enable:
                raise ValueError(
                    "Cannot import a legacy objective-split state while "
                    "target-block direction shadow is enabled."
                )
        else:
            if not isinstance(direction_shadow_payload, dict):
                raise TypeError(
                    "target_block_direction_shadow must be a dictionary."
                )
            if int(
                direction_shadow_payload.get("state_version", -1)
            ) != TARGET_BLOCK_DIRECTION_SHADOW_STATE_VERSION:
                raise ValueError(
                    "Unsupported target-block direction shadow state "
                    "version."
                )
            if bool(
                direction_shadow_payload.get("enabled", False)
            ) != self.target_block_direction_shadow_enable:
                raise ValueError(
                    "Cannot import objective-split state across target-block "
                    "direction shadow enable modes."
                )
            if (
                int(direction_shadow_payload.get("n_agents", -1))
                != self.n_agents
                or int(
                    direction_shadow_payload.get("target_num", -1)
                )
                != self.committee_target_num
                or int(
                    direction_shadow_payload.get(
                        "coordinate_dim", -1
                    )
                )
                != self.committee_coordinate_dim
                or tuple(
                    direction_shadow_payload.get("sources", ())
                )
                != TARGET_BLOCK_DIRECTION_SHADOW_SOURCES
            ):
                raise ValueError(
                    "Target-block direction shadow layout/source mismatch."
                )
            block_shape = (
                self.n_agents,
                self.committee_target_num,
            )
            source_shape = block_shape + (
                len(TARGET_BLOCK_DIRECTION_SHADOW_SOURCES),
            )
            direction_shape = block_shape + (
                self.committee_coordinate_dim,
            )

            def _direction_shadow_float(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    direction_shadow_payload.get(name, None),
                    dtype=np.float64,
                )
                if value.shape != shape or not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        "Invalid target-block direction shadow floating "
                        f"tensor: {name}, shape={value.shape}."
                    )
                return value.copy()

            def _direction_shadow_bool(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    direction_shadow_payload.get(name, None),
                    dtype=bool,
                )
                if value.shape != shape:
                    raise ValueError(
                        "Invalid target-block direction shadow mask: "
                        f"{name}, shape={value.shape}."
                    )
                return value.copy()

            def _direction_shadow_int(name: str, shape) -> np.ndarray:
                value = np.asarray(
                    direction_shadow_payload.get(name, None),
                    dtype=np.int64,
                )
                if value.shape != shape:
                    raise ValueError(
                        "Invalid target-block direction shadow integer "
                        f"tensor: {name}, shape={value.shape}."
                    )
                return value.copy()

            restored_target_block_direction_shadow = {
                "last_eligible": _direction_shadow_bool(
                    "last_eligible", block_shape
                ),
                "last_selected": _direction_shadow_bool(
                    "last_selected", block_shape
                ),
                "last_candidate_valid": _direction_shadow_bool(
                    "last_candidate_valid", source_shape
                ),
                "last_candidate_angle": _direction_shadow_float(
                    "last_candidate_angle", source_shape
                ),
                "last_candidate_response": _direction_shadow_float(
                    "last_candidate_response", source_shape
                ),
                "last_candidate_positive": _direction_shadow_bool(
                    "last_candidate_positive", source_shape
                ),
                "last_best_response": _direction_shadow_float(
                    "last_best_response", block_shape
                ),
                "last_best_source": _direction_shadow_int(
                    "last_best_source", block_shape
                ),
                "last_best_sign": _direction_shadow_int(
                    "last_best_sign", block_shape
                ),
                "last_best_direction": _direction_shadow_float(
                    "last_best_direction", direction_shape
                ),
                "last_support": _direction_shadow_float(
                    "last_support", block_shape
                ),
                "last_conflict": _direction_shadow_float(
                    "last_conflict", block_shape
                ),
            }
            counters = {
                key: int(direction_shadow_payload.get(key, 0))
                for key in (
                    "events",
                    "eligible_blocks",
                    "probed_blocks",
                    "local_evals",
                    "comm_rounds",
                    "messages",
                    "transmitted_floats",
                )
            }
            if any(value < 0 for value in counters.values()):
                raise ValueError(
                    "Target-block direction shadow counters must be "
                    "non-negative."
                )
            source_num = len(TARGET_BLOCK_DIRECTION_SHADOW_SOURCES)
            if (
                np.any(
                    restored_target_block_direction_shadow[
                        "last_candidate_angle"
                    ]
                    < 0.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_candidate_angle"
                    ]
                    > 180.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_candidate_response"
                    ]
                    < 0.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_best_response"
                    ]
                    < 0.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_support"
                    ]
                    < 0.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_support"
                    ]
                    > 1.0 + 1e-12
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_conflict"
                    ]
                    < 0.0
                )
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_conflict"
                    ]
                    > 1.0 + 1e-12
                )
                or np.any(
                    (
                        restored_target_block_direction_shadow[
                            "last_best_source"
                        ]
                        < -1
                    )
                    | (
                        restored_target_block_direction_shadow[
                            "last_best_source"
                        ]
                        >= source_num
                    )
                )
                or np.any(
                    ~np.isin(
                        restored_target_block_direction_shadow[
                            "last_best_sign"
                        ],
                        (-1, 0, 1),
                    )
                )
            ):
                raise ValueError(
                    "Target-block direction shadow telemetry is invalid."
                )
            expected_messages = int(
                counters["comm_rounds"]
                * (
                    self.graph.directed_edge_count
                    if self.graph is not None
                    else 0
                )
            )
            if (
                counters["eligible_blocks"]
                > counters["events"]
                * self.n_agents
                * self.committee_target_num
                or counters["probed_blocks"]
                > counters["eligible_blocks"]
                or counters["local_evals"]
                < 3 * counters["probed_blocks"]
                or counters["local_evals"]
                > (
                    1
                    + 2
                    * len(TARGET_BLOCK_DIRECTION_SHADOW_SOURCES)
                )
                * counters["probed_blocks"]
                or counters["comm_rounds"] > counters["events"]
                or counters["messages"] != expected_messages
                or counters["transmitted_floats"]
                != expected_messages
                * self.committee_target_num
                * self.committee_coordinate_dim
                or np.any(
                    restored_target_block_direction_shadow[
                        "last_selected"
                    ]
                    & ~restored_target_block_direction_shadow[
                        "last_eligible"
                    ]
                )
                or np.any(
                    np.sum(
                        restored_target_block_direction_shadow[
                            "last_selected"
                        ],
                        axis=1,
                    )
                    > 1
                )
            ):
                raise ValueError(
                    "Target-block direction shadow accounting is "
                    "inconsistent."
                )
            if (
                self.target_block_direction_shadow_enable
                and (
                    restored_target_block_dormancy is None
                    or counters["events"]
                    != restored_target_block_dormancy["events"]
                )
            ):
                raise ValueError(
                    "Target-block direction shadow and dormancy clocks "
                    "disagree."
                )
            if (
                not self.target_block_direction_shadow_enable
                and (
                    any(counters.values())
                    or any(
                        np.any(
                            restored_target_block_direction_shadow[name]
                        )
                        for name in (
                            "last_eligible",
                            "last_selected",
                            "last_candidate_valid",
                            "last_candidate_angle",
                            "last_candidate_response",
                            "last_candidate_positive",
                            "last_best_response",
                            "last_best_sign",
                            "last_best_direction",
                            "last_support",
                            "last_conflict",
                        )
                    )
                    or np.any(
                        restored_target_block_direction_shadow[
                            "last_best_source"
                        ]
                        != -1
                    )
                )
            ):
                raise ValueError(
                    "Disabled target-block direction shadow payload "
                    "contains active state."
                )
            restored_target_block_direction_shadow.update(counters)
        challenge_payload = state.get(
            "target_block_challenge_response", None
        )
        restored_target_block_challenge = None
        if challenge_payload is None:
            if self.target_block_challenge_response_mode != "off":
                raise ValueError(
                    "Checkpoint is missing enabled target-block "
                    "challenge-response state."
                )
        else:
            if not isinstance(challenge_payload, dict):
                raise ValueError(
                    "target_block_challenge_response must be a dictionary."
                )
            if int(challenge_payload.get("state_version", -1)) != int(
                TARGET_BLOCK_CHALLENGE_RESPONSE_STATE_VERSION
            ):
                raise ValueError(
                    "Unsupported target-block challenge-response state version."
                )
            if str(challenge_payload.get("mode", "")) != str(
                self.target_block_challenge_response_mode
            ):
                raise ValueError(
                    "Checkpoint challenge-response mode mismatch."
                )
            if (
                int(challenge_payload.get("n_agents", -1)) != self.n_agents
                or int(challenge_payload.get("target_num", -1))
                != self.committee_target_num
                or int(challenge_payload.get("coordinate_dim", -1))
                != self.committee_coordinate_dim
                or tuple(challenge_payload.get("sources", ()))
                != tuple(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES)
            ):
                raise ValueError(
                    "Checkpoint challenge-response geometry mismatch."
                )
            challenge_block_shape = (
                self.n_agents,
                self.committee_target_num,
            )
            challenge_source_shape = challenge_block_shape + (
                len(TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES),
            )
            challenge_direction_shape = challenge_block_shape + (
                self.committee_coordinate_dim,
            )

            def _challenge_array(name, shape, dtype, *, finite=False):
                value = np.asarray(
                    challenge_payload.get(name, None), dtype=dtype
                )
                if value.shape != shape:
                    raise ValueError(
                        f"Challenge-response {name} has shape "
                        f"{value.shape}, expected {shape}."
                    )
                if finite and not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Challenge-response {name} must be finite."
                    )
                return value.copy()

            restored_target_block_challenge = {
                "last_selected": _challenge_array(
                    "last_selected", challenge_block_shape, bool
                ),
                "last_candidate_valid": _challenge_array(
                    "last_candidate_valid", challenge_source_shape, bool
                ),
                "last_candidate_angle": _challenge_array(
                    "last_candidate_angle", challenge_source_shape,
                    np.float64, finite=True
                ),
                "last_candidate_score": _challenge_array(
                    "last_candidate_score", challenge_source_shape,
                    np.float64, finite=True
                ),
                "last_candidate_sign": _challenge_array(
                    "last_candidate_sign", challenge_source_shape, np.int64
                ),
                "last_path_score": _challenge_array(
                    "last_path_score", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_alternative_score": _challenge_array(
                    "last_alternative_score", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_margin": _challenge_array(
                    "last_margin", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_best_source": _challenge_array(
                    "last_best_source", challenge_block_shape, np.int64
                ),
                "last_best_sign": _challenge_array(
                    "last_best_sign", challenge_block_shape, np.int64
                ),
                "last_best_direction": _challenge_array(
                    "last_best_direction", challenge_direction_shape,
                    np.float64, finite=True
                ),
                "last_coverage": _challenge_array(
                    "last_coverage", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_positive_sources": _challenge_array(
                    "last_positive_sources", challenge_block_shape, np.int64
                ),
                "last_neighbor_positive_sources": _challenge_array(
                    "last_neighbor_positive_sources", challenge_block_shape,
                    np.int64
                ),
                "last_support": _challenge_array(
                    "last_support", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_conflict": _challenge_array(
                    "last_conflict", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_actuation_eligible": _challenge_array(
                    "last_actuation_eligible", challenge_block_shape, bool
                ),
                "last_applied": _challenge_array(
                    "last_applied", challenge_block_shape, bool
                ),
                "last_applied_norm": _challenge_array(
                    "last_applied_norm", challenge_block_shape,
                    np.float64, finite=True
                ),
                "last_boundary_clipped": _challenge_array(
                    "last_boundary_clipped", challenge_block_shape, bool
                ),
            }
            for name in (
                "events", "challenges", "directed_responses",
                "local_evals", "reported_evals", "applied_commits",
                "comm_rounds", "messages", "transmitted_floats",
            ):
                value = int(challenge_payload.get(name, -1))
                if value < 0:
                    raise ValueError(
                        f"Challenge-response {name} must be non-negative."
                    )
                restored_target_block_challenge[name] = value
            if np.any(
                restored_target_block_challenge["last_applied"]
                & ~restored_target_block_challenge[
                    "last_actuation_eligible"
                ]
            ):
                raise ValueError(
                    "Applied challenge commits must be actuation eligible."
                )
            mode = self.target_block_challenge_response_mode
            if mode == "shadow" and (
                restored_target_block_challenge["reported_evals"] != 0
                or restored_target_block_challenge["applied_commits"] != 0
                or np.any(restored_target_block_challenge["last_applied"])
            ):
                raise ValueError(
                    "Shadow challenge-response cannot report or apply probes."
                )
            if mode == "actuate" and (
                restored_target_block_challenge["reported_evals"]
                != restored_target_block_challenge["local_evals"]
            ):
                raise ValueError(
                    "Actuating challenge-response must report every probe."
                )
            if mode == "off" and any(
                restored_target_block_challenge[name] != 0
                for name in (
                    "events", "challenges", "directed_responses",
                    "local_evals", "reported_evals", "applied_commits",
                    "comm_rounds", "messages", "transmitted_floats",
                )
            ):
                raise ValueError(
                    "Disabled challenge-response checkpoint must be empty."
                )
        dual_clock_payload = state.get(
            "target_block_dual_clock",
            None,
        )
        restored_target_block_dual_clock = None
        if dual_clock_payload is None:
            if self.target_block_dual_clock_enable:
                raise ValueError(
                    "Cannot import a legacy objective-split state while "
                    "target-block dual-clock mode is enabled."
                )
        else:
            if not isinstance(dual_clock_payload, dict):
                raise TypeError(
                    "target_block_dual_clock must be a dictionary."
                )
            if int(
                dual_clock_payload.get("state_version", -1)
            ) != TARGET_BLOCK_DUAL_CLOCK_STATE_VERSION:
                raise ValueError(
                    "Unsupported target-block dual-clock state version."
                )
            if bool(
                dual_clock_payload.get("enabled", False)
            ) != self.target_block_dual_clock_enable:
                raise ValueError(
                    "Cannot import objective-split state across "
                    "target-block dual-clock enable modes."
                )
            if bool(
                dual_clock_payload.get("commit_lock_enabled", False)
            ) != self.target_block_dual_clock_commit_lock_enable:
                raise ValueError(
                    "Cannot import objective-split state across target-block "
                    "dual-clock commit-lock modes."
                )
            saved_commit_credit_mode = str(
                dual_clock_payload.get("commit_credit_mode", "off")
            ).lower()
            if (
                saved_commit_credit_mode
                != self.target_block_commit_credit_mode
            ):
                raise ValueError(
                    "Cannot import objective-split state across target-block "
                    "commit-credit modes."
                )
            if int(
                dual_clock_payload.get("n_agents", -1)
            ) != self.n_agents:
                raise ValueError(
                    "Target-block dual-clock agent layout mismatch."
                )
            dual_counters = {
                key: int(dual_clock_payload.get(key, 0))
                for key in (
                    "outer_events",
                    "local_generation_ticks",
                    "communication_ticks",
                    "commit_ticks",
                    "last_microcycles",
                )
            }
            if any(value < 0 for value in dual_counters.values()):
                raise ValueError(
                    "Target-block dual-clock counters must be "
                    "non-negative."
                )
            last_generation_fes = np.asarray(
                dual_clock_payload.get(
                    "last_generation_fes_per_agent",
                    None,
                ),
                dtype=np.int64,
            )
            if (
                last_generation_fes.shape != (self.n_agents,)
                or np.any(last_generation_fes < 0)
            ):
                raise ValueError(
                    "Invalid target-block dual-clock generation-FEs "
                    "telemetry."
                )
            ticks = dual_counters["local_generation_ticks"]
            if (
                dual_counters["communication_ticks"] != ticks
                or dual_counters["commit_ticks"] != ticks
            ):
                raise ValueError(
                    "Target-block dual-clock tick accounting is "
                    "inconsistent."
                )
            if dual_counters["outer_events"] == 0:
                if (
                    ticks != 0
                    or dual_counters["last_microcycles"] != 0
                    or np.any(last_generation_fes != 0)
                ):
                    raise ValueError(
                        "Empty target-block dual-clock state contains "
                        "active telemetry."
                    )
            elif (
                dual_counters["last_microcycles"] < 2
                or ticks < dual_counters["last_microcycles"]
                or np.any(last_generation_fes <= 0)
            ):
                raise ValueError(
                    "Active target-block dual-clock telemetry is "
                    "inconsistent."
                )
            if (
                not self.target_block_dual_clock_enable
                and (
                    any(dual_counters.values())
                    or np.any(last_generation_fes != 0)
                )
            ):
                raise ValueError(
                    "Disabled target-block dual-clock payload contains "
                    "active state."
                )
            if (
                self.target_block_dual_clock_enable
                and (
                    restored_target_block_field is None
                    or restored_target_block_field["events"] != ticks
                    or restored_target_block_field["comm_rounds"]
                    != ticks
                )
            ):
                raise ValueError(
                    "Target-block dual-clock and field clocks disagree."
                )
            commit_credit_events = int(
                dual_clock_payload.get("commit_credit_events", 0)
            )
            if commit_credit_events < 0:
                raise ValueError(
                    "Target-block commit-credit event count must be "
                    "non-negative."
                )
            expected_credit_events = (
                int(self.n_agents * dual_counters["commit_ticks"])
                if saved_commit_credit_mode != "off"
                else 0
            )
            if commit_credit_events != expected_credit_events:
                raise ValueError(
                    "Target-block commit-credit and dual-clock event counts "
                    "disagree."
                )

            def _commit_credit_array(
                name: str,
                *,
                lower: float = 0.0,
                upper: float = np.inf,
                default: float = 0.0,
            ) -> np.ndarray:
                value = np.asarray(
                    dual_clock_payload.get(
                        name,
                        np.full((self.n_agents,), default),
                    ),
                    dtype=np.float64,
                )
                if (
                    value.shape != (self.n_agents,)
                    or not np.all(np.isfinite(value))
                    or np.any(value < lower)
                    or np.any(value > upper)
                ):
                    raise ValueError(
                        "Invalid target-block commit-credit telemetry: "
                        f"{name}."
                    )
                return value.copy()

            commit_credit_arrays = {
                "last_commit_credit_active": _commit_credit_array(
                    "last_commit_credit_active", upper=1.0
                ),
                "last_commit_credit": _commit_credit_array(
                    "last_commit_credit", upper=1.0
                ),
                "last_commit_credit_cosine": _commit_credit_array(
                    "last_commit_credit_cosine",
                    lower=-1.0,
                    upper=1.0,
                ),
                "last_commit_credit_proposal_norm": _commit_credit_array(
                    "last_commit_credit_proposal_norm"
                ),
                "last_commit_credit_commit_norm": _commit_credit_array(
                    "last_commit_credit_commit_norm"
                ),
                "last_commit_credit_correction_norm": _commit_credit_array(
                    "last_commit_credit_correction_norm"
                ),
                "last_commit_credit_path_retention": _commit_credit_array(
                    "last_commit_credit_path_retention",
                    upper=1.0,
                    default=1.0,
                ),
                "last_commit_credit_scale_retention": _commit_credit_array(
                    "last_commit_credit_scale_retention",
                    upper=1.0,
                    default=1.0,
                ),
                "last_commit_credit_axis_rms_before": _commit_credit_array(
                    "last_commit_credit_axis_rms_before"
                ),
                "last_commit_credit_axis_rms_after": _commit_credit_array(
                    "last_commit_credit_axis_rms_after"
                ),
                "last_commit_credit_sigma_path_before": _commit_credit_array(
                    "last_commit_credit_sigma_path_before"
                ),
                "last_commit_credit_sigma_path_after": _commit_credit_array(
                    "last_commit_credit_sigma_path_after"
                ),
                "last_commit_credit_cov_path_before": _commit_credit_array(
                    "last_commit_credit_cov_path_before"
                ),
                "last_commit_credit_cov_path_after": _commit_credit_array(
                    "last_commit_credit_cov_path_after"
                ),
            }
            if (
                saved_commit_credit_mode == "off"
                and (
                    np.any(
                        commit_credit_arrays[
                            "last_commit_credit_active"
                        ]
                        != 0.0
                    )
                    or np.any(
                        commit_credit_arrays[
                            "last_commit_credit_path_retention"
                        ]
                        != 1.0
                    )
                    or np.any(
                        commit_credit_arrays[
                            "last_commit_credit_scale_retention"
                        ]
                        != 1.0
                    )
                )
            ):
                raise ValueError(
                    "Disabled target-block commit credit contains active "
                    "telemetry."
                )
            restored_target_block_dual_clock = {
                **dual_counters,
                "commit_credit_events": commit_credit_events,
                **commit_credit_arrays,
                "last_generation_fes_per_agent": (
                    last_generation_fes.copy()
                ),
            }
        persistent_payload = state.get("persistent_sepcmaes_bank", None)
        restored_persistent_bank = None
        if persistent_payload is None:
            if self.persistent_sepcmaes_enable:
                raise ValueError(
                    "Cannot import a legacy objective-split state while the "
                    "persistent SepCMAES bank is enabled."
                )
        else:
            if not isinstance(persistent_payload, dict):
                raise TypeError(
                    "persistent_sepcmaes_bank must be a dictionary."
                )
            if int(
                persistent_payload.get("state_version", -1)
            ) != PERSISTENT_SEPCMAES_BANK_STATE_VERSION:
                raise ValueError(
                    "Unsupported persistent SepCMAES bank state version."
                )
            saved_enabled = bool(
                persistent_payload.get("enabled", False)
            )
            if saved_enabled != self.persistent_sepcmaes_enable:
                raise ValueError(
                    "Cannot import objective-split state across persistent "
                    "SepCMAES enable modes."
                )
            if (
                int(persistent_payload.get("dimension", -1)) != self.D
                or int(persistent_payload.get("n_agents", -1))
                != self.n_agents
            ):
                raise ValueError(
                    "Persistent SepCMAES bank dimension/agent count mismatch."
                )
            saved_ratio = float(
                persistent_payload.get("recenter_max_ratio", np.nan)
            )
            if not np.isclose(
                saved_ratio,
                self.persistent_sepcmaes_recenter_max_ratio,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(
                    "Persistent SepCMAES recenter ratio mismatch."
                )
            states = persistent_payload.get("states", [])
            signatures = persistent_payload.get("signatures", [])
            if (
                not isinstance(states, list)
                or not isinstance(signatures, list)
                or len(states) != self.n_agents
                or len(signatures) != self.n_agents
            ):
                raise ValueError(
                    "Persistent SepCMAES bank entry count mismatch."
                )
            for agent_id, (entry, signature) in enumerate(
                zip(states, signatures)
            ):
                if (entry is None) != (signature is None):
                    raise ValueError(
                        "Persistent SepCMAES state/signature presence "
                        f"mismatch for agent {agent_id}."
                    )
                if entry is not None:
                    if not saved_enabled:
                        raise ValueError(
                            "Disabled persistent SepCMAES bank contains "
                            "active state."
                        )
                    validate_persistent_sepcmaes_snapshot(entry, signature)
            counters = {
                key: int(persistent_payload.get(key, 0))
                for key in (
                    "events",
                    "fresh_events",
                    "resume_events",
                    "config_resets",
                )
            }
            if any(value < 0 for value in counters.values()):
                raise ValueError(
                    "Persistent SepCMAES counters must be non-negative."
                )
            if (
                counters["fresh_events"] + counters["resume_events"]
                != counters["events"]
                or counters["config_resets"] > counters["fresh_events"]
            ):
                raise ValueError(
                    "Persistent SepCMAES counters are inconsistent."
                )

            def _persistent_array(name: str) -> np.ndarray:
                value = np.asarray(
                    persistent_payload.get(
                        name, np.zeros((self.n_agents,))
                    ),
                    dtype=np.float64,
                ).reshape(self.n_agents)
                if not np.all(np.isfinite(value)) or np.any(value < 0.0):
                    raise ValueError(
                        f"Persistent SepCMAES telemetry is invalid: {name}."
                    )
                return value.copy()

            restored_persistent_bank = {
                "states": copy.deepcopy(states),
                "signatures": copy.deepcopy(signatures),
                "events": counters["events"],
                "fresh_events": counters["fresh_events"],
                "resume_events": counters["resume_events"],
                "config_resets": counters["config_resets"],
                "last_active": _persistent_array("last_active"),
                "last_fresh": _persistent_array("last_fresh"),
                "last_config_reset": _persistent_array(
                    "last_config_reset"
                ),
                "last_recenter_requested": _persistent_array(
                    "last_recenter_requested"
                ),
                "last_recenter_applied": _persistent_array(
                    "last_recenter_applied"
                ),
                "last_recenter_remaining": _persistent_array(
                    "last_recenter_remaining"
                ),
            }
            for name in (
                "last_active",
                "last_fresh",
                "last_config_reset",
            ):
                if np.any(restored_persistent_bank[name] > 1.0):
                    raise ValueError(
                        "Persistent SepCMAES binary telemetry is outside "
                        f"[0,1]: {name}."
                    )
        if self.target_block_dual_clock_enable:
            if (
                restored_target_block_dual_clock is None
                or restored_persistent_bank is None
            ):
                raise ValueError(
                    "Target-block dual-clock checkpoint requires both "
                    "field-clock and persistent-bank payloads."
                )
            expected_bank_events = int(
                self.n_agents
                * restored_target_block_dual_clock[
                    "local_generation_ticks"
                ]
            )
            if (
                restored_persistent_bank["events"]
                != expected_bank_events
            ):
                raise ValueError(
                    "Target-block dual-clock and persistent-bank clocks "
                    "disagree."
                )
        saved_information_mode = str(
            state.get("information_mode", "legacy_global")
        ).lower()
        if saved_information_mode != self.information_mode:
            raise ValueError(
                "Cannot import objective-split state across information "
                f"semantics: saved={saved_information_mode}, "
                f"current={self.information_mode}."
            )
        replacement_state_version = state.get(
            "guide_replacement_state_version", None
        )
        if replacement_state_version is not None:
            if int(replacement_state_version) != 1:
                raise ValueError(
                    "Unsupported guide replacement state version: "
                    f"{replacement_state_version}."
                )
            saved_mode = str(
                state.get("guide_replacement_mode", "off")
            ).lower()
            saved_scope = str(
                state.get(
                    "guide_replacement_scope", "guide_optimizers"
                )
            ).lower()
            if saved_mode != self.guide_replacement_mode:
                raise ValueError(
                    "Cannot import guide replacement state across modes: "
                    f"saved={saved_mode}, current={self.guide_replacement_mode}."
                )
            if saved_scope != self.guide_replacement_scope:
                raise ValueError(
                    "Cannot import guide replacement state across scopes: "
                    f"saved={saved_scope}, current={self.guide_replacement_scope}."
                )
            if int(state.get("guide_replacement_dimension", -1)) != self.D or int(
                state.get("guide_replacement_n_agents", -1)
            ) != self.n_agents:
                raise ValueError(
                    "Guide replacement state dimension/agent count mismatch."
                )
        collective_state_version = state.get(
            "collective_guide_state_version", None
        )
        if collective_state_version is None:
            if self.collective_guide_enabled:
                raise ValueError(
                    "Cannot import legacy state into an active collective guide mode."
                )
        else:
            if int(collective_state_version) != 1:
                raise ValueError(
                    "Unsupported collective guide state version: "
                    f"{collective_state_version}."
                )
            saved_collective_mode = str(
                state.get("collective_guide_mode", "off")
            ).lower()
            if saved_collective_mode != self.collective_guide_mode:
                raise ValueError(
                    "Cannot import collective guide state across modes: "
                    f"saved={saved_collective_mode}, current={self.collective_guide_mode}."
                )
            saved_null_rate = float(
                state.get("collective_guide_null_veto_rate", 0.0)
            )
            if not np.isclose(
                saved_null_rate,
                self.collective_guide_null_veto_rate,
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    "Cannot import collective guide state across null veto rates."
                )
            if int(state.get("collective_guide_dimension", -1)) != self.D or int(
                state.get("collective_guide_n_agents", -1)
            ) != self.n_agents:
                raise ValueError(
                    "Collective guide state dimension/agent count mismatch."
                )
        restored_rvcpd_state = None
        rvcpd_state_version = state.get("rvcpd_state_version", None)
        if rvcpd_state_version is not None:
            if int(rvcpd_state_version) != 1:
                raise ValueError(
                    f"Unsupported RVCPD state version: {rvcpd_state_version}."
                )
            saved_arm = str(state.get("rvcpd_arm", "off")).lower()
            if saved_arm != self.rvcpd_arm:
                raise ValueError(
                    "Cannot import RVCPD state across arm contracts: "
                    f"saved={saved_arm}, current={self.rvcpd_arm}."
                )
            saved_integration_mode = str(
                state.get("rvcpd_integration_mode", "isolated")
            ).lower()
            if saved_integration_mode != self.rvcpd_integration_mode:
                raise ValueError(
                    "Cannot import RVCPD state across integration contracts: "
                    f"saved={saved_integration_mode}, "
                    f"current={self.rvcpd_integration_mode}."
                )
            if int(state.get("rvcpd_dimension", -1)) != self.D or int(
                state.get("rvcpd_n_agents", -1)
            ) != self.n_agents:
                raise ValueError(
                    "RVCPD state dimension/agent count mismatch."
                )
            if (
                str(state.get("rvcpd_message_schema", ""))
                != "unit_direction_valid_age_step_ratio_v1"
            ):
                raise ValueError("RVCPD message schema mismatch.")
            array_specs = {
                "path": ("rvcpd_path", np.float64, (self.n_agents, self.D)),
                "support": (
                    "rvcpd_support_ema",
                    np.float64,
                    (self.n_agents,),
                ),
                "conflict": (
                    "rvcpd_conflict_ema",
                    np.float64,
                    (self.n_agents,),
                ),
                "uncertainty": (
                    "rvcpd_uncertainty_ema",
                    np.float64,
                    (self.n_agents,),
                ),
                "trust": ("rvcpd_trust", np.float64, (self.n_agents,)),
                "scale": ("rvcpd_scale", np.float64, (self.n_agents,)),
                "age": ("rvcpd_age", np.int64, (self.n_agents,)),
                "last_direction": (
                    "last_rvcpd_direction",
                    np.float64,
                    (self.n_agents, self.D),
                ),
                "last_sign": (
                    "last_rvcpd_sign",
                    np.int64,
                    (self.n_agents,),
                ),
                "last_active": (
                    "last_rvcpd_active",
                    np.float64,
                    (self.n_agents,),
                ),
                "last_agreement": (
                    "last_rvcpd_agreement",
                    np.float64,
                    (self.n_agents,),
                ),
                "last_requested": (
                    "last_rvcpd_requested_radius",
                    np.float64,
                    (self.n_agents,),
                ),
                "last_applied": (
                    "last_rvcpd_applied_radius",
                    np.float64,
                    (self.n_agents,),
                ),
                "last_plus": (
                    "last_rvcpd_plus_gain",
                    np.float64,
                    (self.n_agents,),
                ),
                "last_minus": (
                    "last_rvcpd_minus_gain",
                    np.float64,
                    (self.n_agents,),
                ),
            }
            restored_arrays = {}
            for short_name, (state_name, dtype, shape) in array_specs.items():
                value = np.asarray(state[state_name], dtype=dtype)
                if value.shape != shape:
                    raise ValueError(
                        f"{state_name} shape mismatch: "
                        f"expected {shape}, got {value.shape}."
                    )
                if np.issubdtype(value.dtype, np.floating) and not np.all(
                    np.isfinite(value)
                ):
                    raise ValueError(
                        f"{state_name} contains non-finite values."
                    )
                restored_arrays[short_name] = value.copy()
            if np.any(
                np.linalg.norm(restored_arrays["path"], axis=1) > 1.0 + 1e-10
            ):
                raise ValueError("RVCPD path norm exceeds its contract.")
            if np.any(restored_arrays["age"] < 0):
                raise ValueError("RVCPD age must be nonnegative.")
            if np.any(restored_arrays["trust"] < 0.0) or np.any(
                restored_arrays["trust"] > 1.0
            ):
                raise ValueError("RVCPD trust is outside [0,1].")
            if np.any(
                restored_arrays["scale"] < self.rvcpd_scale_min - 1e-12
            ) or np.any(
                restored_arrays["scale"] > self.rvcpd_scale_max + 1e-12
            ):
                raise ValueError("RVCPD scale is outside configured bounds.")
            if not np.all(
                np.isin(restored_arrays["last_sign"], [-1, 0, 1])
            ):
                raise ValueError("RVCPD sign metadata is invalid.")
            counter_names = (
                "rvcpd_events",
                "rvcpd_valid_event_count",
                "rvcpd_commit_count",
                "rvcpd_reverse_count",
                "rvcpd_noop_count",
                "rvcpd_probe_local_evals",
                "rvcpd_messages",
                "rvcpd_transmitted_floats",
            )
            restored_counters = {
                name: int(state.get(name, 0)) for name in counter_names
            }
            if any(value < 0 for value in restored_counters.values()):
                raise ValueError("RVCPD counters must be nonnegative.")
            restored_rvcpd_state = {
                **restored_arrays,
                **restored_counters,
            }
        restored_multisecant_state = None
        multisecant_state_version = state.get(
            "candidate_multisecant_state_version",
            None,
        )
        if multisecant_state_version is not None:
            if int(multisecant_state_version) != 1:
                raise ValueError(
                    "Unsupported candidate multisecant state version: "
                    f"{multisecant_state_version}."
                )
            saved_generator = str(
                state.get(
                    "candidate_multisecant_generator",
                    "",
                )
            ).lower()
            if saved_generator != self.candidate_generator:
                raise ValueError(
                    "Cannot import candidate multisecant history across "
                    "generator contracts: "
                    f"saved={saved_generator}, "
                    f"current={self.candidate_generator}."
                )
            history_shape = (
                self.n_agents,
                self.committee_target_num,
                self.candidate_history_size,
            )
            step_shape = history_shape + (
                self.committee_coordinate_dim,
            )
            restored_step = np.asarray(
                state["candidate_multisecant_history_step"],
                dtype=np.float64,
            )
            restored_response = np.asarray(
                state["candidate_multisecant_history_response"],
                dtype=np.float64,
            )
            restored_center = np.asarray(
                state["candidate_multisecant_history_center"],
                dtype=np.float64,
            )
            restored_event_id = np.asarray(
                state["candidate_multisecant_history_event_id"],
                dtype=np.int64,
            )
            restored_valid = np.asarray(
                state["candidate_multisecant_history_valid"],
                dtype=bool,
            )
            restored_cursor = np.asarray(
                state["candidate_multisecant_history_cursor"],
                dtype=np.int64,
            )
            restored_count = np.asarray(
                state["candidate_multisecant_history_count"],
                dtype=np.int64,
            )
            restored_active = np.asarray(
                state["last_candidate_multisecant_active"],
                dtype=bool,
            )
            restored_reason = np.asarray(
                state["last_candidate_multisecant_fallback_reason"],
                dtype=np.int64,
            )
            expected_shapes = {
                "candidate_multisecant_history_step": (
                    restored_step.shape,
                    step_shape,
                ),
                "candidate_multisecant_history_response": (
                    restored_response.shape,
                    history_shape,
                ),
                "candidate_multisecant_history_center": (
                    restored_center.shape,
                    step_shape,
                ),
                "candidate_multisecant_history_event_id": (
                    restored_event_id.shape,
                    history_shape,
                ),
                "candidate_multisecant_history_valid": (
                    restored_valid.shape,
                    history_shape,
                ),
                "candidate_multisecant_history_cursor": (
                    restored_cursor.shape,
                    history_shape[:2],
                ),
                "candidate_multisecant_history_count": (
                    restored_count.shape,
                    history_shape[:2],
                ),
                "last_candidate_multisecant_active": (
                    restored_active.shape,
                    history_shape[:2],
                ),
                "last_candidate_multisecant_fallback_reason": (
                    restored_reason.shape,
                    history_shape[:2],
                ),
            }
            for name, (actual, expected) in expected_shapes.items():
                if actual != expected:
                    raise ValueError(
                        f"{name} shape mismatch: "
                        f"expected {expected}, got {actual}."
                    )
            if (
                not np.all(np.isfinite(restored_step))
                or not np.all(np.isfinite(restored_response))
                or not np.all(np.isfinite(restored_center))
            ):
                raise ValueError(
                    "Candidate multisecant history contains non-finite values."
                )
            saved_candidate_events = int(
                state.get("candidate_response_events", 0)
            )
            valid_counts = np.count_nonzero(restored_valid, axis=2)
            if (
                saved_candidate_events < 0
                or np.any(restored_cursor < 0)
                or np.any(restored_cursor >= self.candidate_history_size)
                or np.any(restored_count < 0)
                or np.any(restored_count > self.candidate_history_size)
                or np.any(valid_counts != restored_count)
                or np.any(
                    (restored_count < self.candidate_history_size)
                    & (restored_cursor != restored_count)
                )
                or np.any(restored_event_id[restored_valid] < 0)
                or np.any(
                    restored_event_id[restored_valid]
                    >= saved_candidate_events
                )
                or np.any(restored_reason < 0)
                or np.any(restored_reason > 6)
            ):
                raise ValueError(
                    "Candidate multisecant history metadata is inconsistent."
                )
            restored_multisecant_state = {
                "step": restored_step.copy(),
                "response": restored_response.copy(),
                "center": restored_center.copy(),
                "event_id": restored_event_id.copy(),
                "valid": restored_valid.copy(),
                "cursor": restored_cursor.copy(),
                "count": restored_count.copy(),
                "active": restored_active.copy(),
                "reason": restored_reason.copy(),
            }
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
        self.last_committed_local_improve = np.asarray(
            state.get(
                "last_committed_local_improve",
                self.last_committed_local_improve,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        committed_history = state.get("committed_local_improve_hist", None)
        if committed_history is not None:
            if len(committed_history) != self.n_agents:
                raise ValueError(
                    "committed_local_improve_hist agent count mismatch: "
                    f"expected {self.n_agents}, got {len(committed_history)}."
                )
            self.committed_local_improve_hist = [
                [float(value) for value in history]
                for history in committed_history
            ]
        self.last_consensus_local_effect = np.asarray(
            state.get(
                "last_consensus_local_effect",
                self.last_consensus_local_effect,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_local_commit_success = np.asarray(
            state.get(
                "last_local_commit_success",
                self.last_local_commit_success,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.report_current_x = np.asarray(
            state.get("report_current_x", self.current_x),
            dtype=np.float64,
        ).reshape(self.D)
        self.report_current_f = float(
            state.get("report_current_f", self.current_f)
        )
        self.report_best_x = np.asarray(
            state.get("report_best_x", self.gbest_x),
            dtype=np.float64,
        ).reshape(self.D)
        self.report_best_f = float(state.get("report_best_f", self.gbest_f))
        self._set_report_monitor_state(
            self.report_current_x,
            self.report_current_f,
        )
        if np.isfinite(self.report_best_f) and self.report_best_f < self.gbest_f:
            self.gbest_f = float(self.report_best_f)
            self.gbest_x = self.report_best_x.copy()
        self.last_neighbor_summary = np.asarray(
            state.get("last_neighbor_summary", self.last_neighbor_summary),
            dtype=np.float64,
        ).reshape(self.n_agents, 3)
        self.last_sigma_value = np.asarray(
            state.get("last_sigma_value", self.last_sigma_value),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.sigma_state_age = np.asarray(
            state.get("sigma_state_age", self.sigma_state_age),
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
        self.last_cmaes_numeric_fail_soft = np.asarray(
            state.get(
                "last_cmaes_numeric_fail_soft",
                self.last_cmaes_numeric_fail_soft,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_cmaes_numeric_fail_soft_generation = np.asarray(
            state.get(
                "last_cmaes_numeric_fail_soft_generation",
                self.last_cmaes_numeric_fail_soft_generation,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.cmaes_numeric_fail_soft_events = int(
            state.get(
                "cmaes_numeric_fail_soft_events",
                self.cmaes_numeric_fail_soft_events,
            )
        )
        self.last_consensus_shift_norm = np.asarray(
            state.get("last_consensus_shift_norm", self.last_consensus_shift_norm),
            dtype=np.float64,
        ).reshape(self.n_agents)
        def _restore_state_message(name: str, fallback: np.ndarray) -> np.ndarray:
            raw = np.asarray(state.get(name, fallback), dtype=np.float64)
            expected = (self.n_agents, self.state_msg_dim)
            if raw.size == int(np.prod(expected)):
                return raw.reshape(expected)
            if self.state_msg_dim == 0:
                return np.zeros(expected, dtype=np.float64)
            if raw.ndim == 2 and raw.shape[0] == self.n_agents:
                restored = np.zeros(expected, dtype=np.float64)
                width = min(self.state_msg_dim, int(raw.shape[1]))
                restored[:, :width] = raw[:, :width]
                return restored
            raise ValueError(
                f"{name} shape mismatch: expected {expected}, got {raw.shape}."
            )

        self.last_state_neighbor_message = _restore_state_message(
            "last_state_neighbor_message",
            self.last_state_neighbor_message,
        )
        self.last_state_delta_message = _restore_state_message(
            "last_state_delta_message",
            self.last_state_delta_message,
        )
        metrics = state.get("last_consensus_metrics", None)
        if isinstance(metrics, dict):
            self.last_consensus_metrics = {
                key: float(metrics.get(key, 0.0))
                for key in self._empty_consensus_metrics()
            }
        self.committee_direction = np.asarray(
            state.get("committee_direction", self.committee_direction),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.committee_confidence = np.asarray(
            state.get("committee_confidence", self.committee_confidence),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.committee_verified_x = np.asarray(
            state.get("committee_verified_x", self.committee_verified_x),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.committee_selected_source = np.asarray(
            state.get(
                "committee_selected_source",
                self.committee_selected_source,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents, self.committee_target_num)
        self.committee_candidate_score = np.asarray(
            state.get(
                "committee_candidate_score",
                self.committee_candidate_score,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents, self.committee_target_num)
        self.committee_acceptance_active = bool(
            state.get(
                "committee_acceptance_active",
                self.committee_acceptance_active,
            )
        )
        self.committee_candidate_global_f = np.asarray(
            state.get(
                "committee_candidate_global_f",
                self.committee_candidate_global_f,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.committee_candidate_report_f = float(
            state.get(
                "committee_candidate_report_f",
                self.committee_candidate_report_f,
            )
        )
        self.committee_candidate_vs_report_log_improve = float(
            state.get(
                "committee_candidate_vs_report_log_improve",
                self.committee_candidate_vs_report_log_improve,
            )
        )
        self.committee_effective_beta = np.asarray(
            state.get(
                "committee_effective_beta",
                self.committee_effective_beta,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.committee_base_alignment = np.asarray(
            state.get(
                "committee_base_alignment",
                self.committee_base_alignment,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        committee_metrics = state.get("last_committee_metrics", None)
        if isinstance(committee_metrics, dict):
            self.last_committee_metrics = {
                key: float(committee_metrics.get(key, 0.0))
                for key in self._empty_committee_metrics()
            }
        self.candidate_response_verified_x = np.asarray(
            state.get(
                "candidate_response_verified_x",
                self.candidate_response_verified_x,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.candidate_response_selected_source = np.asarray(
            state.get(
                "candidate_response_selected_source",
                self.candidate_response_selected_source,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents, self.committee_target_num)
        self.candidate_response_accepted_mask = np.asarray(
            state.get(
                "candidate_response_accepted_mask",
                self.candidate_response_accepted_mask,
            ),
            dtype=bool,
        ).reshape(self.n_agents, self.committee_target_num)
        self.last_candidate_accept_ratio = np.asarray(
            state.get(
                "last_candidate_accept_ratio",
                self.last_candidate_accept_ratio,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_candidate_selection_confidence = np.asarray(
            state.get(
                "last_candidate_selection_confidence",
                self.last_candidate_selection_confidence,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_candidate_support_ratio = np.asarray(
            state.get(
                "last_candidate_support_ratio",
                self.last_candidate_support_ratio,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_candidate_requested_shift_norm = np.asarray(
            state.get(
                "last_candidate_requested_shift_norm",
                self.last_candidate_requested_shift_norm,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_candidate_actuator_action = np.asarray(
            state.get(
                "last_candidate_actuator_action",
                self.last_candidate_actuator_action,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_candidate_actuator_beta = np.asarray(
            state.get(
                "last_candidate_actuator_beta",
                self.last_candidate_actuator_beta,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.candidate_response_shadow_base_x = np.asarray(
            state.get(
                "candidate_response_shadow_base_x",
                self.candidate_response_shadow_base_x,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.candidate_response_shadow_candidate_x = np.asarray(
            state.get(
                "candidate_response_shadow_candidate_x",
                self.candidate_response_shadow_candidate_x,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.candidate_response_shadow_due = bool(
            state.get(
                "candidate_response_shadow_due",
                self.candidate_response_shadow_due,
            )
        )
        candidate_metrics = state.get("last_candidate_response_metrics", None)
        if isinstance(candidate_metrics, dict):
            self.last_candidate_response_metrics = {
                key: float(candidate_metrics.get(key, default))
                for key, default in self._empty_candidate_response_metrics().items()
            }
        if restored_multisecant_state is not None:
            self.candidate_multisecant_history_step = (
                restored_multisecant_state["step"]
            )
            self.candidate_multisecant_history_response = (
                restored_multisecant_state["response"]
            )
            self.candidate_multisecant_history_center = (
                restored_multisecant_state["center"]
            )
            self.candidate_multisecant_history_event_id = (
                restored_multisecant_state["event_id"]
            )
            self.candidate_multisecant_history_valid = (
                restored_multisecant_state["valid"]
            )
            self.candidate_multisecant_history_cursor = (
                restored_multisecant_state["cursor"]
            )
            self.candidate_multisecant_history_count = (
                restored_multisecant_state["count"]
            )
            self.last_candidate_multisecant_active = (
                restored_multisecant_state["active"]
            )
            self.last_candidate_multisecant_fallback_reason = (
                restored_multisecant_state["reason"]
            )
        else:
            self.candidate_multisecant_history_step.fill(0.0)
            self.candidate_multisecant_history_response.fill(0.0)
            self.candidate_multisecant_history_center.fill(0.0)
            self.candidate_multisecant_history_event_id.fill(-1)
            self.candidate_multisecant_history_valid.fill(False)
            self.candidate_multisecant_history_cursor.fill(0)
            self.candidate_multisecant_history_count.fill(0)
            self.last_candidate_multisecant_active.fill(False)
            self.last_candidate_multisecant_fallback_reason.fill(1)
        if restored_rvcpd_state is None:
            self.rvcpd_path.fill(0.0)
            self.rvcpd_support_ema.fill(0.0)
            self.rvcpd_conflict_ema.fill(0.0)
            self.rvcpd_uncertainty_ema.fill(0.0)
            self.rvcpd_trust.fill(1.0)
            self.rvcpd_scale.fill(self.rvcpd_initial_scale)
            self.rvcpd_age.fill(0)
            self.last_rvcpd_direction.fill(0.0)
            self.last_rvcpd_sign.fill(0)
            self.last_rvcpd_active.fill(0.0)
            self.last_rvcpd_agreement.fill(0.0)
            self.last_rvcpd_requested_radius.fill(0.0)
            self.last_rvcpd_applied_radius.fill(0.0)
            self.last_rvcpd_plus_gain.fill(0.0)
            self.last_rvcpd_minus_gain.fill(0.0)
            self.rvcpd_events = 0
            self.rvcpd_valid_event_count = 0
            self.rvcpd_commit_count = 0
            self.rvcpd_reverse_count = 0
            self.rvcpd_noop_count = 0
            self.rvcpd_probe_local_evals = 0
            self.rvcpd_messages = 0
            self.rvcpd_transmitted_floats = 0
        else:
            self.rvcpd_path = restored_rvcpd_state["path"]
            self.rvcpd_support_ema = restored_rvcpd_state["support"]
            self.rvcpd_conflict_ema = restored_rvcpd_state["conflict"]
            self.rvcpd_uncertainty_ema = restored_rvcpd_state[
                "uncertainty"
            ]
            self.rvcpd_trust = restored_rvcpd_state["trust"]
            self.rvcpd_scale = restored_rvcpd_state["scale"]
            self.rvcpd_age = restored_rvcpd_state["age"]
            self.last_rvcpd_direction = restored_rvcpd_state[
                "last_direction"
            ]
            self.last_rvcpd_sign = restored_rvcpd_state["last_sign"]
            self.last_rvcpd_active = restored_rvcpd_state["last_active"]
            self.last_rvcpd_agreement = restored_rvcpd_state[
                "last_agreement"
            ]
            self.last_rvcpd_requested_radius = restored_rvcpd_state[
                "last_requested"
            ]
            self.last_rvcpd_applied_radius = restored_rvcpd_state[
                "last_applied"
            ]
            self.last_rvcpd_plus_gain = restored_rvcpd_state["last_plus"]
            self.last_rvcpd_minus_gain = restored_rvcpd_state["last_minus"]
            self.rvcpd_events = restored_rvcpd_state["rvcpd_events"]
            self.rvcpd_valid_event_count = restored_rvcpd_state[
                "rvcpd_valid_event_count"
            ]
            self.rvcpd_commit_count = restored_rvcpd_state[
                "rvcpd_commit_count"
            ]
            self.rvcpd_reverse_count = restored_rvcpd_state[
                "rvcpd_reverse_count"
            ]
            self.rvcpd_noop_count = restored_rvcpd_state[
                "rvcpd_noop_count"
            ]
            self.rvcpd_probe_local_evals = restored_rvcpd_state[
                "rvcpd_probe_local_evals"
            ]
            self.rvcpd_messages = restored_rvcpd_state["rvcpd_messages"]
            self.rvcpd_transmitted_floats = restored_rvcpd_state[
                "rvcpd_transmitted_floats"
            ]
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
        self.candidate_validation_local_evals = int(
            state.get(
                "candidate_validation_local_evals",
                self.candidate_validation_local_evals,
            )
        )
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
        self.state_comm_rounds = int(
            state.get("state_comm_rounds", self.state_comm_rounds)
        )
        self.state_comm_messages = int(
            state.get("state_comm_messages", self.state_comm_messages)
        )
        self.state_comm_transmitted_floats = int(
            state.get(
                "state_comm_transmitted_floats",
                self.state_comm_transmitted_floats,
            )
        )
        self.graph_comm_rounds = int(
            state.get("graph_comm_rounds", self.graph_comm_rounds)
        )
        self.event_slot_interleaving_events = int(
            state.get(
                "event_slot_interleaving_events",
                self.event_slot_interleaving_events,
            )
        )
        self.event_slot_interleaving_slots = int(
            state.get(
                "event_slot_interleaving_slots",
                self.event_slot_interleaving_slots,
            )
        )
        self.event_slot_native_local_evals = int(
            state.get(
                "event_slot_native_local_evals",
                self.event_slot_native_local_evals,
            )
        )
        self.event_slot_reported_local_evals = int(
            state.get(
                "event_slot_reported_local_evals",
                self.event_slot_reported_local_evals,
            )
        )
        self.event_slot_physical_local_evals = int(
            state.get(
                "event_slot_physical_local_evals",
                self.event_slot_physical_local_evals,
            )
        )
        self.last_event_slot_count = int(
            state.get("last_event_slot_count", self.last_event_slot_count)
        )
        raw_packet_units = state.get(
            "last_event_slot_packet_units",
            self.last_event_slot_packet_units,
        )
        self.last_event_slot_packet_units = np.asarray(
            raw_packet_units, dtype=np.int64
        ).reshape(-1, self.n_agents)
        self.last_event_slot_distribution_updates = np.asarray(
            state.get(
                "last_event_slot_distribution_updates",
                self.last_event_slot_distribution_updates,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_event_slot_native_evals = np.asarray(
            state.get(
                "last_event_slot_native_evals",
                self.last_event_slot_native_evals,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_event_slot_physical_evals = np.asarray(
            state.get(
                "last_event_slot_physical_evals",
                self.last_event_slot_physical_evals,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_event_slot_center_shift_norms = np.asarray(
            state.get(
                "last_event_slot_center_shift_norms",
                self.last_event_slot_center_shift_norms,
            ),
            dtype=np.float64,
        ).reshape(-1, self.n_agents)
        self.last_event_slot_packet_improvements = np.asarray(
            state.get(
                "last_event_slot_packet_improvements",
                np.zeros(
                    (int(self.last_event_slot_count), self.n_agents),
                    dtype=np.float64,
                ),
            ),
            dtype=np.float64,
        ).reshape(-1, self.n_agents)
        if self.last_event_slot_packet_improvements.shape != (
            int(self.last_event_slot_count), self.n_agents
        ) or not np.all(
            np.isfinite(self.last_event_slot_packet_improvements)
        ):
            raise ValueError(
                "Event-slot packet-improvement checkpoint is malformed."
            )
        raw_commit_audit = state.get(
            "last_event_slot_commit_audit", {}
        )
        if raw_commit_audit is None:
            raw_commit_audit = {}
        if not isinstance(raw_commit_audit, dict):
            raise ValueError(
                "Event-slot commit audit checkpoint must be a mapping."
            )
        restored_commit_audit = {}
        expected_audit_shape = (
            int(self.last_event_slot_count), self.n_agents
        )
        for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS:
            raw_value = raw_commit_audit.get(
                field,
                np.zeros(expected_audit_shape, dtype=np.float64),
            )
            value = np.asarray(raw_value, dtype=np.float64)
            if value.size != int(np.prod(expected_audit_shape)):
                raise ValueError(
                    "Event-slot commit audit checkpoint shape mismatch for "
                    f"{field}: {value.shape} != {expected_audit_shape}."
                )
            value = value.reshape(expected_audit_shape)
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    "Event-slot commit audit checkpoint contains non-finite "
                    f"values for {field}."
                )
            restored_commit_audit[field] = value.copy()
        self.last_event_slot_commit_audit = restored_commit_audit
        self.last_event_slot_mmes_neutral_success = np.asarray(
            state.get(
                "last_event_slot_mmes_neutral_success",
                self.last_event_slot_mmes_neutral_success,
            ),
            dtype=np.int64,
        ).reshape(self.n_agents)
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
        self.committee_events = int(
            state.get("committee_events", self.committee_events)
        )
        self.committee_decision_local_evals = int(
            state.get(
                "committee_decision_local_evals",
                self.committee_decision_local_evals,
            )
        )
        self.committee_shadow_local_evals = int(
            state.get(
                "committee_shadow_local_evals",
                self.committee_shadow_local_evals,
            )
        )
        self.committee_shadow_global_local_evals = int(
            state.get(
                "committee_shadow_global_local_evals",
                self.committee_shadow_global_local_evals,
            )
        )
        self.committee_messages = int(
            state.get("committee_messages", self.committee_messages)
        )
        self.committee_transmitted_floats = int(
            state.get(
                "committee_transmitted_floats",
                self.committee_transmitted_floats,
            )
        )
        self.candidate_response_events = int(
            state.get("candidate_response_events", self.candidate_response_events)
        )
        self.candidate_response_actuated_events = int(
            state.get(
                "candidate_response_actuated_events",
                self.candidate_response_actuated_events,
            )
        )
        self.candidate_response_probe_local_evals = int(
            state.get(
                "candidate_response_probe_local_evals",
                self.candidate_response_probe_local_evals,
            )
        )
        self.candidate_response_verification_local_evals = int(
            state.get(
                "candidate_response_verification_local_evals",
                self.candidate_response_verification_local_evals,
            )
        )
        self.candidate_response_shadow_global_local_evals = int(
            state.get(
                "candidate_response_shadow_global_local_evals",
                self.candidate_response_shadow_global_local_evals,
            )
        )
        self.candidate_response_rounds = int(
            state.get("candidate_response_rounds", self.candidate_response_rounds)
        )
        self.candidate_response_messages = int(
            state.get("candidate_response_messages", self.candidate_response_messages)
        )
        self.candidate_response_transmitted_floats = int(
            state.get(
                "candidate_response_transmitted_floats",
                self.candidate_response_transmitted_floats,
            )
        )
        self.committee_acceptance_events = int(
            state.get(
                "committee_acceptance_events",
                self.committee_acceptance_events,
            )
        )
        self.committee_acceptance_accepted_events = int(
            state.get(
                "committee_acceptance_accepted_events",
                self.committee_acceptance_accepted_events,
            )
        )
        self.committee_acceptance_rejected_events = int(
            state.get(
                "committee_acceptance_rejected_events",
                self.committee_acceptance_rejected_events,
            )
        )
        self.committee_acceptance_local_evals = int(
            state.get(
                "committee_acceptance_local_evals",
                self.committee_acceptance_local_evals,
            )
        )
        self.committee_acceptance_messages = int(
            state.get(
                "committee_acceptance_messages",
                self.committee_acceptance_messages,
            )
        )
        self.committee_acceptance_transmitted_floats = int(
            state.get(
                "committee_acceptance_transmitted_floats",
                self.committee_acceptance_transmitted_floats,
            )
        )
        self.committee_acceptance_stage_events = np.asarray(
            state.get(
                "committee_acceptance_stage_events",
                self.committee_acceptance_stage_events,
            ),
            dtype=np.int64,
        ).reshape(3)
        self.committee_acceptance_stage_accepted_events = np.asarray(
            state.get(
                "committee_acceptance_stage_accepted_events",
                self.committee_acceptance_stage_accepted_events,
            ),
            dtype=np.int64,
        ).reshape(3)
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
        self.optimizer_guide_direction = np.asarray(
            state.get("optimizer_guide_direction", self.optimizer_guide_direction),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.last_optimizer_guide_norm = np.asarray(
            state.get("last_optimizer_guide_norm", self.last_optimizer_guide_norm),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_applied = np.asarray(
            state.get("last_optimizer_guide_applied", self.last_optimizer_guide_applied),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_alignment = np.asarray(
            state.get("last_optimizer_guide_alignment", self.last_optimizer_guide_alignment),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_source_active_ratio = float(
            state.get(
                "last_optimizer_guide_source_active_ratio",
                self.last_optimizer_guide_source_active_ratio,
            )
        )
        replacement_arrays = {
            "guide_replacement_direction": (np.float64, (self.n_agents, self.D)),
            "last_guide_replacement_eligible": (np.float64, (self.n_agents,)),
            "last_guide_replacement_valid": (np.float64, (self.n_agents,)),
            "last_guide_replacement_old_guide_suppressed": (
                np.float64,
                (self.n_agents,),
            ),
            "last_guide_replacement_sign": (np.int64, (self.n_agents,)),
            "last_guide_replacement_strength": (np.float64, (self.n_agents,)),
            "last_guide_replacement_sigma": (np.float64, (self.n_agents,)),
            "last_guide_replacement_requested_radius": (
                np.float64,
                (self.n_agents,),
            ),
            "last_guide_replacement_applied_radius": (
                np.float64,
                (self.n_agents,),
            ),
            "last_guide_replacement_plus_gain": (np.float64, (self.n_agents,)),
            "last_guide_replacement_minus_gain": (np.float64, (self.n_agents,)),
        }
        for attr, (dtype, shape) in replacement_arrays.items():
            current = getattr(self, attr)
            setattr(
                self,
                attr,
                np.asarray(state.get(attr, current), dtype=dtype).reshape(shape),
            )
        for attr in (
            "guide_replacement_events",
            "guide_replacement_eligible_count",
            "guide_replacement_valid_event_count",
            "guide_replacement_forward_count",
            "guide_replacement_reverse_count",
            "guide_replacement_noop_count",
            "guide_replacement_commit_count",
            "guide_replacement_probe_local_evals",
        ):
            setattr(self, attr, int(state.get(attr, getattr(self, attr))))
        collective_arrays = {
            "last_collective_guide_shared_base": (
                np.float64,
                (self.n_agents, self.D),
            ),
            "last_collective_guide_direction": (
                np.float64,
                (self.n_agents, self.D),
            ),
            "last_collective_guide_radius": (np.float64, (self.n_agents,)),
            "last_collective_guide_vote": (np.int64, (self.n_agents,)),
            "last_collective_guide_source_valid": (
                np.float64,
                (self.n_agents,),
            ),
            "last_collective_guide_vote_valid": (
                np.float64,
                (self.n_agents,),
            ),
            "last_collective_guide_suppressed": (
                np.float64,
                (self.n_agents,),
            ),
        }
        for attr, (dtype, shape) in collective_arrays.items():
            current = getattr(self, attr)
            setattr(
                self,
                attr,
                np.asarray(state.get(attr, current), dtype=dtype).reshape(shape),
            )
        for attr in (
            "last_collective_guide_shared_base",
            "last_collective_guide_direction",
            "last_collective_guide_radius",
            "last_collective_guide_source_valid",
            "last_collective_guide_vote_valid",
            "last_collective_guide_suppressed",
        ):
            if not np.all(np.isfinite(getattr(self, attr))):
                raise ValueError(
                    f"Collective guide state contains non-finite {attr}."
                )
        if np.any(self.last_collective_guide_radius < 0.0):
            raise ValueError("Collective guide radius must be nonnegative.")
        if not np.all(
            np.isin(self.last_collective_guide_vote, (-1, 0, 1))
        ):
            raise ValueError("Collective guide votes must be in {-1,0,1}.")
        for attr in (
            "last_collective_guide_source_valid",
            "last_collective_guide_vote_valid",
            "last_collective_guide_suppressed",
        ):
            values = getattr(self, attr)
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(
                    f"Collective guide binary state is outside [0,1]: {attr}."
                )
        self.last_collective_guide_valid = bool(
            state.get(
                "last_collective_guide_valid",
                self.last_collective_guide_valid,
            )
        )
        self.last_collective_guide_vote_sum = int(
            state.get(
                "last_collective_guide_vote_sum",
                self.last_collective_guide_vote_sum,
            )
        )
        self.last_collective_guide_hypothetical_veto = bool(
            state.get(
                "last_collective_guide_hypothetical_veto",
                self.last_collective_guide_hypothetical_veto,
            )
        )
        self.last_collective_guide_actual_veto = bool(
            state.get(
                "last_collective_guide_actual_veto",
                self.last_collective_guide_actual_veto,
            )
        )
        self.last_collective_guide_null_veto = bool(
            state.get(
                "last_collective_guide_null_veto",
                self.last_collective_guide_null_veto,
            )
        )
        for attr in (
            "collective_guide_events",
            "collective_guide_valid_events",
            "collective_guide_hypothetical_veto_events",
            "collective_guide_actual_veto_events",
            "collective_guide_probe_local_evals",
            "collective_guide_comm_rounds",
            "collective_guide_messages",
            "collective_guide_transmitted_floats",
        ):
            setattr(self, attr, int(state.get(attr, getattr(self, attr))))
        if any(
            getattr(self, attr) < 0
            for attr in (
                "collective_guide_events",
                "collective_guide_valid_events",
                "collective_guide_hypothetical_veto_events",
                "collective_guide_actual_veto_events",
                "collective_guide_probe_local_evals",
                "collective_guide_comm_rounds",
                "collective_guide_messages",
                "collective_guide_transmitted_floats",
            )
        ):
            raise ValueError("Collective guide counters must be nonnegative.")
        self.last_optimizer_guide_internal_active = np.asarray(
            state.get(
                "last_optimizer_guide_internal_active",
                self.last_optimizer_guide_internal_active,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_internal_mean_step_norm = np.asarray(
            state.get(
                "last_optimizer_guide_internal_mean_step_norm",
                self.last_optimizer_guide_internal_mean_step_norm,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_internal_alignment = np.asarray(
            state.get(
                "last_optimizer_guide_internal_alignment",
                self.last_optimizer_guide_internal_alignment,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.anchor_points = np.asarray(
            state.get("anchor_points", self.anchor_points),
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        self.last_anchor_applied = np.asarray(
            state.get("last_anchor_applied", self.last_anchor_applied),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_anchor_dist = np.asarray(
            state.get("last_anchor_dist", self.last_anchor_dist),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_anchor_direction_norm = np.asarray(
            state.get("last_anchor_direction_norm", self.last_anchor_direction_norm),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_anchor_applied = np.asarray(
            state.get("last_optimizer_anchor_applied", self.last_optimizer_anchor_applied),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_anchor_mean_step_norm = np.asarray(
            state.get(
                "last_optimizer_anchor_mean_step_norm",
                self.last_optimizer_anchor_mean_step_norm,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_anchor_sample_applied = np.asarray(
            state.get(
                "last_optimizer_anchor_sample_applied",
                self.last_optimizer_anchor_sample_applied,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_collab_mode_idx = np.asarray(
            state.get("last_collab_mode_idx", self.last_collab_mode_idx),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_guide_scale_idx = np.asarray(
            state.get("last_guide_scale_idx", self.last_guide_scale_idx),
            dtype=np.int64,
        ).reshape(self.n_agents)
        self.last_guide_scale_value = np.asarray(
            state.get("last_guide_scale_value", self.last_guide_scale_value),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_collab_leader_active = np.asarray(
            state.get("last_collab_leader_active", self.last_collab_leader_active),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_collab_strength_multiplier = np.asarray(
            state.get(
                "last_collab_strength_multiplier",
                self.last_collab_strength_multiplier,
            ),
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_optimizer_guide_strength_effective = float(
            state.get(
                "last_optimizer_guide_strength_effective",
                self.last_optimizer_guide_strength_effective,
            )
        )
        self.last_optimizer_guide_strength_scale = float(
            state.get(
                "last_optimizer_guide_strength_scale",
                self.last_optimizer_guide_strength_scale,
            )
        )
        self.last_optimizer_guide_schedule_metric = float(
            state.get(
                "last_optimizer_guide_schedule_metric",
                self.last_optimizer_guide_schedule_metric,
            )
        )
        if restored_persistent_bank is not None:
            self.persistent_sepcmaes_states = restored_persistent_bank[
                "states"
            ]
            self.persistent_sepcmaes_signatures = restored_persistent_bank[
                "signatures"
            ]
            self.persistent_sepcmaes_events = restored_persistent_bank[
                "events"
            ]
            self.persistent_sepcmaes_fresh_events = restored_persistent_bank[
                "fresh_events"
            ]
            self.persistent_sepcmaes_resume_events = restored_persistent_bank[
                "resume_events"
            ]
            self.persistent_sepcmaes_config_resets = restored_persistent_bank[
                "config_resets"
            ]
            self.last_persistent_sepcmaes_active = restored_persistent_bank[
                "last_active"
            ]
            self.last_persistent_sepcmaes_fresh = restored_persistent_bank[
                "last_fresh"
            ]
            self.last_persistent_sepcmaes_config_reset = restored_persistent_bank[
                "last_config_reset"
            ]
            self.last_persistent_sepcmaes_recenter_requested = restored_persistent_bank[
                "last_recenter_requested"
            ]
            self.last_persistent_sepcmaes_recenter_applied = restored_persistent_bank[
                "last_recenter_applied"
            ]
            self.last_persistent_sepcmaes_recenter_remaining = restored_persistent_bank[
                "last_recenter_remaining"
            ]
        if restored_target_block_field is not None:
            self.target_block_field_path = (
                restored_target_block_field["path"]
            )
            self.target_block_field_radius = (
                restored_target_block_field["radius"]
            )
            self.target_block_field_initialized = (
                restored_target_block_field["initialized"]
            )
            self.last_target_block_field_secant = (
                restored_target_block_field["last_secant"]
            )
            self.last_target_block_field_secant_valid = (
                restored_target_block_field["last_secant_valid"]
            )
            self.last_target_block_field_secant_alignment = (
                restored_target_block_field["last_secant_alignment"]
            )
            self.last_target_block_field_direction = (
                restored_target_block_field["last_direction"]
            )
            self.last_target_block_field_source_diversity = (
                restored_target_block_field["last_source_diversity"]
            )
            self.last_target_block_field_path_norm = (
                restored_target_block_field["last_path_norm"]
            )
            self.last_target_block_field_path_alignment = (
                restored_target_block_field["last_path_alignment"]
            )
            self.last_target_block_field_conflict = (
                restored_target_block_field["last_conflict"]
            )
            self.last_target_block_field_radius_expand = (
                restored_target_block_field["last_radius_expand"]
            )
            self.last_target_block_field_radius_shrink = (
                restored_target_block_field["last_radius_shrink"]
            )
            self.last_target_block_field_radius_clip_min = (
                restored_target_block_field["last_radius_clip_min"]
            )
            self.last_target_block_field_radius_clip_max = (
                restored_target_block_field["last_radius_clip_max"]
            )
            self.last_target_block_field_requested_commit_norm = (
                restored_target_block_field[
                    "last_requested_commit_norm"
                ]
            )
            self.last_target_block_field_applied_commit_norm = (
                restored_target_block_field[
                    "last_applied_commit_norm"
                ]
            )
            self.last_target_block_field_boundary_clipped = (
                restored_target_block_field[
                    "last_boundary_clipped"
                ]
            )
            self.last_target_block_field_path_injection = float(
                restored_target_block_field[
                    "last_path_injection"
                ]
            )
            self.last_target_block_field_path_angle_degrees = float(
                restored_target_block_field[
                    "last_path_angle_degrees"
                ]
            )
            self.target_block_field_events = int(
                restored_target_block_field["events"]
            )
            self.target_block_field_local_evals = int(
                restored_target_block_field["local_evals"]
            )
            self.target_block_field_comm_rounds = int(
                restored_target_block_field["comm_rounds"]
            )
            self.target_block_field_messages = int(
                restored_target_block_field["messages"]
            )
            self.target_block_field_transmitted_floats = int(
                restored_target_block_field["transmitted_floats"]
            )
        if restored_target_block_dormancy is not None:
            self.target_block_dormancy_residual_reference = (
                restored_target_block_dormancy[
                    "residual_reference"
                ]
            )
            self.target_block_dormancy_reserve_radius = (
                restored_target_block_dormancy["reserve_radius"]
            )
            self.target_block_dormancy_initialized = (
                restored_target_block_dormancy["initialized"]
            )
            self.target_block_dormancy_floor_age = (
                restored_target_block_dormancy["floor_age"]
            )
            self.target_block_dormancy_stagnation_age = (
                restored_target_block_dormancy[
                    "stagnation_age"
                ]
            )
            self.target_block_dormancy_cooldown = (
                restored_target_block_dormancy["cooldown"]
            )
            self.target_block_dormancy_activation_count = (
                restored_target_block_dormancy[
                    "activation_count"
                ]
            )
            self.last_target_block_dormancy_active = (
                restored_target_block_dormancy["last_active"]
            )
            self.last_target_block_dormancy_unresolved = (
                restored_target_block_dormancy[
                    "last_unresolved"
                ]
            )
            self.last_target_block_dormancy_material_progress = (
                restored_target_block_dormancy[
                    "last_material_progress"
                ]
            )
            self.last_target_block_dormancy_reliable_scale = (
                restored_target_block_dormancy[
                    "last_reliable_scale"
                ]
            )
            self.last_target_block_dormancy_residual_ratio = (
                restored_target_block_dormancy[
                    "last_residual_ratio"
                ]
            )
            self.last_target_block_dormancy_progress_ratio = (
                restored_target_block_dormancy[
                    "last_progress_ratio"
                ]
            )
            self.last_target_block_dormancy_disagreement = (
                restored_target_block_dormancy[
                    "last_disagreement"
                ]
            )
            self.last_target_block_dormancy_disagreement_ratio = (
                restored_target_block_dormancy[
                    "last_disagreement_ratio"
                ]
            )
            self.last_target_block_dormancy_restore_radius = (
                restored_target_block_dormancy[
                    "last_restore_radius"
                ]
            )
            self.target_block_dormancy_events = int(
                restored_target_block_dormancy["events"]
            )
            self.target_block_dormancy_activations = int(
                restored_target_block_dormancy["activations"]
            )
        if restored_target_block_direction_shadow is not None:
            self.last_target_block_direction_shadow_eligible = (
                restored_target_block_direction_shadow[
                    "last_eligible"
                ]
            )
            self.last_target_block_direction_shadow_selected = (
                restored_target_block_direction_shadow[
                    "last_selected"
                ]
            )
            self.last_target_block_direction_shadow_candidate_valid = (
                restored_target_block_direction_shadow[
                    "last_candidate_valid"
                ]
            )
            self.last_target_block_direction_shadow_candidate_angle = (
                restored_target_block_direction_shadow[
                    "last_candidate_angle"
                ]
            )
            self.last_target_block_direction_shadow_candidate_response = (
                restored_target_block_direction_shadow[
                    "last_candidate_response"
                ]
            )
            self.last_target_block_direction_shadow_candidate_positive = (
                restored_target_block_direction_shadow[
                    "last_candidate_positive"
                ]
            )
            self.last_target_block_direction_shadow_best_response = (
                restored_target_block_direction_shadow[
                    "last_best_response"
                ]
            )
            self.last_target_block_direction_shadow_best_source = (
                restored_target_block_direction_shadow[
                    "last_best_source"
                ]
            )
            self.last_target_block_direction_shadow_best_sign = (
                restored_target_block_direction_shadow[
                    "last_best_sign"
                ]
            )
            self.last_target_block_direction_shadow_best_direction = (
                restored_target_block_direction_shadow[
                    "last_best_direction"
                ]
            )
            self.last_target_block_direction_shadow_support = (
                restored_target_block_direction_shadow[
                    "last_support"
                ]
            )
            self.last_target_block_direction_shadow_conflict = (
                restored_target_block_direction_shadow[
                    "last_conflict"
                ]
            )
            self.target_block_direction_shadow_events = int(
                restored_target_block_direction_shadow["events"]
            )
            self.target_block_direction_shadow_eligible_blocks = int(
                restored_target_block_direction_shadow[
                    "eligible_blocks"
                ]
            )
            self.target_block_direction_shadow_probed_blocks = int(
                restored_target_block_direction_shadow[
                    "probed_blocks"
                ]
            )
            self.target_block_direction_shadow_local_evals = int(
                restored_target_block_direction_shadow["local_evals"]
            )
            self.target_block_direction_shadow_comm_rounds = int(
                restored_target_block_direction_shadow["comm_rounds"]
            )
            self.target_block_direction_shadow_messages = int(
                restored_target_block_direction_shadow["messages"]
            )
            self.target_block_direction_shadow_transmitted_floats = int(
                restored_target_block_direction_shadow[
                    "transmitted_floats"
                ]
            )
        if restored_target_block_challenge is not None:
            challenge_restore_map = {
                "last_target_block_challenge_selected": "last_selected",
                "last_target_block_challenge_candidate_valid": "last_candidate_valid",
                "last_target_block_challenge_candidate_angle": "last_candidate_angle",
                "last_target_block_challenge_candidate_score": "last_candidate_score",
                "last_target_block_challenge_candidate_sign": "last_candidate_sign",
                "last_target_block_challenge_path_score": "last_path_score",
                "last_target_block_challenge_alternative_score": "last_alternative_score",
                "last_target_block_challenge_margin": "last_margin",
                "last_target_block_challenge_best_source": "last_best_source",
                "last_target_block_challenge_best_sign": "last_best_sign",
                "last_target_block_challenge_best_direction": "last_best_direction",
                "last_target_block_challenge_coverage": "last_coverage",
                "last_target_block_challenge_positive_sources": "last_positive_sources",
                "last_target_block_challenge_neighbor_positive_sources": "last_neighbor_positive_sources",
                "last_target_block_challenge_support": "last_support",
                "last_target_block_challenge_conflict": "last_conflict",
                "last_target_block_challenge_actuation_eligible": "last_actuation_eligible",
                "last_target_block_challenge_applied": "last_applied",
                "last_target_block_challenge_applied_norm": "last_applied_norm",
                "last_target_block_challenge_boundary_clipped": "last_boundary_clipped",
            }
            for attribute, key in challenge_restore_map.items():
                setattr(
                    self,
                    attribute,
                    restored_target_block_challenge[key].copy(),
                )
            challenge_counter_map = {
                "target_block_challenge_events": "events",
                "target_block_challenge_challenges": "challenges",
                "target_block_challenge_directed_responses": "directed_responses",
                "target_block_challenge_local_evals": "local_evals",
                "target_block_challenge_reported_evals": "reported_evals",
                "target_block_challenge_applied_commits": "applied_commits",
                "target_block_challenge_comm_rounds": "comm_rounds",
                "target_block_challenge_messages": "messages",
                "target_block_challenge_transmitted_floats": "transmitted_floats",
            }
            for attribute, key in challenge_counter_map.items():
                setattr(
                    self,
                    attribute,
                    int(restored_target_block_challenge[key]),
                )
        if restored_target_block_dual_clock is not None:
            self.target_block_dual_clock_outer_events = int(
                restored_target_block_dual_clock["outer_events"]
            )
            self.target_block_dual_clock_local_generation_ticks = int(
                restored_target_block_dual_clock[
                    "local_generation_ticks"
                ]
            )
            self.target_block_dual_clock_communication_ticks = int(
                restored_target_block_dual_clock[
                    "communication_ticks"
                ]
            )
            self.target_block_dual_clock_commit_ticks = int(
                restored_target_block_dual_clock["commit_ticks"]
            )
            self.last_target_block_dual_clock_microcycles = int(
                restored_target_block_dual_clock[
                    "last_microcycles"
                ]
            )
            self.last_target_block_dual_clock_generation_fes = (
                restored_target_block_dual_clock[
                    "last_generation_fes_per_agent"
                ]
            )
            self.target_block_commit_credit_events = int(
                restored_target_block_dual_clock[
                    "commit_credit_events"
                ]
            )
            self.last_target_block_commit_credit_active = (
                restored_target_block_dual_clock[
                    "last_commit_credit_active"
                ]
            )
            self.last_target_block_commit_credit = (
                restored_target_block_dual_clock["last_commit_credit"]
            )
            self.last_target_block_commit_credit_cosine = (
                restored_target_block_dual_clock[
                    "last_commit_credit_cosine"
                ]
            )
            self.last_target_block_commit_credit_proposal_norm = (
                restored_target_block_dual_clock[
                    "last_commit_credit_proposal_norm"
                ]
            )
            self.last_target_block_commit_credit_commit_norm = (
                restored_target_block_dual_clock[
                    "last_commit_credit_commit_norm"
                ]
            )
            self.last_target_block_commit_credit_correction_norm = (
                restored_target_block_dual_clock[
                    "last_commit_credit_correction_norm"
                ]
            )
            self.last_target_block_commit_credit_path_retention = (
                restored_target_block_dual_clock[
                    "last_commit_credit_path_retention"
                ]
            )
            self.last_target_block_commit_credit_scale_retention = (
                restored_target_block_dual_clock[
                    "last_commit_credit_scale_retention"
                ]
            )
            self.last_target_block_commit_credit_axis_rms_before = (
                restored_target_block_dual_clock[
                    "last_commit_credit_axis_rms_before"
                ]
            )
            self.last_target_block_commit_credit_axis_rms_after = (
                restored_target_block_dual_clock[
                    "last_commit_credit_axis_rms_after"
                ]
            )
            self.last_target_block_commit_credit_sigma_path_before = (
                restored_target_block_dual_clock[
                    "last_commit_credit_sigma_path_before"
                ]
            )
            self.last_target_block_commit_credit_sigma_path_after = (
                restored_target_block_dual_clock[
                    "last_commit_credit_sigma_path_after"
                ]
            )
            self.last_target_block_commit_credit_cov_path_before = (
                restored_target_block_dual_clock[
                    "last_commit_credit_cov_path_before"
                ]
            )
            self.last_target_block_commit_credit_cov_path_after = (
                restored_target_block_dual_clock[
                    "last_commit_credit_cov_path_after"
                ]
            )

    def _step_transition(self, actions: np.ndarray):
        actions = np.asarray(actions)
        comm_actions = None
        collab_actions = None
        guide_scale_actions = None
        actuator_actions = None
        if actions.ndim == 2 and actions.shape == (self.n_agents, 3):
            if self.candidate_actuator_action_enable:
                raise ValueError(
                    "The D4-B actuator head requires the full current action "
                    "columns; legacy [A,3] actions cannot encode per-agent beta."
                )
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions = np.clip(actions[:, 1].astype(np.int64), 0, len(self.profile_candidates) - 1)
            res_actions = np.clip(actions[:, 2].astype(np.int64), 0, len(self.resource_factors) - 1)
            cfg_actions_block = np.repeat(cfg_actions[:, None], self.cfg_param_num, axis=1)
        elif actions.ndim == 2 and actions.shape[0] == self.n_agents and actions.shape[1] >= 2 + self.cfg_param_num:
            opt_actions = np.clip(actions[:, 0].astype(np.int64), 0, len(self.optimizer_candidates) - 1)
            cfg_actions_block = np.clip(
                actions[:, 1 : 1 + self.cfg_param_num].astype(np.int64),
                0,
                len(self.profile_candidates) - 1,
            )
            cfg_actions = cfg_actions_block[:, 0].copy()
            res_actions = np.clip(actions[:, 1 + self.cfg_param_num].astype(np.int64), 0, len(self.resource_factors) - 1)
            expected_cols = (
                2
                + self.cfg_param_num
                + (1 if self.comm_action_enable else 0)
                + (1 if self.collab_action_enable else 0)
                + (1 if self.guide_scale_action_enable else 0)
                + (1 if self.candidate_actuator_action_enable else 0)
            )
            if actions.shape[1] != expected_cols:
                raise ValueError(
                    f"Expected actions shape [A,{expected_cols}] with current action-head switches, got {actions.shape}."
                )
            cursor = 2 + self.cfg_param_num
            if self.comm_action_enable:
                comm_actions = actions[:, cursor].astype(np.int64)
                cursor += 1
            if self.collab_action_enable:
                collab_actions = actions[:, cursor].astype(np.int64)
                cursor += 1
            if self.guide_scale_action_enable:
                guide_scale_actions = actions[:, cursor].astype(np.int64)
                cursor += 1
            if self.candidate_actuator_action_enable:
                actuator_actions = actions[:, cursor].astype(np.int64)
        else:
            raise ValueError(
                f"Expected actions shape [A,3] or current action columns with A={self.n_agents}, got {actions.shape}."
            )
        if collab_actions is None:
            collab_actions = np.zeros((self.n_agents,), dtype=np.int64)
        else:
            collab_actions = np.clip(
                collab_actions.astype(np.int64), 0, len(self.collab_modes) - 1
            )
        if guide_scale_actions is None:
            guide_scale_actions = np.zeros((self.n_agents,), dtype=np.int64)
        else:
            guide_scale_actions = np.clip(
                guide_scale_actions.astype(np.int64), 0, len(self.guide_scale_candidates) - 1
            )
        if actuator_actions is None:
            actuator_actions = np.zeros((self.n_agents,), dtype=np.int64)
            actuator_beta_per_agent = np.full(
                (self.n_agents,), self.candidate_actuator_beta, dtype=np.float64
            )
        else:
            actuator_actions = np.clip(
                actuator_actions.astype(np.int64),
                0,
                len(self.candidate_actuator_candidates) - 1,
            )
            if self.candidate_actuator_forced_action >= 0:
                actuator_actions.fill(
                    int(
                        np.clip(
                            self.candidate_actuator_forced_action,
                            0,
                            len(self.candidate_actuator_candidates) - 1,
                        )
                    )
                )
            actuator_beta_per_agent = np.asarray(
                [
                    self.candidate_actuator_candidates[int(x)]
                    for x in actuator_actions
                ],
                dtype=np.float64,
            )

        self.last_optimizer_idx = opt_actions.copy()
        self.last_action_idx = cfg_actions.copy()
        self.last_resource_idx = res_actions.copy()
        self.last_collab_mode_idx = collab_actions.copy()
        self.last_guide_scale_idx = guide_scale_actions.copy()
        self.last_guide_scale_value = np.asarray(
            [self._guide_scale_value(int(x)) for x in guide_scale_actions],
            dtype=np.float64,
        )
        self.last_candidate_actuator_action = actuator_actions.copy()
        self.last_collab_leader_active.fill(0.0)
        self.last_collab_strength_multiplier.fill(1.0)
        self.last_optimizer_guide_applied.fill(0.0)
        self.last_optimizer_guide_internal_active.fill(0.0)
        self.last_optimizer_guide_internal_mean_step_norm.fill(0.0)
        self.last_optimizer_guide_internal_alignment.fill(0.0)
        self.last_anchor_applied.fill(0.0)
        self.last_anchor_dist.fill(0.0)
        self.last_anchor_direction_norm.fill(0.0)
        self.last_optimizer_anchor_applied.fill(0.0)
        self.last_optimizer_anchor_mean_step_norm.fill(0.0)
        self.last_optimizer_anchor_sample_applied.fill(0.0)
        self.last_persistent_sepcmaes_active.fill(0.0)
        self.last_persistent_sepcmaes_fresh.fill(0.0)
        self.last_persistent_sepcmaes_config_reset.fill(0.0)
        self.last_persistent_sepcmaes_recenter_requested.fill(0.0)
        self.last_persistent_sepcmaes_recenter_applied.fill(0.0)
        self.last_persistent_sepcmaes_recenter_remaining.fill(0.0)
        if self.guide_replacement_enabled:
            self._reset_guide_replacement_event()
        requested_comm_rounds = self._resolve_comm_rounds_from_actions(comm_actions)

        agent_base_x = self.agent_x.copy()
        collective_guide_reported_evals = (
            self._run_collective_guide_validation_event(
                base_states=agent_base_x,
                optimizer_actions=opt_actions,
                config_actions=cfg_actions_block,
            )
            if self.collective_guide_enabled
            else 0
        )
        self._refresh_anchor_points(agent_base_x)
        local_prev = self.agent_local_f.copy()
        f_prev = None
        if not self.local_only_information:
            f_prev = float(self.current_f)
        if self.target_block_dual_clock_enable:
            optimizer_batch = self._run_target_block_dual_clock_event(
                base_states=agent_base_x,
                local_reference=local_prev,
                opt_actions=opt_actions,
                cfg_actions_block=cfg_actions_block,
                res_actions=res_actions,
                collab_actions=collab_actions,
                guide_scale_actions=guide_scale_actions,
            )
        elif self.event_slot_interleaving_enable:
            optimizer_batch = self._run_event_slot_interleaving_event(
                base_states=agent_base_x,
                local_reference=local_prev,
                opt_actions=opt_actions,
                cfg_actions_block=cfg_actions_block,
                res_actions=res_actions,
                collab_actions=collab_actions,
                guide_scale_actions=guide_scale_actions,
                comm_rounds=requested_comm_rounds,
            )
        else:
            optimizer_batch = self._run_objective_split_optimizer_batch(
                base_states=agent_base_x,
                local_reference=local_prev,
                opt_actions=opt_actions,
                cfg_actions_block=cfg_actions_block,
                res_actions=res_actions,
                collab_actions=collab_actions,
                guide_scale_actions=guide_scale_actions,
            )
        proposal_states = np.asarray(
            optimizer_batch["proposal_states"],
            dtype=np.float64,
        ).reshape(self.n_agents, self.D)
        proposal_local_values = np.asarray(
            optimizer_batch["proposal_local_values"],
            dtype=np.float64,
        ).reshape(self.n_agents)
        local_improvements = np.asarray(
            optimizer_batch["local_improvements"],
            dtype=np.float64,
        ).reshape(self.n_agents)
        total_evals = int(optimizer_batch["total_evals"])
        candidate_validation_evals = int(
            optimizer_batch["candidate_validation_evals"]
        )
        candidate_shape_fallbacks = int(
            optimizer_batch["candidate_shape_fallbacks"]
        )
        self.last_cmaes_numeric_fail_soft = np.asarray(
            optimizer_batch["cmaes_numeric_fail_soft_mask"],
            dtype=np.float64,
        ).reshape(self.n_agents)
        self.last_cmaes_numeric_fail_soft_generation = np.asarray(
            optimizer_batch["cmaes_numeric_fail_soft_generations"],
            dtype=np.int64,
        ).reshape(self.n_agents)
        cmaes_fail_soft_evaluations = np.asarray(
            optimizer_batch["cmaes_numeric_fail_soft_evaluations"],
            dtype=np.int64,
        ).reshape(self.n_agents)
        cmaes_fail_soft_reasons = list(
            optimizer_batch["cmaes_numeric_fail_soft_reasons"]
        )
        cmaes_fail_soft_step_count = int(
            np.sum(self.last_cmaes_numeric_fail_soft)
        )
        self.cmaes_numeric_fail_soft_events += cmaes_fail_soft_step_count
        self.last_actual_fes_per_agent = np.asarray(
            optimizer_batch["evals_per_agent"],
            dtype=np.float64,
        ).reshape(self.n_agents)
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
        target_block_field_reported_evals = 0
        if self.target_block_dual_clock_enable:
            next_agent_x = np.asarray(
                optimizer_batch["next_agent_x"],
                dtype=np.float64,
            ).reshape(self.n_agents, self.D)
            target_block_field_reported_evals = int(
                optimizer_batch[
                    "target_block_field_reported_evals"
                ]
            )
            communication_applied = True
        elif self.event_slot_interleaving_enable:
            next_agent_x = np.asarray(
                optimizer_batch["next_agent_x"],
                dtype=np.float64,
            ).reshape(self.n_agents, self.D)
            communication_applied = bool(
                optimizer_batch["communication_applied"]
            )
        elif self.target_block_field_enable:
            (
                next_agent_x,
                target_block_field_reported_evals,
            ) = self._run_target_block_field_event(
                base_states=agent_base_x,
                proposal_states=proposal_states,
            )
            communication_applied = True
        else:
            next_agent_x, communication_applied = self._apply_consensus(
                base_states=agent_base_x,
                proposal_states=proposal_states,
                local_improvements=np.asarray(
                    local_improvements, dtype=np.float64
                ),
                comm_rounds=requested_comm_rounds,
            )
        rvcpd_next_local = np.asarray(
            proposal_local_values, dtype=np.float64
        ).reshape(self.n_agents)
        rvcpd_reported_evals = 0
        if self.rvcpd_enabled and not self.rvcpd_d5_post_commit:
            (
                next_agent_x,
                rvcpd_next_local,
                rvcpd_reported_evals,
            ) = self._run_rvcpd_event(
                base_states=agent_base_x,
                proposal_states=proposal_states,
                proposal_local_f=rvcpd_next_local,
            )
            communication_applied = bool(
                np.any(self.last_rvcpd_active > 0.0)
            )
        if (
            self.ccsa_momentum_update_mode == "always"
            and self.consensus_mode not in {"ccsa_lite", "ccsa_masoie_lite"}
            and not self.event_slot_interleaving_enable
        ):
            self._update_ccsa_direction_momentum(
                base_states=agent_base_x,
                proposals=proposal_states,
                local_improvements=np.asarray(local_improvements, dtype=np.float64),
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
        self.last_optimizer_guide_schedule_metric = float(post_mean_disagreement)
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

        d6_generator_actions = None
        d6_multisecant_preview = None
        if self.d6_pre_generator_selector_enable:
            d6_multisecant_preview = (
                self._candidate_response_multisecant_preview(next_agent_x)
            )
            (
                generator_obs,
                generator_actor_mask,
                generator_critic_mask,
                generator_prepare_info,
            ) = self._build_d6_generator_context(
                primary_actions=np.asarray(actions[:, :-1], dtype=np.int64),
                optimizer_base_states=agent_base_x,
                proposal_states=proposal_states,
                base_states=next_agent_x,
                local_improvements=np.asarray(
                    local_improvements,
                    dtype=np.float64,
                ),
                communication_applied=communication_applied,
                multisecant_preview=d6_multisecant_preview,
            )
            d6_generator_override = yield (
                np.asarray(generator_obs, dtype=np.float32).copy(),
                np.asarray(generator_actor_mask, dtype=np.float32).copy(),
                np.asarray(generator_critic_mask, dtype=np.float32).copy(),
                generator_prepare_info,
            )
            if d6_generator_override is None:
                raise RuntimeError(
                    "D6 generator stage requires explicit SPSA/Hybrid actions."
                )
            d6_generator_actions = np.clip(
                np.asarray(
                    d6_generator_override,
                    dtype=np.int64,
                ).reshape(self.n_agents),
                0,
                1,
            )

        candidate_pending = self._prepare_d5_candidate_response_event(
            optimizer_base_states=agent_base_x,
            proposal_states=proposal_states,
            base_states=next_agent_x,
            communication_applied=communication_applied,
            generator_actions=d6_generator_actions,
            multisecant_preview=d6_multisecant_preview,
        )
        prepare_info = {
            "d5_two_stage": True,
            "d6_three_stage": bool(self.d6_pre_generator_selector_enable),
            "d6_generator_actions": (
                np.asarray(d6_generator_actions, dtype=np.int64).copy()
                if d6_generator_actions is not None
                else np.empty((0,), dtype=np.int64)
            ),
            "candidate_event_active": bool(candidate_pending["active"]),
            "actuator_obs_schema": list(self.d5_actuator_obs_schema),
            "opportunity_mask": np.asarray(
                candidate_pending["opportunity_mask"],
                dtype=np.float32,
            ).copy(),
            "consensus_base_x": np.asarray(
                candidate_pending["base_states"],
                dtype=np.float64,
            ).copy(),
            "verified_x": np.asarray(
                (
                    candidate_pending["selection"].verified_x
                    if candidate_pending["active"]
                    else candidate_pending["base_states"]
                ),
                dtype=np.float64,
            ).copy(),
            "accepted_mask": np.asarray(
                (
                    candidate_pending["selection"].accepted_mask
                    if candidate_pending["active"]
                    else np.zeros(
                        (
                            self.n_agents,
                            self.committee_target_num,
                        ),
                        dtype=bool,
                    )
                ),
                dtype=bool,
            ).copy(),
            "probe_direction": np.asarray(
                (
                    candidate_pending["geometry"].direction
                    if candidate_pending["active"]
                    else np.zeros(
                        (
                            self.n_agents,
                            self.committee_target_num,
                            self.committee_coordinate_dim,
                        ),
                        dtype=np.float64,
                    )
                ),
                dtype=np.float64,
            ).copy(),
            "candidate_x": np.asarray(
                (
                    candidate_pending["generation"].candidate_x
                    if candidate_pending["active"]
                    else candidate_pending["base_states"]
                ),
                dtype=np.float64,
            ).copy(),
            "generator_diagnostics": {
                key: np.asarray(value).copy()
                for key, value in candidate_pending.get(
                    "generator_diagnostics",
                    {},
                ).items()
            },
            "base_own_local_f": np.asarray(
                candidate_pending["base_own_local_f"],
                dtype=np.float64,
            ).copy(),
            "candidate_probe_local_evals": int(
                candidate_pending["probe_evals"]
            ),
            "candidate_verification_local_evals": int(
                candidate_pending["verification_evals"]
            ),
            "reported_sum_fes_before_commit": int(self.sum_fes),
            "step_before_commit": int(self.step_count),
        }
        actuator_override = yield (
            np.asarray(
                candidate_pending["actuator_obs"],
                dtype=np.float32,
            ).copy(),
            np.asarray(
                candidate_pending["opportunity_mask"],
                dtype=np.float32,
            ).copy(),
            prepare_info,
        )
        if actuator_override is not None:
            actuator_actions = np.clip(
                np.asarray(
                    actuator_override,
                    dtype=np.int64,
                ).reshape(self.n_agents),
                0,
                len(self.candidate_actuator_candidates) - 1,
            )
            if self.candidate_actuator_forced_action >= 0:
                actuator_actions.fill(
                    int(
                        np.clip(
                            self.candidate_actuator_forced_action,
                            0,
                            len(self.candidate_actuator_candidates) - 1,
                        )
                    )
                )
            actuator_beta_per_agent = np.asarray(
                [
                    self.candidate_actuator_candidates[int(x)]
                    for x in actuator_actions
                ],
                dtype=np.float64,
            )
        if self.d6_pre_generator_selector_enable:
            actuator_actions = np.full(
                (self.n_agents,),
                int(self.d6_post_actuator_forced_action),
                dtype=np.int64,
            )
            actuator_beta_per_agent = np.asarray(
                [
                    self.candidate_actuator_candidates[int(x)]
                    for x in actuator_actions
                ],
                dtype=np.float64,
            )
        self.last_candidate_actuator_action = actuator_actions.copy()
        (
            next_agent_x,
            candidate_response_reported_evals,
            d5_base_own_local_f,
        ) = self._commit_d5_candidate_response_event(
            candidate_pending,
            actuator_beta_per_agent,
        )
        if not np.all(np.isfinite(next_agent_x)):
            raise FloatingPointError(
                "D3-P candidate-response states contain NaN or infinity."
            )
        if np.any(next_agent_x < self.lb - 1e-10) or np.any(
            next_agent_x > self.ub + 1e-10
        ):
            raise FloatingPointError(
                "D3-P candidate-response states escaped problem bounds."
            )

        x_report = (
            next_agent_x[0].copy()
            if self.consensus_mode == "full_mean"
            else np.mean(next_agent_x, axis=0)
        )
        d5_committed_local_f = None
        guide_replacement_reported_evals = 0
        if self.local_only_information:
            if self.guide_replacement_enabled:
                d5_committed_local_f = self._eval_agent_local_states(
                    next_agent_x
                )
                agent_state_eval_count = self.n_agents
                (
                    next_agent_x,
                    next_agent_local,
                    guide_replacement_reported_evals,
                ) = self._run_guide_replacement_event(
                    commit_base_states=next_agent_x,
                    commit_base_local_f=d5_committed_local_f,
                )
                x_report = (
                    next_agent_x[0].copy()
                    if self.consensus_mode == "full_mean"
                    else np.mean(next_agent_x, axis=0)
                )
            elif self.rvcpd_d5_post_commit:
                d5_committed_local_f = self._eval_agent_local_states(
                    next_agent_x
                )
                agent_state_eval_count = self.n_agents
                (
                    next_agent_x,
                    next_agent_local,
                    rvcpd_reported_evals,
                ) = self._run_rvcpd_event(
                    base_states=agent_base_x,
                    proposal_states=proposal_states,
                    proposal_local_f=np.asarray(
                        proposal_local_values,
                        dtype=np.float64,
                    ),
                    commit_base_states=next_agent_x,
                    commit_base_local_f=d5_committed_local_f,
                )
                x_report = (
                    next_agent_x[0].copy()
                    if self.consensus_mode == "full_mean"
                    else np.mean(next_agent_x, axis=0)
                )
            elif self.rvcpd_enabled:
                next_agent_local = rvcpd_next_local.copy()
                agent_state_eval_count = 0
            else:
                next_agent_local = self._eval_agent_local_states(next_agent_x)
                agent_state_eval_count = self.n_agents
            f_report = float("nan")
            report_local = np.empty((0,), dtype=np.float64)
        else:
            if self.consensus_mode == "full_mean":
                f_report, report_local = self._eval_global_and_all_local(x_report)
                next_agent_local = report_local.copy()
                agent_state_eval_count = 0
            else:
                next_agent_local = self._eval_agent_local_states(next_agent_x)
                agent_state_eval_count = self.n_agents
                f_report, report_local = self._eval_global_and_all_local(x_report)

        committee_reported_evals = 0
        if not self.local_only_information or self.committee_mode == "guide":
            committee_reported_evals = self._run_committee_event(
                proposal_states=proposal_states,
                next_agent_x=next_agent_x,
                f_report=f_report,
                communication_applied=communication_applied,
            )

        self.agent_x = next_agent_x
        self.agent_local_f = next_agent_local
        self.agent_best_local_f = np.minimum(self.agent_best_local_f, next_agent_local)
        committed_improvements = np.asarray(
            [
                _safe_log_improvement(float(before), float(after))
                for before, after in zip(local_prev, next_agent_local)
            ],
            dtype=np.float64,
        )
        d5_actuator_local_credit = np.zeros(
            (self.n_agents,),
            dtype=np.float64,
        )
        d5_opportunity_mask = np.asarray(
            candidate_pending["opportunity_mask"],
            dtype=bool,
        ).reshape(self.n_agents)
        d5_credit_local_f = (
            next_agent_local
            if d5_committed_local_f is None
            else np.asarray(
                d5_committed_local_f,
                dtype=np.float64,
            ).reshape(self.n_agents)
        )
        for agent_id in np.flatnonzero(d5_opportunity_mask):
            d5_actuator_local_credit[int(agent_id)] = _safe_log_improvement(
                float(d5_base_own_local_f[int(agent_id)]),
                float(d5_credit_local_f[int(agent_id)]),
            )
        proposal_arr = np.asarray(local_improvements, dtype=np.float64)
        self.last_committed_local_improve = committed_improvements.copy()
        self.last_consensus_local_effect = (
            committed_improvements - proposal_arr
        )
        self.last_local_commit_success = (
            np.asarray(next_agent_local, dtype=np.float64)
            <= np.asarray(local_prev, dtype=np.float64)
        ).astype(np.float64)
        if self.local_only_information:
            self.last_team_improve = float(np.mean(committed_improvements))
            self.last_commit_success = float(
                np.mean(self.last_local_commit_success)
            )
        else:
            self.current_x = x_report
            self.current_f = f_report
            if self.current_f <= self.gbest_f:
                self.gbest_f = self.current_f
                self.gbest_x = self.current_x.copy()
            self.report_current_x = self.current_x.copy()
            self.report_current_f = float(self.current_f)
            self.report_best_x = self.gbest_x.copy()
            self.report_best_f = float(self.gbest_f)
            self.last_team_improve = _safe_log_improvement(
                float(f_prev),
                self.current_f,
            )
            self.last_commit_success = 1.0 if f_report <= float(f_prev) else 0.0
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
            "optimizer_guide_norm_mean": float(np.mean(self.last_optimizer_guide_norm)),
            "optimizer_guide_norm_max": float(np.max(self.last_optimizer_guide_norm)),
            "optimizer_guide_applied_ratio": float(np.mean(self.last_optimizer_guide_applied)),
            "optimizer_guide_alignment_mean": float(
                np.mean(self.last_optimizer_guide_alignment)
            ),
            "optimizer_guide_source_active_ratio": float(
                self.last_optimizer_guide_source_active_ratio
            ),
            "optimizer_guide_strength_effective": float(
                self.last_optimizer_guide_strength_effective
            ),
            "optimizer_guide_strength_scale": float(
                self.last_optimizer_guide_strength_scale
            ),
            "optimizer_guide_schedule_metric": float(
                self.last_optimizer_guide_schedule_metric
            ),
            "optimizer_guide_internal_active_ratio": float(
                np.mean(self.last_optimizer_guide_internal_active)
            ),
            "optimizer_guide_internal_mean_step_norm_mean": float(
                np.mean(self.last_optimizer_guide_internal_mean_step_norm)
            ),
            "optimizer_guide_internal_mean_step_norm_max": float(
                np.max(self.last_optimizer_guide_internal_mean_step_norm)
            ),
            "optimizer_guide_internal_alignment_mean": float(
                np.mean(self.last_optimizer_guide_internal_alignment)
            ),
            "anchor_enabled": float(1.0 if self.anchor_enable else 0.0),
            "anchor_applied_ratio": float(np.mean(self.last_anchor_applied)),
            "anchor_dist_mean": float(np.mean(self.last_anchor_dist)),
            "anchor_dist_max": float(np.max(self.last_anchor_dist)),
            "anchor_direction_norm_mean": float(
                np.mean(self.last_anchor_direction_norm)
            ),
            "anchor_direction_norm_max": float(
                np.max(self.last_anchor_direction_norm)
            ),
            "optimizer_anchor_applied_ratio": float(
                np.mean(self.last_optimizer_anchor_applied)
            ),
            "optimizer_anchor_mean_step_norm_mean": float(
                np.mean(self.last_optimizer_anchor_mean_step_norm)
            ),
            "optimizer_anchor_mean_step_norm_max": float(
                np.max(self.last_optimizer_anchor_mean_step_norm)
            ),
            "optimizer_anchor_sample_applied_ratio": float(
                np.mean(self.last_optimizer_anchor_sample_applied)
            ),
            "collab_mode_idx_mean": float(np.mean(self.last_collab_mode_idx)),
            "guide_scale_idx_mean": float(np.mean(self.last_guide_scale_idx)),
            "guide_scale_value_mean": float(np.mean(self.last_guide_scale_value)),
            "collab_leader_active_ratio": float(np.mean(self.last_collab_leader_active)),
            "collab_strength_multiplier_mean": float(
                np.mean(self.last_collab_strength_multiplier)
            ),
        }

        self._update_optimizer_guide_direction(
            base_states=agent_base_x,
            proposal_states=proposal_states,
            local_improvements=np.asarray(local_improvements, dtype=np.float64),
        )
        self._fuse_committee_with_optimizer_guide()

        for i, li in enumerate(local_improvements):
            self.last_local_improve[i] = float(li)
            self.local_improve_hist[i].append(float(li))
            committed_value = float(committed_improvements[i])
            self.committed_local_improve_hist[i].append(committed_value)

        local_arr = np.asarray(local_improvements, dtype=np.float64)
        mean_local_improve = float(np.mean(local_arr)) if local_arr.size else 0.0
        team_improve = float(self.last_team_improve)
        self.last_joint_vs_mean_local_gap = float(team_improve - mean_local_improve)
        self.last_local_vs_joint_gap = (local_arr - team_improve).astype(np.float64, copy=False)
        if self.local_only_information:
            committed_mean = (
                float(np.mean(committed_improvements))
                if committed_improvements.size
                else 0.0
            )
            self.last_local_centered = (
                committed_improvements - committed_mean
            ).astype(np.float64, copy=False)
        else:
            self.last_local_centered = (local_arr - mean_local_improve).astype(
                np.float64,
                copy=False,
            )

        self.local_search_fes += int(total_evals)
        self.candidate_validation_local_evals += int(
            candidate_validation_evals
        )
        self.agent_state_local_evals += int(agent_state_eval_count)
        if not self.local_only_information:
            self.global_monitor_local_evals += int(self.n_agents)
            self.global_monitor_rounds += 1
        # Preserve historical cadence: local-search evaluations plus one legacy
        # report-round budget token, irrespective of detached monitor calls.
        reported_step_evals = (
            int(total_evals)
            + 1
            + int(committee_reported_evals)
            + int(candidate_response_reported_evals)
            + int(rvcpd_reported_evals)
            + int(guide_replacement_reported_evals)
            + int(collective_guide_reported_evals)
            + int(target_block_field_reported_evals)
        )
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
        self._record_state_communication_event(
            communication_applied=communication_applied
        )
        obs_next = self._build_obs()

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
        reward_semantics = (
            "committed_local_proxy"
            if self.local_only_information
            else "legacy_exact_global"
        )

        # In local_only, exact-global evaluation is deliberately delayed until
        # agent state, reward, done, and next observation are already fixed.
        if self.local_only_information:
            f_report, report_local = self._run_detached_global_monitor(x_report)
            self._run_candidate_response_detached_shadow_monitor()
            if self.committee_mode == "shadow":
                self._run_committee_event(
                    proposal_states=proposal_states,
                    next_agent_x=next_agent_x,
                    f_report=f_report,
                    communication_applied=communication_applied,
                )

        step_fitness_record = np.asarray(
            [self.report_current_f],
            dtype=np.float64,
        )
        if self.record_eval_individual:
            step_individual_record = self.report_current_x.astype(
                np.float32,
                copy=False,
            )[None, :]
        else:
            step_individual_record = np.empty((0, self.D), dtype=np.float32)

        verification_local_evals = int(
            self.committee_decision_local_evals
            + self.committee_shadow_local_evals
            + self.committee_shadow_global_local_evals
            + self.committee_acceptance_local_evals
            + self.candidate_response_probe_local_evals
            + self.candidate_response_verification_local_evals
            + self.candidate_response_shadow_global_local_evals
            + self.rvcpd_probe_local_evals
            + self.guide_replacement_probe_local_evals
            + self.collective_guide_probe_local_evals
        )
        total_physical_local_calls = int(
            self.local_search_fes
            + self.event_slot_physical_local_evals
            - self.event_slot_reported_local_evals
            + self.candidate_validation_local_evals
            + self.agent_state_local_evals
            + self.global_monitor_local_evals
            + verification_local_evals
            + self.target_block_field_local_evals
            + self.target_block_direction_shadow_local_evals
            + self.target_block_challenge_local_evals
        )
        info = {
            "team_improve": self.last_team_improve,
            "local_improve": np.asarray(local_improvements, dtype=np.float32),
            "proposal_local_improve": np.asarray(
                local_improvements,
                dtype=np.float32,
            ),
            "committed_local_improve": committed_improvements.astype(
                np.float32,
                copy=False,
            ),
            "consensus_local_effect": (
                committed_improvements - local_arr
            ).astype(np.float32, copy=False),
            "local_commit_success": (
                np.asarray(next_agent_local, dtype=np.float64)
                <= np.asarray(local_prev, dtype=np.float64)
            ).astype(np.float32),
            "current_f": self.report_current_f,
            "report_x_fitness": self.report_current_f,
            "gbest_fitness": self.report_best_f,
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
            "reward_semantics": reward_semantics,
            "d5_two_stage_actuator_enable": bool(
                self.d5_two_stage_actuator_enable
            ),
            "d5_actuator_obs_schema": list(self.d5_actuator_obs_schema),
            "d5_actuator_obs": np.asarray(
                candidate_pending["actuator_obs"],
                dtype=np.float32,
            ).copy(),
            "d5_opportunity_mask": d5_opportunity_mask.astype(
                np.float32,
                copy=False,
            ),
            "d5_base_own_local_f": np.asarray(
                d5_base_own_local_f,
                dtype=np.float64,
            ).copy(),
            "d5_committed_local_f": np.asarray(
                d5_credit_local_f,
                dtype=np.float64,
            ).copy(),
            "d5_actuator_local_credit": d5_actuator_local_credit.astype(
                np.float32,
                copy=False,
            ),
            "d6_generator_trajectory_reward": committed_improvements.astype(
                np.float32,
                copy=False,
            ),
            "d6_generator_actions": (
                np.asarray(d6_generator_actions, dtype=np.int64).copy()
                if d6_generator_actions is not None
                else np.empty((0,), dtype=np.int64)
            ),
            "team_improve_semantics": (
                "mean_committed_local_proxy"
                if self.local_only_information
                else "exact_global_log_improvement"
            ),
            "information_mode": str(self.information_mode),
            "global_monitor_enable": bool(self.global_monitor_enable),
            "env_mode": self.env_mode_name,
            "agent_parallel_workers": int(self.agent_parallel_workers),
            "event_slot_interleaving_enable": bool(
                self.event_slot_interleaving_enable
            ),
            "event_slot_interleaving_events": int(
                self.event_slot_interleaving_events
            ),
            "event_slot_interleaving_slots": int(
                self.event_slot_interleaving_slots
            ),
            "event_slot_native_local_evals": int(
                self.event_slot_native_local_evals
            ),
            "event_slot_reported_local_evals": int(
                self.event_slot_reported_local_evals
            ),
            "event_slot_physical_local_evals": int(
                self.event_slot_physical_local_evals
            ),
            "event_slot_count": int(self.last_event_slot_count),
            "event_slot_packet_units": (
                self.last_event_slot_packet_units.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_distribution_updates": (
                self.last_event_slot_distribution_updates.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_native_evals_per_agent": (
                self.last_event_slot_native_evals.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_physical_evals_per_agent": (
                self.last_event_slot_physical_evals.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_center_shift_norms": (
                self.last_event_slot_center_shift_norms.astype(
                    np.float64, copy=True
                )
            ),
            "event_slot_sigma_diagnostics": {
                key: value.copy() for key, value in self.last_sigma_diagnostics.items()
            },
            **({
                "event_slot_vkd_diagnostics": {
                    key: value.copy()
                    for key, value in self.last_event_slot_vkd_diagnostics.items()
                },
            "vkd_ps_outlet_mode": str(self.vkd_ps_outlet_mode),
            "vkd_boundary_update_mode": str(self.vkd_boundary_update_mode),
            } if getattr(self, "last_event_slot_vkd_diagnostics", None) is not None
               and self.event_slot_interleaving_enable else {}),
            **({
                "event_slot_mmes_diagnostics": {
                    key: value.copy()
                    for key, value in self.last_event_slot_mmes_diagnostics.items()
                },
            } if getattr(self, "last_event_slot_mmes_diagnostics", None) is not None
               and self.event_slot_interleaving_enable else {}),
            **({
                "event_slot_native_state_trace": copy.deepcopy(
                    self.last_event_slot_native_state_trace
                ),
            } if getattr(self, "last_event_slot_native_state_trace", None) is not None
               and self.event_slot_interleaving_enable else {}),
            **({
                "event_slot_vkd_slot_start_centers": self.last_event_slot_vector_centers["slot_start"].copy(),
                "event_slot_vkd_slot_proposal_centers": self.last_event_slot_vector_centers["proposal"].copy(),
                "event_slot_vkd_slot_committed_centers": self.last_event_slot_vector_centers["committed"].copy(),
                "event_slot_vkd_slot_shift_vectors": self.last_event_slot_vector_centers["shift"].copy(),
                "event_slot_vkd_generation_traces": copy.deepcopy(self.last_event_slot_vkd_generation_traces),
            } if getattr(self, "last_event_slot_centers", None) is not None
               and int(getattr(self.opts, "eval_save_vkd_state_trace", 0)) else {}),
            "event_slot_packet_improvements": (
                self.last_event_slot_packet_improvements.astype(
                    np.float64, copy=True
                )
            ),
            "event_slot_reported_evals_per_agent": (
                self.last_actual_fes_per_agent.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_commit_audit": {
                field: self.last_event_slot_commit_audit[field].astype(
                    np.float64, copy=True
                )
                for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS
            },
            # Forensic capture (docs 28 card).  Absent entirely when the switch
            # is off, so the env-side info dict is unchanged in that case.
            **(
                {
                    FORENSIC_INFO_PREFIX + field: value.astype(
                        np.float64, copy=True
                    )
                    for field, value in self.last_event_slot_forensic.items()
                }
                if getattr(self, "last_event_slot_forensic", None) is not None
                else {}
            ),
            **(
                {
                    "event_slot_forensic_vector_details": [
                        [dict(item) for item in slot]
                        for slot in self.last_event_slot_forensic_details
                    ]
                }
                if getattr(
                    self, "last_event_slot_forensic_details", None
                ) is not None
                else {}
            ),
            "event_slot_primary_memory_semantics": {
                "cmaes": "covariance_path_p_c",
                "mmes": "evolution_path_p",
                "sepcmaes": "covariance_path_p",
                "vkd": "covariance_path_pc",
            },
            "event_slot_secondary_memory_semantics": {
                "cmaes": "step_size_path_p_s_mapped_to_coordinates",
                "mmes": "direction_archive_q0",
                "sepcmaes": "step_size_path_s_mapped_to_coordinates",
                "vkd": "not_vector_valued_zero_sentinel",
            },
            "event_slot_effective_scale_semantics": {
                "cmaes": "sigma_times_covariance_axis_rms",
                "mmes": "sigma_isotropic_proxy",
                "sepcmaes": "sigma_times_diagonal_axis_rms",
                "vkd": "sigma_times_low_rank_diagonal_axis_rms",
            },
            "event_slot_mmes_neutral_success": (
                self.last_event_slot_mmes_neutral_success.astype(
                    np.int64, copy=True
                )
            ),
            "event_slot_packet_units_min": float(
                np.min(self.last_event_slot_packet_units)
                if self.last_event_slot_packet_units.size
                else 0.0
            ),
            "event_slot_packet_units_mean": float(
                np.mean(self.last_event_slot_packet_units)
                if self.last_event_slot_packet_units.size
                else 0.0
            ),
            "event_slot_distribution_updates_mean": float(
                np.mean(self.last_event_slot_distribution_updates)
            ),
            "event_slot_physical_evals_mean": float(
                np.mean(self.last_event_slot_physical_evals)
            ),
            "event_slot_center_shift_mean": float(
                np.mean(self.last_event_slot_center_shift_norms)
                if self.last_event_slot_center_shift_norms.size
                else 0.0
            ),
            "event_slot_center_shift_max": float(
                np.max(self.last_event_slot_center_shift_norms)
                if self.last_event_slot_center_shift_norms.size
                else 0.0
            ),
            "event_slot_proposal_commit_cosine_mean": float(
                np.mean(
                    self.last_event_slot_commit_audit[
                        "proposal_commit_cosine"
                    ]
                )
                if self.last_event_slot_count > 0
                else 0.0
            ),
            "event_slot_commit_credit_proxy_mean": float(
                np.mean(
                    self.last_event_slot_commit_audit[
                        "commit_credit_proxy"
                    ]
                )
                if self.last_event_slot_count > 0
                else 0.0
            ),
            "event_slot_primary_memory_commit_cosine_mean": float(
                np.mean(
                    self.last_event_slot_commit_audit[
                        "primary_memory_commit_cosine"
                    ]
                )
                if self.last_event_slot_count > 0
                else 0.0
            ),
            "event_slot_guide_commit_cosine_mean": float(
                np.mean(
                    self.last_event_slot_commit_audit[
                        "guide_commit_cosine"
                    ]
                )
                if self.last_event_slot_count > 0
                else 0.0
            ),
            "event_slot_mmes_neutral_success_total": int(
                np.sum(self.last_event_slot_mmes_neutral_success)
            ),
            "event_slot_execution_mode": (
                "serial_event_local"
                if self.event_slot_interleaving_enable
                else "disabled"
            ),
            "persistent_sepcmaes_enable": bool(
                self.persistent_sepcmaes_enable
            ),
            "persistent_sepcmaes_recenter_max_ratio": float(
                self.persistent_sepcmaes_recenter_max_ratio
            ),
            "persistent_sepcmaes_recenter_max_shift": float(
                self.persistent_sepcmaes_recenter_max_shift
            ),
            "persistent_sepcmaes_active": self.last_persistent_sepcmaes_active.astype(
                np.float32, copy=True
            ),
            "persistent_sepcmaes_fresh": self.last_persistent_sepcmaes_fresh.astype(
                np.float32, copy=True
            ),
            "persistent_sepcmaes_config_reset": self.last_persistent_sepcmaes_config_reset.astype(
                np.float32, copy=True
            ),
            "persistent_sepcmaes_active_ratio": float(
                np.mean(self.last_persistent_sepcmaes_active)
            ),
            "persistent_sepcmaes_fresh_ratio": float(
                np.mean(self.last_persistent_sepcmaes_fresh)
            ),
            "persistent_sepcmaes_config_reset_ratio": float(
                np.mean(self.last_persistent_sepcmaes_config_reset)
            ),
            "persistent_sepcmaes_recenter_requested": self.last_persistent_sepcmaes_recenter_requested.astype(
                np.float64, copy=True
            ),
            "persistent_sepcmaes_recenter_applied": self.last_persistent_sepcmaes_recenter_applied.astype(
                np.float64, copy=True
            ),
            "persistent_sepcmaes_recenter_remaining": self.last_persistent_sepcmaes_recenter_remaining.astype(
                np.float64, copy=True
            ),
            "persistent_sepcmaes_recenter_requested_mean": float(
                np.mean(self.last_persistent_sepcmaes_recenter_requested)
            ),
            "persistent_sepcmaes_recenter_requested_max": float(
                np.max(self.last_persistent_sepcmaes_recenter_requested)
            ),
            "persistent_sepcmaes_recenter_applied_mean": float(
                np.mean(self.last_persistent_sepcmaes_recenter_applied)
            ),
            "persistent_sepcmaes_recenter_applied_max": float(
                np.max(self.last_persistent_sepcmaes_recenter_applied)
            ),
            "persistent_sepcmaes_recenter_remaining_mean": float(
                np.mean(self.last_persistent_sepcmaes_recenter_remaining)
            ),
            "persistent_sepcmaes_recenter_remaining_max": float(
                np.max(self.last_persistent_sepcmaes_recenter_remaining)
            ),
            "persistent_sepcmaes_events": int(
                self.persistent_sepcmaes_events
            ),
            "persistent_sepcmaes_fresh_events": int(
                self.persistent_sepcmaes_fresh_events
            ),
            "persistent_sepcmaes_resume_events": int(
                self.persistent_sepcmaes_resume_events
            ),
            "persistent_sepcmaes_config_resets": int(
                self.persistent_sepcmaes_config_resets
            ),
            **self._persistent_sepcmaes_diagnostic_summary(),
            "target_block_field_enable": bool(
                self.target_block_field_enable
            ),
            "target_block_field_state_version": int(
                TARGET_BLOCK_FIELD_STATE_VERSION
            ),
            "target_block_field_path_decay": float(
                self.target_block_field_path_decay
            ),
            "target_block_field_step_rate": float(
                self.target_block_field_step_rate
            ),
            "target_block_field_radius_min_ratio": float(
                self.target_block_field_radius_min_ratio
            ),
            "target_block_field_radius_max_ratio": float(
                self.target_block_field_radius_max_ratio
            ),
            "target_block_field_events": int(
                self.target_block_field_events
            ),
            "target_block_field_local_generations_per_agent": (
                np.full(
                    (self.n_agents,),
                    self.target_block_field_events,
                    dtype=np.int64,
                )
            ),
            "target_block_field_local_evals": int(
                self.target_block_field_local_evals
            ),
            "target_block_field_secant_valid_ratio": float(
                np.mean(self.last_target_block_field_secant_valid)
                if self.last_target_block_field_secant_valid.size
                else 0.0
            ),
            "target_block_field_secant_norm_mean": float(
                np.mean(
                    np.linalg.norm(
                        self.last_target_block_field_secant,
                        axis=2,
                    )
                )
                if self.last_target_block_field_secant.size
                else 0.0
            ),
            "target_block_field_secant_alignment_mean": float(
                np.mean(
                    self.last_target_block_field_secant_alignment[
                        self.last_target_block_field_secant_valid
                    ]
                )
                if np.any(
                    self.last_target_block_field_secant_valid
                )
                else 0.0
            ),
            "target_block_field_source_diversity_mean": float(
                np.mean(
                    self.last_target_block_field_source_diversity
                )
                if self.last_target_block_field_source_diversity.size
                else 0.0
            ),
            "target_block_field_path_norm_mean": float(
                np.mean(self.last_target_block_field_path_norm)
                if self.last_target_block_field_path_norm.size
                else 0.0
            ),
            "target_block_field_path_norm_max": float(
                np.max(self.last_target_block_field_path_norm)
                if self.last_target_block_field_path_norm.size
                else 0.0
            ),
            "target_block_field_path_alignment_mean": float(
                np.mean(
                    self.last_target_block_field_path_alignment
                )
                if self.last_target_block_field_path_alignment.size
                else 0.0
            ),
            "target_block_field_conflict_ratio": float(
                np.mean(self.last_target_block_field_conflict)
                if self.last_target_block_field_conflict.size
                else 0.0
            ),
            "target_block_field_radius_mean": float(
                np.mean(self.target_block_field_radius)
                if self.target_block_field_radius.size
                else 0.0
            ),
            "target_block_field_radius_expand_ratio": float(
                np.mean(self.last_target_block_field_radius_expand)
                if self.last_target_block_field_radius_expand.size
                else 0.0
            ),
            "target_block_field_radius_shrink_ratio": float(
                np.mean(self.last_target_block_field_radius_shrink)
                if self.last_target_block_field_radius_shrink.size
                else 0.0
            ),
            "target_block_field_radius_clip_min_ratio": float(
                np.mean(self.last_target_block_field_radius_clip_min)
                if self.last_target_block_field_radius_clip_min.size
                else 0.0
            ),
            "target_block_field_radius_clip_max_ratio": float(
                np.mean(self.last_target_block_field_radius_clip_max)
                if self.last_target_block_field_radius_clip_max.size
                else 0.0
            ),
            "target_block_field_commit_norm_mean": float(
                np.mean(
                    self.last_target_block_field_applied_commit_norm
                )
                if self.last_target_block_field_applied_commit_norm.size
                else 0.0
            ),
            "target_block_field_commit_norm_max": float(
                np.max(
                    self.last_target_block_field_applied_commit_norm
                )
                if self.last_target_block_field_applied_commit_norm.size
                else 0.0
            ),
            "target_block_field_boundary_clip_ratio": float(
                np.mean(
                    self.last_target_block_field_boundary_clipped
                )
                if self.last_target_block_field_boundary_clipped.size
                else 0.0
            ),
            "target_block_field_path_injection": float(
                self.last_target_block_field_path_injection
            ),
            "target_block_field_path_angle_degrees": float(
                self.last_target_block_field_path_angle_degrees
            ),
            "target_block_field_comm_rounds": int(
                self.target_block_field_comm_rounds
            ),
            "target_block_field_messages": int(
                self.target_block_field_messages
            ),
            "target_block_field_transmitted_floats": int(
                self.target_block_field_transmitted_floats
            ),
            "target_block_field_transmitted_bytes": int(
                self.target_block_field_transmitted_floats * 8
            ),
            "target_block_dormancy_recovery_enable": bool(
                self.target_block_dormancy_recovery_enable
            ),
            "target_block_dormancy_state_version": int(
                TARGET_BLOCK_DORMANCY_STATE_VERSION
            ),
            "target_block_dormancy_progress_ratio": float(
                TARGET_BLOCK_DORMANCY_PROGRESS_RATIO
            ),
            "target_block_dormancy_unresolved_ratio": float(
                TARGET_BLOCK_DORMANCY_UNRESOLVED_RATIO
            ),
            "target_block_dormancy_patience": int(
                TARGET_BLOCK_DORMANCY_PATIENCE
            ),
            "target_block_dormancy_cooldown_events": int(
                TARGET_BLOCK_DORMANCY_COOLDOWN
            ),
            "target_block_dormancy_min_source_diversity": int(
                TARGET_BLOCK_DORMANCY_MIN_SOURCE_DIVERSITY
            ),
            "target_block_dormancy_events": int(
                self.target_block_dormancy_events
            ),
            "target_block_dormancy_activations": int(
                self.target_block_dormancy_activations
            ),
            "target_block_dormancy_activation_ratio": float(
                self.target_block_dormancy_activations
                / float(
                    max(
                        1,
                        self.target_block_dormancy_events
                        * self.n_agents
                        * self.committee_target_num,
                    )
                )
            ),
            "target_block_dormancy_ever_active_ratio": float(
                np.mean(
                    self.target_block_dormancy_activation_count > 0
                )
                if self.target_block_dormancy_activation_count.size
                else 0.0
            ),
            "target_block_dormancy_last_active_ratio": float(
                np.mean(self.last_target_block_dormancy_active)
                if self.last_target_block_dormancy_active.size
                else 0.0
            ),
            "target_block_dormancy_unresolved_block_ratio": float(
                np.mean(self.last_target_block_dormancy_unresolved)
                if self.last_target_block_dormancy_unresolved.size
                else 0.0
            ),
            "target_block_dormancy_material_progress_ratio": float(
                np.mean(
                    self.last_target_block_dormancy_material_progress
                )
                if self.last_target_block_dormancy_material_progress.size
                else 0.0
            ),
            "target_block_dormancy_reliable_scale_ratio": float(
                np.mean(
                    self.last_target_block_dormancy_reliable_scale
                )
                if self.last_target_block_dormancy_reliable_scale.size
                else 0.0
            ),
            "target_block_dormancy_residual_ratio_mean": float(
                np.mean(
                    self.last_target_block_dormancy_residual_ratio
                )
                if self.last_target_block_dormancy_residual_ratio.size
                else 0.0
            ),
            "target_block_dormancy_residual_ratio_max": float(
                np.max(
                    self.last_target_block_dormancy_residual_ratio
                )
                if self.last_target_block_dormancy_residual_ratio.size
                else 0.0
            ),
            "target_block_dormancy_progress_ratio_mean": float(
                np.mean(
                    self.last_target_block_dormancy_progress_ratio
                )
                if self.last_target_block_dormancy_progress_ratio.size
                else 0.0
            ),
            "target_block_dormancy_disagreement_ratio_mean": float(
                np.mean(
                    self.last_target_block_dormancy_disagreement_ratio
                )
                if self.last_target_block_dormancy_disagreement_ratio.size
                else 0.0
            ),
            "target_block_dormancy_disagreement_ratio_max": float(
                np.max(
                    self.last_target_block_dormancy_disagreement_ratio
                )
                if self.last_target_block_dormancy_disagreement_ratio.size
                else 0.0
            ),
            "target_block_dormancy_floor_age_mean": float(
                np.mean(self.target_block_dormancy_floor_age)
                if self.target_block_dormancy_floor_age.size
                else 0.0
            ),
            "target_block_dormancy_floor_age_max": int(
                np.max(self.target_block_dormancy_floor_age)
                if self.target_block_dormancy_floor_age.size
                else 0
            ),
            "target_block_dormancy_stagnation_age_mean": float(
                np.mean(self.target_block_dormancy_stagnation_age)
                if self.target_block_dormancy_stagnation_age.size
                else 0.0
            ),
            "target_block_dormancy_stagnation_age_max": int(
                np.max(self.target_block_dormancy_stagnation_age)
                if self.target_block_dormancy_stagnation_age.size
                else 0
            ),
            "target_block_dormancy_cooldown_mean": float(
                np.mean(self.target_block_dormancy_cooldown)
                if self.target_block_dormancy_cooldown.size
                else 0.0
            ),
            "target_block_dormancy_cooldown_max": int(
                np.max(self.target_block_dormancy_cooldown)
                if self.target_block_dormancy_cooldown.size
                else 0
            ),
            "target_block_dormancy_reserve_radius_mean": float(
                np.mean(self.target_block_dormancy_reserve_radius)
                if self.target_block_dormancy_reserve_radius.size
                else 0.0
            ),
            "target_block_dormancy_reserve_radius_max": float(
                np.max(self.target_block_dormancy_reserve_radius)
                if self.target_block_dormancy_reserve_radius.size
                else 0.0
            ),
            "target_block_dormancy_restore_radius_mean": float(
                np.mean(
                    self.last_target_block_dormancy_restore_radius
                )
                if self.last_target_block_dormancy_restore_radius.size
                else 0.0
            ),
            "target_block_dormancy_restore_radius_max": float(
                np.max(
                    self.last_target_block_dormancy_restore_radius
                )
                if self.last_target_block_dormancy_restore_radius.size
                else 0.0
            ),
            "target_block_direction_shadow_enable": bool(
                self.target_block_direction_shadow_enable
            ),
            "target_block_direction_shadow_state_version": int(
                TARGET_BLOCK_DIRECTION_SHADOW_STATE_VERSION
            ),
            "target_block_direction_shadow_sources": list(
                TARGET_BLOCK_DIRECTION_SHADOW_SOURCES
            ),
            "target_block_direction_shadow_events": int(
                self.target_block_direction_shadow_events
            ),
            "target_block_direction_shadow_eligible_blocks": int(
                self.target_block_direction_shadow_eligible_blocks
            ),
            "target_block_direction_shadow_probed_blocks": int(
                self.target_block_direction_shadow_probed_blocks
            ),
            "target_block_direction_shadow_local_evals": int(
                self.target_block_direction_shadow_local_evals
            ),
            "target_block_direction_shadow_reported_evals": 0,
            "target_block_direction_shadow_candidate_availability_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_valid[
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_positive_response_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_positive[
                        self.last_target_block_direction_shadow_candidate_valid
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_candidate_valid
                )
                else 0.0
            ),
            "target_block_direction_shadow_alternative_positive_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_positive[
                        ..., 1:
                    ][
                        self.last_target_block_direction_shadow_candidate_valid[
                            ..., 1:
                        ]
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 1:
                    ]
                )
                else 0.0
            ),
            "target_block_direction_shadow_best_response_mean": float(
                np.mean(
                    self.last_target_block_direction_shadow_best_response[
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_best_response_max": float(
                np.max(
                    self.last_target_block_direction_shadow_best_response
                )
                if self.last_target_block_direction_shadow_best_response.size
                else 0.0
            ),
            "target_block_direction_shadow_alternative_win_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_best_source[
                        self.last_target_block_direction_shadow_selected
                    ]
                    > 0
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_path_win_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_best_source[
                        self.last_target_block_direction_shadow_selected
                    ]
                    == 0
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_elite_available_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 1
                    ][
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_elite_positive_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_positive[
                        ..., 1
                    ][
                        self.last_target_block_direction_shadow_candidate_valid[
                            ..., 1
                        ]
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 1
                    ]
                )
                else 0.0
            ),
            "target_block_direction_shadow_elite_win_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_best_source[
                        self.last_target_block_direction_shadow_selected
                    ]
                    == 1
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_disagreement_available_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 2
                    ][
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_disagreement_positive_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_positive[
                        ..., 2
                    ][
                        self.last_target_block_direction_shadow_candidate_valid[
                            ..., 2
                        ]
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 2
                    ]
                )
                else 0.0
            ),
            "target_block_direction_shadow_disagreement_win_ratio": float(
                np.mean(
                    self.last_target_block_direction_shadow_best_source[
                        self.last_target_block_direction_shadow_selected
                    ]
                    == 2
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_alternative_angle_mean": float(
                np.mean(
                    self.last_target_block_direction_shadow_candidate_angle[
                        ..., 1:
                    ][
                        self.last_target_block_direction_shadow_candidate_valid[
                            ..., 1:
                        ]
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_candidate_valid[
                        ..., 1:
                    ]
                )
                else 0.0
            ),
            "target_block_direction_shadow_neighbor_support_mean": float(
                np.mean(
                    self.last_target_block_direction_shadow_support[
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_neighbor_conflict_mean": float(
                np.mean(
                    self.last_target_block_direction_shadow_conflict[
                        self.last_target_block_direction_shadow_selected
                    ]
                )
                if np.any(
                    self.last_target_block_direction_shadow_selected
                )
                else 0.0
            ),
            "target_block_direction_shadow_applied_commits": 0,
            "target_block_direction_shadow_comm_rounds": int(
                self.target_block_direction_shadow_comm_rounds
            ),
            "target_block_direction_shadow_messages": int(
                self.target_block_direction_shadow_messages
            ),
            "target_block_direction_shadow_transmitted_floats": int(
                self.target_block_direction_shadow_transmitted_floats
            ),
            "target_block_direction_shadow_transmitted_bytes": int(
                self.target_block_direction_shadow_transmitted_floats * 8
            ),
            "target_block_challenge_response_mode": str(
                self.target_block_challenge_response_mode
            ),
            "target_block_challenge_response_state_version": int(
                TARGET_BLOCK_CHALLENGE_RESPONSE_STATE_VERSION
            ),
            "target_block_challenge_response_sources": list(
                TARGET_BLOCK_CHALLENGE_RESPONSE_SOURCES
            ),
            "target_block_challenge_events": int(
                self.target_block_challenge_events
            ),
            "target_block_challenge_challenges": int(
                self.target_block_challenge_challenges
            ),
            "target_block_challenge_directed_responses": int(
                self.target_block_challenge_directed_responses
            ),
            "target_block_challenge_local_evals": int(
                self.target_block_challenge_local_evals
            ),
            "target_block_challenge_reported_evals": int(
                self.target_block_challenge_reported_evals
            ),
            "target_block_challenge_candidate_availability_ratio": float(
                np.mean(
                    self.last_target_block_challenge_candidate_valid[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_alternative_win_ratio": float(
                np.mean(
                    self.last_target_block_challenge_margin[
                        self.last_target_block_challenge_selected
                    ] > 0.0
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_path_score_mean": float(
                np.mean(
                    self.last_target_block_challenge_path_score[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_alternative_score_mean": float(
                np.mean(
                    self.last_target_block_challenge_alternative_score[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_margin_mean": float(
                np.mean(
                    self.last_target_block_challenge_margin[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_response_coverage_mean": float(
                np.mean(
                    self.last_target_block_challenge_coverage[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_positive_sources_mean": float(
                np.mean(
                    self.last_target_block_challenge_positive_sources[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_neighbor_positive_sources_mean": float(
                np.mean(
                    self.last_target_block_challenge_neighbor_positive_sources[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_support_mean": float(
                np.mean(
                    self.last_target_block_challenge_support[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_conflict_mean": float(
                np.mean(
                    self.last_target_block_challenge_conflict[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_actuation_eligible_ratio": float(
                np.mean(
                    self.last_target_block_challenge_actuation_eligible[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_applied_ratio": float(
                np.mean(
                    self.last_target_block_challenge_applied[
                        self.last_target_block_challenge_selected
                    ]
                )
                if np.any(self.last_target_block_challenge_selected)
                else 0.0
            ),
            "target_block_challenge_applied_norm_mean": float(
                np.mean(
                    self.last_target_block_challenge_applied_norm[
                        self.last_target_block_challenge_applied
                    ]
                )
                if np.any(self.last_target_block_challenge_applied)
                else 0.0
            ),
            "target_block_challenge_applied_commits": int(
                self.target_block_challenge_applied_commits
            ),
            "target_block_challenge_comm_rounds": int(
                self.target_block_challenge_comm_rounds
            ),
            "target_block_challenge_messages": int(
                self.target_block_challenge_messages
            ),
            "target_block_challenge_transmitted_floats": int(
                self.target_block_challenge_transmitted_floats
            ),
            "target_block_challenge_transmitted_bytes": int(
                self.target_block_challenge_transmitted_floats * 8
            ),
            "target_block_dual_clock_enable": bool(
                self.target_block_dual_clock_enable
            ),
            "target_block_dual_clock_commit_lock_enable": bool(
                self.target_block_dual_clock_commit_lock_enable
            ),
            "target_block_dual_clock_state_version": int(
                TARGET_BLOCK_DUAL_CLOCK_STATE_VERSION
            ),
            "target_block_dual_clock_outer_events": int(
                self.target_block_dual_clock_outer_events
            ),
            "target_block_dual_clock_local_generation_ticks": int(
                self.target_block_dual_clock_local_generation_ticks
            ),
            "target_block_dual_clock_communication_ticks": int(
                self.target_block_dual_clock_communication_ticks
            ),
            "target_block_dual_clock_commit_ticks": int(
                self.target_block_dual_clock_commit_ticks
            ),
            "target_block_dual_clock_last_microcycles": int(
                self.last_target_block_dual_clock_microcycles
            ),
            "target_block_dual_clock_generation_fes_per_agent": (
                self.last_target_block_dual_clock_generation_fes.astype(
                    np.int64,
                    copy=True,
                )
            ),
            "target_block_commit_credit_mode": str(
                self.target_block_commit_credit_mode
            ),
            "target_block_commit_credit_events": int(
                self.target_block_commit_credit_events
            ),
            "target_block_commit_credit_active_ratio": float(
                np.mean(self.last_target_block_commit_credit_active)
            ),
            "target_block_commit_credit_mean": float(
                np.mean(self.last_target_block_commit_credit)
            ),
            "target_block_commit_credit_min": float(
                np.min(self.last_target_block_commit_credit)
            ),
            "target_block_commit_credit_max": float(
                np.max(self.last_target_block_commit_credit)
            ),
            "target_block_commit_credit_cosine_mean": float(
                np.mean(self.last_target_block_commit_credit_cosine)
            ),
            "target_block_commit_credit_proposal_norm_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_proposal_norm
                )
            ),
            "target_block_commit_credit_commit_norm_mean": float(
                np.mean(self.last_target_block_commit_credit_commit_norm)
            ),
            "target_block_commit_credit_correction_norm_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_correction_norm
                )
            ),
            "target_block_commit_credit_path_retention_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_path_retention
                )
            ),
            "target_block_commit_credit_scale_retention_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_scale_retention
                )
            ),
            "target_block_commit_credit_scale_contraction_mean": float(
                np.mean(
                    1.0
                    - self.last_target_block_commit_credit_scale_retention
                )
            ),
            "target_block_commit_credit_scale_contraction_max": float(
                np.max(
                    1.0
                    - self.last_target_block_commit_credit_scale_retention
                )
            ),
            "target_block_commit_credit_axis_rms_before_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_axis_rms_before
                )
            ),
            "target_block_commit_credit_axis_rms_after_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_axis_rms_after
                )
            ),
            "target_block_commit_credit_sigma_path_norm_before_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_sigma_path_before
                )
            ),
            "target_block_commit_credit_sigma_path_norm_after_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_sigma_path_after
                )
            ),
            "target_block_commit_credit_cov_path_norm_before_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_cov_path_before
                )
            ),
            "target_block_commit_credit_cov_path_norm_after_mean": float(
                np.mean(
                    self.last_target_block_commit_credit_cov_path_after
                )
            ),
            "candidate_shape_fallbacks": int(candidate_shape_fallbacks),
            "cmaes_numeric_fail_soft_enabled": bool(
                self.cmaes_numeric_fail_soft
            ),
            "cmaes_numeric_fail_soft_step_count": int(
                cmaes_fail_soft_step_count
            ),
            "cmaes_numeric_fail_soft_events": int(
                self.cmaes_numeric_fail_soft_events
            ),
            "cmaes_numeric_fail_soft_agent_mask": (
                self.last_cmaes_numeric_fail_soft.astype(
                    np.float64, copy=True
                ).tolist()
            ),
            "cmaes_numeric_fail_soft_generations": (
                self.last_cmaes_numeric_fail_soft_generation.astype(
                    np.int64, copy=True
                ).tolist()
            ),
            "cmaes_numeric_fail_soft_evaluations": (
                cmaes_fail_soft_evaluations.astype(
                    np.int64, copy=True
                ).tolist()
            ),
            "cmaes_numeric_fail_soft_reasons": list(
                cmaes_fail_soft_reasons
            ),
            "consensus_mode": self.consensus_mode,
            "consensus_strength": float(self.consensus_strength),
            "communication_applied": bool(communication_applied),
            "did_communicate": bool(communication_applied),
            "state_comm_mode": self.state_comm_mode,
            "state_comm_enabled": bool(self.state_comm_enabled),
            "state_comm_cost_mode": str(self.state_comm_cost_mode),
            "state_comm_include_delta": bool(self.state_comm_include_delta),
            "state_comm_include_actual_fes": bool(self.state_comm_include_actual_fes),
            "state_comm_msg_dim": int(self.state_msg_dim),
            "candidate_history_obs_enable": bool(
                self.candidate_history_obs_enable
            ),
            "candidate_history_obs_dim": int(self.candidate_history_obs_dim),
            "candidate_actuator_action_enable": bool(
                self.candidate_actuator_action_enable
            ),
            "candidate_actuator_action_idx_mean": float(
                np.mean(self.last_candidate_actuator_action)
            ),
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
            **self.last_committee_metrics,
            **self.last_candidate_response_metrics,
            "mean_disagreement": float(post_mean_disagreement),
            "max_edge_disagreement": float(post_max_edge),
            "proposal_disagreement": float(proposal_mean_disagreement),
            "consensus_improve": float(consensus_improvement),
            "consensus_operator_improve": float(
                consensus_operator_improvement
            ),
            "local_search_fes": int(self.local_search_fes),
            "candidate_validation_local_evals": int(
                self.candidate_validation_local_evals
            ),
            "agent_state_local_evals": int(self.agent_state_local_evals),
            "global_monitor_local_evals": int(self.global_monitor_local_evals),
            "global_monitor_rounds": int(self.global_monitor_rounds),
            "verification_local_evals": int(verification_local_evals),
            "total_physical_local_calls": int(total_physical_local_calls),
            "state_comm_rounds": int(self.state_comm_rounds),
            "state_comm_messages": int(self.state_comm_messages),
            "state_comm_transmitted_floats": int(
                self.state_comm_transmitted_floats
            ),
            "state_comm_transmitted_bytes": int(
                self.state_comm_transmitted_floats * 8
            ),
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
            "rvcpd_arm": str(self.rvcpd_arm),
            "rvcpd_integration_mode": str(self.rvcpd_integration_mode),
            "rvcpd_enabled": bool(self.rvcpd_enabled),
            "rvcpd_active_ratio": float(np.mean(self.last_rvcpd_active)),
            "rvcpd_forward_ratio": float(
                np.mean(self.last_rvcpd_sign > 0)
            ),
            "rvcpd_reverse_ratio": float(
                np.mean(self.last_rvcpd_sign < 0)
            ),
            "rvcpd_noop_ratio": float(
                np.mean(
                    (self.last_rvcpd_active > 0.0)
                    & (self.last_rvcpd_sign == 0)
                )
            ),
            "rvcpd_agreement_mean": float(
                np.mean(self.last_rvcpd_agreement)
            ),
            "rvcpd_path_norm_mean": float(
                np.mean(np.linalg.norm(self.rvcpd_path, axis=1))
            ),
            "rvcpd_path_norm_max": float(
                np.max(np.linalg.norm(self.rvcpd_path, axis=1))
            ),
            "rvcpd_support_ema_mean": float(
                np.mean(self.rvcpd_support_ema)
            ),
            "rvcpd_conflict_ema_mean": float(
                np.mean(self.rvcpd_conflict_ema)
            ),
            "rvcpd_uncertainty_ema_mean": float(
                np.mean(self.rvcpd_uncertainty_ema)
            ),
            "rvcpd_trust_mean": float(np.mean(self.rvcpd_trust)),
            "rvcpd_scale_mean": float(np.mean(self.rvcpd_scale)),
            "rvcpd_scale_min_fraction": float(
                np.mean(
                    self.rvcpd_scale
                    <= self.rvcpd_scale_min + 1e-12
                )
            ),
            "rvcpd_scale_max_fraction": float(
                np.mean(
                    self.rvcpd_scale
                    >= self.rvcpd_scale_max - 1e-12
                )
            ),
            "rvcpd_age_mean": float(np.mean(self.rvcpd_age)),
            "rvcpd_requested_radius_mean": float(
                np.mean(self.last_rvcpd_requested_radius)
            ),
            "rvcpd_applied_radius_mean": float(
                np.mean(self.last_rvcpd_applied_radius)
            ),
            "rvcpd_plus_gain_mean": float(
                np.mean(self.last_rvcpd_plus_gain)
            ),
            "rvcpd_minus_gain_mean": float(
                np.mean(self.last_rvcpd_minus_gain)
            ),
            "rvcpd_events": int(self.rvcpd_events),
            "rvcpd_valid_event_count": int(
                self.rvcpd_valid_event_count
            ),
            "rvcpd_commit_count": int(self.rvcpd_commit_count),
            "rvcpd_reverse_count": int(self.rvcpd_reverse_count),
            "rvcpd_noop_count": int(self.rvcpd_noop_count),
            "rvcpd_probe_local_evals": int(
                self.rvcpd_probe_local_evals
            ),
            "rvcpd_messages": int(self.rvcpd_messages),
            "rvcpd_transmitted_floats": int(
                self.rvcpd_transmitted_floats
            ),
            "rvcpd_transmitted_bytes": int(
                self.rvcpd_transmitted_floats * 8
            ),
            "guide_replacement_mode": str(self.guide_replacement_mode),
            "guide_replacement_scope": str(self.guide_replacement_scope),
            "guide_replacement_enabled": bool(self.guide_replacement_enabled),
            "guide_replacement_eligible_ratio": float(
                np.mean(self.last_guide_replacement_eligible)
            ),
            "guide_replacement_active_ratio": float(
                np.mean(self.last_guide_replacement_valid)
            ),
            "guide_replacement_old_guide_suppressed_ratio": float(
                np.mean(self.last_guide_replacement_old_guide_suppressed)
            ),
            "guide_replacement_forward_ratio": float(
                np.mean(self.last_guide_replacement_sign > 0)
            ),
            "guide_replacement_reverse_ratio": float(
                np.mean(self.last_guide_replacement_sign < 0)
            ),
            "guide_replacement_noop_ratio": float(
                np.mean(
                    (self.last_guide_replacement_eligible > 0.0)
                    & (self.last_guide_replacement_sign == 0)
                )
            ),
            "guide_replacement_strength_mean": float(
                np.mean(self.last_guide_replacement_strength)
            ),
            "guide_replacement_sigma_mean": float(
                np.mean(self.last_guide_replacement_sigma)
            ),
            "guide_replacement_requested_radius_mean": float(
                np.mean(self.last_guide_replacement_requested_radius)
            ),
            "guide_replacement_applied_radius_mean": float(
                np.mean(self.last_guide_replacement_applied_radius)
            ),
            "guide_replacement_plus_gain_mean": float(
                np.mean(self.last_guide_replacement_plus_gain)
            ),
            "guide_replacement_minus_gain_mean": float(
                np.mean(self.last_guide_replacement_minus_gain)
            ),
            "guide_replacement_events": int(self.guide_replacement_events),
            "guide_replacement_eligible_count": int(
                self.guide_replacement_eligible_count
            ),
            "guide_replacement_valid_event_count": int(
                self.guide_replacement_valid_event_count
            ),
            "guide_replacement_forward_count": int(
                self.guide_replacement_forward_count
            ),
            "guide_replacement_reverse_count": int(
                self.guide_replacement_reverse_count
            ),
            "guide_replacement_noop_count": int(
                self.guide_replacement_noop_count
            ),
            "guide_replacement_commit_count": int(
                self.guide_replacement_commit_count
            ),
            "guide_replacement_probe_local_evals": int(
                self.guide_replacement_probe_local_evals
            ),
            "collective_guide_mode": str(self.collective_guide_mode),
            "collective_guide_enabled": bool(self.collective_guide_enabled),
            "collective_guide_valid": bool(self.last_collective_guide_valid),
            "collective_guide_source_valid_ratio": float(
                np.mean(self.last_collective_guide_source_valid)
            ),
            "collective_guide_vote_valid_ratio": float(
                np.mean(self.last_collective_guide_vote_valid)
            ),
            "collective_guide_positive_vote_ratio": float(
                np.mean(self.last_collective_guide_vote > 0)
            ),
            "collective_guide_negative_vote_ratio": float(
                np.mean(self.last_collective_guide_vote < 0)
            ),
            "collective_guide_zero_vote_ratio": float(
                np.mean(self.last_collective_guide_vote == 0)
            ),
            "collective_guide_vote_sum": int(
                self.last_collective_guide_vote_sum
            ),
            "collective_guide_hypothetical_veto": bool(
                self.last_collective_guide_hypothetical_veto
            ),
            "collective_guide_null_veto": bool(
                self.last_collective_guide_null_veto
            ),
            "collective_guide_actual_veto": bool(
                self.last_collective_guide_actual_veto
            ),
            "collective_guide_suppressed_ratio": float(
                np.mean(self.last_collective_guide_suppressed)
            ),
            "collective_guide_radius": float(
                np.mean(self.last_collective_guide_radius)
            ),
            "collective_guide_shared_base_receiver_max_diff": float(
                np.max(
                    np.abs(
                        self.last_collective_guide_shared_base
                        - self.last_collective_guide_shared_base[0:1]
                    )
                )
            ),
            "collective_guide_direction_receiver_max_diff": float(
                np.max(
                    np.abs(
                        self.last_collective_guide_direction
                        - self.last_collective_guide_direction[0:1]
                    )
                )
            ),
            "collective_guide_events": int(self.collective_guide_events),
            "collective_guide_valid_events": int(
                self.collective_guide_valid_events
            ),
            "collective_guide_hypothetical_veto_events": int(
                self.collective_guide_hypothetical_veto_events
            ),
            "collective_guide_actual_veto_events": int(
                self.collective_guide_actual_veto_events
            ),
            "collective_guide_probe_local_evals": int(
                self.collective_guide_probe_local_evals
            ),
            "collective_guide_comm_rounds": int(
                self.collective_guide_comm_rounds
            ),
            "collective_guide_messages": int(
                self.collective_guide_messages
            ),
            "collective_guide_transmitted_floats": int(
                self.collective_guide_transmitted_floats
            ),
            "collective_guide_transmitted_bytes": int(
                self.collective_guide_transmitted_floats * 8
            ),
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
            "committee_mode": str(self.committee_mode),
            "committee_selection": str(self.committee_selection),
            "committee_acceptance_mode": str(self.committee_acceptance_mode),
            "committee_acceptance_min_log_improve": float(
                self.committee_acceptance_min_log_improve
            ),
            "committee_events": int(self.committee_events),
            "committee_decision_local_evals": int(
                self.committee_decision_local_evals
            ),
            "committee_shadow_local_evals": int(
                self.committee_shadow_local_evals
            ),
            "committee_shadow_global_local_evals": int(
                self.committee_shadow_global_local_evals
            ),
            "committee_messages": int(self.committee_messages),
            "committee_transmitted_floats": int(
                self.committee_transmitted_floats
            ),
            "committee_transmitted_bytes": int(
                self.committee_transmitted_floats * 8
            ),
            "committee_acceptance_events": int(
                self.committee_acceptance_events
            ),
            "committee_acceptance_accepted_events": int(
                self.committee_acceptance_accepted_events
            ),
            "committee_acceptance_rejected_events": int(
                self.committee_acceptance_rejected_events
            ),
            "committee_acceptance_local_evals": int(
                self.committee_acceptance_local_evals
            ),
            "committee_acceptance_messages": int(
                self.committee_acceptance_messages
            ),
            "committee_acceptance_transmitted_floats": int(
                self.committee_acceptance_transmitted_floats
            ),
            "committee_acceptance_transmitted_bytes": int(
                self.committee_acceptance_transmitted_floats * 8
            ),
            "candidate_response_mode": str(self.candidate_response_mode),
            "candidate_generator": str(self.candidate_generator),
            "candidate_response_events": int(self.candidate_response_events),
            "candidate_response_actuated_events": int(
                self.candidate_response_actuated_events
            ),
            "candidate_response_probe_local_evals": int(
                self.candidate_response_probe_local_evals
            ),
            "candidate_response_verification_local_evals": int(
                self.candidate_response_verification_local_evals
            ),
            "candidate_response_shadow_global_local_evals": int(
                self.candidate_response_shadow_global_local_evals
            ),
            "candidate_response_rounds": int(self.candidate_response_rounds),
            "candidate_response_messages": int(self.candidate_response_messages),
            "candidate_response_transmitted_floats": int(
                self.candidate_response_transmitted_floats
            ),
            "candidate_response_transmitted_bytes": int(
                self.candidate_response_transmitted_floats * 8
            ),
            "committee_acceptance_early_events": int(
                self.committee_acceptance_stage_events[0]
            ),
            "committee_acceptance_mid_events": int(
                self.committee_acceptance_stage_events[1]
            ),
            "committee_acceptance_late_events": int(
                self.committee_acceptance_stage_events[2]
            ),
            "committee_acceptance_early_accepted_events": int(
                self.committee_acceptance_stage_accepted_events[0]
            ),
            "committee_acceptance_mid_accepted_events": int(
                self.committee_acceptance_stage_accepted_events[1]
            ),
            "committee_acceptance_late_accepted_events": int(
                self.committee_acceptance_stage_accepted_events[2]
            ),
            "committee_acceptance_ratio_early": float(
                self.committee_acceptance_stage_accepted_events[0]
                / max(1, self.committee_acceptance_stage_events[0])
            ),
            "committee_acceptance_ratio_mid": float(
                self.committee_acceptance_stage_accepted_events[1]
                / max(1, self.committee_acceptance_stage_events[1])
            ),
            "committee_acceptance_ratio_late": float(
                self.committee_acceptance_stage_accepted_events[2]
                / max(1, self.committee_acceptance_stage_events[2])
            ),
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

    def _expand_d5_primary_actions(self, primary_actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(primary_actions)
        if not self.candidate_actuator_action_enable:
            raise RuntimeError(
                "D5 primary-action expansion requires the legacy actuator "
                "column to remain enabled."
            )
        expected_primary_cols = int(self.action_space.nvec.shape[1] - 1)
        expected = (self.n_agents, expected_primary_cols)
        if actions.shape != expected:
            raise ValueError(
                f"Expected D5 primary actions with shape {expected}, "
                f"got {actions.shape}."
            )
        placeholder = np.full(
            (self.n_agents, 1),
            int(self.candidate_actuator_initial_action),
            dtype=np.int64,
        )
        return np.concatenate(
            [actions.astype(np.int64, copy=False), placeholder],
            axis=1,
        )

    @staticmethod
    def _finish_step_generator(generator, actuator_actions):
        try:
            generator.send(actuator_actions)
        except StopIteration as stop:
            return stop.value
        raise RuntimeError("D5 transition generator yielded more than once.")

    def prepare_step(
        self,
        primary_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if not self.d5_two_stage_actuator_enable:
            raise RuntimeError(
                "prepare_step requires "
                "objective_split_d5_two_stage_actuator_enable=1."
            )
        if self.d6_pre_generator_selector_enable:
            raise RuntimeError(
                "D6 is enabled; call prepare_generator_step(), "
                "prepare_actuator_step(), then commit_step()."
            )
        self._assert_no_pending_transition("prepare another step")
        full_actions = self._expand_d5_primary_actions(primary_actions)
        generator = self._step_transition(full_actions)
        try:
            payload = next(generator)
        except StopIteration as stop:
            raise RuntimeError(
                "D5 transition completed before producing verifier context."
            ) from stop
        self._pending_step_generator = generator
        self._pending_prepare_payload = payload
        self._pending_transition_phase = "PENDING_ACTUATOR"
        return payload

    def prepare_generator_step(
        self,
        primary_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        if not self.d6_pre_generator_selector_enable:
            raise RuntimeError(
                "prepare_generator_step requires "
                "objective_split_d6_pre_generator_selector_enable=1."
            )
        self._assert_no_pending_transition("prepare another generator stage")
        full_actions = self._expand_d5_primary_actions(primary_actions)
        generator = self._step_transition(full_actions)
        try:
            payload = next(generator)
        except StopIteration as stop:
            raise RuntimeError(
                "D6 transition completed before producing generator context."
            ) from stop
        if not isinstance(payload, tuple) or len(payload) != 4:
            raise RuntimeError("D6 generator stage returned an invalid payload.")
        self._pending_step_generator = generator
        self._pending_prepare_payload = payload
        self._pending_transition_phase = "PENDING_GENERATOR"
        return payload

    def prepare_actuator_step(
        self,
        generator_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        if not self.d6_pre_generator_selector_enable:
            raise RuntimeError(
                "prepare_actuator_step requires D6 pre-generator mode."
            )
        if self._pending_transition_phase != "PENDING_GENERATOR":
            raise RuntimeError(
                "Cannot prepare actuator stage without a pending D6 "
                "generator contract."
            )
        generator = self._pending_step_generator
        if generator is None:
            raise RuntimeError("D6 generator contract is missing.")
        actions = np.asarray(
            generator_actions,
            dtype=np.int64,
        ).reshape(self.n_agents)
        try:
            payload = generator.send(actions)
        except StopIteration as stop:
            self._pending_step_generator = None
            self._pending_prepare_payload = None
            self._pending_transition_phase = "READY"
            self._d5_transition_broken = True
            raise RuntimeError(
                "D6 transition completed before producing actuator context; "
                "the environment must be reconstructed."
            ) from stop
        except Exception as exc:
            self._pending_step_generator = None
            self._pending_prepare_payload = None
            self._pending_transition_phase = "READY"
            self._d5_transition_broken = True
            raise RuntimeError(
                "D6 generator dispatch failed; the environment must be "
                "reconstructed."
            ) from exc
        if not isinstance(payload, tuple) or len(payload) != 3:
            self._pending_step_generator = None
            self._pending_prepare_payload = None
            self._pending_transition_phase = "READY"
            self._d5_transition_broken = True
            raise RuntimeError(
                "D6 actuator stage returned an invalid payload."
            )
        self._pending_prepare_payload = payload
        self._pending_transition_phase = "PENDING_ACTUATOR"
        return payload

    def commit_step(
        self,
        actuator_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        generator = self._pending_step_generator
        if generator is None:
            raise RuntimeError("Cannot commit D5 transition without prepare_step.")
        if self._pending_transition_phase != "PENDING_ACTUATOR":
            raise RuntimeError(
                "Cannot commit before the actuator stage is prepared."
            )
        actions = np.asarray(
            actuator_actions,
            dtype=np.int64,
        ).reshape(self.n_agents)
        try:
            result = self._finish_step_generator(generator, actions)
        except Exception as exc:
            self._pending_step_generator = None
            self._pending_prepare_payload = None
            self._pending_transition_phase = "READY"
            self._d5_transition_broken = True
            raise RuntimeError(
                "D5 commit failed; the environment is now unusable and must "
                "be reconstructed."
            ) from exc
        self._pending_step_generator = None
        self._pending_prepare_payload = None
        self._pending_transition_phase = "READY"
        return result

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        if (
            self.d5_two_stage_actuator_enable
            or self.d6_pre_generator_selector_enable
        ):
            raise RuntimeError(
                "Legacy step() is disabled while a staged D5/D6 protocol is "
                "enabled."
            )
        self._assert_no_pending_transition("run a legacy step")
        generator = self._step_transition(actions)
        try:
            next(generator)
        except StopIteration as stop:
            raise RuntimeError(
                "Objective-split transition completed before its commit point."
            ) from stop
        return self._finish_step_generator(generator, None)


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

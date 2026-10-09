"""Detached row builder for event-slot evaluation diagnostics.

This module only validates and reshapes values that were already produced by
the environment.  It must not call an objective or mutate environment state.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import json

import numpy as np


EVENT_SLOT_COMMIT_AUDIT_FIELDS = (
    "proposal_step_norm",
    "commit_step_norm",
    "correction_norm",
    "proposal_commit_cosine",
    "commit_credit_proxy",
    "internal_center_step_norm",
    "internal_center_commit_cosine",
    "primary_memory_norm",
    "primary_memory_commit_cosine",
    "secondary_memory_norm",
    "secondary_memory_commit_cosine",
    "effective_scale_rms",
    "effective_scale_commit_ratio",
    "guide_norm",
    "guide_strength",
    "guide_mix_strength",
    "guide_commit_cosine",
)


# Optional VKD observations must not change the strict historical audit schema.
EVENT_SLOT_VKD_FIELDS = (
    "vkd_core_available",
    "vkd_sigma",
    "vkd_ps",
    "vkd_alpha",
    "vkd_hsig",
    "vkd_sigma_consumer",
    "vkd_shape_consumer",
    "vkd_pc_norm",
    "vkd_D_rms",
    "vkd_S_norm",
    "vkd_V_norm",
    "vkd_state_finite",
    "vkd_candidate_finite",
    "vkd_fitness_finite",
    "vkd_objective_calls",
)

MMES_FIELDS = (
    "mmes_state_transition_mode", "mmes_neutralize_success", "mmes_credit_action",
    "mmes_y_bak_comparable", "mmes_previous_slot_shift_norm",
    "mmes_stale_credit_pending_at_start", "mmes_stale_source_shift_norm",
    "mmes_y_bak_incomparable_updates", "mmes_y_bak_comparable_updates",
    "mmes_neutralized_updates", "mmes_paired_updates",
    "mmes_sigma_before", "mmes_sigma_after",
    "mmes_success_stat_before", "mmes_success_stat_after",
)


EVENT_SLOT_FORENSIC_FIELDS = (
    "objective_local_f_before",
    "objective_local_f_after",
    "objective_call_count_event",
    "sigma_preview_f64",
    "sigma_cast_overflow",
    "sigma_exp_arg_raw",
    "optimizer_lambda",
    "optimizer_generation_index",
    "optimizer_terminated",
    "internal_mean_norm",
    "internal_step_norm",
    "guide_injected",
    "guide_candidate_finite",
    "remaining_budget_fes",
    "event_f_spread",
    "packet_candidate_finite",
    "packet_candidate_is_nan",
    "packet_candidate_is_inf",
    "packet_candidate_max_abs",
    "packet_candidate_stable_norm",
    "packet_candidate_norm_overflow",
    "internal_mean_finite",
    "internal_mean_is_nan",
    "internal_mean_is_inf",
    "internal_mean_max_abs",
    "internal_mean_stable_norm",
    "internal_mean_norm_overflow",
    "internal_shift_finite",
    "internal_shift_is_nan",
    "internal_shift_is_inf",
    "internal_shift_max_abs",
    "internal_shift_stable_norm",
    "internal_shift_norm_overflow",
)

# Prefix used by the environment when publishing the forensic matrices.
FORENSIC_INFO_PREFIX = "event_slot_forensic_"


def _vkd_row(info: Dict, slot_id: int, agent_id: int, shape) -> dict:
    raw = info.get("event_slot_vkd_diagnostics")
    if not isinstance(raw, dict):
        return {}
    row = {
        field: float(_optional_matrix(raw, field, shape)[slot_id, agent_id])
        for field in EVENT_SLOT_VKD_FIELDS
    }
    row["vkd_ps_outlet_mode"] = str(info.get("vkd_ps_outlet_mode", "native"))
    for key in (
        "event_slot_vkd_slot_start_centers",
        "event_slot_vkd_slot_proposal_centers",
        "event_slot_vkd_slot_committed_centers",
        "event_slot_vkd_slot_shift_vectors",
    ):
        values = info.get(key)
        if values is not None:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim == 3 and array.shape[:2] == shape:
                row[key] = json.dumps(array[slot_id, agent_id].tolist())
    traces = info.get("event_slot_vkd_generation_traces")
    if traces is not None:
        try:
            row["event_slot_vkd_generation_trace"] = json.dumps(
                traces[slot_id][agent_id], ensure_ascii=False
            )
        except Exception:
            row["event_slot_vkd_generation_trace"] = ""
    return row


def _mmes_row(info: Dict, slot_id: int, agent_id: int, shape) -> dict:
    raw = info.get("event_slot_mmes_diagnostics")
    if not isinstance(raw, dict):
        return {}
    row = {}
    for field in MMES_FIELDS:
        values = raw.get(field)
        if values is None:
            continue
        values = np.asarray(values, dtype=object)
        if values.shape != tuple(shape):
            raise ValueError(
                f"MMES diagnostic field {field} has shape {values.shape}; expected {tuple(shape)}."
            )
        row[field] = values[slot_id, agent_id]
    return row


def _native_state_row(info: Dict, slot_id: int, agent_id: int, shape) -> dict:
    trace = info.get("event_slot_native_state_trace")
    if trace is None:
        return {}
    if len(trace) != int(shape[0]) or any(len(slot) != int(shape[1]) for slot in trace):
        raise ValueError("Event-slot native state trace has the wrong slot/agent shape.")
    record = trace[slot_id][agent_id]
    if record is None:
        return {}
    return {"event_slot_native_state_trace": json.dumps(record, ensure_ascii=False)}


def _flush_live_jump_windows(reason: str) -> None:
    """Best-effort pre-failure jump line for the live optimizer windows.

    A non-finite diagnostic aborts the event before its optimizer core is asked
    to flush, so the reserved failure line is written from the process-wide
    registry of live windows instead.  Diagnostic only; never raises.
    """
    try:
        from optimizers.cmaes.numeric_forensics import flush_registered_windows

        flush_registered_windows(str(reason))
    except Exception:
        pass


def _matrix(info: Dict, key: str, shape, dtype) -> np.ndarray:
    if key not in info:
        raise ValueError(f"Event-slot diagnostics are missing required field {key}.")
    value = np.asarray(info[key], dtype=dtype)
    if value.shape != tuple(shape):
        raise ValueError(
            f"Event-slot diagnostic field {key} has shape {value.shape}; "
            f"expected {tuple(shape)}."
        )
    if np.issubdtype(value.dtype, np.floating) and not np.all(
        np.isfinite(value)
    ):
        if info.get("vkd_origin_trace_dir") and value.ndim in (1, 2):
            try:
                from optimizers.unified_opt.vkd_origin_trace import failure_from_matrix
                failure_from_matrix(info["vkd_origin_trace_dir"],
                                    info["vkd_origin_identity"], key, value)
            except Exception:
                pass
        _flush_live_jump_windows(f"event_slot_diagnostics_non_finite:{key}")
        raise ValueError(
            f"Event-slot diagnostic field {key} contains non-finite values."
        )
    return value


def _agent_vector(info: Dict, key: str, n_agents: int, dtype) -> np.ndarray:
    return _matrix(info, key, (n_agents,), dtype)


def _optional_matrix(info: Dict, key: str, shape) -> np.ndarray:
    """Lenient sibling of ``_matrix`` for the forensic columns.

    The forensic trace is best-effort by contract (docs 28 card): a missing or
    non-finite value must never abort a run, otherwise the trace would itself
    become a new crash source.  Missing or mis-shaped values become NaN, which
    is recorded verbatim so "not captured" stays distinguishable from zero.
    """
    value = info.get(key)
    if value is None:
        return np.full(shape, np.nan, dtype=np.float64)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return np.full(shape, np.nan, dtype=np.float64)
    if arr.shape != tuple(shape):
        return np.full(shape, np.nan, dtype=np.float64)
    return arr


def build_event_slot_diagnostic_records(
    *,
    info: Dict,
    actions,
    rewards,
    optimizer_candidates: Sequence[str],
    resource_factors: Sequence[float],
    cfg_param_num: int,
    comm_action_enable: bool,
    collab_action_enable: bool,
    guide_scale_action_enable: bool,
    seed: int,
    function_id: int,
    strategy: str,
    rollout_step: int,
    pre_sum_fes: int,
    post_sum_fes: int,
    phase: str,
) -> List[dict]:
    """Expand one already-completed event into validated slot-agent rows."""

    if not bool(info.get("event_slot_interleaving_enable", False)):
        raise ValueError(
            "Event-slot diagnostics require event-slot interleaving to be enabled."
        )
    slot_count = int(info.get("event_slot_count", 0))
    if slot_count <= 0:
        raise ValueError("Event-slot diagnostics received an empty event.")

    action_matrix = np.asarray(actions, dtype=np.int64)
    if action_matrix.ndim != 2:
        raise ValueError(
            "Event-slot diagnostic actions must have shape [agent, action_column]."
        )
    n_agents = int(action_matrix.shape[0])
    cfg_param_num = int(cfg_param_num)
    resource_col = 1 + cfg_param_num
    comm_col = resource_col + 1
    collab_col = comm_col + int(bool(comm_action_enable))
    guide_scale_col = collab_col + int(bool(collab_action_enable))
    required_cols = guide_scale_col + int(bool(guide_scale_action_enable))
    if action_matrix.shape[1] < required_cols:
        raise ValueError(
            "Event-slot diagnostic action matrix is missing configured action columns."
        )

    packet_units = _matrix(
        info,
        "event_slot_packet_units",
        (slot_count, n_agents),
        np.int64,
    )
    packet_improvements = _matrix(
        info,
        "event_slot_packet_improvements",
        (slot_count, n_agents),
        np.float64,
    )
    distribution_updates = _agent_vector(
        info,
        "event_slot_distribution_updates",
        n_agents,
        np.int64,
    )
    native_evals = _agent_vector(
        info,
        "event_slot_native_evals_per_agent",
        n_agents,
        np.int64,
    )
    reported_evals = _agent_vector(
        info,
        "event_slot_reported_evals_per_agent",
        n_agents,
        np.int64,
    )
    physical_evals = _agent_vector(
        info,
        "event_slot_physical_evals_per_agent",
        n_agents,
        np.int64,
    )
    center_shift = _matrix(
        info,
        "event_slot_center_shift_norms",
        (slot_count, n_agents),
        np.float64,
    )
    proposal_improve = _agent_vector(
        info, "proposal_local_improve", n_agents, np.float64
    )
    committed_improve = _agent_vector(
        info, "committed_local_improve", n_agents, np.float64
    )
    consensus_effect = _agent_vector(
        info, "consensus_local_effect", n_agents, np.float64
    )
    reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if reward_values.size == 1 and n_agents > 1:
        reward_values = np.repeat(reward_values, n_agents)
    if reward_values.shape != (n_agents,) or not np.all(
        np.isfinite(reward_values)
    ):
        raise ValueError(
            "Event-slot diagnostic rewards do not match the agent dimension."
        )

    raw_audit = info.get("event_slot_commit_audit")
    raw_sigma = info.get("event_slot_sigma_diagnostics")
    if not isinstance(raw_sigma, dict):
        raise ValueError("Event-slot diagnostics are missing sigma diagnostics.")
    sigma_fields = ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "sigma_preview", "sigma_used", "evolved_sigma", "inherit_applied", "reset_reason", "signature_match", "stale_gate_shift", "stale_gate_scale")
    if set(raw_sigma) != set(sigma_fields):
        raise ValueError("Event-slot sigma diagnostic schema does not match.")
    for field in ("sigma_preview", "sigma_used", "evolved_sigma", "stale_gate_shift", "stale_gate_scale"):
        raw_sigma[field] = _agent_vector(raw_sigma, field, n_agents, np.float64)
    for field in ("inherit_applied", "signature_match"):
        raw_sigma[field] = _agent_vector(raw_sigma, field, n_agents, np.int8)
    for field in ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "reset_reason"):
        values = np.asarray(raw_sigma[field], dtype=object).reshape(-1)
        if values.shape != (n_agents,):
            raise ValueError(f"Event-slot sigma field {field} has wrong shape.")
        raw_sigma[field] = values
    if np.any(~np.isin(raw_sigma["inherit_applied"], (0, 1))):
        raise ValueError("Event-slot sigma inherit_applied must contain only 0 or 1.")
    if np.any(~np.isin(raw_sigma["signature_match"], (0, 1))):
        raise ValueError("Event-slot sigma signature_match must contain only 0 or 1.")
    for agent_id in range(n_agents):
        applied = bool(raw_sigma["inherit_applied"][agent_id])
        resolved = str(raw_sigma["resolved_profile"][agent_id]).lower()
        used = float(raw_sigma["sigma_used"][agent_id])
        if applied and (resolved != "numeric_inherit" or not np.isfinite(used) or used <= 0):
            raise ValueError("Applied sigma inheritance has inconsistent diagnostics.")
    if not isinstance(raw_audit, dict):
        raise ValueError("Event-slot diagnostics are missing the commit audit.")
    if set(raw_audit) != set(EVENT_SLOT_COMMIT_AUDIT_FIELDS):
        raise ValueError("Event-slot commit-audit schema does not match.")
    audit = {
        field: _matrix(
            raw_audit,
            field,
            (slot_count, n_agents),
            np.float64,
        )
        for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS
    }

    optimizer_names = [str(value).lower() for value in optimizer_candidates]
    resources = [float(value) for value in resource_factors]
    rows: List[dict] = []
    for slot_id in range(slot_count):
        for agent_id in range(n_agents):
            optimizer_action = int(action_matrix[agent_id, 0])
            resource_action = int(action_matrix[agent_id, resource_col])
            if not 0 <= optimizer_action < len(optimizer_names):
                raise ValueError(
                    f"Invalid optimizer action {optimizer_action} for agent {agent_id}."
                )
            if not 0 <= resource_action < len(resources):
                raise ValueError(
                    f"Invalid resource action {resource_action} for agent {agent_id}."
                )
            row = {
                "seed": int(seed),
                "function_id": int(function_id),
                "strategy": str(strategy),
                "rollout_step": int(rollout_step),
                "environment_step": int(info.get("step", rollout_step)),
                "pre_sum_fes": int(pre_sum_fes),
                "post_sum_fes": int(post_sum_fes),
                "phase": str(phase),
                "slot_id": int(slot_id),
                "slot_count": int(slot_count),
                "agent_id": int(agent_id),
                "optimizer_action": optimizer_action,
                "optimizer_name": optimizer_names[optimizer_action],
                "resource_action": resource_action,
                "resource_factor": resources[resource_action],
                "communication_action": (
                    int(action_matrix[agent_id, comm_col])
                    if comm_action_enable
                    else -1
                ),
                "collaboration_action": (
                    int(action_matrix[agent_id, collab_col])
                    if collab_action_enable
                    else -1
                ),
                "guide_scale_action": (
                    int(action_matrix[agent_id, guide_scale_col])
                    if guide_scale_action_enable
                    else -1
                ),
                "packet_population_units": int(
                    packet_units[slot_id, agent_id]
                ),
                "packet_local_improvement": float(
                    packet_improvements[slot_id, agent_id]
                ),
                "event_proposal_local_improvement": float(
                    proposal_improve[agent_id]
                ),
                "event_committed_local_improvement": float(
                    committed_improve[agent_id]
                ),
                "event_consensus_local_effect": float(
                    consensus_effect[agent_id]
                ),
                "event_reward": float(reward_values[agent_id]),
                "requested_profile": str(raw_sigma["requested_profile"][agent_id]),
                "resolved_profile": str(raw_sigma["resolved_profile"][agent_id]),
                "previous_optimizer": str(raw_sigma["previous_optimizer"][agent_id]),
                "current_optimizer": str(raw_sigma["current_optimizer"][agent_id]),
                "sigma_preview": float(raw_sigma["sigma_preview"][agent_id]),
                "sigma_used": float(raw_sigma["sigma_used"][agent_id]),
                "evolved_sigma": float(raw_sigma["evolved_sigma"][agent_id]),
                "inherit_applied": int(raw_sigma["inherit_applied"][agent_id]),
                "reset_reason": str(raw_sigma["reset_reason"][agent_id]),
                "signature_match": int(raw_sigma["signature_match"][agent_id]),
                "stale_gate_shift": float(raw_sigma["stale_gate_shift"][agent_id]),
                "stale_gate_scale": float(raw_sigma["stale_gate_scale"][agent_id]),
                "event_distribution_updates": int(
                    distribution_updates[agent_id]
                ),
                "event_native_evals": int(native_evals[agent_id]),
                "event_reported_evals": int(reported_evals[agent_id]),
                "event_physical_evals": int(physical_evals[agent_id]),
                "center_shift_norm": float(
                    center_shift[slot_id, agent_id]
                ),
            }
            for cfg_id in range(cfg_param_num):
                row[f"cfg_action_{cfg_id}"] = int(
                    action_matrix[agent_id, 1 + cfg_id]
                )
            for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS:
                row[field] = float(audit[field][slot_id, agent_id])
            row.update(_vkd_row(info, slot_id, agent_id, (slot_count, n_agents)))
            row.update(_mmes_row(info, slot_id, agent_id, (slot_count, n_agents)))
            row.update(_native_state_row(info, slot_id, agent_id, (slot_count, n_agents)))
            rows.append(row)
    return rows


def find_event_slot_diagnostic_failure(
    *, info: Dict, actions, rewards, cfg_param_num: int,
    comm_action_enable: bool, collab_action_enable: bool,
    guide_scale_action_enable: bool, optimizer_count: int,
    resource_count: int,
) -> Dict:
    """Return the first strict-builder failure without constructing strict rows."""
    try:
        if not bool(info.get("event_slot_interleaving_enable", False)):
            raise ValueError("Event-slot diagnostics require event-slot interleaving to be enabled.")
        slot_count = int(info.get("event_slot_count", 0))
        if slot_count <= 0:
            raise ValueError("Event-slot diagnostics received an empty event.")
        action_matrix = np.asarray(actions, dtype=np.int64)
        if action_matrix.ndim != 2:
            raise ValueError("Event-slot diagnostic actions must have shape [agent, action_column].")
        n_agents = int(action_matrix.shape[0])
        resource_col = 1 + int(cfg_param_num)
        comm_col = resource_col + 1
        collab_col = comm_col + int(bool(comm_action_enable))
        guide_scale_col = collab_col + int(bool(collab_action_enable))
        required_cols = guide_scale_col + int(bool(guide_scale_action_enable))
        if action_matrix.shape[1] < required_cols:
            raise ValueError("Event-slot diagnostic action matrix is missing configured action columns.")
        checks = (
            ("event_slot_packet_units", (slot_count, n_agents), np.int64),
            ("event_slot_packet_improvements", (slot_count, n_agents), np.float64),
            ("event_slot_distribution_updates", (n_agents,), np.int64),
            ("event_slot_native_evals_per_agent", (n_agents,), np.int64),
            ("event_slot_reported_evals_per_agent", (n_agents,), np.int64),
            ("event_slot_physical_evals_per_agent", (n_agents,), np.int64),
            ("event_slot_center_shift_norms", (slot_count, n_agents), np.float64),
            ("proposal_local_improve", (n_agents,), np.float64),
            ("committed_local_improve", (n_agents,), np.float64),
            ("consensus_local_effect", (n_agents,), np.float64),
        )
        for key, shape, dtype in checks:
            _matrix(info, key, shape, dtype)
        reward_values = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if reward_values.size == 1 and n_agents > 1:
            reward_values = np.repeat(reward_values, n_agents)
        if reward_values.shape != (n_agents,) or not np.all(np.isfinite(reward_values)):
            raise ValueError("Event-slot diagnostic rewards do not match the agent dimension.")
        raw_sigma = info.get("event_slot_sigma_diagnostics")
        if not isinstance(raw_sigma, dict):
            raise ValueError("Event-slot diagnostics are missing sigma diagnostics.")
        sigma_fields = ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "sigma_preview", "sigma_used", "evolved_sigma", "inherit_applied", "reset_reason", "signature_match", "stale_gate_shift", "stale_gate_scale")
        if set(raw_sigma) != set(sigma_fields):
            raise ValueError("Event-slot sigma diagnostic schema does not match.")
        for field in ("sigma_preview", "sigma_used", "evolved_sigma", "stale_gate_shift", "stale_gate_scale"):
            _agent_vector(raw_sigma, field, n_agents, np.float64)
        inherit_applied = _agent_vector(
            raw_sigma, "inherit_applied", n_agents, np.int8
        )
        signature_match = _agent_vector(
            raw_sigma, "signature_match", n_agents, np.int8
        )
        string_values = {}
        for field in ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "reset_reason"):
            values = np.asarray(raw_sigma[field], dtype=object).reshape(-1)
            if values.shape != (n_agents,):
                raise ValueError(f"Event-slot sigma field {field} has wrong shape.")
            string_values[field] = values
        if np.any(~np.isin(inherit_applied, (0, 1))):
            raise ValueError("Event-slot sigma inherit_applied must contain only 0 or 1.")
        if np.any(~np.isin(signature_match, (0, 1))):
            raise ValueError("Event-slot sigma signature_match must contain only 0 or 1.")
        sigma_used = np.asarray(raw_sigma["sigma_used"], dtype=np.float64)
        for agent_id in range(n_agents):
            if bool(inherit_applied[agent_id]) and (
                str(string_values["resolved_profile"][agent_id]).lower()
                != "numeric_inherit"
                or not np.isfinite(sigma_used[agent_id])
                or sigma_used[agent_id] <= 0
            ):
                raise ValueError("Applied sigma inheritance has inconsistent diagnostics.")
        raw_audit = info.get("event_slot_commit_audit")
        if not isinstance(raw_audit, dict):
            raise ValueError("Event-slot diagnostics are missing the commit audit.")
        if set(raw_audit) != set(EVENT_SLOT_COMMIT_AUDIT_FIELDS):
            raise ValueError("Event-slot commit-audit schema does not match.")
        for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS:
            _matrix(raw_audit, field, (slot_count, n_agents), np.float64)
        for agent_id in range(n_agents):
            optimizer_action = int(action_matrix[agent_id, 0])
            resource_action = int(action_matrix[agent_id, resource_col])
            if not 0 <= optimizer_action < int(optimizer_count):
                raise ValueError(
                    f"Invalid optimizer action {optimizer_action} for agent {agent_id}."
                )
            if not 0 <= resource_action < int(resource_count):
                raise ValueError(
                    f"Invalid resource action {resource_action} for agent {agent_id}."
                )
    except Exception as exc:
        message = str(exc)
        field = message.split("field ", 1)[1].split(" ", 1)[0] if "field " in message else ""
        return {"exception_type": type(exc).__name__, "exception": message, "field": field}
    return {}


def build_forensic_trace_records(*, info: Dict, **kwargs) -> List[dict]:
    """Build lenient slot-agent rows independently of the strict builder."""
    try:
        actions = np.asarray(kwargs.get("actions"), dtype=np.int64)
    except Exception:
        return []
    if actions.ndim != 2:
        return []
    slot_count = int(info.get("event_slot_count", 0))
    if slot_count <= 0:
        return []
    n_agents = int(actions.shape[0])
    optimizer_candidates = [str(value).lower() for value in kwargs.get("optimizer_candidates", ())]
    resource_factors = [float(value) for value in kwargs.get("resource_factors", ())]
    cfg_param_num = int(kwargs.get("cfg_param_num", 0))
    resource_col = 1 + cfg_param_num
    comm_col = resource_col + 1
    collab_col = comm_col + int(bool(kwargs.get("comm_action_enable")))
    guide_scale_col = collab_col + int(bool(kwargs.get("collab_action_enable")))
    raw_sigma = info.get("event_slot_sigma_diagnostics")
    if not isinstance(raw_sigma, dict):
        raw_sigma = {}
    packet_units = _optional_matrix(info, "event_slot_packet_units", (slot_count, n_agents))
    packet_improvements = _optional_matrix(info, "event_slot_packet_improvements", (slot_count, n_agents))
    center_shift = _optional_matrix(info, "event_slot_center_shift_norms", (slot_count, n_agents))
    forensic = {
        field: _optional_matrix(info, FORENSIC_INFO_PREFIX + field, (slot_count, n_agents))
        for field in EVENT_SLOT_FORENSIC_FIELDS
    }
    details = info.get("event_slot_forensic_vector_details")
    if not isinstance(details, list):
        details = []
    raw_audit = info.get("event_slot_commit_audit")
    if not isinstance(raw_audit, dict):
        raw_audit = {}
    audit = {
        field: _optional_matrix(raw_audit, field, (slot_count, n_agents))
        for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS
    }

    def agent_value(mapping, field, agent_id, default):
        try:
            values = np.asarray(mapping.get(field, ()), dtype=object).reshape(-1)
            return values[agent_id] if values.size == n_agents else default
        except Exception:
            return default

    rows: List[dict] = []
    for slot_id in range(slot_count):
        for agent_id in range(n_agents):
            optimizer_action = int(actions[agent_id, 0])
            resource_action = int(actions[agent_id, resource_col])
            row = {
                "seed": int(kwargs.get("seed", -1)),
                "function_id": int(kwargs.get("function_id", -1)),
                "strategy": str(kwargs.get("strategy", "")),
                "rollout_step": int(kwargs.get("rollout_step", -1)),
                "environment_step": int(info.get("step", kwargs.get("rollout_step", -1))),
                "pre_sum_fes": int(kwargs.get("pre_sum_fes", 0)),
                "post_sum_fes": int(kwargs.get("post_sum_fes", 0)),
                "phase": str(kwargs.get("phase", "")),
                "slot_id": slot_id,
                "slot_count": slot_count,
                "agent_id": agent_id,
                "optimizer_action": optimizer_action,
                "optimizer_name": optimizer_candidates[optimizer_action] if 0 <= optimizer_action < len(optimizer_candidates) else "",
                "resource_action": resource_action,
                "resource_factor": resource_factors[resource_action] if 0 <= resource_action < len(resource_factors) else float("nan"),
                "communication_action": int(actions[agent_id, comm_col]) if kwargs.get("comm_action_enable") else -1,
                "collaboration_action": int(actions[agent_id, collab_col]) if kwargs.get("collab_action_enable") else -1,
                "guide_scale_action": int(actions[agent_id, guide_scale_col]) if kwargs.get("guide_scale_action_enable") else -1,
                "packet_population_units": float(packet_units[slot_id, agent_id]),
                "packet_local_improvement": float(packet_improvements[slot_id, agent_id]),
                "center_shift_norm": float(center_shift[slot_id, agent_id]),
            }
            for cfg_id in range(cfg_param_num):
                row[f"cfg_action_{cfg_id}"] = int(actions[agent_id, 1 + cfg_id])
            for field in ("requested_profile", "resolved_profile", "previous_optimizer", "current_optimizer", "reset_reason"):
                row[field] = str(agent_value(raw_sigma, field, agent_id, ""))
            for field in ("sigma_preview", "sigma_used", "evolved_sigma", "stale_gate_shift", "stale_gate_scale", "inherit_applied", "signature_match"):
                value = agent_value(raw_sigma, field, agent_id, float("nan"))
                try:
                    row[field] = float(value)
                except (TypeError, ValueError, OverflowError):
                    row[field] = float("nan")
            for field in EVENT_SLOT_COMMIT_AUDIT_FIELDS:
                row[field] = float(audit[field][slot_id, agent_id])
            for field in EVENT_SLOT_FORENSIC_FIELDS:
                row[field] = float(forensic[field][slot_id, agent_id])
            if slot_id < len(details) and isinstance(details[slot_id], list) and agent_id < len(details[slot_id]) and isinstance(details[slot_id][agent_id], dict):
                row.update(details[slot_id][agent_id])
            for field in ("sigma_preview", "sigma_used", "evolved_sigma"):
                value = float(row.get(field, float("nan")))
                finite = bool(np.isfinite(value))
                row[f"{field}_is_finite"] = int(finite)
                row[f"{field}_is_nan"] = int(np.isnan(value))
                row[f"{field}_is_inf"] = int(np.isinf(value))
                row[f"{field}_max_abs"] = abs(value) if finite else float("nan")
                row[f"{field}_stable_norm"] = abs(value) if finite else float("nan")
                row[f"{field}_norm_overflow"] = 0
            row.update(_vkd_row(info, slot_id, agent_id, (slot_count, n_agents)))
            row.update(_mmes_row(info, slot_id, agent_id, (slot_count, n_agents)))
            rows.append(row)
    return rows

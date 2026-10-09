import csv
import json
import math
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from env.agent.utils.logger import TrainingLogger
from env.parallel.venvs import SubprocVectorEnv
from env.optimizer.env_factory import make_opt_env
from .buffer import MAPPOBuffer
from .checkpoint import MAPPO_RNG_STATE_VERSION, capture_rng_state


class MAPPORunner:
    def __init__(self, opts, policy, trainer, tb_logger=None):
        self.opts = opts
        self.policy = policy
        self.trainer = trainer
        self.device = torch.device("cuda" if opts.use_cuda else "cpu")
        self.tb_logger = tb_logger

        self.n_env = int(opts.each_question_batch_num)
        self.fun_ids: List[int] = list(opts.train_function_ids)
        self.start_epoch = int(opts.epoch_start)
        self.end_epoch = int(opts.epoch_end)
        self.train_epochs = int(max(0, self.end_epoch - self.start_epoch))
        self.schedule_start_epoch = int(
            getattr(opts, "resume_schedule_start_epoch", self.start_epoch)
        )
        self.schedule_end_epoch = int(
            getattr(opts, "resume_schedule_end_epoch", self.end_epoch)
        )
        self.schedule_train_epochs = int(
            max(0, self.schedule_end_epoch - self.schedule_start_epoch)
        )
        self.horizon = int(opts.episode_steps)
        self.action_dim = int(len(getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])))  # profile head bins
        self.mappo_action_arch = str(getattr(opts, "mappo_action_arch", "current")).lower()
        self.cfg_param_num = int(getattr(opts, "mappo_cfg_param_num", 4))
        if self.mappo_action_arch == "pre_caf4a62":
            self.cfg_param_num = 1
        self.comm_action_enable = bool(
            int(getattr(opts, "objective_split_comm_action_enable", 0))
        )
        self.legacy_comm_old_logp_zero = bool(
            int(getattr(opts, "mappo_legacy_comm_old_logp_zero", 0))
        )
        if self.legacy_comm_old_logp_zero and not self.comm_action_enable:
            raise ValueError(
                "mappo_legacy_comm_old_logp_zero=1 requires "
                "objective_split_comm_action_enable=1."
            )
        self.collab_action_enable = bool(
            int(getattr(opts, "objective_split_collab_action_enable", 0))
        )
        self.guide_scale_action_enable = bool(
            int(getattr(opts, "objective_split_guide_scale_action_enable", 0))
        )
        self.actuator_action_enable = bool(
            int(
                getattr(
                    opts,
                    "objective_split_candidate_actuator_action_enable",
                    0,
                )
            )
        )
        raw_comm_candidates = getattr(opts, "objective_split_comm_round_candidates", [1, 2, 4, 8])
        if isinstance(raw_comm_candidates, str):
            raw_comm_candidates = [x.strip() for x in raw_comm_candidates.split(",") if x.strip()]
        self.comm_action_dim = (
            int(max(1, len(list(raw_comm_candidates))))
            if self.comm_action_enable
            else 1
        )
        raw_collab_modes = getattr(opts, "objective_split_collab_modes", ["consensus", "self", "leader", "soft_diversify"])
        if isinstance(raw_collab_modes, str):
            raw_collab_modes = [x.strip() for x in raw_collab_modes.split(",") if x.strip()]
        raw_guide_scale_candidates = getattr(opts, "objective_split_guide_scale_candidates", [1.0, 0.5, 0.75, 1.25])
        if isinstance(raw_guide_scale_candidates, str):
            raw_guide_scale_candidates = [x.strip() for x in raw_guide_scale_candidates.split(",") if x.strip()]
        self.collab_action_dim = (
            int(max(1, len(list(raw_collab_modes))))
            if self.collab_action_enable
            else 1
        )
        self.guide_scale_action_dim = (
            int(max(1, len(list(raw_guide_scale_candidates))))
            if self.guide_scale_action_enable
            else 1
        )
        raw_actuator_candidates = getattr(
            opts,
            "objective_split_candidate_actuator_candidates",
            [0.0, 0.25, 0.5],
        )
        if isinstance(raw_actuator_candidates, str):
            raw_actuator_candidates = [
                x.strip() for x in raw_actuator_candidates.split(",") if x.strip()
            ]
        self.actuator_action_dim = (
            int(max(1, len(list(raw_actuator_candidates))))
            if self.actuator_action_enable
            else 1
        )
        self.forced_actuator_idx = int(
            getattr(
                opts,
                "objective_split_candidate_actuator_forced_action",
                -1,
            )
        )
        if not self.actuator_action_enable:
            self.forced_actuator_idx = -1
        self.action_cols = int(
            2
            + self.cfg_param_num
            + (1 if self.comm_action_enable else 0)
            + (1 if self.collab_action_enable else 0)
            + (1 if self.guide_scale_action_enable else 0)
            + (1 if self.actuator_action_enable else 0)
        )
        self.opt_action_dim = int(len(getattr(opts, "optimizer_candidates", ["mmes", "vkd"])))
        self.res_action_dim = int(len(getattr(opts, "resource_factors", [0.5, 1.0, 2.0])))
        self.optimizer_candidates = [str(x).lower() for x in getattr(opts, "optimizer_candidates", ["mmes", "vkd"])]
        self.forced_optimizer_enable = bool(int(getattr(opts, "forced_optimizer_enable", 0)))
        self.forced_optimizer_prob = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_prob", 0.0))))
        self.forced_optimizer_warmup_ratio = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_warmup_ratio", 0.0))))
        self.forced_optimizer_mode = str(getattr(opts, "forced_optimizer_mode", "cycle")).lower()
        if self.mappo_action_arch == "pre_caf4a62":
            self.cfg_ratio_keys = [f"cfg_ratio_{i}" for i in range(self.action_dim)]
        else:
            self.cfg_ratio_keys = [f"cfg_p{p}_ratio_{i}" for p in range(self.cfg_param_num) for i in range(self.action_dim)]
        self.opt_ratio_keys = [f"opt_ratio_{i}" for i in range(self.opt_action_dim)]
        self.res_ratio_keys = [f"res_ratio_{i}" for i in range(self.res_action_dim)]
        self.comm_ratio_keys = (
            [f"comm_ratio_{i}" for i in range(self.comm_action_dim)]
            if self.comm_action_enable
            else []
        )
        self.collab_ratio_keys = (
            [f"collab_ratio_{i}" for i in range(self.collab_action_dim)]
            if self.collab_action_enable
            else []
        )
        self.guide_scale_ratio_keys = (
            [f"guide_scale_ratio_{i}" for i in range(self.guide_scale_action_dim)]
            if self.guide_scale_action_enable
            else []
        )
        self.actuator_ratio_keys = (
            [f"actuator_ratio_{i}" for i in range(self.actuator_action_dim)]
            if self.actuator_action_enable
            else []
        )
        self.stats_header = self._build_stats_header()

        self.mappo_log_enable = bool(int(getattr(opts, "mappo_log_enable", 1)))
        self.mappo_log_interval = int(max(1, getattr(opts, "mappo_log_interval", 1)))
        self.mappo_log_action_hist = bool(int(getattr(opts, "mappo_log_action_hist", 1)))
        self.mappo_log_adv_stats = bool(int(getattr(opts, "mappo_log_adv_stats", 1)))
        self.training_logger = TrainingLogger(tb_logger) if self.mappo_log_enable else None

        self.saving_enabled = not bool(getattr(opts, "no_saving", False))
        if self.saving_enabled:
            os.makedirs(opts.modal_save_dir, exist_ok=True)
            os.makedirs(opts.data_save_dir, exist_ok=True)
            self.stats_path = os.path.join(opts.data_save_dir, "mappo_training_stats.csv")
            self.best_ckpt_path = os.path.join(opts.modal_save_dir, "mappo-best.pt")
            self._ensure_stats_header()
        else:
            self.stats_path = None
            self.best_ckpt_path = None

    def _build_stats_header(self):
        base = [
            "epoch",
            "reward_mean",
            "team_reward_mean",
            "mixed_reward_mean",
            "local_centered_mean",
            "local_centered_std",
            "policy_loss",
            "policy_loss_opt",
            "policy_loss_cfg",
            "policy_loss_res",
            "policy_loss_comm",
            "policy_loss_collab",
            "policy_loss_guide_scale",
            "policy_loss_actuator",
            "value_loss",
            "value_loss_ref",
            "value_loss_opt",
            "value_loss_cfg",
            "value_loss_res",
            "value_loss_comm",
            "value_loss_collab",
            "value_loss_guide_scale",
            "value_loss_actuator",
            "entropy",
            "entropy_opt",
            "entropy_cfg",
            "entropy_res",
            "entropy_comm",
            "entropy_collab",
            "entropy_guide_scale",
            "entropy_actuator",
            "approx_kl",
            "approx_kl_opt",
            "approx_kl_cfg",
            "approx_kl_res",
            "approx_kl_comm",
            "approx_kl_collab",
            "approx_kl_guide_scale",
            "approx_kl_actuator",
            "clip_frac",
            "clip_frac_opt",
            "clip_frac_cfg",
            "clip_frac_res",
            "clip_frac_comm",
            "clip_frac_collab",
            "clip_frac_guide_scale",
            "clip_frac_actuator",
            "ratio_mean",
            "ratio_std",
            "ratio_mean_opt",
            "ratio_mean_cfg",
            "ratio_mean_res",
            "ratio_mean_comm",
            "ratio_mean_collab",
            "ratio_mean_guide_scale",
            "ratio_mean_actuator",
            "ratio_std_opt",
            "ratio_std_cfg",
            "ratio_std_res",
            "ratio_std_comm",
            "ratio_std_collab",
            "ratio_std_guide_scale",
            "ratio_std_actuator",
            "delta_logp_opt_mean",
            "delta_logp_cfg_mean",
            "delta_logp_res_mean",
            "delta_logp_comm_mean",
            "delta_logp_collab_mean",
            "delta_logp_guide_scale_mean",
            "delta_logp_actuator_mean",
            "delta_logp_opt_std",
            "delta_logp_cfg_std",
            "delta_logp_res_std",
            "delta_logp_comm_std",
            "delta_logp_collab_std",
            "delta_logp_guide_scale_std",
            "delta_logp_actuator_std",
            "adv_opt_mean",
            "adv_opt_std",
            "adv_cfg_mean",
            "adv_cfg_std",
            "adv_res_mean",
            "adv_res_std",
            "adv_comm_mean",
            "adv_comm_std",
            "adv_collab_mean",
            "adv_collab_std",
            "adv_guide_scale_mean",
            "adv_guide_scale_std",
            "adv_actuator_mean",
            "adv_actuator_std",
            "adv_mean",
            "adv_std",
            "ret_mean",
            "ret_std",
            "value_mean",
            "value_std",
            "value_ref_mean",
            "value_ref_std",
            "adv_agent_mean_std",
            "adv_agent_std_mean",
            "adv_agent_std_max",
            "adv_agent_abs_mean_min",
            "ret_agent_mean_std",
            "value_agent_mean_std",
            "forced_optimizer_active",
            "forced_optimizer_idx",
            "forced_optimizer_warmup",
            "proposal_mean_disagreement",
            "proposal_max_edge_disagreement",
            "pre_state_mean_disagreement",
            "pre_state_max_edge_disagreement",
            "post_mean_disagreement",
            "post_max_edge_disagreement",
            "consensus_improvement",
            "consensus_operator_improvement",
            "direction_agreement_mean",
            "neighbor_avg_improve_mean",
            "ccsa_momentum_norm_mean",
            "ccsa_scale_mean",
            "ccsa_scale_std",
            "masoie_velocity_norm_mean",
            "masoie_neighbor_pull_norm_mean",
            "optimizer_guide_norm_mean",
            "optimizer_guide_norm_max",
            "optimizer_guide_applied_ratio",
            "optimizer_guide_alignment_mean",
            "optimizer_guide_source_active_ratio",
            "optimizer_guide_internal_active_ratio",
            "optimizer_guide_internal_mean_step_norm_mean",
            "optimizer_guide_internal_mean_step_norm_max",
            "optimizer_guide_internal_alignment_mean",
            "anchor_enabled",
            "anchor_applied_ratio",
            "anchor_dist_mean",
            "anchor_dist_max",
            "anchor_direction_norm_mean",
            "anchor_direction_norm_max",
            "optimizer_anchor_applied_ratio",
            "optimizer_anchor_mean_step_norm_mean",
            "optimizer_anchor_mean_step_norm_max",
            "optimizer_anchor_sample_applied_ratio",
            "committee_supported",
            "committee_active",
            "committee_selection_target_block",
            "committee_best_score_mean",
            "committee_best_score_max",
            "committee_confidence_mean",
            "committee_confidence_max",
            "committee_guide_norm_mean",
            "committee_guide_norm_max",
            "committee_self_source_ratio",
            "committee_source_diversity_mean",
            "committee_source_diversity_max",
            "committee_shadow_whole_report_f",
            "committee_shadow_target_block_report_f",
            "committee_shadow_whole_report_win",
            "committee_shadow_target_block_report_win",
            "committee_shadow_whole_report_log_improve",
            "committee_shadow_target_block_report_log_improve",
            "committee_shadow_whole_agent_f_mean",
            "committee_shadow_whole_agent_f_min",
            "committee_shadow_target_block_agent_f_mean",
            "committee_shadow_target_block_agent_f_min",
            "committee_shadow_whole_agent_win_ratio",
            "committee_shadow_target_block_agent_win_ratio",
            "committee_shadow_target_block_win_vs_whole",
            "committee_acceptance_supported",
            "committee_acceptance_ratio",
            "committee_rejected_ratio",
            "committee_candidate_report_f",
            "committee_candidate_global_f_mean",
            "committee_candidate_global_f_min",
            "committee_candidate_vs_report_log_improve",
            "committee_effective_beta_mean",
            "committee_effective_beta_max",
            "committee_base_alignment_mean",
            "committee_noop_ratio",
            "candidate_response_supported",
            "candidate_response_active",
            "candidate_response_actuated",
            "candidate_response_probe_active_ratio",
            "candidate_response_generator_active_ratio",
            "candidate_response_accept_ratio",
            "candidate_response_selection_confidence_mean",
            "candidate_response_support_mean",
            "candidate_response_confidence_mean",
            "candidate_response_trust_radius_mean",
            "candidate_response_probe_radius_mean",
            "candidate_response_requested_shift_mean",
            "candidate_response_applied_shift_mean",
            "candidate_response_noop_ratio",
            "candidate_response_rollback_ratio",
            "candidate_response_agent_opportunity_ratio",
            "candidate_response_agent_requested_ratio",
            "candidate_response_agent_effective_ratio",
            "candidate_response_actuator_beta_mean",
            "candidate_response_actuator_beta_min",
            "candidate_response_actuator_beta_max",
            "candidate_response_shadow_base_report_f",
            "candidate_response_shadow_candidate_report_f",
            "candidate_response_shadow_report_win",
            "candidate_response_shadow_report_log_improve",
            "committee_acceptance_ratio_early",
            "committee_acceptance_ratio_mid",
            "committee_acceptance_ratio_late",
            "collab_mode_idx_mean",
            "guide_scale_idx_mean",
            "guide_scale_value_mean",
            "collab_leader_active_ratio",
            "collab_strength_multiplier_mean",
            "graph_comm_rounds",
            "comm_rounds_per_event",
            "comm_force_rounds",
            "last_comm_rounds_applied",
            "last_comm_rounds_requested",
            "comm_round_idx_mean",
            "comm_round_idx_max",
            "state_comm_enabled",
            "state_comm_msg_dim",
            "last_actual_fes_mean",
            "last_actual_fes_max",
            "cumulative_actual_fes_mean",
            "cmaes_numeric_fail_soft_step_count",
            "cmaes_numeric_fail_soft_events",
            "last_consensus_shift_norm_mean",
            "last_consensus_shift_norm_max",
            "ccsa_lite_rounds",
            "masoie_lite_rounds",
            "centralized_full_mean_rounds",
            "total_comm_rounds_applied",
            "total_comm_events",
            "step_comm_rounds_applied",
            "step_comm_events",
            "graph_messages",
            "graph_transmitted_floats",
            "graph_transmitted_bytes",
            "candidate_response_events",
            "candidate_response_actuated_events",
            "candidate_response_probe_local_evals",
            "candidate_response_verification_local_evals",
            "candidate_response_shadow_global_local_evals",
            "candidate_response_rounds",
            "candidate_response_messages",
            "candidate_response_transmitted_floats",
            "candidate_response_transmitted_bytes",
            "local_search_fes",
            "candidate_validation_local_evals",
            "agent_state_local_evals",
            "global_monitor_local_evals",
            "global_monitor_rounds",
            "verification_local_evals",
            "total_physical_local_calls",
            "reported_sum_fes",
            "state_comm_rounds",
            "state_comm_messages",
            "state_comm_transmitted_floats",
            "state_comm_transmitted_bytes",
            "rollout_actual_local_search_fes",
            "rollout_reported_fes",
            "rollout_total_physical_local_calls",
            "completed_episodes_per_rollout",
            "env_resets_per_rollout",
            "truncated_env_count",
            "meta_steps_per_episode",
        ]
        tail = [
            "opt_max_ratio",
            "cfg_max_ratio",
            "res_max_ratio",
            "comm_max_ratio",
            "collab_max_ratio",
            "guide_scale_max_ratio",
            "actuator_max_ratio",
            "actor_grad_norm",
            "critic_grad_norm",
        ]
        return (
            base
            + self.opt_ratio_keys
            + self.cfg_ratio_keys
            + self.res_ratio_keys
            + self.comm_ratio_keys
            + self.collab_ratio_keys
            + self.guide_scale_ratio_keys
            + self.actuator_ratio_keys
            + tail
        )

    def _ensure_stats_header(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), [])
            if existing_header == self.stats_header:
                return
            self.stats_path = os.path.join(
                self.opts.data_save_dir,
                "mappo_training_stats_graph.csv",
            )
            if os.path.exists(self.stats_path):
                with open(
                    self.stats_path,
                    "r",
                    newline="",
                    encoding="utf-8",
                ) as f:
                    graph_header = next(csv.reader(f), [])
                if graph_header == self.stats_header:
                    return
        with open(self.stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.stats_header)

    def _append_stats(self, epoch: int, stats: dict):
        if not self.saving_enabled or self.stats_path is None:
            return
        with open(self.stats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [int(epoch)]
            for k in self.stats_header[1:]:
                row.append(float(stats.get(k, 0.0)))
            writer.writerow(row)

    def _make_envs(self, batch_fun_ids: List[int]):
        env_fns = []
        for fid in batch_fun_ids:
            env_fns.append(lambda f=fid, o=self.opts: make_opt_env(f, opts_in=o))
        return SubprocVectorEnv(env_fns)

    def _forced_optimizer_for_epoch(self, epoch_i: int):
        if (not self.forced_optimizer_enable) or self.opt_action_dim <= 0:
            return False, -1, False

        local_epoch = int(max(0, epoch_i - self.schedule_start_epoch))
        warmup_epochs = int(
            math.floor(
                float(self.schedule_train_epochs)
                * self.forced_optimizer_warmup_ratio
                + 1e-9
            )
        )
        warmup_epochs = int(
            max(0, min(self.schedule_train_epochs, warmup_epochs))
        )
        in_warmup = local_epoch < warmup_epochs

        rng = np.random.default_rng(int(self.opts.seed + 1000003 + epoch_i))
        active = bool(in_warmup)
        if not active:
            active = bool(rng.random() < self.forced_optimizer_prob)
        if not active:
            return False, -1, False

        if in_warmup and self.forced_optimizer_mode == "cycle":
            opt_idx = int(local_epoch % self.opt_action_dim)
        else:
            opt_idx = int(rng.integers(0, self.opt_action_dim))
        return True, opt_idx, bool(in_warmup)

    @staticmethod
    def _window_mean(vals, frac=0.2, tail=False):
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return 0.0
        w = max(1, int(np.ceil(arr.size * float(frac))))
        if tail:
            arr = arr[-w:]
        else:
            arr = arr[:w]
        return float(np.mean(arr))

    def _analyze_training(self, history_stats: List[dict]) -> dict:
        frac = float(getattr(self.opts, "mappo_diag_reward_frac", 0.25))
        reward_tol_ratio = float(getattr(self.opts, "mappo_diag_reward_tol_ratio", 0.02))
        kl_low = float(getattr(self.opts, "mappo_diag_kl_low", 1e-4))
        kl_high = float(getattr(self.opts, "mappo_diag_kl_high", 5e-2))
        clip_low = float(getattr(self.opts, "mappo_diag_clip_low", 1e-3))
        clip_high = float(getattr(self.opts, "mappo_diag_clip_high", 5e-1))
        entropy_low = float(getattr(self.opts, "mappo_diag_entropy_low", 0.02))
        collapse_ratio = float(getattr(self.opts, "mappo_diag_collapse_ratio", 0.95))
        adv_imbalance_factor = float(getattr(self.opts, "mappo_diag_adv_imbalance_factor", 3.0))
        adv_abs_mean_min_threshold = float(getattr(self.opts, "mappo_diag_adv_abs_mean_min", 0.05))

        if len(history_stats) == 0:
            return {
                "status": "unknown",
                "score": 0,
                "max_score": 5,
                "message": "No training stats were collected.",
                "signals": {},
                "warnings": ["empty_history"],
                "recommendations": ["Check whether training loop produced any epoch updates."],
            }

        rewards = [float(s.get("reward_mean", 0.0)) for s in history_stats]
        entropies = [float(s.get("entropy", 0.0)) for s in history_stats]
        kls = [float(s.get("approx_kl", 0.0)) for s in history_stats]
        clips = [float(s.get("clip_frac", 0.0)) for s in history_stats]
        kl_opt = [float(s.get("approx_kl_opt", 0.0)) for s in history_stats]
        kl_cfg = [float(s.get("approx_kl_cfg", 0.0)) for s in history_stats]
        kl_res = [float(s.get("approx_kl_res", 0.0)) for s in history_stats]
        clip_opt = [float(s.get("clip_frac_opt", 0.0)) for s in history_stats]
        clip_cfg = [float(s.get("clip_frac_cfg", 0.0)) for s in history_stats]
        clip_res = [float(s.get("clip_frac_res", 0.0)) for s in history_stats]
        value_losses = [float(s.get("value_loss", 0.0)) for s in history_stats]
        opt_maxes = [float(s.get("opt_max_ratio", 0.0)) for s in history_stats]
        cfg_maxes = [float(s.get("cfg_max_ratio", 0.0)) for s in history_stats]
        res_maxes = [float(s.get("res_max_ratio", 0.0)) for s in history_stats]
        adv_std_means = [float(s.get("adv_agent_std_mean", 0.0)) for s in history_stats]
        adv_std_maxes = [float(s.get("adv_agent_std_max", 0.0)) for s in history_stats]
        adv_abs_min = [float(s.get("adv_agent_abs_mean_min", 0.0)) for s in history_stats]

        reward_head = self._window_mean(rewards, frac=frac, tail=False)
        reward_tail = self._window_mean(rewards, frac=frac, tail=True)
        reward_delta = reward_tail - reward_head
        reward_tol = max(1e-3, reward_tol_ratio * max(1.0, abs(reward_head)))

        kl_tail = self._window_mean(kls, frac=frac, tail=True)
        clip_tail = self._window_mean(clips, frac=frac, tail=True)
        entropy_tail = self._window_mean(entropies, frac=frac, tail=True)
        value_head = self._window_mean(value_losses, frac=frac, tail=False)
        value_tail = self._window_mean(value_losses, frac=frac, tail=True)
        opt_max_tail = self._window_mean(opt_maxes, frac=frac, tail=True)
        cfg_max_tail = self._window_mean(cfg_maxes, frac=frac, tail=True)
        res_max_tail = self._window_mean(res_maxes, frac=frac, tail=True)
        adv_std_mean_tail = self._window_mean(adv_std_means, frac=frac, tail=True)
        adv_std_max_tail = self._window_mean(adv_std_maxes, frac=frac, tail=True)
        adv_abs_min_tail = self._window_mean(adv_abs_min, frac=frac, tail=True)
        kl_opt_tail = self._window_mean(kl_opt, frac=frac, tail=True)
        kl_cfg_tail = self._window_mean(kl_cfg, frac=frac, tail=True)
        kl_res_tail = self._window_mean(kl_res, frac=frac, tail=True)
        clip_opt_tail = self._window_mean(clip_opt, frac=frac, tail=True)
        clip_cfg_tail = self._window_mean(clip_cfg, frac=frac, tail=True)
        clip_res_tail = self._window_mean(clip_res, frac=frac, tail=True)

        score = 0
        warnings = []
        recommendations = []

        # signal 1: reward trend
        if reward_delta > reward_tol:
            score += 2
        else:
            warnings.append("weak_reward_improvement")
            recommendations.append("Increase reward signal quality (e.g., mixed team+local reward) or train longer.")

        # signal 2: update health (KL + clip fraction)
        kl_ok = (kl_low <= kl_tail <= kl_high)
        clip_ok = (clip_low <= clip_tail <= clip_high)
        if kl_ok and clip_ok:
            score += 1
        else:
            warnings.append("ppo_update_unhealthy")
            recommendations.append("Tune lr/eps_clip/K_epochs to keep KL and clip_frac in a healthy range.")

        # signal 3: entropy (exploration not fully dead)
        if entropy_tail >= entropy_low:
            score += 1
        else:
            warnings.append("low_entropy")
            recommendations.append("Increase entropy_coef or slow its decay to avoid premature policy collapse.")

        # signal 4: head-wise collapse
        if (opt_max_tail < collapse_ratio) and (cfg_max_tail < collapse_ratio) and (res_max_tail < collapse_ratio):
            score += 1
        else:
            if opt_max_tail >= collapse_ratio:
                warnings.append("optimizer_head_collapse_risk")
            if cfg_max_tail >= collapse_ratio:
                warnings.append("cfg_head_collapse_risk")
            if res_max_tail >= collapse_ratio:
                warnings.append("resource_head_collapse_risk")
            recommendations.append("Tune reward shaping / entropy_coef to reduce per-head collapse risk.")

        # value loss check: do not add score, only warning
        if not np.isfinite(value_tail) or (value_head > 0 and value_tail > 2.0 * value_head):
            warnings.append("value_learning_unstable")
            recommendations.append("Stabilize critic (lr_critic, value_coef, return scaling).")
        if (adv_std_mean_tail > 0.0 and adv_std_max_tail > adv_imbalance_factor * adv_std_mean_tail) or (
            adv_abs_min_tail < adv_abs_mean_min_threshold
        ):
            warnings.append("agent_advantage_imbalance")
            recommendations.append("Lower local reward mixing or improve optimizer-head exploration regularization.")
        # head-wise PPO health
        def _unhealthy(kl_t, clip_t):
            return not ((kl_low <= kl_t <= kl_high) and (clip_low <= clip_t <= clip_high))
        if _unhealthy(kl_opt_tail, clip_opt_tail):
            warnings.append("optimizer_head_update_unhealthy")
        if _unhealthy(kl_cfg_tail, clip_cfg_tail):
            warnings.append("cfg_head_update_unhealthy")
        if _unhealthy(kl_res_tail, clip_res_tail):
            warnings.append("resource_head_update_unhealthy")

        if score >= 4:
            status = "likely_success"
            message = "Training signals look healthy; policy likely learned useful behavior."
        elif score >= 2:
            status = "uncertain"
            message = "Training shows partial learning signals but still has notable risks."
        else:
            status = "likely_failed"
            message = "Training signals suggest learning is weak or policy may have collapsed."

        analysis = {
            "status": status,
            "score": int(score),
            "max_score": 5,
            "message": message,
            "signals": {
                "reward_head_mean": reward_head,
                "reward_tail_mean": reward_tail,
                "reward_delta": reward_delta,
                "reward_tolerance": reward_tol,
                "kl_tail_mean": kl_tail,
                "clip_frac_tail_mean": clip_tail,
                "entropy_tail_mean": entropy_tail,
                "opt_max_ratio_tail_mean": opt_max_tail,
                "cfg_max_ratio_tail_mean": cfg_max_tail,
                "res_max_ratio_tail_mean": res_max_tail,
                "value_loss_head_mean": value_head,
                "value_loss_tail_mean": value_tail,
                "adv_agent_std_mean_tail": adv_std_mean_tail,
                "adv_agent_std_max_tail": adv_std_max_tail,
                "adv_agent_abs_mean_min_tail": adv_abs_min_tail,
                "kl_opt_tail_mean": kl_opt_tail,
                "kl_cfg_tail_mean": kl_cfg_tail,
                "kl_res_tail_mean": kl_res_tail,
                "clip_opt_tail_mean": clip_opt_tail,
                "clip_cfg_tail_mean": clip_cfg_tail,
                "clip_res_tail_mean": clip_res_tail,
            },
            "warnings": warnings,
            "recommendations": recommendations,
        }
        return analysis

    def _save_analysis(self, analysis: dict):
        print("[MAPPO Diagnosis]", analysis.get("status"), "|", analysis.get("message"))
        print(
            "[MAPPO Diagnosis] score={}/{} reward_delta={:.6f} kl={:.6f} clip={:.6f} entropy={:.6f} "
            "opt_max={:.6f} cfg_max={:.6f} res_max={:.6f}".format(
                int(analysis.get("score", 0)),
                int(analysis.get("max_score", 5)),
                float(analysis.get("signals", {}).get("reward_delta", 0.0)),
                float(analysis.get("signals", {}).get("kl_tail_mean", 0.0)),
                float(analysis.get("signals", {}).get("clip_frac_tail_mean", 0.0)),
                float(analysis.get("signals", {}).get("entropy_tail_mean", 0.0)),
                float(analysis.get("signals", {}).get("opt_max_ratio_tail_mean", 0.0)),
                float(analysis.get("signals", {}).get("cfg_max_ratio_tail_mean", 0.0)),
                float(analysis.get("signals", {}).get("res_max_ratio_tail_mean", 0.0)),
            )
        )
        if self.saving_enabled:
            json_path = os.path.join(self.opts.data_save_dir, "mappo_training_diagnosis.json")
            txt_path = os.path.join(self.opts.data_save_dir, "mappo_training_diagnosis.txt")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"status: {analysis.get('status')}\n")
                f.write(f"score: {analysis.get('score')}/{analysis.get('max_score')}\n")
                f.write(f"message: {analysis.get('message')}\n\n")
                f.write("signals:\n")
                for k, v in analysis.get("signals", {}).items():
                    f.write(f"- {k}: {v}\n")
                f.write("\nwarnings:\n")
                for x in analysis.get("warnings", []):
                    f.write(f"- {x}\n")
                f.write("\nrecommendations:\n")
                for x in analysis.get("recommendations", []):
                    f.write(f"- {x}\n")

    def train(self):
        if self.train_epochs <= 0:
            print(
                f"[MAPPO] No training epochs to run: epoch_start={self.start_epoch}, epoch_end={self.end_epoch}."
            )
            return
        if len(self.fun_ids) == 0:
            raise ValueError("train_function_ids is empty.")
        if self.n_env <= 0:
            raise ValueError("each_question_batch_num must be positive.")

        best_reward = -float("inf")
        history_stats: List[dict] = []
        n = len(self.fun_ids)
        m = int(self.n_env)
        batches_per_epoch = int(math.ceil(float(n) / float(m)))
        total_batches = int(self.train_epochs * batches_per_epoch)
        pbar = tqdm(total=total_batches, disable=bool(self.opts.no_progress_bar), desc="mappo-train")
        progress_every = max(0, int(os.environ.get("MAPPO_ACCEPT_PROGRESS_EVERY", "0")))

        for epoch_i in range(self.start_epoch, self.end_epoch):
            # per-epoch shuffle of train set
            shuffled = list(self.fun_ids)
            rng = np.random.default_rng(int(self.opts.seed + epoch_i))
            rng.shuffle(shuffled)
            forced_active, forced_opt_idx, forced_in_warmup = self._forced_optimizer_for_epoch(epoch_i)

            epoch_reward_means = []
            last_stats = None

            for bidx in range(batches_per_epoch):
                l = bidx * m
                r = min((bidx + 1) * m, n)
                batch_fun_ids = shuffled[l:r]
                if progress_every:
                    print(
                        f"[accept-progress] epoch={epoch_i} batch={bidx + 1}/{batches_per_epoch} "
                        f"rollout=0/{self.horizon} functions={batch_fun_ids} phase=batch_start",
                        flush=True,
                    )
                envs = self._make_envs(batch_fun_ids=batch_fun_ids)

                obs = envs.reset()  # [E,A,F]
                if int(obs.shape[-1]) != int(self.policy.obs_dim):
                    raise ValueError(
                        "Environment/policy observation mismatch: "
                        f"env={obs.shape[-1]}, policy={self.policy.obs_dim}."
                    )
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)

                buffer = MAPPOBuffer(action_cols=self.action_cols)
                ep_rewards = []
                team_reward_steps = []
                mixed_reward_steps = []
                local_centered_step_means = []
                local_centered_step_stds = []
                graph_state_metric_steps = {
                    "proposal_mean_disagreement": [],
                    "proposal_max_edge_disagreement": [],
                    "pre_state_mean_disagreement": [],
                    "pre_state_max_edge_disagreement": [],
                    "post_mean_disagreement": [],
                    "post_max_edge_disagreement": [],
                    "consensus_improvement": [],
                    "consensus_operator_improvement": [],
                    "direction_agreement_mean": [],
                    "neighbor_avg_improve_mean": [],
                    "ccsa_momentum_norm_mean": [],
                    "ccsa_scale_mean": [],
                    "ccsa_scale_std": [],
                    "masoie_velocity_norm_mean": [],
                    "masoie_neighbor_pull_norm_mean": [],
                    "optimizer_guide_norm_mean": [],
                    "optimizer_guide_norm_max": [],
                    "optimizer_guide_applied_ratio": [],
                    "optimizer_guide_alignment_mean": [],
                    "optimizer_guide_source_active_ratio": [],
                    "optimizer_guide_strength_effective": [],
                    "optimizer_guide_strength_scale": [],
                    "optimizer_guide_schedule_metric": [],
                    "optimizer_guide_internal_active_ratio": [],
                    "optimizer_guide_internal_mean_step_norm_mean": [],
                    "optimizer_guide_internal_mean_step_norm_max": [],
                    "optimizer_guide_internal_alignment_mean": [],
                    "anchor_enabled": [],
                    "anchor_applied_ratio": [],
                    "anchor_dist_mean": [],
                    "anchor_dist_max": [],
                    "anchor_direction_norm_mean": [],
                    "anchor_direction_norm_max": [],
                    "optimizer_anchor_applied_ratio": [],
                    "optimizer_anchor_mean_step_norm_mean": [],
                    "optimizer_anchor_mean_step_norm_max": [],
                    "optimizer_anchor_sample_applied_ratio": [],
                    "committee_supported": [],
                    "committee_active": [],
                    "committee_selection_target_block": [],
                    "committee_best_score_mean": [],
                    "committee_best_score_max": [],
                    "committee_confidence_mean": [],
                    "committee_confidence_max": [],
                    "committee_guide_norm_mean": [],
                    "committee_guide_norm_max": [],
                    "committee_self_source_ratio": [],
                    "committee_source_diversity_mean": [],
                    "committee_source_diversity_max": [],
                    "committee_shadow_whole_report_f": [],
                    "committee_shadow_target_block_report_f": [],
                    "committee_shadow_whole_report_win": [],
                    "committee_shadow_target_block_report_win": [],
                    "committee_shadow_whole_report_log_improve": [],
                    "committee_shadow_target_block_report_log_improve": [],
                    "committee_shadow_whole_agent_f_mean": [],
                    "committee_shadow_whole_agent_f_min": [],
                    "committee_shadow_target_block_agent_f_mean": [],
                    "committee_shadow_target_block_agent_f_min": [],
                    "committee_shadow_whole_agent_win_ratio": [],
                    "committee_shadow_target_block_agent_win_ratio": [],
                    "committee_shadow_target_block_win_vs_whole": [],
                    "committee_acceptance_supported": [],
                    "committee_acceptance_ratio": [],
                    "committee_rejected_ratio": [],
                    "committee_candidate_report_f": [],
                    "committee_candidate_global_f_mean": [],
                    "committee_candidate_global_f_min": [],
                    "committee_candidate_vs_report_log_improve": [],
                    "committee_effective_beta_mean": [],
                    "committee_effective_beta_max": [],
                    "committee_base_alignment_mean": [],
                    "committee_noop_ratio": [],
                    "candidate_response_supported": [],
                    "candidate_response_active": [],
                    "candidate_response_actuated": [],
                    "candidate_response_probe_active_ratio": [],
                    "candidate_response_generator_active_ratio": [],
                    "candidate_response_accept_ratio": [],
                    "candidate_response_selection_confidence_mean": [],
                    "candidate_response_support_mean": [],
                    "candidate_response_confidence_mean": [],
                    "candidate_response_trust_radius_mean": [],
                    "candidate_response_probe_radius_mean": [],
                    "candidate_response_requested_shift_mean": [],
                    "candidate_response_applied_shift_mean": [],
                    "candidate_response_noop_ratio": [],
                    "candidate_response_rollback_ratio": [],
                    "candidate_response_agent_opportunity_ratio": [],
                    "candidate_response_agent_requested_ratio": [],
                    "candidate_response_agent_effective_ratio": [],
                    "candidate_response_actuator_beta_mean": [],
                    "candidate_response_actuator_beta_min": [],
                    "candidate_response_actuator_beta_max": [],
                    "candidate_response_shadow_base_report_f": [],
                    "candidate_response_shadow_candidate_report_f": [],
                    "candidate_response_shadow_report_win": [],
                    "candidate_response_shadow_report_log_improve": [],
                    "committee_acceptance_ratio_early": [],
                    "committee_acceptance_ratio_mid": [],
                    "committee_acceptance_ratio_late": [],
                    "collab_mode_idx_mean": [],
                    "guide_scale_idx_mean": [],
                    "guide_scale_value_mean": [],
                    "collab_leader_active_ratio": [],
                    "collab_strength_multiplier_mean": [],
                    "comm_rounds_per_event": [],
                    "comm_force_rounds": [],
                    "last_comm_rounds_applied": [],
                    "last_comm_rounds_requested": [],
                    "comm_round_idx_mean": [],
                    "comm_round_idx_max": [],
                    "state_comm_enabled": [],
                    "state_comm_msg_dim": [],
                    "last_actual_fes_mean": [],
                    "last_actual_fes_max": [],
                    "cumulative_actual_fes_mean": [],
                    "cmaes_numeric_fail_soft_step_count": [],
                    "cmaes_numeric_fail_soft_events": [],
                    "last_consensus_shift_norm_mean": [],
                    "last_consensus_shift_norm_max": [],
                    "step_comm_rounds_applied": [],
                    "step_comm_events": [],
                }
                graph_cumulative_names = (
                    "graph_comm_rounds",
                    "ccsa_lite_rounds",
                    "masoie_lite_rounds",
                    "centralized_full_mean_rounds",
                    "total_comm_rounds_applied",
                    "total_comm_events",
                    "graph_messages",
                    "graph_transmitted_floats",
                    "graph_transmitted_bytes",
                    "committee_events",
                    "committee_decision_local_evals",
                    "committee_shadow_local_evals",
                    "committee_shadow_global_local_evals",
                    "committee_messages",
                    "committee_transmitted_floats",
                    "committee_transmitted_bytes",
                    "committee_acceptance_events",
                    "committee_acceptance_accepted_events",
                    "committee_acceptance_rejected_events",
                    "committee_acceptance_local_evals",
                    "committee_acceptance_messages",
                    "committee_acceptance_transmitted_floats",
                    "committee_acceptance_transmitted_bytes",
                    "committee_acceptance_early_events",
                    "committee_acceptance_mid_events",
                    "committee_acceptance_late_events",
                    "committee_acceptance_early_accepted_events",
                    "committee_acceptance_mid_accepted_events",
                    "committee_acceptance_late_accepted_events",
                    "candidate_response_events",
                    "candidate_response_actuated_events",
                    "candidate_response_probe_local_evals",
                    "candidate_response_verification_local_evals",
                    "candidate_response_shadow_global_local_evals",
                    "candidate_response_rounds",
                    "candidate_response_messages",
                    "candidate_response_transmitted_floats",
                    "candidate_response_transmitted_bytes",
                    "local_search_fes",
                    "candidate_validation_local_evals",
                    "agent_state_local_evals",
                    "global_monitor_local_evals",
                    "global_monitor_rounds",
                    "verification_local_evals",
                    "total_physical_local_calls",
                    "reported_sum_fes",
                    "state_comm_rounds",
                    "state_comm_messages",
                    "state_comm_transmitted_floats",
                    "state_comm_transmitted_bytes",
                    "cmaes_numeric_fail_soft_events",
                )
                graph_cumulative_prev = None
                graph_cumulative_totals = None
                rollout_completed_episodes = 0
                rollout_env_resets = 0
                rollout_episode_lengths = []
                rollout_steps_since_reset = None

                for rollout_step in range(self.horizon):
                    global_obs = obs.reshape(obs.shape[0], -1)  # [E, A*F]
                    with torch.no_grad():
                        forced_idx = forced_opt_idx if forced_active else None
                        actions, log_probs, _, act_parts = self.policy.act(
                            obs,
                            return_parts=True,
                            forced_opt_idx=forced_idx,
                            forced_actuator_idx=(
                                self.forced_actuator_idx
                                if self.forced_actuator_idx >= 0
                                else None
                            ),
                        )
                        values = self.policy.get_values_detailed(global_obs, actions)

                    actions_np = actions.detach().cpu().numpy()
                    next_obs, rewards, dones, infos = envs.step(actions_np)
                    if progress_every and ((rollout_step + 1) % progress_every == 0):
                        print(
                            f"[accept-progress] epoch={epoch_i} batch={bidx + 1}/{batches_per_epoch} "
                            f"rollout={rollout_step + 1}/{self.horizon} functions={batch_fun_ids}",
                            flush=True,
                        )
                    done_ids = np.where(np.asarray(dones).astype(bool))[0]
                    if done_ids.size > 0:
                        # BaseVectorEnv requires caller-side reset after done.
                        reset_obs = envs.reset(id=done_ids.tolist())
                        next_obs[done_ids] = reset_obs

                    rewards_np = np.asarray(rewards, dtype=np.float32)
                    if rewards_np.ndim != 2:
                        rewards_np = rewards_np.reshape(rewards_np.shape[0], -1)
                    e_num, a_num = rewards_np.shape
                    if rollout_steps_since_reset is None:
                        rollout_steps_since_reset = np.zeros(
                            (e_num,),
                            dtype=np.int64,
                        )
                    rollout_steps_since_reset += 1
                    if done_ids.size > 0:
                        rollout_completed_episodes += int(done_ids.size)
                        rollout_env_resets += int(done_ids.size)
                        rollout_episode_lengths.extend(
                            int(rollout_steps_since_reset[int(env_id)])
                            for env_id in done_ids
                        )
                        rollout_steps_since_reset[done_ids] = 0
                    if graph_cumulative_prev is None:
                        graph_cumulative_prev = {
                            name: np.zeros((e_num,), dtype=np.float64)
                            for name in graph_cumulative_names
                        }
                        graph_cumulative_totals = {
                            name: np.zeros((e_num,), dtype=np.float64)
                            for name in graph_cumulative_names
                        }
                    team_rewards_np = np.mean(rewards_np, axis=1).astype(np.float32, copy=False)
                    local_centered_np = np.zeros((e_num, a_num), dtype=np.float32)
                    mixed_rewards_np = rewards_np.copy()
                    if isinstance(infos, (list, tuple, np.ndarray)):
                        infos_list = list(infos)
                    else:
                        infos_list = [infos for _ in range(e_num)]
                    for ei in range(min(e_num, len(infos_list))):
                        info_i = infos_list[ei] if isinstance(infos_list[ei], dict) else {}
                        team_rewards_np[ei] = float(info_i.get("team_reward", info_i.get("team_improve", team_rewards_np[ei])))
                        local_i = info_i.get("local_centered", None)
                        if local_i is not None:
                            arr_i = np.asarray(local_i, dtype=np.float32).reshape(-1)
                            if arr_i.size == a_num:
                                local_centered_np[ei] = arr_i
                        mixed_i = info_i.get("mixed_reward", None)
                        if mixed_i is not None:
                            arr_m = np.asarray(mixed_i, dtype=np.float32).reshape(-1)
                            if arr_m.size == a_num:
                                mixed_rewards_np[ei] = arr_m
                        for metric_name in graph_state_metric_steps:
                            if metric_name in info_i:
                                graph_state_metric_steps[metric_name].append(
                                    float(info_i[metric_name])
                                )
                        for metric_name in graph_cumulative_names:
                            if metric_name not in info_i:
                                continue
                            current_value = float(info_i[metric_name])
                            previous_value = float(graph_cumulative_prev[metric_name][ei])
                            increment = (
                                current_value - previous_value
                                if current_value >= previous_value
                                else current_value
                            )
                            graph_cumulative_totals[metric_name][ei] += increment
                            graph_cumulative_prev[metric_name][ei] = current_value

                    rewards_t = torch.as_tensor(rewards_np, dtype=torch.float32, device=self.device)  # [E,A]
                    dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)  # [E]
                    team_rewards_t = torch.as_tensor(team_rewards_np, dtype=torch.float32, device=self.device)
                    local_centered_t = torch.as_tensor(local_centered_np, dtype=torch.float32, device=self.device)
                    mixed_rewards_t = torch.as_tensor(mixed_rewards_np, dtype=torch.float32, device=self.device)
                    ep_rewards.append(float(rewards_t.mean().item()))
                    team_reward_steps.append(float(team_rewards_t.mean().item()))
                    mixed_reward_steps.append(float(mixed_rewards_t.mean().item()))
                    local_centered_step_means.append(float(local_centered_t.mean().item()))
                    local_centered_step_stds.append(float(local_centered_t.std(unbiased=False).item()))

                    log_probs_comm = act_parts.get(
                        "logp_comm",
                        torch.zeros_like(act_parts["logp_res"]),
                    )
                    if self.legacy_comm_old_logp_zero:
                        log_probs_comm = torch.zeros_like(log_probs_comm)

                    buffer.add(
                        obs,
                        global_obs,
                        actions,
                        log_probs,
                        rewards_t,
                        dones_t,
                        values_opt=values["value_opt"],
                        values_cfg=values["value_cfg"],
                        values_res=values["value_res"],
                        values_ref=values["value_ref"],
                        values_comm=values.get("value_comm", torch.zeros_like(values["value_res"])),
                        values_collab=values.get("value_collab", torch.zeros_like(values["value_res"])),
                        values_guide_scale=values.get("value_guide_scale", torch.zeros_like(values["value_res"])),
                        values_actuator=values.get("value_actuator", torch.zeros_like(values["value_res"])),
                        log_probs_opt=act_parts["logp_opt"],
                        log_probs_cfg=act_parts["logp_cfg"],
                        log_probs_res=act_parts["logp_res"],
                        log_probs_comm=log_probs_comm,
                        log_probs_collab=act_parts.get("logp_collab", torch.zeros_like(act_parts["logp_res"])),
                        log_probs_guide_scale=act_parts.get("logp_guide_scale", torch.zeros_like(act_parts["logp_res"])),
                        log_probs_actuator=act_parts.get("logp_actuator", torch.zeros_like(act_parts["logp_res"])),
                    )

                    obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                    obs = torch.where(torch.isnan(obs), torch.zeros_like(obs), obs)

                with torch.no_grad():
                    next_global_obs = obs.reshape(obs.shape[0], -1)
                    next_forced_idx = forced_opt_idx if forced_active else None
                    next_actions, _, _ = self.policy.act(
                        obs,
                        forced_opt_idx=next_forced_idx,
                        forced_actuator_idx=(
                            self.forced_actuator_idx
                            if self.forced_actuator_idx >= 0
                            else None
                        ),
                    )
                    next_values = self.policy.get_values_detailed(next_global_obs, next_actions)

                buffer.compute_returns_advantages(
                    next_values=next_values,
                    gamma=float(self.opts.gamma),
                    gae_lambda=float(getattr(self.opts, "gae_lambda", 0.95)),
                    normalize_scope=(
                        "env"
                        if (
                            str(getattr(self.opts, "mappo_update_mode", "joint")).lower() == "split_env"
                            and int(getattr(self.opts, "mappo_split_env_adv_norm", 1))
                        )
                        else "global"
                    ),
                )
                batch = buffer.as_tensors()
                if str(getattr(self.opts, "mappo_update_mode", "joint")).lower() == "split_env":
                    split_update_stats = []
                    env_count = int(batch["returns_all"].shape[1])
                    for env_i in range(env_count):
                        env_batch = buffer.as_tensors(env_indices=env_i)
                        split_update_stats.append(self.trainer.update(env_batch))
                    stats = {}
                    if len(split_update_stats) > 0:
                        keys = set().union(*(x.keys() for x in split_update_stats))
                        for key in keys:
                            vals = [
                                float(x[key])
                                for x in split_update_stats
                                if key in x and isinstance(x[key], (int, float, np.floating))
                            ]
                            if vals:
                                stats[key] = float(np.mean(vals))
                    stats["split_env_updates"] = float(env_count)
                else:
                    stats = self.trainer.update(batch)
                    stats["split_env_updates"] = 0.0
                stats["mappo_update_mode_split_env"] = (
                    1.0
                    if str(getattr(self.opts, "mappo_update_mode", "joint")).lower() == "split_env"
                    else 0.0
                )
                stats["reward_mean"] = float(np.mean(ep_rewards))
                stats["team_reward_mean"] = float(np.mean(team_reward_steps)) if len(team_reward_steps) > 0 else 0.0
                stats["mixed_reward_mean"] = float(np.mean(mixed_reward_steps)) if len(mixed_reward_steps) > 0 else 0.0
                stats["local_centered_mean"] = float(np.mean(local_centered_step_means)) if len(local_centered_step_means) > 0 else 0.0
                stats["local_centered_std"] = float(np.mean(local_centered_step_stds)) if len(local_centered_step_stds) > 0 else 0.0
                stats["forced_optimizer_active"] = 1.0 if forced_active else 0.0
                stats["forced_optimizer_idx"] = float(forced_opt_idx if forced_active else -1)
                stats["forced_optimizer_warmup"] = 1.0 if forced_in_warmup else 0.0
                for metric_name, metric_values in graph_state_metric_steps.items():
                    stats[metric_name] = (
                        float(np.mean(metric_values)) if metric_values else 0.0
                    )
                for metric_name in graph_cumulative_names:
                    totals = (
                        graph_cumulative_totals[metric_name]
                        if graph_cumulative_totals is not None
                        else np.zeros((1,), dtype=np.float64)
                    )
                    stats[metric_name] = float(np.mean(totals))
                stats["rollout_actual_local_search_fes"] = float(
                    stats.get("local_search_fes", 0.0)
                )
                stats["rollout_reported_fes"] = float(
                    stats.get("reported_sum_fes", 0.0)
                )
                stats["rollout_total_physical_local_calls"] = float(
                    stats.get("total_physical_local_calls", 0.0)
                )
                stats["completed_episodes_per_rollout"] = float(
                    rollout_completed_episodes
                )
                stats["env_resets_per_rollout"] = float(rollout_env_resets)
                stats["truncated_env_count"] = float(
                    np.count_nonzero(rollout_steps_since_reset)
                    if rollout_steps_since_reset is not None
                    else 0
                )
                stats["meta_steps_per_episode"] = float(
                    np.mean(rollout_episode_lengths)
                    if rollout_episode_lengths
                    else 0.0
                )
                with torch.no_grad():
                    adv_opt = batch["adv_opt_all"]
                    adv_cfg = batch["adv_cfg_all"]
                    adv_res = batch["adv_res_all"]
                    adv_comm = batch["adv_comm_all"]
                    ret = batch["returns_all"]
                    values_ref = batch["values_ref_all"]
                    values_opt = batch["values_opt_all"]
                    values_cfg = batch["values_cfg_all"]
                    values_res = batch["values_res_all"]
                    values_comm = batch["values_comm_all"]
                    values_collab = batch["values_collab_all"]
                    values_guide_scale = batch["values_guide_scale_all"]
                    values_actuator = batch["values_actuator_all"]
                    value_parts = [values_ref, values_opt, values_cfg, values_res]
                    if self.comm_action_enable:
                        value_parts.append(values_comm)
                    if self.collab_action_enable:
                        value_parts.append(values_collab)
                    if self.guide_scale_action_enable:
                        value_parts.append(values_guide_scale)
                    if self.actuator_action_enable:
                        value_parts.append(values_actuator)
                    values_mean = sum(value_parts) / float(len(value_parts))
                    if self.mappo_log_adv_stats and (adv_res is not None):
                        stats["adv_opt_mean"] = float(adv_opt.mean().item())
                        stats["adv_opt_std"] = float(adv_opt.std(unbiased=False).item())
                        stats["adv_cfg_mean"] = float(adv_cfg.mean().item())
                        stats["adv_cfg_std"] = float(adv_cfg.std(unbiased=False).item())
                        stats["adv_res_mean"] = float(adv_res.mean().item())
                        stats["adv_res_std"] = float(adv_res.std(unbiased=False).item())
                        stats["adv_comm_mean"] = float(adv_comm.mean().item()) if self.comm_action_enable else 0.0
                        stats["adv_comm_std"] = float(adv_comm.std(unbiased=False).item()) if self.comm_action_enable else 0.0
                        adv_collab = batch["adv_collab_all"]
                        adv_guide_scale = batch["adv_guide_scale_all"]
                        adv_actuator = batch["adv_actuator_all"]
                        stats["adv_collab_mean"] = float(adv_collab.mean().item()) if self.collab_action_enable else 0.0
                        stats["adv_collab_std"] = float(adv_collab.std(unbiased=False).item()) if self.collab_action_enable else 0.0
                        stats["adv_guide_scale_mean"] = float(adv_guide_scale.mean().item()) if self.guide_scale_action_enable else 0.0
                        stats["adv_guide_scale_std"] = float(adv_guide_scale.std(unbiased=False).item()) if self.guide_scale_action_enable else 0.0
                        stats["adv_actuator_mean"] = float(adv_actuator.mean().item()) if self.actuator_action_enable else 0.0
                        stats["adv_actuator_std"] = float(adv_actuator.std(unbiased=False).item()) if self.actuator_action_enable else 0.0
                        adv_head_parts = [adv_opt, adv_cfg, adv_res]
                        if self.comm_action_enable:
                            adv_head_parts.append(adv_comm)
                        if self.collab_action_enable:
                            adv_head_parts.append(adv_collab)
                        if self.guide_scale_action_enable:
                            adv_head_parts.append(adv_guide_scale)
                        if self.actuator_action_enable:
                            adv_head_parts.append(adv_actuator)
                        adv_all_heads = sum(adv_head_parts) / float(len(adv_head_parts))
                        stats["adv_mean"] = float(adv_all_heads.mean().item())
                        stats["adv_std"] = float(adv_all_heads.std(unbiased=False).item())
                        # Across-agent imbalance metrics.
                        adv_ref = adv_res
                        adv_agent_mean = adv_ref.mean(dim=(0, 1))  # [A]
                        adv_agent_std = adv_ref.std(dim=(0, 1), unbiased=False)  # [A]
                        adv_agent_abs_mean = adv_ref.abs().mean(dim=(0, 1))  # [A]
                        ret_agent_mean = ret.mean(dim=(0, 1)) if ret is not None else None
                        value_agent_mean = values_mean.mean(dim=(0, 1))
                        stats["adv_agent_mean_std"] = float(adv_agent_mean.std(unbiased=False).item())
                        stats["adv_agent_std_mean"] = float(adv_agent_std.mean().item())
                        stats["adv_agent_std_max"] = float(adv_agent_std.max().item())
                        stats["adv_agent_abs_mean_min"] = float(adv_agent_abs_mean.min().item())
                        stats["ret_agent_mean_std"] = float(ret_agent_mean.std(unbiased=False).item()) if ret_agent_mean is not None else 0.0
                        stats["value_agent_mean_std"] = float(value_agent_mean.std(unbiased=False).item())
                    else:
                        stats["adv_opt_mean"] = 0.0
                        stats["adv_opt_std"] = 0.0
                        stats["adv_cfg_mean"] = 0.0
                        stats["adv_cfg_std"] = 0.0
                        stats["adv_res_mean"] = 0.0
                        stats["adv_res_std"] = 0.0
                        stats["adv_comm_mean"] = 0.0
                        stats["adv_comm_std"] = 0.0
                        stats["adv_collab_mean"] = 0.0
                        stats["adv_collab_std"] = 0.0
                        stats["adv_guide_scale_mean"] = 0.0
                        stats["adv_guide_scale_std"] = 0.0
                        stats["adv_actuator_mean"] = 0.0
                        stats["adv_actuator_std"] = 0.0
                        stats["adv_mean"] = 0.0
                        stats["adv_std"] = 0.0
                        stats["adv_agent_mean_std"] = 0.0
                        stats["adv_agent_std_mean"] = 0.0
                        stats["adv_agent_std_max"] = 0.0
                        stats["adv_agent_abs_mean_min"] = 0.0
                        stats["ret_agent_mean_std"] = 0.0
                        stats["value_agent_mean_std"] = 0.0
                    stats["ret_mean"] = float(ret.mean().item()) if ret is not None else 0.0
                    stats["ret_std"] = float(ret.std(unbiased=False).item()) if ret is not None else 0.0
                    stats["value_mean"] = float(values_mean.mean().item())
                    stats["value_std"] = float(values_mean.std(unbiased=False).item())
                    stats["value_ref_mean"] = float(values_ref.mean().item())
                    stats["value_ref_std"] = float(values_ref.std(unbiased=False).item())

                if self.mappo_log_action_hist:
                    actions_all = torch.stack(buffer.actions, dim=0).detach().cpu().numpy()  # [T,E,A,C]
                    opt_flat = actions_all[..., 0].reshape(-1)
                    res_flat = actions_all[..., 1 + self.cfg_param_num].reshape(-1)

                    cfg_max_ratio = 0.0
                    cfg_block = actions_all[..., 1 : 1 + self.cfg_param_num]
                    for p in range(self.cfg_param_num):
                        cfg_flat = cfg_block[..., p].reshape(-1)
                        denom_cfg = max(1, cfg_flat.size)
                        counts_cfg = np.bincount(cfg_flat, minlength=self.action_dim).astype(np.float64)
                        ratios_cfg = counts_cfg / float(denom_cfg)
                        cfg_max_ratio = max(cfg_max_ratio, float(np.max(ratios_cfg)) if ratios_cfg.size > 0 else 0.0)
                        for i in range(self.action_dim):
                            key = f"cfg_ratio_{i}" if self.mappo_action_arch == "pre_caf4a62" else f"cfg_p{p}_ratio_{i}"
                            stats[key] = float(ratios_cfg[i])

                    denom_opt = max(1, opt_flat.size)
                    counts_opt = np.bincount(opt_flat, minlength=self.opt_action_dim).astype(np.float64)
                    ratios_opt = counts_opt / float(denom_opt)
                    for i in range(self.opt_action_dim):
                        stats[f"opt_ratio_{i}"] = float(ratios_opt[i])

                    denom_res = max(1, res_flat.size)
                    counts_res = np.bincount(res_flat, minlength=self.res_action_dim).astype(np.float64)
                    ratios_res = counts_res / float(denom_res)
                    for i in range(self.res_action_dim):
                        stats[f"res_ratio_{i}"] = float(ratios_res[i])
                    if self.comm_action_enable:
                        comm_col = 2 + self.cfg_param_num
                        comm_flat = actions_all[..., comm_col].reshape(-1)
                        denom_comm = max(1, comm_flat.size)
                        counts_comm = np.bincount(comm_flat, minlength=self.comm_action_dim).astype(np.float64)
                        ratios_comm = counts_comm / float(denom_comm)
                        for i in range(self.comm_action_dim):
                            stats[f"comm_ratio_{i}"] = float(ratios_comm[i])
                        stats["comm_max_ratio"] = float(np.max(ratios_comm)) if ratios_comm.size > 0 else 0.0
                    else:
                        stats["comm_max_ratio"] = 0.0
                    collab_col = 2 + self.cfg_param_num + (1 if self.comm_action_enable else 0)
                    if self.collab_action_enable:
                        collab_flat = actions_all[..., collab_col].reshape(-1)
                        denom_collab = max(1, collab_flat.size)
                        counts_collab = np.bincount(collab_flat, minlength=self.collab_action_dim).astype(np.float64)
                        ratios_collab = counts_collab / float(denom_collab)
                        for i in range(self.collab_action_dim):
                            stats[f"collab_ratio_{i}"] = float(ratios_collab[i])
                        stats["collab_max_ratio"] = float(np.max(ratios_collab)) if ratios_collab.size > 0 else 0.0
                    else:
                        stats["collab_max_ratio"] = 0.0
                    guide_scale_col = collab_col + (1 if self.collab_action_enable else 0)
                    if self.guide_scale_action_enable:
                        guide_flat = actions_all[..., guide_scale_col].reshape(-1)
                        denom_guide = max(1, guide_flat.size)
                        counts_guide = np.bincount(guide_flat, minlength=self.guide_scale_action_dim).astype(np.float64)
                        ratios_guide = counts_guide / float(denom_guide)
                        for i in range(self.guide_scale_action_dim):
                            stats[f"guide_scale_ratio_{i}"] = float(ratios_guide[i])
                        stats["guide_scale_max_ratio"] = float(np.max(ratios_guide)) if ratios_guide.size > 0 else 0.0
                    else:
                        stats["guide_scale_max_ratio"] = 0.0
                    actuator_col = guide_scale_col + (
                        1 if self.guide_scale_action_enable else 0
                    )
                    if self.actuator_action_enable:
                        actuator_flat = actions_all[..., actuator_col].reshape(-1)
                        denom_actuator = max(1, actuator_flat.size)
                        counts_actuator = np.bincount(
                            actuator_flat,
                            minlength=self.actuator_action_dim,
                        ).astype(np.float64)
                        ratios_actuator = counts_actuator / float(denom_actuator)
                        for i in range(self.actuator_action_dim):
                            stats[f"actuator_ratio_{i}"] = float(
                                ratios_actuator[i]
                            )
                        stats["actuator_max_ratio"] = (
                            float(np.max(ratios_actuator))
                            if ratios_actuator.size > 0
                            else 0.0
                        )
                    else:
                        stats["actuator_max_ratio"] = 0.0
                    stats["opt_max_ratio"] = float(np.max(ratios_opt)) if ratios_opt.size > 0 else 0.0
                    stats["cfg_max_ratio"] = float(cfg_max_ratio)
                    stats["res_max_ratio"] = float(np.max(ratios_res)) if ratios_res.size > 0 else 0.0
                else:
                    for i in range(self.opt_action_dim):
                        stats[f"opt_ratio_{i}"] = 0.0
                    for p in range(self.cfg_param_num):
                        for i in range(self.action_dim):
                            key = f"cfg_ratio_{i}" if self.mappo_action_arch == "pre_caf4a62" else f"cfg_p{p}_ratio_{i}"
                            stats[key] = 0.0
                    for i in range(self.res_action_dim):
                        stats[f"res_ratio_{i}"] = 0.0
                    for i in range(self.comm_action_dim if self.comm_action_enable else 0):
                        stats[f"comm_ratio_{i}"] = 0.0
                    for i in range(self.collab_action_dim if self.collab_action_enable else 0):
                        stats[f"collab_ratio_{i}"] = 0.0
                    for i in range(self.guide_scale_action_dim if self.guide_scale_action_enable else 0):
                        stats[f"guide_scale_ratio_{i}"] = 0.0
                    for i in range(self.actuator_action_dim if self.actuator_action_enable else 0):
                        stats[f"actuator_ratio_{i}"] = 0.0
                    stats["opt_max_ratio"] = 0.0
                    stats["cfg_max_ratio"] = 0.0
                    stats["res_max_ratio"] = 0.0
                    stats["comm_max_ratio"] = 0.0
                    stats["collab_max_ratio"] = 0.0
                    stats["guide_scale_max_ratio"] = 0.0
                    stats["actuator_max_ratio"] = 0.0

                self._append_stats(epoch_i, stats)
                history_stats.append({k: float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))})
                last_stats = stats
                epoch_reward_means.append(float(stats["reward_mean"]))
                if self.training_logger is not None:
                    self.training_logger.store_mappo_update_stats(stats)

                pbar.set_postfix_str(
                    f"e={epoch_i + 1}/{self.end_epoch} b={bidx + 1}/{batches_per_epoch} "
                    f"r={stats['reward_mean']:.4f} pi={stats['policy_loss']:.4f} v={stats['value_loss']:.4f}"
                )
                pbar.update(1)
                envs.close()

            # per-epoch tb logging
            if self.training_logger is not None:
                if ((epoch_i - self.start_epoch + 1) % self.mappo_log_interval) == 0:
                    self.training_logger.write_mappo_to_tb(
                        step=epoch_i,
                        lr_actor=self.policy.actor_optimizer.param_groups[0]["lr"],
                        lr_critic=self.policy.critic_optimizer.param_groups[0]["lr"],
                    )
                    self.training_logger.reset_mappo_stats()

            # simple periodic checkpoint (per epoch)
            if (last_stats is not None) and self.saving_enabled and ((epoch_i + 1) % int(max(1, self.opts.checkpoint_epochs)) == 0):
                ckpt = {
                    "checkpoint_version": 2,
                    "actor": self.policy.actor.state_dict(),
                    "critic": self.policy.critic.state_dict(),
                    "actor_opt": self.policy.actor_optimizer.state_dict(),
                    "critic_opt": self.policy.critic_optimizer.state_dict(),
                    "policy_signature": getattr(self.policy, "policy_signature", None),
                    "policy_signature_str": getattr(self.policy, "policy_signature_str", ""),
                    "epoch": int(epoch_i),
                    "stats": last_stats,
                    "rng_state_version": MAPPO_RNG_STATE_VERSION,
                    "rng_state": capture_rng_state(),
                    "training_schedule": {
                        "start_epoch": int(self.schedule_start_epoch),
                        "end_epoch": int(self.schedule_end_epoch),
                    },
                }
                torch.save(ckpt, os.path.join(self.opts.modal_save_dir, f"mappo-epoch-{epoch_i}.pt"))
                epoch_reward = float(np.mean(epoch_reward_means)) if len(epoch_reward_means) > 0 else -float("inf")
                if epoch_reward > best_reward and self.best_ckpt_path is not None:
                    best_reward = epoch_reward
                    torch.save(ckpt, self.best_ckpt_path)

        pbar.close()

        analysis = self._analyze_training(history_stats)
        self._save_analysis(analysis)

from typing import Dict

import torch
import torch.nn as nn


class MAPPOTrainer:
    def __init__(self, opts, policy):
        self.opts = opts
        self.policy = policy
        self.clip_param = float(opts.eps_clip)
        self.ppo_epoch = int(opts.K_epochs)
        self.value_coef = float(getattr(opts, "vf_coef", 1.0))
        self.max_grad_norm = float(opts.max_grad_norm)

        # Entropy coefficients (optimizer head should be largest in stage C).
        base_ent = float(getattr(opts, "entropy_coef", 0.01))
        self.beta_opt = float(getattr(opts, "mappo_entropy_coef_opt", base_ent))
        self.beta_cfg = float(getattr(opts, "mappo_entropy_coef_cfg", base_ent))
        self.beta_res = float(getattr(opts, "mappo_entropy_coef_res", base_ent))
        self.beta_comm = float(getattr(opts, "mappo_entropy_coef_comm", base_ent))
        self.beta_collab = float(getattr(opts, "mappo_entropy_coef_collab", base_ent))
        self.beta_guide_scale = float(getattr(opts, "mappo_entropy_coef_guide_scale", base_ent))
        self.beta_actuator = float(getattr(opts, "mappo_entropy_coef_actuator", base_ent))

        # Head loss weights.
        self.comm_action_enable = bool(
            int(getattr(opts, "objective_split_comm_action_enable", 0))
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
        self.lambda_opt = float(getattr(opts, "mappo_policy_loss_weight_opt", 0.8))
        self.lambda_cfg = float(getattr(opts, "mappo_policy_loss_weight_cfg", 1.0))
        self.lambda_res = float(getattr(opts, "mappo_policy_loss_weight_res", 1.0))
        self.lambda_comm = float(getattr(opts, "mappo_policy_loss_weight_comm", 1.0))
        self.lambda_collab = float(getattr(opts, "mappo_policy_loss_weight_collab", 1.0))
        self.lambda_guide_scale = float(getattr(opts, "mappo_policy_loss_weight_guide_scale", 1.0))
        self.lambda_actuator = float(getattr(opts, "mappo_policy_loss_weight_actuator", 1.0))
        self.eta_ref = float(getattr(opts, "mappo_value_loss_weight_ref", 1.0))
        self.eta_opt = float(getattr(opts, "mappo_value_loss_weight_opt", 0.5))
        self.eta_cfg = float(getattr(opts, "mappo_value_loss_weight_cfg", 0.8))
        self.eta_res = float(getattr(opts, "mappo_value_loss_weight_res", 1.0))
        self.eta_comm = float(getattr(opts, "mappo_value_loss_weight_comm", 1.0))
        self.eta_collab = float(getattr(opts, "mappo_value_loss_weight_collab", 1.0))
        self.eta_guide_scale = float(getattr(opts, "mappo_value_loss_weight_guide_scale", 1.0))
        self.eta_actuator = float(getattr(opts, "mappo_value_loss_weight_actuator", 1.0))

    @staticmethod
    def _ppo_terms(new_logp, old_logp, adv, clip_param):
        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        with torch.no_grad():
            log_ratio = new_logp - old_logp
            approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
            clip_frac = (torch.abs(ratio - 1.0) > clip_param).float().mean()
            ratio_mean = ratio.mean()
            ratio_std = ratio.std(unbiased=False)
            delta_mean = log_ratio.mean()
            delta_std = log_ratio.std(unbiased=False)
        return policy_loss, approx_kl, clip_frac, ratio_mean, ratio_std, delta_mean, delta_std

    @staticmethod
    def _clipped_value_loss(values, old_values, returns, clip_param):
        values_clipped = old_values + torch.clamp(values - old_values, -clip_param, clip_param)
        v_loss1 = (values - returns) ** 2
        v_loss2 = (values_clipped - returns) ** 2
        return torch.max(v_loss1, v_loss2).mean()

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs_actor = batch["obs_actor"]
        actions_actor = batch["actions_actor"].long()
        old_logp_total = batch["old_log_probs_actor"]
        old_logp_opt = batch["old_log_probs_opt_actor"]
        old_logp_cfg = batch["old_log_probs_cfg_actor"]
        old_logp_res = batch["old_log_probs_res_actor"]
        old_logp_comm = batch["old_log_probs_comm_actor"]
        old_logp_collab = batch["old_log_probs_collab_actor"]
        old_logp_guide_scale = batch["old_log_probs_guide_scale_actor"]
        old_logp_actuator = batch["old_log_probs_actuator_actor"]
        adv_opt = batch["adv_opt_actor"]
        adv_cfg = batch["adv_cfg_actor"]
        adv_res = batch["adv_res_actor"]
        adv_comm = batch["adv_comm_actor"]
        adv_collab = batch["adv_collab_actor"]
        adv_guide_scale = batch["adv_guide_scale_actor"]
        adv_actuator = batch["adv_actuator_actor"]
        adv_parts = [adv_opt, adv_cfg, adv_res]
        if self.comm_action_enable:
            adv_parts.append(adv_comm)
        if self.collab_action_enable:
            adv_parts.append(adv_collab)
        if self.guide_scale_action_enable:
            adv_parts.append(adv_guide_scale)
        if self.actuator_action_enable:
            adv_parts.append(adv_actuator)
        adv_total = sum(adv_parts) / float(len(adv_parts))

        global_obs_critic = batch["global_obs_critic"]
        actions_critic = batch["actions_critic"].long()
        old_values_ref = batch["old_values_ref_critic"]
        old_values_opt = batch["old_values_opt_critic"]
        old_values_cfg = batch["old_values_cfg_critic"]
        old_values_res = batch["old_values_res_critic"]
        old_values_comm = batch["old_values_comm_critic"]
        old_values_collab = batch["old_values_collab_critic"]
        old_values_guide_scale = batch["old_values_guide_scale_critic"]
        old_values_actuator = batch["old_values_actuator_critic"]
        returns_critic = batch["returns_critic"]

        info = {
            "policy_loss": 0.0,
            "policy_loss_opt": 0.0,
            "policy_loss_cfg": 0.0,
            "policy_loss_res": 0.0,
            "policy_loss_comm": 0.0,
            "policy_loss_collab": 0.0,
            "policy_loss_guide_scale": 0.0,
            "policy_loss_actuator": 0.0,
            "value_loss": 0.0,
            "value_loss_ref": 0.0,
            "value_loss_opt": 0.0,
            "value_loss_cfg": 0.0,
            "value_loss_res": 0.0,
            "value_loss_comm": 0.0,
            "value_loss_collab": 0.0,
            "value_loss_guide_scale": 0.0,
            "value_loss_actuator": 0.0,
            "entropy": 0.0,
            "entropy_opt": 0.0,
            "entropy_cfg": 0.0,
            "entropy_res": 0.0,
            "entropy_comm": 0.0,
            "entropy_collab": 0.0,
            "entropy_guide_scale": 0.0,
            "entropy_actuator": 0.0,
            "approx_kl": 0.0,
            "approx_kl_opt": 0.0,
            "approx_kl_cfg": 0.0,
            "approx_kl_res": 0.0,
            "approx_kl_comm": 0.0,
            "approx_kl_collab": 0.0,
            "approx_kl_guide_scale": 0.0,
            "approx_kl_actuator": 0.0,
            "clip_frac": 0.0,
            "clip_frac_opt": 0.0,
            "clip_frac_cfg": 0.0,
            "clip_frac_res": 0.0,
            "clip_frac_comm": 0.0,
            "clip_frac_collab": 0.0,
            "clip_frac_guide_scale": 0.0,
            "clip_frac_actuator": 0.0,
            "ratio_mean": 0.0,
            "ratio_std": 0.0,
            "ratio_mean_opt": 0.0,
            "ratio_mean_cfg": 0.0,
            "ratio_mean_res": 0.0,
            "ratio_mean_comm": 0.0,
            "ratio_mean_collab": 0.0,
            "ratio_mean_guide_scale": 0.0,
            "ratio_mean_actuator": 0.0,
            "ratio_std_opt": 0.0,
            "ratio_std_cfg": 0.0,
            "ratio_std_res": 0.0,
            "ratio_std_comm": 0.0,
            "ratio_std_collab": 0.0,
            "ratio_std_guide_scale": 0.0,
            "ratio_std_actuator": 0.0,
            "delta_logp_opt_mean": 0.0,
            "delta_logp_cfg_mean": 0.0,
            "delta_logp_res_mean": 0.0,
            "delta_logp_comm_mean": 0.0,
            "delta_logp_collab_mean": 0.0,
            "delta_logp_guide_scale_mean": 0.0,
            "delta_logp_actuator_mean": 0.0,
            "delta_logp_opt_std": 0.0,
            "delta_logp_cfg_std": 0.0,
            "delta_logp_res_std": 0.0,
            "delta_logp_comm_std": 0.0,
            "delta_logp_collab_std": 0.0,
            "delta_logp_guide_scale_std": 0.0,
            "delta_logp_actuator_std": 0.0,
            "adv_opt_mean": 0.0,
            "adv_opt_std": 0.0,
            "adv_cfg_mean": 0.0,
            "adv_cfg_std": 0.0,
            "adv_res_mean": 0.0,
            "adv_res_std": 0.0,
            "adv_comm_mean": 0.0,
            "adv_comm_std": 0.0,
            "adv_collab_mean": 0.0,
            "adv_collab_std": 0.0,
            "adv_guide_scale_mean": 0.0,
            "adv_guide_scale_std": 0.0,
            "adv_actuator_mean": 0.0,
            "adv_actuator_std": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
        }

        for _ in range(self.ppo_epoch):
            # ----- actor -----
            parts = self.policy.actor.evaluate_actions_detailed(obs_actor, actions_actor)
            pl_opt, kl_opt, cf_opt, rmean_opt, rstd_opt, dmean_opt, dstd_opt = self._ppo_terms(
                parts["logp_opt"], old_logp_opt, adv_opt, self.clip_param
            )
            pl_cfg, kl_cfg, cf_cfg, rmean_cfg, rstd_cfg, dmean_cfg, dstd_cfg = self._ppo_terms(
                parts["logp_cfg"], old_logp_cfg, adv_cfg, self.clip_param
            )
            pl_res, kl_res, cf_res, rmean_res, rstd_res, dmean_res, dstd_res = self._ppo_terms(
                parts["logp_res"], old_logp_res, adv_res, self.clip_param
            )
            if self.comm_action_enable:
                pl_comm, kl_comm, cf_comm, rmean_comm, rstd_comm, dmean_comm, dstd_comm = self._ppo_terms(
                    parts["logp_comm"], old_logp_comm, adv_comm, self.clip_param
                )
            else:
                zero = torch.zeros((), device=obs_actor.device, dtype=obs_actor.dtype)
                pl_comm = kl_comm = cf_comm = rmean_comm = rstd_comm = dmean_comm = dstd_comm = zero
            if self.collab_action_enable:
                pl_collab, kl_collab, cf_collab, rmean_collab, rstd_collab, dmean_collab, dstd_collab = self._ppo_terms(
                    parts["logp_collab"], old_logp_collab, adv_collab, self.clip_param
                )
            else:
                zero = torch.zeros((), device=obs_actor.device, dtype=obs_actor.dtype)
                pl_collab = kl_collab = cf_collab = rmean_collab = rstd_collab = dmean_collab = dstd_collab = zero
            if self.guide_scale_action_enable:
                pl_guide_scale, kl_guide_scale, cf_guide_scale, rmean_guide_scale, rstd_guide_scale, dmean_guide_scale, dstd_guide_scale = self._ppo_terms(
                    parts["logp_guide_scale"], old_logp_guide_scale, adv_guide_scale, self.clip_param
                )
            else:
                zero = torch.zeros((), device=obs_actor.device, dtype=obs_actor.dtype)
                pl_guide_scale = kl_guide_scale = cf_guide_scale = rmean_guide_scale = rstd_guide_scale = dmean_guide_scale = dstd_guide_scale = zero
            if self.actuator_action_enable:
                pl_actuator, kl_actuator, cf_actuator, rmean_actuator, rstd_actuator, dmean_actuator, dstd_actuator = self._ppo_terms(
                    parts["logp_actuator"],
                    old_logp_actuator,
                    adv_actuator,
                    self.clip_param,
                )
            else:
                zero = torch.zeros((), device=obs_actor.device, dtype=obs_actor.dtype)
                pl_actuator = kl_actuator = cf_actuator = rmean_actuator = rstd_actuator = dmean_actuator = dstd_actuator = zero
            # keep total ratio diagnostics for continuity
            _, kl_total, cf_total, rmean_total, rstd_total, _, _ = self._ppo_terms(
                parts["logp_total"], old_logp_total, adv_total, self.clip_param
            )

            policy_loss = (
                self.lambda_opt * pl_opt
                + self.lambda_cfg * pl_cfg
                + self.lambda_res * pl_res
            )
            if self.comm_action_enable:
                policy_loss = policy_loss + self.lambda_comm * pl_comm
            if self.collab_action_enable:
                policy_loss = policy_loss + self.lambda_collab * pl_collab
            if self.guide_scale_action_enable:
                policy_loss = policy_loss + self.lambda_guide_scale * pl_guide_scale
            if self.actuator_action_enable:
                policy_loss = policy_loss + self.lambda_actuator * pl_actuator
            entropy_reg = (
                self.beta_opt * parts["entropy_opt"].mean()
                + self.beta_cfg * parts["entropy_cfg"].mean()
                + self.beta_res * parts["entropy_res"].mean()
            )
            if self.comm_action_enable:
                entropy_reg = entropy_reg + self.beta_comm * parts["entropy_comm"].mean()
            if self.collab_action_enable:
                entropy_reg = entropy_reg + self.beta_collab * parts["entropy_collab"].mean()
            if self.guide_scale_action_enable:
                entropy_reg = entropy_reg + self.beta_guide_scale * parts["entropy_guide_scale"].mean()
            if self.actuator_action_enable:
                entropy_reg = entropy_reg + self.beta_actuator * parts["entropy_actuator"].mean()

            self.policy.actor_optimizer.zero_grad()
            (policy_loss - entropy_reg).backward()
            actor_gn = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            self.policy.actor_optimizer.step()

            # ----- critic -----
            values = self.policy.critic(global_obs_critic, actions_critic)
            v_ref = values["value_ref"]
            v_opt = values["value_opt"]
            v_cfg = values["value_cfg"]
            v_res = values["value_res"]
            v_comm = values.get("value_comm", torch.zeros_like(v_res))
            v_collab = values.get("value_collab", torch.zeros_like(v_res))
            v_guide_scale = values.get("value_guide_scale", torch.zeros_like(v_res))
            v_actuator = values.get("value_actuator", torch.zeros_like(v_res))

            vl_ref = self._clipped_value_loss(v_ref, old_values_ref, returns_critic, self.clip_param)
            vl_opt = self._clipped_value_loss(v_opt, old_values_opt, returns_critic, self.clip_param)
            vl_cfg = self._clipped_value_loss(v_cfg, old_values_cfg, returns_critic, self.clip_param)
            vl_res = self._clipped_value_loss(v_res, old_values_res, returns_critic, self.clip_param)
            if self.comm_action_enable:
                vl_comm = self._clipped_value_loss(v_comm, old_values_comm, returns_critic, self.clip_param)
            else:
                vl_comm = torch.zeros((), device=returns_critic.device, dtype=returns_critic.dtype)
            if self.collab_action_enable:
                vl_collab = self._clipped_value_loss(v_collab, old_values_collab, returns_critic, self.clip_param)
            else:
                vl_collab = torch.zeros((), device=returns_critic.device, dtype=returns_critic.dtype)
            if self.guide_scale_action_enable:
                vl_guide_scale = self._clipped_value_loss(v_guide_scale, old_values_guide_scale, returns_critic, self.clip_param)
            else:
                vl_guide_scale = torch.zeros((), device=returns_critic.device, dtype=returns_critic.dtype)
            if self.actuator_action_enable:
                vl_actuator = self._clipped_value_loss(
                    v_actuator,
                    old_values_actuator,
                    returns_critic,
                    self.clip_param,
                )
            else:
                vl_actuator = torch.zeros((), device=returns_critic.device, dtype=returns_critic.dtype)
            value_loss = (
                self.eta_ref * vl_ref
                + self.eta_opt * vl_opt
                + self.eta_cfg * vl_cfg
                + self.eta_res * vl_res
            )
            if self.comm_action_enable:
                value_loss = value_loss + self.eta_comm * vl_comm
            if self.collab_action_enable:
                value_loss = value_loss + self.eta_collab * vl_collab
            if self.guide_scale_action_enable:
                value_loss = value_loss + self.eta_guide_scale * vl_guide_scale
            if self.actuator_action_enable:
                value_loss = value_loss + self.eta_actuator * vl_actuator

            self.policy.critic_optimizer.zero_grad()
            (self.value_coef * value_loss).backward()
            critic_gn = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            self.policy.critic_optimizer.step()

            # ----- logging -----
            info["policy_loss"] += float(policy_loss.item())
            info["policy_loss_opt"] += float(pl_opt.item())
            info["policy_loss_cfg"] += float(pl_cfg.item())
            info["policy_loss_res"] += float(pl_res.item())
            info["policy_loss_comm"] += float(pl_comm.item())
            info["policy_loss_collab"] += float(pl_collab.item())
            info["policy_loss_guide_scale"] += float(pl_guide_scale.item())
            info["policy_loss_actuator"] += float(pl_actuator.item())
            info["value_loss"] += float(value_loss.item())
            info["value_loss_ref"] += float(vl_ref.item())
            info["value_loss_opt"] += float(vl_opt.item())
            info["value_loss_cfg"] += float(vl_cfg.item())
            info["value_loss_res"] += float(vl_res.item())
            info["value_loss_comm"] += float(vl_comm.item())
            info["value_loss_collab"] += float(vl_collab.item())
            info["value_loss_guide_scale"] += float(vl_guide_scale.item())
            info["value_loss_actuator"] += float(vl_actuator.item())

            ent_opt = parts["entropy_opt"].mean()
            ent_cfg = parts["entropy_cfg"].mean()
            ent_res = parts["entropy_res"].mean()
            ent_comm = parts["entropy_comm"].mean()
            ent_collab = parts["entropy_collab"].mean()
            ent_guide_scale = parts["entropy_guide_scale"].mean()
            ent_actuator = parts["entropy_actuator"].mean()
            ent_parts = [ent_opt, ent_cfg, ent_res]
            if self.comm_action_enable:
                ent_parts.append(ent_comm)
            if self.collab_action_enable:
                ent_parts.append(ent_collab)
            if self.guide_scale_action_enable:
                ent_parts.append(ent_guide_scale)
            if self.actuator_action_enable:
                ent_parts.append(ent_actuator)
            ent_total = sum(ent_parts) / float(len(ent_parts))
            info["entropy"] += float(ent_total.item())
            info["entropy_opt"] += float(ent_opt.item())
            info["entropy_cfg"] += float(ent_cfg.item())
            info["entropy_res"] += float(ent_res.item())
            info["entropy_comm"] += float(ent_comm.item())
            info["entropy_collab"] += float(ent_collab.item())
            info["entropy_guide_scale"] += float(ent_guide_scale.item())
            info["entropy_actuator"] += float(ent_actuator.item())

            info["approx_kl"] += float(kl_total.item())
            info["approx_kl_opt"] += float(kl_opt.item())
            info["approx_kl_cfg"] += float(kl_cfg.item())
            info["approx_kl_res"] += float(kl_res.item())
            info["approx_kl_comm"] += float(kl_comm.item())
            info["approx_kl_collab"] += float(kl_collab.item())
            info["approx_kl_guide_scale"] += float(kl_guide_scale.item())
            info["approx_kl_actuator"] += float(kl_actuator.item())

            info["clip_frac"] += float(cf_total.item())
            info["clip_frac_opt"] += float(cf_opt.item())
            info["clip_frac_cfg"] += float(cf_cfg.item())
            info["clip_frac_res"] += float(cf_res.item())
            info["clip_frac_comm"] += float(cf_comm.item())
            info["clip_frac_collab"] += float(cf_collab.item())
            info["clip_frac_guide_scale"] += float(cf_guide_scale.item())
            info["clip_frac_actuator"] += float(cf_actuator.item())

            info["ratio_mean"] += float(rmean_total.item())
            info["ratio_std"] += float(rstd_total.item())
            info["ratio_mean_opt"] += float(rmean_opt.item())
            info["ratio_mean_cfg"] += float(rmean_cfg.item())
            info["ratio_mean_res"] += float(rmean_res.item())
            info["ratio_mean_comm"] += float(rmean_comm.item())
            info["ratio_mean_collab"] += float(rmean_collab.item())
            info["ratio_mean_guide_scale"] += float(rmean_guide_scale.item())
            info["ratio_mean_actuator"] += float(rmean_actuator.item())
            info["ratio_std_opt"] += float(rstd_opt.item())
            info["ratio_std_cfg"] += float(rstd_cfg.item())
            info["ratio_std_res"] += float(rstd_res.item())
            info["ratio_std_comm"] += float(rstd_comm.item())
            info["ratio_std_collab"] += float(rstd_collab.item())
            info["ratio_std_guide_scale"] += float(rstd_guide_scale.item())
            info["ratio_std_actuator"] += float(rstd_actuator.item())

            info["delta_logp_opt_mean"] += float(dmean_opt.item())
            info["delta_logp_cfg_mean"] += float(dmean_cfg.item())
            info["delta_logp_res_mean"] += float(dmean_res.item())
            info["delta_logp_comm_mean"] += float(dmean_comm.item())
            info["delta_logp_collab_mean"] += float(dmean_collab.item())
            info["delta_logp_guide_scale_mean"] += float(dmean_guide_scale.item())
            info["delta_logp_actuator_mean"] += float(dmean_actuator.item())
            info["delta_logp_opt_std"] += float(dstd_opt.item())
            info["delta_logp_cfg_std"] += float(dstd_cfg.item())
            info["delta_logp_res_std"] += float(dstd_res.item())
            info["delta_logp_comm_std"] += float(dstd_comm.item())
            info["delta_logp_collab_std"] += float(dstd_collab.item())
            info["delta_logp_guide_scale_std"] += float(dstd_guide_scale.item())
            info["delta_logp_actuator_std"] += float(dstd_actuator.item())

            info["adv_opt_mean"] += float(adv_opt.mean().item())
            info["adv_opt_std"] += float(adv_opt.std(unbiased=False).item())
            info["adv_cfg_mean"] += float(adv_cfg.mean().item())
            info["adv_cfg_std"] += float(adv_cfg.std(unbiased=False).item())
            info["adv_res_mean"] += float(adv_res.mean().item())
            info["adv_res_std"] += float(adv_res.std(unbiased=False).item())
            info["adv_comm_mean"] += float(adv_comm.mean().item())
            info["adv_comm_std"] += float(adv_comm.std(unbiased=False).item())
            info["adv_collab_mean"] += float(adv_collab.mean().item())
            info["adv_collab_std"] += float(adv_collab.std(unbiased=False).item())
            info["adv_guide_scale_mean"] += float(adv_guide_scale.mean().item())
            info["adv_guide_scale_std"] += float(adv_guide_scale.std(unbiased=False).item())
            info["adv_actuator_mean"] += float(adv_actuator.mean().item())
            info["adv_actuator_std"] += float(adv_actuator.std(unbiased=False).item())

            info["actor_grad_norm"] += float(actor_gn.item() if torch.is_tensor(actor_gn) else actor_gn)
            info["critic_grad_norm"] += float(critic_gn.item() if torch.is_tensor(critic_gn) else critic_gn)

        for k in list(info.keys()):
            info[k] /= self.ppo_epoch
        return info

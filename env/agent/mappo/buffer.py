from typing import Dict

import torch


class MAPPOBuffer:
    def __init__(self, action_cols: int = 3):
        self.action_cols = int(action_cols)
        self.obs = []
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.log_probs_opt = []
        self.log_probs_cfg = []
        self.log_probs_res = []
        self.log_probs_comm = []
        self.rewards = []
        self.dones = []
        self.values_opt = []
        self.values_cfg = []
        self.values_res = []
        self.values_comm = []
        self.values_ref = []

        self.returns = None
        self.adv_opt = None
        self.adv_cfg = None
        self.adv_res = None
        self.adv_comm = None

    def add(
        self,
        obs,
        global_obs,
        actions,
        log_probs,
        rewards,
        dones,
        values_opt,
        values_cfg,
        values_res,
        values_ref,
        values_comm=None,
        log_probs_opt=None,
        log_probs_cfg=None,
        log_probs_res=None,
        log_probs_comm=None,
    ):
        self.obs.append(obs.detach())
        self.global_obs.append(global_obs.detach())
        self.actions.append(actions.detach())
        self.log_probs.append(log_probs.detach())

        if log_probs_opt is None:
            log_probs_opt = torch.zeros_like(log_probs)
        if log_probs_cfg is None:
            log_probs_cfg = torch.zeros_like(log_probs)
        if log_probs_res is None:
            log_probs_res = torch.zeros_like(log_probs)
        if log_probs_comm is None:
            log_probs_comm = torch.zeros_like(log_probs)
        self.log_probs_opt.append(log_probs_opt.detach())
        self.log_probs_cfg.append(log_probs_cfg.detach())
        self.log_probs_res.append(log_probs_res.detach())
        self.log_probs_comm.append(log_probs_comm.detach())

        self.rewards.append(rewards.detach())
        self.dones.append(dones.detach())
        if values_comm is None:
            values_comm = torch.zeros_like(values_res)
        self.values_opt.append(values_opt.detach())
        self.values_cfg.append(values_cfg.detach())
        self.values_res.append(values_res.detach())
        self.values_comm.append(values_comm.detach())
        self.values_ref.append(values_ref.detach())

    def compute_returns_advantages(
        self,
        next_values: Dict[str, torch.Tensor],
        gamma: float,
        gae_lambda: float,
        normalize_scope: str = "global",
    ):
        """
        Stage-C target:
        - one shared per-agent return target [T,E,A] bootstrapped by dedicated V_ref
        - three advantages: A_opt/A_cfg/A_res from the shared return minus each old value
        """
        rewards = torch.stack(self.rewards, dim=0)  # [T,E,A]
        dones = torch.stack(self.dones, dim=0)  # [T,E]
        values_opt = torch.stack(self.values_opt, dim=0)  # [T,E,A]
        values_cfg = torch.stack(self.values_cfg, dim=0)  # [T,E,A]
        values_res = torch.stack(self.values_res, dim=0)  # [T,E,A]
        values_comm = torch.stack(self.values_comm, dim=0)  # [T,E,A]
        values_ref = torch.stack(self.values_ref, dim=0)  # [T,E,A]

        next_ref = next_values["value_ref"]  # [E,A]

        t, e, a = rewards.shape
        adv_ref = torch.zeros_like(rewards)
        last_adv = torch.zeros((e, a), device=rewards.device, dtype=rewards.dtype)
        last_value = next_ref
        for k in reversed(range(t)):
            nonterminal = (1.0 - dones[k]).unsqueeze(-1)  # [E,1]
            delta = rewards[k] + gamma * last_value * nonterminal - values_ref[k]
            last_adv = delta + gamma * gae_lambda * nonterminal * last_adv
            adv_ref[k] = last_adv
            last_value = values_ref[k]
        returns = adv_ref + values_ref

        def _normalize(x: torch.Tensor) -> torch.Tensor:
            scope = str(normalize_scope).lower()
            if scope in {"env", "per_env", "split_env"}:
                mean = x.mean(dim=(0, 2), keepdim=True)
                std = x.std(dim=(0, 2), unbiased=False, keepdim=True)
                return (x - mean) / (std + 1e-8)
            return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)

        adv_opt = _normalize(returns - values_opt)
        adv_cfg = _normalize(returns - values_cfg)
        adv_res = _normalize(returns - values_res)
        adv_comm = _normalize(returns - values_comm)

        self.returns = returns
        self.adv_opt = adv_opt
        self.adv_cfg = adv_cfg
        self.adv_res = adv_res
        self.adv_comm = adv_comm

    def as_tensors(self, env_indices=None) -> Dict[str, torch.Tensor]:
        obs = torch.stack(self.obs, dim=0)  # [T,E,A,F]
        global_obs = torch.stack(self.global_obs, dim=0)  # [T,E,G]
        actions = torch.stack(self.actions, dim=0)  # [T,E,A,C]

        log_probs = torch.stack(self.log_probs, dim=0)  # [T,E,A]
        log_probs_opt = torch.stack(self.log_probs_opt, dim=0)  # [T,E,A]
        log_probs_cfg = torch.stack(self.log_probs_cfg, dim=0)  # [T,E,A]
        log_probs_res = torch.stack(self.log_probs_res, dim=0)  # [T,E,A]
        log_probs_comm = torch.stack(self.log_probs_comm, dim=0)  # [T,E,A]

        values_opt = torch.stack(self.values_opt, dim=0)  # [T,E,A]
        values_cfg = torch.stack(self.values_cfg, dim=0)  # [T,E,A]
        values_res = torch.stack(self.values_res, dim=0)  # [T,E,A]
        values_comm = torch.stack(self.values_comm, dim=0)  # [T,E,A]
        values_ref = torch.stack(self.values_ref, dim=0)  # [T,E,A]

        returns = self.returns  # [T,E,A]
        adv_opt = self.adv_opt  # [T,E,A]
        adv_cfg = self.adv_cfg  # [T,E,A]
        adv_res = self.adv_res  # [T,E,A]
        adv_comm = self.adv_comm  # [T,E,A]

        if env_indices is not None:
            if isinstance(env_indices, int):
                env_indices = [env_indices]
            idx = torch.as_tensor(env_indices, dtype=torch.long, device=obs.device)
            obs = obs.index_select(1, idx)
            global_obs = global_obs.index_select(1, idx)
            actions = actions.index_select(1, idx)
            log_probs = log_probs.index_select(1, idx)
            log_probs_opt = log_probs_opt.index_select(1, idx)
            log_probs_cfg = log_probs_cfg.index_select(1, idx)
            log_probs_res = log_probs_res.index_select(1, idx)
            log_probs_comm = log_probs_comm.index_select(1, idx)
            values_opt = values_opt.index_select(1, idx)
            values_cfg = values_cfg.index_select(1, idx)
            values_res = values_res.index_select(1, idx)
            values_comm = values_comm.index_select(1, idx)
            values_ref = values_ref.index_select(1, idx)
            returns = returns.index_select(1, idx)
            adv_opt = adv_opt.index_select(1, idx)
            adv_cfg = adv_cfg.index_select(1, idx)
            adv_res = adv_res.index_select(1, idx)
            adv_comm = adv_comm.index_select(1, idx)

        t, e, a, f = obs.shape
        g = global_obs.shape[-1]

        return {
            # actor side (agent-level flatten)
            "obs_actor": obs.reshape(t * e * a, f),
            "actions_actor": actions.reshape(t * e * a, self.action_cols),
            "old_log_probs_actor": log_probs.reshape(t * e * a),
            "old_log_probs_opt_actor": log_probs_opt.reshape(t * e * a),
            "old_log_probs_cfg_actor": log_probs_cfg.reshape(t * e * a),
            "old_log_probs_res_actor": log_probs_res.reshape(t * e * a),
            "old_log_probs_comm_actor": log_probs_comm.reshape(t * e * a),
            "adv_opt_actor": adv_opt.reshape(t * e * a),
            "adv_cfg_actor": adv_cfg.reshape(t * e * a),
            "adv_res_actor": adv_res.reshape(t * e * a),
            "adv_comm_actor": adv_comm.reshape(t * e * a),
            # critic side (env-level flatten, keep agent dim)
            "global_obs_critic": global_obs.reshape(t * e, g),
            "actions_critic": actions.reshape(t * e, a, self.action_cols),
            "old_values_ref_critic": values_ref.reshape(t * e, a),
            "old_values_opt_critic": values_opt.reshape(t * e, a),
            "old_values_cfg_critic": values_cfg.reshape(t * e, a),
            "old_values_res_critic": values_res.reshape(t * e, a),
            "old_values_comm_critic": values_comm.reshape(t * e, a),
            "returns_critic": returns.reshape(t * e, a),
            # diagnostics
            "returns_all": returns,
            "values_ref_all": values_ref,
            "adv_opt_all": adv_opt,
            "adv_cfg_all": adv_cfg,
            "adv_res_all": adv_res,
            "adv_comm_all": adv_comm,
            "values_opt_all": values_opt,
            "values_cfg_all": values_cfg,
            "values_res_all": values_res,
            "values_comm_all": values_comm,
        }

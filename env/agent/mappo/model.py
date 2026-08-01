from typing import Dict, Tuple
import json

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _comm_candidate_count(opts) -> int:
    raw = getattr(opts, "objective_split_comm_round_candidates", [1, 2, 4, 8])
    if isinstance(raw, str):
        vals = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        vals = list(raw)
    return int(max(1, len(vals)))


class HierarchicalActor(nn.Module):
    """
    Hierarchical policy:
      1) optimizer choice a_opt
      2) profile choice a_cfg conditioned on a_opt
      3) resource choice a_res conditioned on a_opt, a_cfg
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_opt: int,
        n_cfg: int,
        n_res: int,
        opt_names: Tuple[str, ...],
        n_comm: int = 1,
        comm_action_enable: bool = False,
        cfg_param_num: int = 4,
        cfg_bin_num: int = 4,
        opt_emb_dim: int = 8,
        cfg_emb_dim: int = 8,
    ):
        super().__init__()
        self.n_opt = int(n_opt)
        self.n_cfg = int(n_cfg)
        self.n_res = int(n_res)
        self.n_comm = int(max(1, n_comm))
        self.comm_action_enable = bool(comm_action_enable)
        self.cfg_param_num = int(cfg_param_num)
        self.cfg_bin_num = int(cfg_bin_num)
        self.opt_names = tuple([str(x).lower() for x in opt_names])
        if len(self.opt_names) != self.n_opt:
            raise ValueError(
                f"opt_names length mismatch: len(opt_names)={len(self.opt_names)} vs n_opt={self.n_opt}"
            )
        if len(set(self.opt_names)) != len(self.opt_names):
            raise ValueError(f"optimizer names contain duplicates: {self.opt_names}")

        # shared observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # stage-1: optimizer selection
        self.optimizer_head = nn.Linear(hidden_dim, self.n_opt)
        self.optimizer_embedding = nn.Embedding(self.n_opt, opt_emb_dim)

        # stage-2: profile selection
        self.config_backbone = nn.Sequential(
            nn.Linear(hidden_dim + opt_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.config_heads = nn.ModuleList([nn.Linear(hidden_dim, self.n_cfg) for _ in range(self.n_opt)])
        self.config_embedding = nn.Embedding(self.cfg_bin_num, cfg_emb_dim)

        # stage-3: resource selection
        self.resource_head = nn.Sequential(
            nn.Linear(hidden_dim + opt_emb_dim + cfg_emb_dim * self.cfg_param_num, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_res),
        )
        if self.comm_action_enable:
            self.resource_embedding = nn.Embedding(self.n_res, cfg_emb_dim)
            self.comm_head = nn.Sequential(
                nn.Linear(
                    hidden_dim
                    + opt_emb_dim
                    + cfg_emb_dim * self.cfg_param_num
                    + cfg_emb_dim,
                    hidden_dim,
                ),
                nn.Tanh(),
                nn.Linear(hidden_dim, self.n_comm),
            )

    def _flatten_obs(self, obs: torch.Tensor):
        # obs: [E, A, F]
        e, a, f = obs.shape
        obs_flat = obs.reshape(e * a, f)
        return obs_flat, e, a

    def _select_cfg_logits(self, g_flat: torch.Tensor, a_opt_flat: torch.Tensor):
        # select optimizer-specific parameter head output
        logits_all = torch.stack([head(g_flat) for head in self.config_heads], dim=1)  # [N,O,C]
        sel = a_opt_flat.long().view(-1, 1, 1).expand(-1, 1, self.n_cfg)
        cfg_logits = torch.gather(logits_all, dim=1, index=sel).squeeze(1)  # [N,C]
        return cfg_logits

    def dist_and_logprob_entropy(
        self,
        obs_flat: torch.Tensor,
        actions_flat: torch.Tensor = None,
        deterministic: bool = False,
        return_parts: bool = False,
        forced_opt_idx: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          obs_flat: [N,F]
          actions_flat: [N,2+cfg_param_num(+1 if comm enabled)] optional fixed actions
        Returns:
          actions_flat: [N,2+cfg_param_num(+1 if comm enabled)]
          logp_total: [N]
          entropy_total: [N]
        """
        h = self.obs_encoder(obs_flat)

        # stage-1 optimizer
        opt_logits = self.optimizer_head(h)
        dist_opt = Categorical(logits=opt_logits)
        if actions_flat is None:
            if forced_opt_idx is not None:
                a_opt = torch.full(
                    (obs_flat.shape[0],),
                    int(forced_opt_idx),
                    dtype=torch.long,
                    device=obs_flat.device,
                )
            elif deterministic:
                a_opt = torch.argmax(opt_logits, dim=-1)
            else:
                a_opt = dist_opt.sample()
        else:
            a_opt = actions_flat[:, 0].long()
        logp_opt = dist_opt.log_prob(a_opt)
        ent_opt = dist_opt.entropy()

        # stage-2 optimizer-parameter bins (condition on optimizer)
        opt_emb = self.optimizer_embedding(a_opt)
        g = self.config_backbone(torch.cat([h, opt_emb], dim=-1))
        cfg_logits = self._select_cfg_logits(g, a_opt)
        cfg_logits = cfg_logits.view(-1, self.cfg_param_num, self.cfg_bin_num)
        dist_cfg = Categorical(logits=cfg_logits)
        if actions_flat is None:
            if deterministic:
                a_cfg = torch.argmax(cfg_logits, dim=-1)  # [N,P]
            else:
                a_cfg = dist_cfg.sample()  # [N,P]
        else:
            a_cfg = actions_flat[:, 1 : 1 + self.cfg_param_num].long()
        logp_cfg_each = dist_cfg.log_prob(a_cfg)  # [N,P]
        ent_cfg_each = dist_cfg.entropy()  # [N,P]
        logp_cfg = torch.sum(logp_cfg_each, dim=-1)
        ent_cfg = torch.mean(ent_cfg_each, dim=-1)

        # stage-3 resource (condition on optimizer + cfg)
        cfg_emb = self.config_embedding(a_cfg).reshape(a_cfg.shape[0], -1)
        res_logits = self.resource_head(torch.cat([h, opt_emb, cfg_emb], dim=-1))
        dist_res = Categorical(logits=res_logits)
        if actions_flat is None:
            if deterministic:
                a_res = torch.argmax(res_logits, dim=-1)
            else:
                a_res = dist_res.sample()
        else:
            a_res = actions_flat[:, 1 + self.cfg_param_num].long()
        logp_res = dist_res.log_prob(a_res)
        ent_res = dist_res.entropy()

        actions_parts = [a_opt.unsqueeze(-1), a_cfg, a_res.unsqueeze(-1)]
        logp_total = logp_opt + logp_cfg + logp_res
        entropy_sum = ent_opt + ent_cfg + ent_res
        entropy_heads = 3.0
        if self.comm_action_enable:
            res_emb = self.resource_embedding(a_res)
            comm_logits = self.comm_head(torch.cat([h, opt_emb, cfg_emb, res_emb], dim=-1))
            dist_comm = Categorical(logits=comm_logits)
            if actions_flat is None:
                if deterministic:
                    a_comm = torch.argmax(comm_logits, dim=-1)
                else:
                    a_comm = dist_comm.sample()
            else:
                a_comm = actions_flat[:, 2 + self.cfg_param_num].long()
            logp_comm = dist_comm.log_prob(a_comm)
            ent_comm = dist_comm.entropy()
            actions_parts.append(a_comm.unsqueeze(-1))
            logp_total = logp_total + logp_comm
            entropy_sum = entropy_sum + ent_comm
            entropy_heads = 4.0
        else:
            logp_comm = torch.zeros_like(logp_res)
            ent_comm = torch.zeros_like(ent_res)

        actions = torch.cat(actions_parts, dim=-1)
        # use mean entropy across active heads to keep entropy regularization scale stable
        entropy_total = entropy_sum / entropy_heads
        if not return_parts:
            return actions, logp_total, entropy_total

        parts: Dict[str, torch.Tensor] = {
            "logp_total": logp_total,
            "entropy_total": entropy_total,
            "logp_opt": logp_opt,
            "logp_cfg": logp_cfg,
            "logp_res": logp_res,
            "logp_comm": logp_comm,
            "entropy_opt": ent_opt,
            "entropy_cfg": ent_cfg,
            "entropy_res": ent_res,
            "entropy_comm": ent_comm,
        }
        return actions, logp_total, entropy_total, parts

    def act(
        self, obs: torch.Tensor, deterministic: bool = False, return_parts: bool = False, forced_opt_idx: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [E,A,F]
        obs_flat, e, a = self._flatten_obs(obs)
        out = self.dist_and_logprob_entropy(
            obs_flat,
            actions_flat=None,
            deterministic=deterministic,
            return_parts=return_parts,
            forced_opt_idx=forced_opt_idx,
        )
        if return_parts:
            act_flat, logp_flat, ent_flat, parts = out
        else:
            act_flat, logp_flat, ent_flat = out
            parts = None
        action_cols = 2 + self.cfg_param_num + (1 if self.comm_action_enable else 0)
        actions = act_flat.view(e, a, action_cols)
        logp = logp_flat.view(e, a)
        entropy = ent_flat.view(e, a)
        if not return_parts:
            return actions, logp, entropy

        parts_view = {
            "logp_total": parts["logp_total"].view(e, a),
            "entropy_total": parts["entropy_total"].view(e, a),
            "logp_opt": parts["logp_opt"].view(e, a),
            "logp_cfg": parts["logp_cfg"].view(e, a),
            "logp_res": parts["logp_res"].view(e, a),
            "entropy_opt": parts["entropy_opt"].view(e, a),
            "entropy_cfg": parts["entropy_cfg"].view(e, a),
            "entropy_res": parts["entropy_res"].view(e, a),
        }
        return actions, logp, entropy, parts_view

    def evaluate_actions(self, obs_flat: torch.Tensor, actions_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs_flat: [N,F], actions_flat: [N,2+cfg_param_num]
        _, logp, entropy = self.dist_and_logprob_entropy(
            obs_flat, actions_flat=actions_flat.long(), deterministic=False
        )
        return logp, entropy

    def evaluate_actions_detailed(self, obs_flat: torch.Tensor, actions_flat: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, _, _, parts = self.dist_and_logprob_entropy(
            obs_flat, actions_flat=actions_flat.long(), deterministic=False, return_parts=True
        )
        return parts


class CentralCritic(nn.Module):
    def __init__(
        self,
        global_obs_dim: int,
        hidden_dim: int,
        n_agents: int,
        n_opt: int,
        n_cfg: int,
        n_res: int = 1,
        n_comm: int = 1,
        comm_action_enable: bool = False,
        cfg_param_num: int = 4,
        cfg_bin_num: int = 4,
        agent_emb_dim: int = 16,
        action_emb_dim: int = 8,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.cfg_param_num = int(cfg_param_num)
        self.cfg_bin_num = int(cfg_bin_num)
        self.n_res = int(max(1, n_res))
        self.n_comm = int(max(1, n_comm))
        self.comm_action_enable = bool(comm_action_enable)
        self.trunk = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.agent_embedding = nn.Embedding(self.n_agents, int(agent_emb_dim))
        self.opt_embedding = nn.Embedding(int(n_opt), int(action_emb_dim))
        self.cfg_embedding = nn.Embedding(int(self.cfg_bin_num), int(action_emb_dim))
        if self.comm_action_enable:
            self.res_embedding = nn.Embedding(self.n_res, int(action_emb_dim))

        base_in = hidden_dim + int(agent_emb_dim)
        self.value_ref_head = nn.Sequential(
            nn.Linear(base_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_opt_head = nn.Sequential(
            nn.Linear(base_in, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_cfg_head = nn.Sequential(
            nn.Linear(base_in + int(action_emb_dim), hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_res_head = nn.Sequential(
            nn.Linear(base_in + int(action_emb_dim) + int(action_emb_dim) * self.cfg_param_num, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        if self.comm_action_enable:
            self.value_comm_head = nn.Sequential(
                nn.Linear(
                    base_in
                    + int(action_emb_dim)
                    + int(action_emb_dim) * self.cfg_param_num
                    + int(action_emb_dim),
                    hidden_dim,
                ),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, global_obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          global_obs: [B,G]
          actions: [B,A,2+cfg_param_num(+1 if comm enabled)]
        Returns:
          dict of [B,A]:
            value_ref / value_opt / value_cfg / value_res
        """
        b = global_obs.shape[0]
        a = self.n_agents
        expected_d = 2 + self.cfg_param_num + (1 if self.comm_action_enable else 0)
        if actions.dim() != 3 or actions.shape[0] != b or actions.shape[1] != a or actions.shape[2] != expected_d:
            raise ValueError(
                f"Expected critic actions shape [B,{a},{expected_d}], got {tuple(actions.shape)} for B={b}."
            )

        h = self.trunk(global_obs)  # [B,H]
        h = h.unsqueeze(1).expand(b, a, h.shape[-1])  # [B,A,H]

        agent_ids = torch.arange(a, device=global_obs.device, dtype=torch.long).unsqueeze(0).expand(b, a)
        agent_emb = self.agent_embedding(agent_ids)  # [B,A,Ea]
        base = torch.cat([h, agent_emb], dim=-1)  # [B,A,H+Ea]

        a_opt = actions[..., 0].long()
        a_cfg = actions[..., 1 : 1 + self.cfg_param_num].long()
        a_res = actions[..., 1 + self.cfg_param_num].long()
        opt_emb = self.opt_embedding(a_opt)  # [B,A,Eact]
        cfg_emb = self.cfg_embedding(a_cfg).reshape(b, a, -1)  # [B,A,P*Eact]

        value_ref = self.value_ref_head(base).squeeze(-1)
        value_opt = self.value_opt_head(base).squeeze(-1)
        value_cfg = self.value_cfg_head(torch.cat([base, opt_emb], dim=-1)).squeeze(-1)
        value_res = self.value_res_head(torch.cat([base, opt_emb, cfg_emb], dim=-1)).squeeze(-1)
        out = {
            "value_ref": value_ref,
            "value_opt": value_opt,
            "value_cfg": value_cfg,
            "value_res": value_res,
        }
        if self.comm_action_enable:
            res_emb = self.res_embedding(a_res)
            out["value_comm"] = self.value_comm_head(
                torch.cat([base, opt_emb, cfg_emb, res_emb], dim=-1)
            ).squeeze(-1)
        return out

    def forward_mean(self, global_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        vals = self.forward(global_obs, actions)
        return vals["value_ref"]


class MAPPOPolicy:
    def __init__(self, opts, obs_dim: int, n_agents: int, action_dim: int = 0):
        self.opts = opts
        self.device = torch.device("cuda" if opts.use_cuda else "cpu")
        self.obs_dim = int(obs_dim)
        self.n_agents = int(n_agents)

        hidden_actor = int(getattr(opts, "actor2_hidden_dim", 320))
        hidden_critic = int(getattr(opts, "actor1_hidden_dim", 256))
        global_obs_dim = obs_dim * n_agents

        n_opt = int(len(getattr(opts, "optimizer_candidates", ["mmes", "vkd", "cmaes", "sepcmaes"])))
        opt_names = tuple([str(x).lower() for x in getattr(opts, "optimizer_candidates", ["mmes", "vkd", "cmaes", "sepcmaes"])])
        cfg_bin_num = int(len(getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])))
        cfg_param_num = int(getattr(opts, "mappo_cfg_param_num", 4))
        if str(getattr(opts, "mappo_action_arch", "current")).lower() == "pre_caf4a62":
            cfg_param_num = 1
        n_cfg = int(cfg_param_num * cfg_bin_num)
        n_res = int(len(getattr(opts, "resource_factors", [0.5, 1.0, 2.0])))
        self.comm_action_enable = bool(
            int(getattr(opts, "objective_split_comm_action_enable", 0))
        )
        n_comm = (
            _comm_candidate_count(opts)
            if self.comm_action_enable
            else 1
        )

        self.actor = HierarchicalActor(
            obs_dim=obs_dim,
            hidden_dim=hidden_actor,
            n_opt=n_opt,
            n_cfg=n_cfg,
            n_res=n_res,
            opt_names=opt_names,
            n_comm=n_comm,
            comm_action_enable=self.comm_action_enable,
            cfg_param_num=cfg_param_num,
            cfg_bin_num=cfg_bin_num,
            opt_emb_dim=int(getattr(opts, "mappo_opt_emb_dim", 8)),
            cfg_emb_dim=int(getattr(opts, "mappo_cfg_emb_dim", 8)),
        ).to(self.device)
        self.critic = CentralCritic(
            global_obs_dim=global_obs_dim,
            hidden_dim=hidden_critic,
            n_agents=n_agents,
            n_opt=n_opt,
            n_cfg=n_cfg,
            n_res=n_res,
            n_comm=n_comm,
            comm_action_enable=self.comm_action_enable,
            cfg_param_num=cfg_param_num,
            cfg_bin_num=cfg_bin_num,
            agent_emb_dim=int(getattr(opts, "mappo_critic_agent_emb_dim", 16)),
            action_emb_dim=int(getattr(opts, "mappo_critic_action_emb_dim", 8)),
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=opts.lr_model)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=opts.lr_critic)

        self.policy_signature = build_mappo_policy_signature(
            opts=opts,
            obs_dim=obs_dim,
            n_agents=n_agents,
            n_opt=n_opt,
            cfg_param_num=cfg_param_num,
            cfg_bin_num=cfg_bin_num,
            n_res=n_res,
            n_comm=n_comm,
            hidden_actor=hidden_actor,
            hidden_critic=hidden_critic,
        )
        self.policy_signature_str = signature_to_string(self.policy_signature)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        return_parts: bool = False,
        forced_opt_idx: int = None,
    ):
        return self.actor.act(
            obs,
            deterministic=deterministic,
            return_parts=return_parts,
            forced_opt_idx=forced_opt_idx,
        )

    @torch.no_grad()
    def get_values(self, global_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.critic.forward_mean(global_obs, actions)

    @torch.no_grad()
    def get_values_detailed(self, global_obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.critic(global_obs, actions)

    @torch.no_grad()
    def evaluate_actions_detailed(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        obs_flat, e, a = self.actor._flatten_obs(obs)
        actions_flat = actions.reshape(e * a, actions.shape[-1])
        parts = self.actor.evaluate_actions_detailed(obs_flat, actions_flat)
        return {
            "logp_total": parts["logp_total"].view(e, a),
            "entropy_total": parts["entropy_total"].view(e, a),
            "logp_opt": parts["logp_opt"].view(e, a),
            "logp_cfg": parts["logp_cfg"].view(e, a),
            "logp_res": parts["logp_res"].view(e, a),
            "logp_comm": parts["logp_comm"].view(e, a),
            "entropy_opt": parts["entropy_opt"].view(e, a),
            "entropy_cfg": parts["entropy_cfg"].view(e, a),
            "entropy_res": parts["entropy_res"].view(e, a),
            "entropy_comm": parts["entropy_comm"].view(e, a),
        }


# IMPORTANT:
# Bump this version string whenever network/action architecture semantics change.
MAPPO_ARCH_SIGNATURE_VERSION = "mappo-arch-v3-4opt-4param-4bin-20260520"
MAPPO_PRE_CAF_SIGNATURE_VERSION = "mappo-arch-v2-profile-20260520-pre-caf4a62"
MAPPO_GRAPH_OBS_SIGNATURE_VERSION = "mappo-arch-v4-graph-neighbor-obs-20260622"
MAPPO_COMM_ACTION_SIGNATURE_VERSION = "mappo-arch-v5-comm-round-head-20260706"
MAPPO_STATE_COMM_SIGNATURE_VERSION = "mappo-arch-v7-compact-state-comm-20260725"


def build_mappo_policy_signature(
    opts,
    obs_dim: int,
    n_agents: int,
    n_opt: int,
    cfg_param_num: int,
    cfg_bin_num: int,
    n_res: int,
    n_comm: int,
    hidden_actor: int,
    hidden_critic: int,
) -> Dict:
    action_arch = str(getattr(opts, "mappo_action_arch", "current")).lower()
    neighbor_obs_mode = str(
        getattr(opts, "objective_split_neighbor_obs_mode", "auto")
    ).lower()
    if neighbor_obs_mode == "auto":
        neighbor_obs_mode = (
            "full"
            if bool(int(getattr(opts, "objective_split_neighbor_obs", 0)))
            else "none"
        )
    objective_split_modes = {
        "wsn_objective",
        "dbo_objective",
        "cdo_objective",
        "masoie_wsn_objective",
    }
    env_mode = str(getattr(opts, "mappo_env_mode", "variable_cc")).lower()
    neighbor_obs = (
        env_mode in objective_split_modes and neighbor_obs_mode != "none"
    )
    comm_action_enable = bool(
        int(getattr(opts, "objective_split_comm_action_enable", 0))
    )
    state_comm_mode = str(
        getattr(opts, "objective_split_state_comm_mode", "none")
    ).lower()
    state_comm_enable = env_mode in objective_split_modes and state_comm_mode != "none"
    if state_comm_enable:
        arch_version = MAPPO_STATE_COMM_SIGNATURE_VERSION
    elif comm_action_enable:
        arch_version = MAPPO_COMM_ACTION_SIGNATURE_VERSION
    elif action_arch == "pre_caf4a62":
        arch_version = MAPPO_PRE_CAF_SIGNATURE_VERSION
    elif neighbor_obs:
        arch_version = MAPPO_GRAPH_OBS_SIGNATURE_VERSION
    else:
        arch_version = MAPPO_ARCH_SIGNATURE_VERSION
    sig = {
        "arch_version": arch_version,
        "obs_dim": int(obs_dim),
        "n_agents": int(n_agents),
        "n_opt": int(n_opt),
        "cfg_param_num": int(cfg_param_num),
        "cfg_bin_num": int(cfg_bin_num),
        "n_res": int(n_res),
        "n_comm": int(n_comm),
        "actor_hidden_dim": int(hidden_actor),
        "critic_hidden_dim": int(hidden_critic),
        "opt_emb_dim": int(getattr(opts, "mappo_opt_emb_dim", 8)),
        "cfg_emb_dim": int(getattr(opts, "mappo_cfg_emb_dim", 8)),
        "critic_agent_emb_dim": int(getattr(opts, "mappo_critic_agent_emb_dim", 16)),
        "critic_action_emb_dim": int(getattr(opts, "mappo_critic_action_emb_dim", 8)),
        "optimizer_candidates": [str(x).lower() for x in getattr(opts, "optimizer_candidates", ["mmes", "vkd", "cmaes", "sepcmaes"])],
        "optimizer_profile_candidates": [str(x).lower() for x in getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])],
        "resource_factors": [float(x) for x in getattr(opts, "resource_factors", [0.5, 1.0, 2.0])],
    }
    if action_arch == "pre_caf4a62":
        sig["mappo_action_arch"] = action_arch
    vkd_action_param_mode = str(
        getattr(opts, "subopt_vkd_action_param_mode", "rank")
    ).lower()
    if vkd_action_param_mode != "rank":
        sig["subopt_vkd_action_param_mode"] = vkd_action_param_mode
        sig["subopt_vkd_cs_scale_levels"] = [
            float(x)
            for x in getattr(opts, "subopt_vkd_cs_scale_levels", [0.5, 1.0, 2.0])
        ]
        sig["subopt_vkd_k_inc_cond_levels"] = [
            float(x)
            for x in getattr(opts, "subopt_vkd_k_inc_cond_levels", [10.0, 30.0, 60.0])
        ]
    if neighbor_obs:
        sig["objective_split_neighbor_obs"] = True
        sig["objective_split_neighbor_obs_mode"] = neighbor_obs_mode
    if state_comm_enable:
        sig["objective_split_state_comm_mode"] = state_comm_mode
        sig["objective_split_state_comm_include_delta"] = int(
            bool(getattr(opts, "objective_split_state_comm_include_delta", 0))
        )
    if comm_action_enable:
        sig["objective_split_comm_action_enable"] = True
        sig["objective_split_comm_round_candidates"] = [
            int(x)
            for x in getattr(opts, "objective_split_comm_round_candidates", [1, 2, 4, 8])
        ]
        sig["objective_split_comm_action_reduce"] = str(
            getattr(opts, "objective_split_comm_action_reduce", "max")
        ).lower()
    return sig


def signature_to_string(sig: Dict) -> str:
    return json.dumps(sig, ensure_ascii=False, sort_keys=True)

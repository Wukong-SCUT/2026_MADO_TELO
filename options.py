import os
import sys
import time
import math
import argparse
import torch
from pathlib import Path
from typing import List
from collections import OrderedDict

# 获取当前脚本所在路径
BASE_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
# 保存结果的根目录
SAVE_DIR = os.path.join(BASE_DIR, 'save_dir')


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_str_list(raw: str) -> List[str]:
    return [str(x.strip()) for x in raw.split(",") if x.strip()]


def _count_str_list(raw, default: int = 0) -> int:
    if raw is None:
        return int(default)
    if isinstance(raw, (list, tuple)):
        return len(raw)
    return len(_parse_str_list(str(raw)))


def _parse_int_list_allow_empty(raw: str) -> List[int]:
    if raw is None:
        return []
    return [int(x.strip()) for x in str(raw).split(",") if str(x).strip()]


def _parse_budget_int_list_allow_empty(raw: str) -> List[int]:
    if raw is None:
        return []
    out = []
    for x in str(raw).split(","):
        s = str(x).strip()
        if not s:
            continue
        out.append(int(float(s)))
    return out


def _mean_positive_or_default(vals: List[int], default_val: int) -> int:
    xs = [float(v) for v in vals if float(v) > 0.0]
    if not xs:
        return int(default_val)
    return int(round(sum(xs) / len(xs)))


def _run_name_budget_value(opts) -> int:
    benchmark = str(getattr(opts, "benchmark_name", ""))
    if benchmark == "WSNLocation":
        return _mean_positive_or_default(list(getattr(opts, "wsn_max_fes_list", [])), int(opts.max_fes))
    if benchmark == "WSNLocationMASOIE":
        return _mean_positive_or_default(list(getattr(opts, "masoie_wsn_max_fes_list", [])), int(opts.max_fes))
    if benchmark == "DBOF1F10":
        return _mean_positive_or_default(list(getattr(opts, "dbof1f10_max_fes_list", [])), int(opts.max_fes))
    if benchmark == "CDOCompetition":
        return _mean_positive_or_default(list(getattr(opts, "cdo_max_fes_list", [])), int(opts.max_fes))
    if benchmark in {"CDOBenchF1F14", "CDOBenchF1F15"}:
        return _mean_positive_or_default(list(getattr(opts, "cdo_bench_max_fes_list", [])), int(opts.max_fes))
    return int(opts.max_fes)


OBJECTIVE_SPLIT_MODES = {
    "wsn_objective",
    "dbo_objective",
    "cdo_objective",
    "masoie_wsn_objective",
}

OBJECTIVE_SPLIT_CONSENSUS_CHOICES = [
    "full_mean",
    "none",
    "graph_mean",
    "relaxed_graph_mean",
    "ccsa_lite",
    "masoie_lite",
    "ccsa_masoie_lite",
]


def resolve_mappo_obs_dim(opts) -> int:
    mode = str(getattr(opts, "mappo_env_mode", "variable_cc")).lower()
    neighbor_mode = str(
        getattr(opts, "objective_split_neighbor_obs_mode", "auto")
    ).lower()
    if neighbor_mode == "auto":
        neighbor_mode = (
            "full"
            if bool(int(getattr(opts, "objective_split_neighbor_obs", 0)))
            else "none"
        )
    extra_dims = {
        "none": 0,
        "improve": 1,
        "improve_disagreement": 2,
        "full": 3,
    }
    if neighbor_mode not in extra_dims:
        raise ValueError(
            f"Unsupported objective_split_neighbor_obs_mode: {neighbor_mode}."
        )
    if mode in OBJECTIVE_SPLIT_MODES:
        base_dim = 16
        obs_dim = base_dim + int(extra_dims[neighbor_mode])
        state_comm_mode = str(
            getattr(opts, "objective_split_state_comm_mode", "none")
        ).lower()
        if state_comm_mode != "none":
            n_opt = _count_str_list(
                getattr(opts, "optimizer_candidates", None),
                default=4,
            )
            state_msg_dim = 1 + int(n_opt) + 5
            obs_dim += state_msg_dim
            if int(getattr(opts, "objective_split_state_comm_include_delta", 0)):
                obs_dim += state_msg_dim
        if int(getattr(opts, "objective_split_candidate_history_obs_enable", 0)):
            obs_dim += 5
        # σ 跨事件继承（14 卡）：ON 且 local_only 时 env 的 _build_obs 会 append
        # 5 维（上一事件 optimizer one-hot ×4 + inherit 预览 σ ×1）——本公式必须
        # 用同一开关同一条件同步 +5，保证签名/网络/checkpoint/评估四处一致；
        # OFF 时不变（零漂移）。
        if (
            int(getattr(opts, "objective_split_sigma_inherit_enable", 0))
            and str(
                getattr(opts, "objective_split_information_mode", "none")
            ).lower()
            == "local_only"
        ):
            obs_dim += 5
            if int(getattr(opts, "objective_split_sigma_state_obs_enable", 0)):
                obs_dim += 3
        return obs_dim
    return 16


def _resolve_latest_checkpoint(resume_dir: str, resume_keyword: str = "") -> str:
    p = Path(str(resume_dir)).expanduser().resolve()
    if (not p.exists()) or (not p.is_dir()):
        raise FileNotFoundError(f"Resume directory does not exist: {p}")
    cands = [x for x in p.rglob("*.pt") if x.is_file()]
    if resume_keyword:
        key = str(resume_keyword).strip().lower()
        cands = [x for x in cands if key in str(x).lower()]
    if len(cands) == 0:
        raise FileNotFoundError(f"No checkpoint .pt found under: {p}")
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(cands[0])

def get_options(args=None):
    parser = argparse.ArgumentParser(description="CMAES_PPO")
    parser.add_argument('--note', type=str, default='',
                        help='运行注释，会在保存的 options 文件最上方显示')

    # =======================================================================
    # 1. CMA-ES 与 问题环境设置 (Problem & Environment Settings)
    # =======================================================================
    parser.add_argument('--backbone', default='cmaes', choices=['cmaes'], help='骨干算法选择')
    parser.add_argument('--m', type=int, default=10, help='子种群数量 (subgroups)')
    parser.add_argument('--subspace_dim', type=int, default=100, help='子空间维度')
    parser.add_argument('--sigma', type=float, default=0.3, help='CMA-ES 的步长控制参数 sigma')
    parser.add_argument('--sub_popsize', type=int, default=20, help='每个子种群的大小')
    parser.add_argument('--max_fes', type=int, default=int(2.6e6), help='最大函数迭代次数 (FEs)')
    parser.add_argument('--subFEs', type=int, default=1000, help='每个子种群每次交互的最小迭代次数')
    parser.add_argument('--initFEs', type=int, default=1000, help='初始化的迭代次数')
    parser.add_argument('--output_init_cma_info', type=bool, default=False, help='是否输出初始 CMA 信息')
    
    parser.add_argument('--benchmark_name', default="CDOBenchF1F15",
                        choices=["CDOBenchF1F15", "WSNLocation"],
                        help='Packaged benchmark: CDOBenchF1F15 or WSNLocation')
    parser.add_argument('--divide_method', default="CEC2013LSGO",
                        choices=["CEC2013LSGO", "WSNLocation", "BNS", "OB_nondep"],
                        help='分组方法（兼容旧参数；WSNLocation 仅作为 benchmark_name 的兼容别名）')
    parser.add_argument('--mappo_env_mode', default="cdo_objective",
                        choices=["cdo_objective", "wsn_objective"],
                        help='Objective split mode for the selected packaged benchmark')
    parser.add_argument('--resource_list', type=list, default=[50000, 100000, 200000], help='资源限制列表 (Action 可选范围)')
    parser.add_argument('--action_space', type=int, default=3, help='动作空间维度')
    parser.add_argument('--wsn_data_root', type=str, default='',
                        help='WSN数据目录（包含 source_WSN_location / target_WSN_location）')
    parser.add_argument('--dbof1f10_data_root', type=str, default='',
                        help='DBOF1F10 数据目录（包含 A_Fk/W_Fk/R_Fk/xopt_Fk, k=1..10）')
    parser.add_argument('--dbof1f10_data_mode', type=str, default='tevc2024',
                        choices=['tevc2024', 'cdo20'],
                        help='DBOF1F10 data naming/source mode: tevc2024 keeps historical A_Fk/W_Fk/R_Fk/xopt_Fk; cdo20 uses CDO competition A_20n100D/W_20n/R_100D/xopt_100D.')
    parser.add_argument('--dbof1f10_node_num', type=int, default=20, help='DBOF1F10 节点数量')
    parser.add_argument('--dbof1f10_dim', type=int, default=100, help='DBOF1F10 决策维度')
    parser.add_argument('--dbof1f10_weight', type=float, default=100.0, help='DBOF1F10 线性项权重')
    parser.add_argument('--dbof1f10_lower_bound', type=float, default=-100.0, help='DBOF1F10 决策下界')
    parser.add_argument('--dbof1f10_upper_bound', type=float, default=100.0, help='DBOF1F10 决策上界')
    parser.add_argument('--dbof1f10_max_fes_list', type=str, default='',
                        help='DBOF1F10 每个问题 max_fes 列表（逗号分隔，支持科学计数法）；为空则统一使用max_fes')
    parser.add_argument('--cdo_data_root', type=str, default='',
                        help='CDOCompetition data directory containing A_20n100D/A_40n100D/A_81n100D/W_20n/W_40n/W_81n/R_100D/xopt_100D')
    parser.add_argument('--cdo_max_fes_list', type=str, default='',
                        help='CDOCompetition per-function max_fes list for F1..F28; empty means using max_fes')
    parser.add_argument('--cdo_bench_data_root', type=str, default='',
                        help='CDOBenchF1F14/F1F15 data directory containing A_20n100D/W_20n/R_100D/xopt_100D')
    parser.add_argument('--cdo_bench_max_fes_list', type=str, default='',
                        help='CDOBenchF1F14/F1F15 per-function max_fes list; empty means using max_fes')
    parser.add_argument('--wsn_node_num', type=int, default=20, help='WSN传感器数量')
    parser.add_argument('--wsn_coordinate_dim', type=int, default=3, help='WSN坐标维度')
    parser.add_argument('--wsn_objective_consensus', type=str, default='full_mean',
                        choices=OBJECTIVE_SPLIT_CONSENSUS_CHOICES,
                        help='WSN目标拆分模式的共识方式；当前实验版仅支持full_mean')
    parser.add_argument('--objective_split_consensus', type=str, default='full_mean',
                        choices=OBJECTIVE_SPLIT_CONSENSUS_CHOICES,
                        help='目标函数拆分模式的通用共识方式；当前仅支持full_mean')
    parser.add_argument('--objective_split_consensus_strength', type=float, default=1.0,
                        help='Relaxed graph consensus strength eta in [0,1].')
    parser.add_argument('--objective_split_comm_interval', type=int, default=1,
                        help='Run graph communication every M environment steps.')
    parser.add_argument('--objective_split_comm_rounds', type=int, default=1,
                        help='Fixed number of repeated graph-consensus rounds per communication event; only graph_mean/relaxed_graph_mean repeat in this version.')
    parser.add_argument('--objective_split_comm_force_rounds', type=int, default=0,
                        help='Force actual graph-consensus rounds per communication event when >0. This overrides actor-selected K while keeping the comm action head/signature unchanged.')
    parser.add_argument('--objective_split_comm_action_enable', type=int, default=0, choices=[0, 1],
                        help='Let MAPPO actor choose repeated graph-consensus rounds as an extra action head.')
    parser.add_argument('--mappo_legacy_comm_old_logp_zero', type=int, default=0, choices=[0, 1],
                        help='Compatibility audit only: store zero as the communication head old log-probability, reproducing the pre-c333129 C9o-r2 PPO semantics. Default off.')
    parser.add_argument('--objective_split_comm_round_candidates', type=str, default='1,2,4,8',
                        help='Candidate K values for actor-selected communication rounds, comma-separated.')
    parser.add_argument('--objective_split_comm_action_reduce', type=str, default='max',
                        choices=['max', 'mean_round', 'median', 'min'],
                        help='How to reduce per-agent K actions to one communication-round count for the environment step.')
    parser.add_argument('--objective_split_graph_source', type=str, default='benchmark_w',
                        choices=['benchmark_w', 'ring'],
                        help='Graph topology source: benchmark W or an explicit ring fallback.')
    parser.add_argument('--objective_split_weight_mode', type=str, default='metropolis',
                        choices=['metropolis', 'raw_validated'],
                        help='Build Metropolis weights from topology or use validated benchmark W.')
    parser.add_argument('--objective_split_graph_threshold', type=float, default=1e-12,
                        help='Absolute threshold used to extract graph edges from benchmark W.')
    parser.add_argument('--objective_split_neighbor_obs', type=int, default=0, choices=[0, 1],
                        help='Append previous-round neighbor summaries to actor observations (16 -> 19).')
    parser.add_argument('--objective_split_neighbor_obs_mode', type=str, default='auto',
                        choices=['auto', 'none', 'improve', 'improve_disagreement', 'full'],
                        help='Neighbor observation ablation: O0 none, O1 improve, O2 +disagreement, O3 full.')
    parser.add_argument('--objective_split_state_comm_mode', type=str, default='none',
                        choices=['none', 'full_mean', 'graph_mean'],
                        help='Append compact communicated objective-split state messages: none disables; full_mean uses all-agent mean; graph_mean uses W-weighted neighbor aggregation.')
    parser.add_argument('--objective_split_state_comm_include_delta', type=int, default=0, choices=[0, 1],
                        help='When state communication is enabled, append neighbor_message - own_message after neighbor_message.')
    parser.add_argument('--objective_split_state_comm_include_actual_fes', type=int, default=0, choices=[0, 1],
                        help='Deprecated compatibility flag. Compact state messages do not include actual FEs.')
    parser.add_argument('--objective_split_consensus_reward_weight', type=float, default=0.0,
                        help='Optional consensus-improvement reward weight; zero preserves legacy reward.')
    parser.add_argument('--objective_split_ccsa_momentum_decay', type=float, default=0.8,
                        help='CCSA-lite direction momentum decay in [0,1].')
    parser.add_argument('--objective_split_ccsa_direction_lr', type=float, default=0.5,
                        help='CCSA-lite weight for current merged neighbor direction.')
    parser.add_argument('--objective_split_ccsa_scale_rate', type=float, default=0.2,
                        help='CCSA-lite step-scale sensitivity from direction-momentum norm.')
    parser.add_argument('--objective_split_ccsa_scale_min', type=float, default=0.5,
                        help='CCSA-lite minimum proposal step multiplier.')
    parser.add_argument('--objective_split_ccsa_scale_max', type=float, default=1.5,
                        help='CCSA-lite maximum proposal step multiplier.')
    parser.add_argument('--objective_split_ccsa_positive_improve', type=int, default=1, choices=[0, 1],
                        help='Use max(local_improvement,0) as CCSA-lite direction strength.')
    parser.add_argument('--objective_split_ccsa_momentum_update_mode', type=str, default='lite_only',
                        choices=['off', 'lite_only', 'always'],
                        help='When to maintain ccsa_direction_momentum: off disables it; lite_only preserves historical CCSA-lite-only updates; always also updates it under graph_mean so guide_source=ccsa_direction_momentum can be used without changing consensus.')
    parser.add_argument('--objective_split_masoie_velocity_decay', type=float, default=0.5,
                        help='MASOIE-lite external velocity inertia in [0,1].')
    parser.add_argument('--objective_split_masoie_velocity_scale', type=float, default=1.0,
                        help='MASOIE-lite neighbor-pull scale.')
    parser.add_argument('--objective_split_masoie_velocity_clip_ratio', type=float, default=1.0,
                        help='MASOIE-lite velocity norm clip relative to current mean neighbor-pull norm; <=0 disables clipping.')
    parser.add_argument('--objective_split_rvcpd_arm', type=str, default='off',
                        choices=['off', 'p0', 'p1', 'p2'],
                        help='Receiver-Validated Cooperative Path Dynamics arm: off, cost-matched no-op P0, memoryless P1, or persistent-path P2.')
    parser.add_argument('--objective_split_rvcpd_integration_mode', type=str, default='isolated',
                        choices=['isolated', 'd5_post_commit'],
                        help='RVCPD integration contract. isolated preserves the NT063 clean-backbone experiment; d5_post_commit allows only frozen D5 evaluation with P0/P1 after the D5 actuator commit.')
    parser.add_argument('--objective_split_rvcpd_path_decay', type=float, default=0.8,
                        help='RVCPD persistent path coefficient rho.')
    parser.add_argument('--objective_split_rvcpd_direction_lr', type=float, default=0.5,
                        help='RVCPD current neighbor-direction coefficient eta.')
    parser.add_argument('--objective_split_rvcpd_initial_scale', type=float, default=0.25,
                        help='Initial dimensionless RVCPD probe/commit scale relative to the local geometric step.')
    parser.add_argument('--objective_split_rvcpd_scale_min', type=float, default=0.05,
                        help='Minimum RVCPD scale while a valid cooperative event remains active.')
    parser.add_argument('--objective_split_rvcpd_scale_max', type=float, default=0.5,
                        help='Maximum RVCPD scale.')
    parser.add_argument('--objective_split_rvcpd_scale_growth', type=float, default=1.05,
                        help='Slow multiplicative scale growth after a supported forward event.')
    parser.add_argument('--objective_split_rvcpd_scale_decay', type=float, default=0.5,
                        help='Fast multiplicative scale decay after reverse/no-op/conflict.')
    parser.add_argument('--objective_split_rvcpd_noop_path_decay', type=float, default=0.25,
                        help='Persistent-path decay after a receiver-local no-op.')
    parser.add_argument('--objective_split_rvcpd_ema_decay', type=float, default=0.8,
                        help='EMA decay for receiver-local support/conflict/uncertainty diagnostics.')
    parser.add_argument('--objective_split_rvcpd_probe_min_ratio', type=float, default=0.0005,
                        help='Minimum requested probe radius as a ratio of the search-box diagonal.')
    parser.add_argument('--objective_split_rvcpd_probe_max_ratio', type=float, default=0.05,
                        help='Maximum requested probe radius as a ratio of the search-box diagonal.')
    parser.add_argument('--objective_split_rvcpd_min_log_improve', type=float, default=0.0,
                        help='Minimum stable signed-log receiver-local improvement required to select plus or minus.')
    parser.add_argument('--objective_split_record_comm_cost', type=int, default=1, choices=[0, 1],
                        help='Record communication rounds, messages, and transmitted-float estimates.')
    parser.add_argument('--objective_split_information_mode', type=str, default='legacy_global',
                        choices=['legacy_global', 'local_only'],
                        help='Information boundary for objective-split control. legacy_global preserves historical exact-global observation/reward semantics; local_only restricts control to own local objectives plus explicit communication.')
    parser.add_argument('--objective_split_global_monitor_enable', type=int, default=1, choices=[0, 1],
                        help='Evaluate the detached exact-global report monitor. In local_only mode this may affect only report fields and monitor-call accounting.')
    parser.add_argument('--objective_split_state_comm_cost_mode', type=str, default='auto',
                        choices=['auto', 'legacy_untracked', 'piggyback', 'independent'],
                        help='Communication accounting for compact state messages. auto preserves legacy untracked accounting in legacy_global and uses independent accounting in local_only.')
    parser.add_argument('--objective_split_agent_parallel_workers', type=int, default=1,
                        help='Objective-split local optimizer workers inside one env step. 1 keeps historical serial agent execution; >1 runs agent local optimizers in a process pool.')
    parser.add_argument('--objective_split_event_slot_interleaving_enable', type=int, default=0, choices=[0, 1],
                        help='Enable the E-series event-local K-slot contract: freeze one Actor action, distribute native work across the existing K communication budget, use communication-only slots when native work is underfilled or terminated, exact-recenter after every slot, and discard optimizer state when the slow event ends. Requires objective_split_cmaes_numeric_fail_soft=1. Default 0 preserves the historical one-shot path.')
    parser.add_argument('--objective_split_persistent_sepcmaes_enable', type=int, default=0, choices=[0, 1],
                        help='Enable one detached persistent SepCMAES state bank per agent. Strict local_only research path; default 0 preserves one-shot optimizer behavior.')
    parser.add_argument('--objective_split_persistent_sepcmaes_recenter_max_ratio', type=float, default=0.05,
                        help='Maximum persistent SepCMAES mean recenter shift per event as ratio * mean coordinate range. Shape, paths, and sigma are preserved.')
    parser.add_argument('--objective_split_target_block_field_enable', type=int, default=0, choices=[0, 1],
                        help='Enable the strict-local WSN 3D target-block secant/path/radius fast layer. It uses one fixed CCSA-derived graph event per environment step; default 0 preserves the historical path.')
    parser.add_argument('--objective_split_target_block_dual_clock_enable', type=int, default=0, choices=[0, 1],
                        help='Enable NT074 generation-synchronous dual-clock execution: hold one slow action, advance persistent SepCMAES by exactly one complete generation per fast tick, then run one target-block communication/commit event. Default 0 preserves NT073 and historical paths.')
    parser.add_argument('--objective_split_target_block_dual_clock_commit_lock_enable', type=int, default=0, choices=[0, 1],
                        help='Opt in to commit-locked persistent SepCMAES means under target-block dual-clock execution. Each cooperative commit becomes the exact next-generation search center while covariance, sigma, and evolution paths persist. Default 0 preserves NT074/NT075 bounded-recenter semantics.')
    parser.add_argument('--objective_split_target_block_commit_credit_mode', type=str, default='off',
                        choices=['off', 'path', 'scale', 'joint'],
                        help='NT078 commit-conditioned persistent SepCMAES credit under dual-clock commit lock. path reconciles both evolution paths with the actual cooperative commit; scale only contracts sigma to the accepted step scale; joint applies both. Default off preserves NT077 optimization dynamics.')
    parser.add_argument('--objective_split_target_block_dormancy_recovery_enable', type=int, default=0, choices=[0, 1],
                        help='Enable NT079 strict-local selective recovery for dormant WSN target blocks. Requires the fixed NT078 scale-credit kernel. Default 0 preserves NT078 exactly.')
    parser.add_argument('--objective_split_target_block_direction_shadow_enable', type=int, default=0, choices=[0, 1],
                        help='Enable NT080 strict-local shadow probes for alternative directions on NT079-active WSN target blocks. Physical calls and response communication are explicitly audited, while reported FEs, commits, and optimizer state remain unchanged.')
    parser.add_argument('--objective_split_target_block_challenge_response_mode', type=str, default='off', choices=['off', 'shadow', 'actuate'],
                        help='NT081 block-addressed one-hop challenge/response on NT079-active WSN blocks. shadow audits detached local probes; actuate charges all decision probes to reported FEs and replaces only qualified dormant-block path commits.')
    parser.add_argument('--objective_split_early_stop_mode', type=str, default='none',
                        choices=['none', 'mean_disagreement', 'max_edge_disagreement', 'consensus_shift', 'masoie_velocity'],
                        help='Optional native-style early termination for objective-split envs. none keeps historical max_fes-only stopping.')
    parser.add_argument('--objective_split_early_stop_threshold', type=float, default=1e-10,
                        help='Threshold for objective_split_early_stop_mode. Smaller means stricter convergence.')
    parser.add_argument('--objective_split_early_stop_patience', type=int, default=1,
                        help='Require this many consecutive checks below threshold before early stopping.')
    parser.add_argument('--objective_split_early_stop_check_interval', type=int, default=1,
                        help='Check early-stop condition every K env steps. Use 100 to mimic CCSA-style sparse checks.')
    parser.add_argument('--objective_split_early_stop_min_steps', type=int, default=0,
                        help='Do not early-stop before this many env steps, even if the metric is below threshold.')
    parser.add_argument('--objective_split_optimizer_guide_enable', type=int, default=0, choices=[0, 1],
                        help='Enable cooperative optimizer guide for objective-split envs. Default 0 preserves historical behavior.')
    parser.add_argument('--objective_split_optimizer_guide_source', type=str, default='neighbor_improve_direction',
                        choices=['neighbor_improve_direction', 'ccsa_direction_momentum'],
                        help='Source of optimizer guide directions passed into bottom optimizers.')
    parser.add_argument('--objective_split_optimizer_guide_strength', type=float, default=0.5,
                        help='Fixed optimizer-guide strength alpha. For directional injection, candidates use mean +/- alpha*sigma*guide.')
    parser.add_argument('--objective_split_optimizer_guide_strength_schedule', type=str, default='fixed',
                        choices=['fixed', 'disagreement', 'budget'],
                        help='Guide strength schedule. fixed keeps objective_split_optimizer_guide_strength; disagreement scales it by the previous post-consensus mean disagreement; budget anneals it linearly to zero over the first 50%% of the budget (strength * max(0, 1 - elapsed_ratio/0.5)).')
    parser.add_argument('--objective_split_sigma_inherit_enable', type=int, default=0, choices=[0, 1],
                        help='Enable per-agent cross-event sigma inheritance (evolved-sigma slot, card 14). 0 keeps historical event-local sigma semantics; 1 upgrades the inherit config action to reuse the previous same-optimizer event sigma (with switch/signature reset). Default off.')
    parser.add_argument('--objective_split_sigma_validity_gate_enable', type=int, default=0, choices=[0, 1],
                        help='When sigma inheritance is on, reject an inherited sigma whose previous event commit displacement exceeds ratio * effective scale (stale-scale gate). Dimension-consistent comparison: commit center-shift RMS (= L2 norm / sqrt(D)) vs the optimizer coordinate-space effective scale RMS (sigma x covariance/diagonal axis RMS). 0 keeps the plain signature-only inheritance; 1 enables the commit-displacement gate. Default off.')
    parser.add_argument('--objective_split_sigma_validity_commit_ratio', type=float, default=3.0,
                        help='Stale-scale gate threshold: reject inheritance when the previous event max center-shift RMS exceeds this ratio times the effective scale RMS. Falls back to ratio * sigma (i.e. L2 norm > ratio * sigma * sqrt(D)) when no effective scale is available.')
    parser.add_argument('--objective_split_mmes_state_transition_mode', type=str, default='native',
                        choices=['native', 'neutralize_credit'],
                        help='MMES post-commit credit mode: native already skips the first stale y_bak comparison after a nonzero center shift; neutralize_credit also skips later paired comparisons within that slot. Diagnostic only; default native.')
    parser.add_argument('--objective_split_mmes_ratio_success_mode', type=str, default='native', choices=['native', 'attenuate'],
                        help='Experimental MMES accumulated success statistic retention at center commit using the selected A-to-C metric. Default native; cannot combine with neutralize_credit.')
    parser.add_argument('--objective_split_mmes_ratio_success_metric', type=str, default='sigma', choices=['sigma', 'full'],
                        help='Experimental MMES w relocation metric: sigma proxy (default) or full pre-commit q/v sampling geometry. Only used with attenuate mode.')
    parser.add_argument('--objective_split_mmes_ratio_success_strength', type=float, default=0.1,
                        help='Experimental MMES success statistic attenuation strength in [0,1]: retention=1-strength*r/(1+r). Default 0.1; no direction-memory or performance claim.')
    parser.add_argument('--objective_split_mmes_ratio_direction_mode', type=str, default='native', choices=['native', 'attenuate'],
                        help='Experimental MMES path p retention at A-to-C commit, measured in the full q/v sampling geometry. Default native; may combine with w attenuation when w uses full metric.')
    parser.add_argument('--objective_split_mmes_ratio_direction_strength', type=float, default=0.1,
                        help='Experimental MMES path p attenuation strength in [0,1]: retention=1-strength*r/(1+r). Default 0.1.')
    parser.add_argument('--objective_split_sigma_state_obs_enable', type=int, default=0, choices=[0, 1],
                        help='Add 3 per-agent sigma-state observation fields (sigma chain age, log scale vs optimizer default, current candidate validity code) when sigma inheritance is on, all computed on the same time base as the sigma preview. Requires objective_split_sigma_inherit_enable=1. Default off.')
    parser.add_argument('--objective_split_vkd_ps_outlet_mode', type=str, default='native',
                        choices=['native', 'sigma', 'shape', 'both'],
                        help='Diagnostic-only VKD ps consumer mode: native keeps both consumers; sigma isolates ps->sigma; shape isolates ps->hsig/shape; both isolates both. Default native preserves historical behavior.')
    parser.add_argument('--objective_split_vkd_boundary_update_mode', type=str, default='native',
                        choices=['native', 'candidate_a'],
                        help='VKD boundary update mode: native uses raw sampled steps; candidate_a updates internal state from the clipped positions that were scored. Default native preserves historical behavior.')
    parser.add_argument('--objective_split_vkd_ratio_ps_mode', type=str, default='native', choices=['native', 'attenuate'],
                        help='Experimental VKD ps retention at event-slot center commit using A-to-C relocation in native VKD search units. Default native; cannot combine with the always-on ps outlet diagnostic.')
    parser.add_argument('--objective_split_vkd_ratio_ps_strength', type=float, default=0.1,
                        help='Experimental VKD ps attenuation strength in [0,1]: retention=1-strength*r/(1+r). Default 0.1. No full repair claim.')
    parser.add_argument('--objective_split_cma_sep_ratio_path_mode', type=str, default='native',
                        choices=['native', 'step_path', 'shape_path', 'both_paths'],
                        help='Experimental CMAES/SepCMAES path retention after event-slot recentering, using the relative shift and the separate strength option; no direct sigma or covariance edit. Default native.')
    parser.add_argument('--objective_split_cmaes_ratio_path_metric', type=str, default='rms',
                        choices=['rms', 'directional'],
                        help='Experimental CMAES path ratio metric. rms preserves prior trial; directional measures A-to-C displacement along current eigenvector sampling axes. Default rms.')
    parser.add_argument('--objective_split_sepcmaes_ratio_path_metric', type=str, default='rms',
                        choices=['rms', 'directional'],
                        help='Experimental SepCMAES path ratio metric. rms preserves prior trial; directional measures A-to-C displacement along current diagonal sampling axes. Default rms.')
    parser.add_argument('--objective_split_cma_sep_ratio_path_strength', type=float, default=1.0,
                        help='Experimental ratio-path attenuation strength in [0,1]: retention=1-strength*r/(1+r). 1 reproduces the original trial; 0.1 caps each nonzero slot reduction below 10%%. Default 1.')
    parser.add_argument('--objective_split_optimizer_guide_strength_scale_min', type=float, default=1.0,
                        help='Minimum multiplier for disagreement guide-strength schedule.')
    parser.add_argument('--objective_split_optimizer_guide_strength_scale_max', type=float, default=1.0,
                        help='Maximum multiplier for disagreement guide-strength schedule.')
    parser.add_argument('--objective_split_optimizer_guide_disagreement_low', type=float, default=0.0,
                        help='Mean-disagreement value mapped to scale_min for disagreement guide-strength schedule.')
    parser.add_argument('--objective_split_optimizer_guide_disagreement_high', type=float, default=1.0,
                        help='Mean-disagreement value mapped to scale_max for disagreement guide-strength schedule.')
    parser.add_argument('--objective_split_optimizer_guide_injection_pairs', type=int, default=1,
                        help='Number of guide +/- sample pairs for optimizers that inject explicit directional candidates.')
    parser.add_argument('--objective_split_optimizer_guide_use_negative_pair', type=int, default=1, choices=[0, 1],
                        help='When injecting guide candidates, also inject the negative direction.')
    parser.add_argument('--objective_split_optimizer_guide_min_improve', type=float, default=0.0,
                        help='Minimum positive local-improvement signal for contributing to neighbor_improve_direction guide.')
    parser.add_argument('--objective_split_optimizer_guide_gate_enable', type=int, default=0, choices=[0, 1],
                        help='Enable confidence gate before passing optimizer guide into bottom optimizers. Default 0 preserves old runs.')
    parser.add_argument('--objective_split_optimizer_guide_min_norm', type=float, default=0.0,
                        help='Guide-source norm threshold for the confidence gate. Only used when guide_gate_enable=1.')
    parser.add_argument('--objective_split_optimizer_guide_min_alignment', type=float, default=-1.0,
                        help='Minimum alignment between local proposal direction and guide direction for the confidence gate. Range is clipped to [-1,1].')
    parser.add_argument('--objective_split_optimizer_guide_apply_optimizers', type=str, default='all',
                        help='Comma list of optimizers that consume guide directions, or all/none. Example: cmaes,sepcmaes.')
    parser.add_argument('--objective_split_optimizer_guide_mix_strength', type=float, default=-1.0,
                        help='Optional VKD/MMES internal guide mix strength beta. Negative reuses objective_split_optimizer_guide_strength.')
    parser.add_argument('--objective_split_guide_replacement_mode', type=str, default='off',
                        choices=['off', 'shadow', 'p0', 'p1'],
                        help='Receiver-validated historical-guide replacement arm. shadow observes without suppressing the old guide; P0/P1 suppress it and pay matched probes; only P1 commits a receiver-local improvement.')
    parser.add_argument('--objective_split_guide_replacement_scope', type=str, default='guide_optimizers',
                        choices=['guide_optimizers', 'all'],
                        help='Agents eligible for guide replacement: only optimizers in the historical guide apply list, or every optimizer as a separate heterogeneous extension.')
    parser.add_argument('--objective_split_collective_guide_mode', type=str, default='off',
                        choices=['off', 'shadow', 'rate_matched_null', 'collective_veto'],
                        help='Shared-base collective historical-guide validation. shadow records a hypothetical gate; rate_matched_null uses a deterministic schedule at an externally frozen veto rate; collective_veto suppresses the eligible historical guide only when common-base ternary votes have negative sum.')
    parser.add_argument('--objective_split_collective_guide_null_veto_rate', type=float, default=0.0,
                        help='Externally frozen veto probability for rate_matched_null. Must be in [0,1]; ignored by other modes and never estimated online.')
    parser.add_argument('--objective_split_anchor_enable', type=int, default=0, choices=[0, 1],
                        help='Enable consensus-anchor guidance for objective-split bottom optimizers. Default 0 preserves historical behavior.')
    parser.add_argument('--objective_split_anchor_source', type=str, default='consensus',
                        choices=['consensus', 'mean', 'self'],
                        help='Anchor point source. consensus uses graph/full consensus of current agent states; mean uses global mean; self disables movement.')
    parser.add_argument('--objective_split_anchor_apply_optimizers', type=str, default='all',
                        help='Comma list of optimizers that consume consensus anchors, or all/none.')
    parser.add_argument('--objective_split_anchor_strength', type=float, default=0.05,
                        help='Anchor mean-pull strength for CMAES/SepCMAES and fallback mix strength for MMES/VKD.')
    parser.add_argument('--objective_split_anchor_mix_strength', type=float, default=-1.0,
                        help='Optional MMES/VKD anchor direction mix strength. Negative reuses objective_split_anchor_strength.')
    parser.add_argument('--objective_split_anchor_sample_ratio', type=float, default=0.25,
                        help='Fraction of CMAES/SepCMAES offspring slots replaced by anchor interpolation candidates. <=0 disables anchor sample injection.')
    parser.add_argument('--objective_split_anchor_sample_clip_ratio', type=float, default=0.25,
                        help='Clip anchor displacement to ratio * mean(search_range). <=0 disables this cap.')
    parser.add_argument('--objective_split_anchor_mean_pull', type=int, default=1, choices=[0, 1],
                        help='Apply conservative post-recombination mean pull toward anchor in CMAES/SepCMAES.')
    parser.add_argument('--objective_split_anchor_sample_injection', type=int, default=1, choices=[0, 1],
                        help='Inject explicit anchor candidates into CMAES/SepCMAES samples.')
    parser.add_argument('--objective_split_committee_mode', type=str, default='off',
                        choices=['off', 'shadow', 'guide'],
                        help='Neighborhood cross-validation mode: off disables it; shadow records counterfactual candidates without changing behavior; guide fuses the verified direction into the next-step optimizer guide.')
    parser.add_argument('--objective_split_committee_selection', type=str, default='target_block',
                        choices=['whole', 'target_block', 'both'],
                        help='Committee candidate construction: whole selects one complete neighbor proposal; target_block selects one source per WSN target; both is shadow-only.')
    parser.add_argument('--objective_split_committee_mix_strength', type=float, default=1.0,
                        help='Maximum confidence-weighted committee contribution when fusing with the base optimizer guide.')
    parser.add_argument('--objective_split_committee_acceptance_mode', type=str, default='off',
                        choices=['off', 'report_improve'],
                        help='Event-level committee guide acceptance: off preserves the original C13 fusion; report_improve globally evaluates the verified report and applies the committee direction only when it strictly improves the normal report.')
    parser.add_argument('--objective_split_committee_acceptance_min_log_improve', type=float, default=0.0,
                        help='Minimum strict signed-log improvement required by committee acceptance_mode=report_improve.')
    parser.add_argument('--objective_split_committee_shadow_global_eval', type=int, default=1, choices=[0, 1],
                        help='In shadow mode, globally score counterfactual committee candidates for diagnostics only. This never affects decisions or reported FEs.')
    parser.add_argument('--objective_split_candidate_response_mode', type=str, default='off',
                        choices=['off', 'shadow', 'actuate'],
                        help='D3-P local candidate-response mode. shadow records counterfactual local-only decisions; actuate commits a bounded verified base-state move. Default off preserves historical behavior.')
    parser.add_argument('--objective_split_candidate_generator', type=str, default='spsa_target',
                        choices=['spsa_target', 'multisecant_target', 'multisecant_hybrid'],
                        help='Local-only candidate generator: legacy SPSA, probe-secant multisecant with calibration-only cold start, or multisecant with SPSA fallback.')
    parser.add_argument('--objective_split_candidate_multisecant_history_size', type=int, default=5,
                        help='Per-agent, per-target probe-secant ring-buffer length.')
    parser.add_argument('--objective_split_candidate_multisecant_min_rank', type=int, default=3,
                        help='Minimum local secant rank required for a multisecant direction; clipped to the target coordinate dimension.')
    parser.add_argument('--objective_split_candidate_multisecant_rank_tolerance', type=float, default=1e-6,
                        help='Relative singular-value threshold used to determine multisecant effective rank.')
    parser.add_argument('--objective_split_candidate_multisecant_condition_max', type=float, default=1e4,
                        help='Maximum accepted multisecant retained-subspace condition proxy.')
    parser.add_argument('--objective_split_candidate_multisecant_center_distance_max_ratio', type=float, default=0.10,
                        help='Maximum history-center distance from the current target block, as search-span ratio times sqrt(coordinate_dim).')
    parser.add_argument('--objective_split_candidate_multisecant_max_age', type=int, default=8,
                        help='Maximum candidate-response event age retained by multisecant fitting; zero disables age filtering.')
    parser.add_argument('--objective_split_candidate_multisecant_gradient_min_norm', type=float, default=1e-8,
                        help='Minimum recovered normalized residual-gradient norm required to activate a multisecant block.')
    parser.add_argument('--objective_split_candidate_probe_scale', type=float, default=0.25,
                        help='D3-P probe radius as a fraction of the target-block trust radius.')
    parser.add_argument('--objective_split_candidate_trust_scale', type=float, default=0.25,
                        help='D3-P trust radius as a fraction of the optimizer proposal displacement.')
    parser.add_argument('--objective_split_candidate_trust_min_ratio', type=float, default=0.002,
                        help='D3-P minimum target-block trust radius as a fraction of the search span.')
    parser.add_argument('--objective_split_candidate_trust_max_ratio', type=float, default=0.05,
                        help='D3-P maximum target-block trust radius as a fraction of the search span.')
    parser.add_argument('--objective_split_candidate_probe_min_ratio', type=float, default=0.0005,
                        help='D3-P minimum local probe radius as a fraction of the search span.')
    parser.add_argument('--objective_split_candidate_probe_max_ratio', type=float, default=0.01,
                        help='D3-P maximum local probe radius as a fraction of the search span.')
    parser.add_argument('--objective_split_candidate_confidence_min', type=float, default=1e-4,
                        help='Minimum normalized two-sided local residual response needed to emit a target block.')
    parser.add_argument('--objective_split_candidate_support_min', type=float, default=0.60,
                        help='Minimum weighted neighborhood verifier support required to accept a target block.')
    parser.add_argument('--objective_split_candidate_actuator_beta', type=float, default=0.50,
                        help='Bounded fraction of an accepted D3-P target-block shift committed to the next optimizer base state.')
    parser.add_argument('--objective_split_candidate_history_obs_enable', type=int, default=0, choices=[0, 1],
                        help='Append five receiver-local previous candidate-response fields to each actor observation.')
    parser.add_argument('--objective_split_candidate_actuator_action_enable', type=int, default=0, choices=[0, 1],
                        help='Let MAPPO choose one candidate actuator strength per agent as the final hierarchical action head.')
    parser.add_argument('--objective_split_candidate_actuator_candidates', type=str, default='0.0,0.25,0.5',
                        help='Per-agent candidate actuator beta choices for the MAPPO actuator head.')
    parser.add_argument('--objective_split_candidate_actuator_initial_probs', type=str, default='0.2,0.7,0.1',
                        help='Initial actuator-head categorical prior, aligned with actuator candidates.')
    parser.add_argument('--objective_split_candidate_actuator_forced_action', type=int, default=-1,
                        help='Force the environment actuator action index when nonnegative; intended only for qualification/evaluation.')
    parser.add_argument('--objective_split_d5_two_stage_actuator_enable', type=int, default=0, choices=[0, 1],
                        help='Enable the D5 verifier-after-prepare actuator protocol. Default 0 preserves the legacy one-stage step path.')
    parser.add_argument('--objective_split_d5_actuator_reward_mode', type=str, default='mixed',
                        choices=['mixed', 'local_credit'],
                        help='D5 actuator training credit: current mixed reward or zero-extra-FEs base-own-local commit credit.')
    parser.add_argument('--objective_split_candidate_shadow_global_eval_interval', type=int, default=20,
                        help='Optional detached exact-global shadow sampling interval. Zero disables it; sampled values never affect control or reported FEs.')
    parser.add_argument('--objective_split_optimizer_guide_numeric_guard', type=int, default=0, choices=[0, 1],
                        help='Enable finite checks, bound repair, and sigma clipping for guided CMAES/SepCMAES. Default 0 preserves historical behavior.')
    parser.add_argument('--objective_split_cmaes_numeric_fail_soft', type=int, default=0, choices=[0, 1],
                        help='Stop only a numerically invalid CMAES local block and return its evaluated best candidate. Default 0 preserves historical failure behavior and finite-path dynamics.')
    parser.add_argument('--objective_split_optimizer_numeric_telemetry', type=int, default=0, choices=[0, 1],
                        help='Enable generation-level CMAES/SepCMAES numeric telemetry. Default 0 has no file I/O and preserves historical behavior.')
    parser.add_argument('--objective_split_optimizer_numeric_telemetry_dir', type=str, default='',
                        help='Optional JSONL output directory for optimizer numeric telemetry. Empty keeps enabled records in memory only.')
    parser.add_argument('--objective_split_optimizer_numeric_counter', type=int, default=0, choices=[0, 1],
                        help='Enable bounded CMA numeric counter audit without per-generation arrays.')
    parser.add_argument('--objective_split_optimizer_numeric_counter_dir', type=str, default='',
                        help='Directory for bounded CMA numeric counter audit JSONL.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics', type=int, default=0, choices=[0, 1],
                        help='Enable bounded CMAES/SepCMAES forensic ring buffers (64 generations / 16 commits) that flush a window only on an evidence trigger. Diagnostic only; default 0 changes nothing.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_dir', type=str, default='',
                        help='Optional JSONL output directory for forensic windows. Empty defaults to data_save_dir/test_dir (next to the run logs).')
    parser.add_argument('--objective_split_vkd_origin_trace', type=int, default=0, choices=[0, 1],
                        help='Bounded VKD origin/commit/failure trace; default off.')
    parser.add_argument('--objective_split_vkd_origin_trace_dir', type=str, default='',
                        help='Output directory for opt-in VKD origin trace.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_log10', type=float, default=2.0,
                        help='Sigma-jump window gate in log10 (>=2.0 means one generation x100). Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_max_windows', type=int, default=1,
                        help='Max jump windows per process per optimizer. Default 1 = historical once-per-process; set higher to keep the first trigger plus the rolling latest.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_alpha_gate', type=int, default=0, choices=[0, 1],
                        help='Let |alpha|>=1 alone open a VKD jump window. Default 0, because normal extremes reach 1.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_early_write', type=int, default=0, choices=[0, 1],
                        help='Write the two-record trigger line before the following generation exists, so a bounded run can stop early. Default 0 = historical write-once-at-next.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_early_milestone_log10', type=float, default=0.5,
                        help='Cumulative log10 sigma growth that selects the EARLY forensic window (0.5 ~ x3.2 total). Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_jump_mid_milestone_log10', type=float, default=2.0,
                        help='Cumulative log10 sigma growth that selects the MID forensic window (2.0 ~ x100 total). Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_target_function', type=int, default=-1,
                        help='Optional target function id for forensic capture; -1 means no filter. Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_target_seed', type=int, default=-1,
                        help='Optional target seed for forensic capture; -1 means no filter. Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_numeric_forensics_target_agent', type=int, default=-1,
                        help='Optional target agent id for forensic capture; -1 means no filter. Diagnostic only.')
    parser.add_argument('--objective_split_optimizer_guide_sigma_exp_clip', type=float, default=20.0,
                        help='When numeric guard is enabled, clip CMAES/SepCMAES sigma-update exponent to [-value, value].')
    parser.add_argument('--objective_split_optimizer_guide_sigma_clip_ratio', type=float, default=0.5,
                        help='When numeric guard is enabled, clip sigma to at most ratio * mean(search_range). Use <=0 to disable this cap.')
    parser.add_argument('--objective_split_optimizer_guide_sample_clip_ratio', type=float, default=0.0,
                        help='Clip only guided CMAES/SepCMAES sample displacement to ratio * mean(search_range). <=0 disables this bounded-sample guard.')
    parser.add_argument('--objective_split_optimizer_guide_internal_mode', type=str, default='off',
                        choices=['off', 'mean', 'mean_path', 'mean_path_covdiag'],
                        help='Internal optimizer guide mode for CMAES/SepCMAES. Default off preserves historical behavior.')
    parser.add_argument('--objective_split_optimizer_guide_internal_apply_optimizers', type=str, default='cmaes,sepcmaes',
                        help='Comma list of optimizers that consume internal guide, or all/none. First implementation targets cmaes,sepcmaes.')
    parser.add_argument('--objective_split_optimizer_guide_internal_mean_lr', type=float, default=0.0,
                        help='Extra mean-step learning rate for internal optimizer guide.')
    parser.add_argument('--objective_split_optimizer_guide_internal_path_lr', type=float, default=0.0,
                        help='Extra covariance-path learning rate for internal optimizer guide. CMAES only in the first implementation.')
    parser.add_argument('--objective_split_optimizer_guide_internal_cov_lr', type=float, default=0.0,
                        help='Reserved covariance internal guide learning rate. Kept at 0 in the first implementation.')
    parser.add_argument('--objective_split_optimizer_guide_internal_agree_cos_min', type=float, default=-0.25,
                        help='Minimum cosine between selected optimizer step and guide for applying internal guide.')
    parser.add_argument('--objective_split_optimizer_guide_internal_max_step_ratio', type=float, default=0.05,
                        help='Absolute cap for internal guide mean step as ratio * mean(search_range).')
    parser.add_argument('--objective_split_optimizer_guide_internal_max_rel_step', type=float, default=0.5,
                        help='Relative cap for internal guide mean step against the optimizer own mean step.')
    parser.add_argument('--objective_split_optimizer_guide_internal_path_max_rel_norm', type=float, default=0.5,
                        help='Relative cap for internal guide path increment against the selected weighted direction norm.')
    parser.add_argument('--objective_split_optimizer_guide_internal_cov_rank1_clip', type=float, default=0.02,
                        help='Reserved max covariance rank-one guide update weight.')
    parser.add_argument('--objective_split_optimizer_guide_internal_disable_sample_injection', type=int, default=0, choices=[0, 1],
                        help='When internal guide is enabled, disable old CMAES/SepCMAES explicit guide sample injection.')
    parser.add_argument('--objective_split_collab_action_enable', type=int, default=0, choices=[0, 1],
                        help='Let MAPPO actor choose how cooperative optimizer-guide information is consumed. Default 0 preserves historical behavior.')
    parser.add_argument('--objective_split_collab_modes', type=str, default='consensus,self,leader,soft_diversify',
                        help='Comma-separated collaboration-mode action candidates. First candidate should preserve historical guide behavior.')
    parser.add_argument('--objective_split_collab_soft_diversify_scale', type=float, default=0.25,
                        help='Guide-strength multiplier for collab mode soft_diversify.')
    parser.add_argument('--objective_split_collab_leader_fallback', type=str, default='consensus',
                        choices=['consensus', 'off'],
                        help='Fallback when leader mode has no better graph neighbor direction.')
    parser.add_argument('--objective_split_guide_scale_action_enable', type=int, default=0, choices=[0, 1],
                        help='Let MAPPO actor choose an extra multiplier for optimizer-guide strength. Default 0 preserves historical behavior.')
    parser.add_argument('--objective_split_guide_scale_candidates', type=str, default='1.0,0.5,0.75,1.25',
                        help='Comma-separated guide-scale action candidates. First candidate should preserve historical guide strength.')
    parser.add_argument('--wsn_target_num_list', type=str, default='10,20,30,40,50',
                        help='WSN问题目标数列表，逗号分隔；func_id=1..K 对应该列表')
    parser.add_argument('--wsn_max_fes_list', type=str, default='',
                        help='WSN每个问题的max_fes列表（逗号分隔，支持科学计数法）；为空则统一使用max_fes')
    parser.add_argument('--wsn_measurement', type=str, default='RSS', choices=['RSS'],
                        help='WSN测量模型')
    parser.add_argument('--wsn_metric_mode', type=str, default='ccsa_readme',
                        choices=['ccsa_readme', 'normalized'],
                        help='WSN目标聚合口径：ccsa_readme(与复现项目一致) / normalized(除以node*target)')
    parser.add_argument('--wsn_noisy', type=int, default=1, help='WSN观测是否加噪（1是，0否）')
    parser.add_argument('--wsn_noisy_degree', type=float, default=0.1, help='WSN观测噪声标准差')
    parser.add_argument('--wsn_lower_bound', type=float, default=-100.0, help='WSN决策下界')
    parser.add_argument('--wsn_upper_bound', type=float, default=100.0, help='WSN决策上界')
    parser.add_argument('--wsn_grouping_mode', type=str, default='equal_split', choices=['equal_split', 'target_block'],
                        help='WSN变量分组模式')
    parser.add_argument('--wsn_id_offset', type=int, default=100,
                        help='Mixed 模式下 WSN 问题ID偏移量：WSN第k题映射为 wsn_id_offset+k')
    parser.add_argument('--masoie_wsn_node_num', type=int, default=20, help='MASOIE-WSN传感器数量')
    parser.add_argument('--masoie_wsn_target_num_list', type=str, default='5',
                        help='MASOIE-WSN问题目标数列表，逗号分隔；func_id=1..K 对应该列表')
    parser.add_argument('--masoie_wsn_max_fes_list', type=str, default='',
                        help='MASOIE-WSN每个问题max_fes列表（逗号分隔，支持科学计数法）；为空则统一使用max_fes')
    parser.add_argument('--masoie_wsn_space_size', type=float, default=100.0, help='MASOIE-WSN空间边长')
    parser.add_argument('--masoie_wsn_noise_std', type=float, default=2.0, help='MASOIE-WSN测量噪声参数')
    parser.add_argument('--masoie_wsn_id_offset', type=int, default=200,
                        help='Mixed 模式下 MASOIE-WSN 问题ID偏移量：第k题映射为 masoie_wsn_id_offset+k')
    # Repro-style MMES knobs (subproblem optimizer)
    parser.add_argument('--subopt_mmes_m', type=int, default=-1, help='MMES m；<=0表示使用默认公式')
    parser.add_argument('--subopt_mmes_c_c', type=float, default=-1.0, help='MMES c_c；<=0表示使用默认公式')
    parser.add_argument('--subopt_mmes_ms', type=int, default=-1, help='MMES ms (mixing strength)；<=0表示自动/按档位')
    parser.add_argument('--subopt_mmes_c_s', type=float, default=0.3, help='MMES c_s')
    parser.add_argument('--subopt_mmes_a_z', type=float, default=0.05, help='MMES a_z')
    parser.add_argument('--subopt_mmes_distance', type=int, default=0, help='MMES distance；<=0表示自动')
    parser.add_argument('--subopt_mmes_c_a', type=float, default=-1.0, help='MMES c_a；<=0表示使用默认3.8/d')
    parser.add_argument('--subopt_mmes_gamma', type=float, default=-1.0, help='MMES gamma；<=0表示自动')
    parser.add_argument('--subopt_mmes_n_individuals', type=int, default=-1, help='MMES n_individuals；<=0表示自动')
    parser.add_argument('--subopt_vkd_n_individuals', type=int, default=-1, help='VKD n_individuals；<=0表示自动')
    parser.add_argument('--subopt_vkd_k_init', type=int, default=-1, help='VKD k_init；<0表示自动')
    parser.add_argument('--subopt_vkd_kmax', type=int, default=-1, help='VKD kmax；<0表示自动')
    parser.add_argument('--subopt_cmaes_n_individuals', type=int, default=-1, help='CMAES n_individuals；<=0表示自动')
    parser.add_argument('--subopt_sepcmaes_n_individuals', type=int, default=-1, help='SepCMAES n_individuals；<=0表示自动')
    parser.add_argument('--subopt_cmaes_cov_lr_scale', type=float, default=-1.0, help='CMAES cov_lr_scale；<=0表示按动作档位')
    parser.add_argument('--subopt_cmaes_c_s_scale', type=float, default=-1.0, help='CMAES c_s_scale；<=0表示按动作档位')
    parser.add_argument('--subopt_sepcmaes_c_cov_scale', type=float, default=-1.0, help='SepCMAES c_cov_scale；<=0表示按动作档位')
    parser.add_argument('--subopt_sepcmaes_c_s_scale', type=float, default=-1.0, help='SepCMAES c_s_scale；<=0表示按动作档位')
    parser.add_argument('--subopt_mmes_ms_levels', type=str, default='2,4,6',
                        help='MMES ms explicit levels for conservative,balanced,aggressive; inherit is unchanged.')
    parser.add_argument('--subopt_vkd_k_init_levels', type=str, default='0,2,4',
                        help='VKD k_init explicit levels for conservative,balanced,aggressive; clipped by dim-1.')
    parser.add_argument('--subopt_vkd_kmax_levels', type=str, default='8,32,64',
                        help='VKD kmax explicit levels for conservative,balanced,aggressive; clipped by dim-1.')
    parser.add_argument('--subopt_vkd_action_param_mode', type=str, default='rank', choices=['rank', 'tpa_rank'],
                        help='VKD cfg slot semantics: rank=sigma,n_individuals,k_init,kmax; '
                             'tpa_rank=sigma,n_individuals,cs_scale,k_inc_cond with k_init=0,kmax=dim-1.')
    parser.add_argument('--subopt_vkd_cs_scale_levels', type=str, default='0.5,1.0,2.0',
                        help='VKD cs scale levels for conservative,balanced,aggressive when subopt_vkd_action_param_mode=tpa_rank.')
    parser.add_argument('--subopt_vkd_k_inc_cond_levels', type=str, default='10,30,60',
                        help='VKD k_inc_cond levels for conservative,balanced,aggressive when subopt_vkd_action_param_mode=tpa_rank; k_dec_cond follows the same value.')
    parser.add_argument('--subopt_cmaes_cov_lr_scale_levels', type=str, default='0.5,1.0,1.5',
                        help='CMAES cov_lr_scale explicit levels for conservative,balanced,aggressive.')
    parser.add_argument('--subopt_cmaes_c_s_scale_levels', type=str, default='0.7,1.0,1.3',
                        help='CMAES c_s_scale explicit levels for conservative,balanced,aggressive.')
    parser.add_argument('--subopt_sepcmaes_cov_lr_scale_levels', type=str, default='0.5,1.0,1.5',
                        help='SepCMAES cov_lr_scale explicit levels for conservative,balanced,aggressive.')
    parser.add_argument('--subopt_sepcmaes_c_s_scale_levels', type=str, default='0.7,1.0,1.3',
                        help='SepCMAES c_s_scale explicit levels for conservative,balanced,aggressive.')
    parser.add_argument('--subopt_manual_override', type=int, default=0,
                        help='是否允许 subopt_* 手动参数覆盖动作映射结果（1允许，0禁止；默认0）')

    # =======================================================================
    # 2. PPO 强化学习核心参数 (PPO Algorithm Settings)
    # =======================================================================
    parser.add_argument('--RL_agent', default='ppo', choices=['ppo', 'mappo'], help='强化学习训练算法')
    parser.add_argument('--gamma', type=float, default=0.999, help='折扣因子 (Reward discount factor)')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--K_epochs', type=int, default=5, help='每次更新时 PPO 的内部迭代次数')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip 比率')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='策略熵系数')
    parser.add_argument('--vf_coef', type=float, default=1.0, help='价值损失系数')
    parser.add_argument('--n_step', type=int, default=5, help='n-step 回报估计')
    parser.add_argument('--v_range', type=float, default=6.0, help='值函数范围限制，用于控制熵')
    parser.add_argument('--decision_interval', type=int, default=1, help='每隔多少代执行一次动作决策')
    parser.add_argument('--state', default=[0.0 for _ in range(16)], help='Actor 的初始状态')

    # =======================================================================
    # 2.1 MAPPO + 并行CC设置 (Phase-1 固定规格)
    # =======================================================================
    parser.add_argument('--cc_execution_mode', default='parallel_sync',
                        choices=['parallel_sync', 'serial'],
                        help='CC执行模式：并行同步提交 / 串行更新')
    parser.add_argument('--fixed_agent_num', type=int, default=20,
                        help='固定智能体数量（第一阶段先固定20）')
    parser.add_argument('--fixed_subfes_per_agent', type=int, default=2500,
                        help='每个agent每步分配的FEs（第一阶段固定2500）')
    parser.add_argument('--sigma_candidates', type=str, default='0.2,0.3,0.6',
                        help='离散sigma候选，逗号分隔')
    parser.add_argument('--mappo_param_semantics', type=str, default='current', choices=['current', 'a1'],
                        help='MAPPO动作到子优化器参数的解释语义：current=当前语义；a1=复刻A1时期的旧语义')
    parser.add_argument('--mappo_action_arch', type=str, default='current', choices=['current'],
                        help='MAPPO动作架构：current=多参数动作；pre_caf4a62=旧三列[optimizer,profile,resource]动作')
    parser.add_argument('--optimizer_profile_candidates', type=str, default='inherit,conservative,balanced,aggressive',
                        help='优化器参数策略档位，逗号分隔')
    parser.add_argument('--mappo_cfg_param_num', type=int, default=4,
                        help='第二阶段参数动作个数（每个优化器）')
    parser.add_argument('--optimizer_candidates', type=str, default='mmes,vkd,cmaes,sepcmaes',
                        help='优化器候选，逗号分隔（支持: mmes,vkd,cmaes,sepcmaes）')
    parser.add_argument('--resource_factors', type=str, default='0.5,1.0,2.0',
                        help='资源挡位系数（乘在fixed_subfes_per_agent上）')
    parser.add_argument('--eval_forced_resource_action', type=int, default=-1,
                        help='评估时覆盖资源动作索引；-1 保持策略动作，默认关闭')
    parser.add_argument('--eval_action_intervention', type=str, default='off',
                        choices=['off', 'F1a', 'F1b', 'F1c', 'F1d', 'F1e', 'F1f', 'F1g', 'F1h', 'F4a', 'F4b', 'F4c', 'F4d'],
                        help='Eval-only F1/F4 action intervention; off preserves the existing path.')
    parser.add_argument('--forced_optimizer_enable', type=int, default=0,
                        help='训练期是否启用强制优化器探索（1开启，0关闭）')
    parser.add_argument('--forced_optimizer_prob', type=float, default=0.0,
                        help='warmup后每个epoch强制优化器的概率；仅forced_optimizer_enable=1时生效')
    parser.add_argument('--forced_optimizer_warmup_ratio', type=float, default=0.0,
                        help='训练前多少比例的epoch总是强制优化器；例如0.333表示前三分之一')
    parser.add_argument('--forced_optimizer_mode', type=str, default='cycle', choices=['cycle', 'random'],
                        help='warmup阶段强制优化器选择方式：cycle按epoch轮换，random按epoch随机；warmup后触发强制时总是随机')
    parser.add_argument('--sub_optimizer', type=str, default='mmes', choices=['mmes', 'vkd', 'cmaes', 'sepcmaes'],
                        help='MAPPO 子问题优化器')
    parser.add_argument('--train_function_ids', type=str, default='',
                        help='训练函数ID列表，逗号分隔；不再提供默认值，需外部显式传入')
    parser.add_argument('--fun_ids', type=str, default='',
                        help='训练函数ID别名；若未显式传 train_function_ids，则用该值覆盖')
    parser.add_argument('--use_team_reward', type=int, default=1,
                        help='是否使用team reward（1是，0否）')
    parser.add_argument('--mappo_share_policy', type=int, default=1,
                        help='MAPPO是否共享actor策略参数（1是，0否）')
    parser.add_argument('--mappo_use_centralized_critic', type=int, default=1,
                        help='MAPPO是否使用中心化critic（1是，0否）')
    parser.add_argument('--record_eval_individual', type=int, default=0,
                        help='是否记录每次评估对应的全空间个体（1开启，内存开销很大）')
    parser.add_argument('--mappo_log_enable', type=int, default=1,
                        help='MAPPO训练日志总开关（1开启，0关闭）')
    parser.add_argument('--eval_stream_flush_fes_interval', type=float, default=2e6,
                        help='评估时按FES间隔分段刷盘（默认2e6）；<=0表示关闭，保持旧行为')
    parser.add_argument('--eval_record_fes_list', type=str, default='',
                        help='评估结果表额外记录的精确reported-FEs点，逗号分隔并支持科学计数法；非空时旁路记录fitness点对应的sumFEs。默认空保持历史汇总语义，超过单题预算的点忽略，最终预算始终记录。')
    parser.add_argument('--eval_save_actions', type=int, default=1,
                        help='评估时是否保存actions.csv（1开启，0关闭）')
    parser.add_argument('--eval_save_running_data', type=int, default=1,
                        help='评估时是否保存running_data.h5（1开启，0关闭）')
    parser.add_argument('--eval_save_actuator_diagnostics', type=int, default=0, choices=[0, 1],
                        help='评估时旁路保存actuator logits/probabilities、observation和candidate telemetry；默认关闭且不改变动作。')
    parser.add_argument('--eval_save_generator_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save D6 generator logits/probabilities, masks, legal pre-probe observations, and geometry telemetry.')
    parser.add_argument('--eval_save_rvcpd_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save per-event RVCPD environment telemetry and an aggregate JSON summary without changing policy actions.')
    parser.add_argument('--eval_save_guide_replacement_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save per-event receiver-validated guide-replacement telemetry. Requires a non-off replacement mode.')
    parser.add_argument('--eval_save_collective_guide_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save per-event shared-base collective-guide telemetry. Requires a non-off collective guide mode.')
    parser.add_argument('--eval_save_event_slot_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save detached per-event, per-slot, per-agent optimizer/commit/path/scale diagnostics. Requires event-slot interleaving.')
    parser.add_argument('--eval_save_boundary_compare_trace', type=int, default=0, choices=[0, 1],
                        help='Bounded per-generation boundary summaries and first-touch windows for four event-slot optimizers; default off.')
    parser.add_argument('--eval_save_vkd_state_trace', type=int, default=0, choices=[0, 1],
                        help='Save full VKD slot/generation state as JSON cells in existing event-slot/forensic CSV. Requires event-slot interleaving and an existing diagnostic writer. Default off; no state/RNG/objective changes.')
    parser.add_argument('--eval_keep_event_slot_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Keep the raw event_slot_diagnostics.csv after writing its summary (1=keep, 0=delete).')
    parser.add_argument('--eval_save_optimizer_forensic_trace', type=int, default=0, choices=[0, 1],
                        help='Save a self-contained, streamed per-event forensic trace (sigma provenance, objective values, optimizer internals) that survives a crashed evaluation unit. Requires event-slot interleaving. Default 0 preserves historical behavior and writes no file.')
    parser.add_argument('--eval_primary_profile_sample', type=int, default=0, choices=[0, 1],
                        help='Eval-only: sample ONLY the profile head (cfg param 0) from its policy distribution while every other head keeps argmax. Default 0 leaves the existing deterministic path unchanged.')
    parser.add_argument('--eval_save_primary_action_diagnostics', type=int, default=0, choices=[0, 1],
                        help='Save detached per-event primary Actor action-head probabilities (optimizer, cfg_p0 profile logits/probs, resource, K) and sigma-state observation; default off.')
    parser.add_argument('--mappo_log_interval', type=int, default=1,
                        help='MAPPO日志写入间隔（按epoch）')
    parser.add_argument('--mappo_log_action_hist', type=int, default=1,
                        help='是否记录动作分布统计（1开启，0关闭）')
    parser.add_argument('--mappo_log_adv_stats', type=int, default=1,
                        help='是否记录advantage统计（1开启，0关闭）')
    parser.add_argument('--mappo_opt_emb_dim', type=int, default=8,
                        help='层级actor中optimizer embedding维度')
    parser.add_argument('--mappo_cfg_emb_dim', type=int, default=8,
                        help='层级actor中config embedding维度')
    parser.add_argument('--mappo_reward_team_weight', type=float, default=0.8,
                        help='阶段A mixed reward 中 team 项权重 alpha')
    parser.add_argument('--mappo_actor_local_beta', type=float, default=0.2,
                        help='阶段A actor advantage 中 local shaping 系数 beta')
    parser.add_argument('--mappo_entropy_coef_opt', type=float, default=0.02,
                        help='阶段B optimizer head 熵正则系数')
    parser.add_argument('--mappo_entropy_coef_cfg', type=float, default=0.01,
                        help='阶段B config head 熵正则系数')
    parser.add_argument('--mappo_entropy_coef_res', type=float, default=0.01,
                        help='阶段B resource head 熵正则系数')
    parser.add_argument('--mappo_entropy_coef_comm', type=float, default=-1.0,
                        help='Communication-round head entropy coefficient; negative means reuse entropy_coef.')
    parser.add_argument('--mappo_entropy_coef_collab', type=float, default=-1.0,
                        help='Collaboration-mode head entropy coefficient; negative means reuse entropy_coef.')
    parser.add_argument('--mappo_entropy_coef_guide_scale', type=float, default=-1.0,
                        help='Guide-scale head entropy coefficient; negative means reuse entropy_coef.')
    parser.add_argument('--mappo_entropy_coef_actuator', type=float, default=-1.0,
                        help='Candidate-actuator head entropy coefficient; negative means reuse entropy_coef.')
    parser.add_argument('--mappo_policy_loss_weight_opt', type=float, default=0.8,
                        help='阶段C optimizer head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_cfg', type=float, default=1.0,
                        help='阶段C config head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_res', type=float, default=1.0,
                        help='阶段C resource head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_comm', type=float, default=1.0,
                        help='Communication-round head policy loss weight.')
    parser.add_argument('--mappo_policy_loss_weight_collab', type=float, default=1.0,
                        help='Collaboration-mode head policy loss weight.')
    parser.add_argument('--mappo_policy_loss_weight_guide_scale', type=float, default=1.0,
                        help='Guide-scale head policy loss weight.')
    parser.add_argument('--mappo_policy_loss_weight_actuator', type=float, default=1.0,
                        help='Candidate-actuator head policy loss weight.')
    parser.add_argument('--mappo_value_loss_weight_ref', type=float, default=1.0,
                        help='阶段C shared reference value loss 权重')
    parser.add_argument('--mappo_value_loss_weight_opt', type=float, default=0.5,
                        help='阶段C optimizer head value loss 权重')
    parser.add_argument('--mappo_value_loss_weight_cfg', type=float, default=0.8,
                        help='阶段C config head value loss 权重')
    parser.add_argument('--mappo_value_loss_weight_res', type=float, default=1.0,
                        help='阶段C resource head value loss 权重')
    parser.add_argument('--mappo_value_loss_weight_comm', type=float, default=1.0,
                        help='Communication-round head value loss weight.')
    parser.add_argument('--mappo_value_loss_weight_collab', type=float, default=1.0,
                        help='Collaboration-mode head value loss weight.')
    parser.add_argument('--mappo_value_loss_weight_guide_scale', type=float, default=1.0,
                        help='Guide-scale head value loss weight.')
    parser.add_argument('--mappo_value_loss_weight_actuator', type=float, default=1.0,
                        help='Candidate-actuator head value loss weight.')
    parser.add_argument('--d5_actuator_hidden_dim', type=int, default=64,
                        help='Hidden width of the independent D5 actuator actor and critic.')
    parser.add_argument('--d5_actuator_action_emb_dim', type=int, default=8,
                        help='Embedding width per frozen-primary action column for the D5 actuator.')
    parser.add_argument('--d5_actuator_initial_probs', type=str, default='1,1,1',
                        help='Independent D5 actuator initial categorical prior; normalized after parsing.')
    parser.add_argument('--d5_actuator_lr_actor', type=float, default=-1.0,
                        help='D5 actuator actor learning rate; negative reuses lr_model.')
    parser.add_argument('--d5_actuator_lr_critic', type=float, default=-1.0,
                        help='D5 actuator critic learning rate; negative reuses lr_critic.')
    parser.add_argument('--d5_primary_checkpoint_path', type=str, default='',
                        help='Frozen D4-B primary checkpoint used by a fresh D5 training run.')
    parser.add_argument('--d5_eval_forced_actuator_action', type=int, default=-1,
                        help='Optional D5-only forced actuator action for bounded qualification/evaluation.')
    parser.add_argument('--objective_split_d6_pre_generator_selector_enable', type=int, default=0, choices=[0, 1],
                        help='Enable the D6 consensus-after, probe-before SPSA/Hybrid selector protocol.')
    parser.add_argument('--d6_generator_hidden_dim', type=int, default=64,
                        help='Hidden width of the independent D6 generator actor and critic.')
    parser.add_argument('--d6_generator_action_emb_dim', type=int, default=8,
                        help='Embedding width per frozen-primary action column for the D6 generator selector.')
    parser.add_argument('--d6_generator_initial_probs', type=str, default='0.5,0.5',
                        help='Initial D6 generator probabilities for SPSA and Hybrid.')
    parser.add_argument('--d6_generator_lr_actor', type=float, default=-1.0,
                        help='D6 generator actor learning rate; negative reuses lr_model.')
    parser.add_argument('--d6_generator_lr_critic', type=float, default=-1.0,
                        help='D6 generator critic learning rate; negative reuses lr_critic.')
    parser.add_argument('--d6_primary_checkpoint_path', type=str, default='',
                        help='Frozen plain-MAPPO or D5-container checkpoint used by a fresh D6 training run.')
    parser.add_argument('--d6_primary_source_kind', type=str, default='auto',
                        choices=['auto', 'plain_mappo', 'd5_container'],
                        help='D6 frozen-primary checkpoint layout; auto inspects checkpoint fields, never its path.')
    parser.add_argument('--d6_eval_forced_generator_action', type=int, default=-1,
                        help='Optional D6 evaluation override: 0=SPSA, 1=Hybrid, -1=policy.')
    parser.add_argument('--d6_post_actuator_forced_action', type=int, default=2,
                        help='Frozen post-verifier actuator action used by D6; default action2 is beta0.5.')
    parser.add_argument('--d6_eval_forced_post_actuator_action', type=int, default=-1,
                        help='Evaluation-only D6 post-actuator override; -1 preserves the checkpoint protocol.')
    parser.add_argument('--mappo_critic_agent_emb_dim', type=int, default=16,
                        help='阶段C critic agent embedding 维度')
    parser.add_argument('--mappo_critic_action_emb_dim', type=int, default=8,
                        help='阶段C critic action embedding 维度')
    parser.add_argument('--mappo_diag_reward_frac', type=float, default=0.25,
                        help='训练诊断中 head/tail 窗口占比')
    parser.add_argument('--mappo_diag_reward_tol_ratio', type=float, default=0.02,
                        help='reward 趋势判断阈值比例')
    parser.add_argument('--mappo_diag_kl_low', type=float, default=1e-4,
                        help='训练诊断 KL 下界')
    parser.add_argument('--mappo_diag_kl_high', type=float, default=5e-2,
                        help='训练诊断 KL 上界')
    parser.add_argument('--mappo_diag_clip_low', type=float, default=1e-3,
                        help='训练诊断 clip_frac 下界')
    parser.add_argument('--mappo_diag_clip_high', type=float, default=5e-1,
                        help='训练诊断 clip_frac 上界')
    parser.add_argument('--mappo_diag_entropy_low', type=float, default=0.02,
                        help='训练诊断熵下界')
    parser.add_argument('--mappo_diag_collapse_ratio', type=float, default=0.95,
                        help='动作塌缩风险阈值（max ratio）')
    parser.add_argument('--mappo_diag_adv_imbalance_factor', type=float, default=3.0,
                        help='advantage 方差失衡阈值倍率')
    parser.add_argument('--mappo_diag_adv_abs_mean_min', type=float, default=0.05,
                        help='advantage 绝对均值最小阈值')

    # =======================================================================
    # 3. 网络结构参数 (Network / Attention Architecture)
    # =======================================================================
    parser.add_argument('--n_encode_layers', type=int, default=1, help='Encoder 的堆叠层数')
    parser.add_argument('--normalization', default='layer', help="归一化类型: 'layer' 或 'batch'")
    
    # 抽取的特征数量
    parser.add_argument('--feature_num_1', type=int, default=16, help='Actor1 的输入特征维度')
    parser.add_argument('--feature_num_2', type=int, default=15, help='Actor2 的输入特征维度')

    # Attention Heads
    parser.add_argument('--encoder_head_num', type=int, default=4, help='Encoder 的多头注意力数')
    parser.add_argument('--decoder_head_num', type=int, default=4, help='Decoder 的多头注意力数')
    parser.add_argument('--critic_head_num', type=int, default=6, help='Critic Encoder 的多头注意力数')

    # Actor 1 模型维度 (子任务分配相关)
    parser.add_argument('--actor1_embedding_dim', type=int, default=128, help='Actor1 输入嵌入维度')
    parser.add_argument('--actor1_hidden_dim', type=int, default=256, help='Actor1 隐藏层维度')
    parser.add_argument('--actor1_action_space', type=int, default=1, help='Actor1 动作空间')

    # Actor 2 模型维度 (资源分配相关)
    parser.add_argument('--actor2_embedding_dim', type=int, default=160, help='Actor2 输入嵌入维度')
    parser.add_argument('--actor2_hidden_dim', type=int, default=320, help='Actor2 隐藏层维度')
    parser.add_argument('--actor2_action_space', type=int, default=3, help='Actor2 动作空间')

    # =======================================================================
    # 4. 优化器与训练配置 (Optimizer & Training Schedule)
    # =======================================================================
    parser.add_argument('--lr_model', type=float, default=2e-4, help="Actor 网络学习率")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Critic 网络学习率")
    parser.add_argument('--lr_decay', type=float, default=0.99, help='学习率每 epoch 的衰减率')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='梯度裁剪的最大 L2 范数')
    
    parser.add_argument('--T_train', type=int, default=2000, help='训练的总迭代次数')
    parser.add_argument('--epoch_start', type=int, default=0, help='起始 epoch')
    parser.add_argument('--epoch_end', type=int, default=30, help='最大训练 epoch')
    parser.add_argument(
        '--schedule_start_epoch',
        type=int,
        default=-1,
        help='训练日程轴起点；-1 表示跟随 --epoch_start。日程轴只决定 forced optimizer 的 warmup 比例分母，短训时用它保持完整日程语义。',
    )
    parser.add_argument(
        '--schedule_end_epoch',
        type=int,
        default=-1,
        help='训练日程轴终点（不含）；-1 表示跟随 --epoch_end。可与 --epoch_end 不同，用于只执行前若干 epoch 而不压缩探索期。',
    )
    parser.add_argument('--epoch_size', type=int, default=200, help='每个 epoch 的样本数量')
    parser.add_argument('--one_problem_batch_size', type=int, default=4, help='每个问题的实例数量')
    parser.add_argument(
        '--each_question_batch_num',
        type=int,
        default=5,
        help='每个训练batch包含的问题数/并行环境数；batches_per_epoch=ceil(len(train_function_ids)/该值)',
    )
    parser.add_argument('--max_learning_step', type=int, default=10000, help='训练迭代步数上限')
    parser.add_argument('--update_best_model_epochs', type=int, default=1, help='检查并更新最佳模型的频率')
    parser.add_argument(
        '--mappo_update_mode',
        type=str,
        default='joint',
        choices=['joint', 'split_env'],
        help='MAPPO update mode: joint updates one PPO batch per vectorized rollout; split_env samples vectorized envs together but updates once per env/problem.',
    )
    parser.add_argument(
        '--mappo_split_env_adv_norm',
        type=int,
        default=1,
        help='When mappo_update_mode=split_env, normalize advantages per env/problem before sequential updates.',
    )

    # =======================================================================
    # 5. 推理、测试与评估 (Inference & Validation)
    # =======================================================================
    parser.add_argument('--test', type=int, default=0, help='是否切换到测试模式 (1为开启)')
    parser.add_argument('--eval_only', action='store_true', default=False, help='仅进行推理模式')
    parser.add_argument('--greedy_rollout', action='store_true', help='推理时是否使用贪心策略')
    parser.add_argument('--Max_Eval', type=int, default=200000, help='推理时的最大目标函数评价次数')
    parser.add_argument('--val_size', type=int, default=1024, help='验证/推理时的实例数量')
    parser.add_argument('--per_eval_time', type=int, default=1, help='每个实例的评估次数')
    parser.add_argument('--inference_interval', type=int, default=3, help='推理间隔')
    parser.add_argument('--dataset_path', default=None, help='测试数据集路径')
    parser.add_argument('--load_path_for_test', 
                        default=os.path.join(SAVE_DIR, 'ppo_model/model_CEC2013LSGO_1.0e+06_10_20_0.0001_20250829T164119/epoch-90.pt'), 
                        help='测试时加载的模型路径')

    # =======================================================================
    # 6. 系统、日志与断点续传 (System, Logs & Resume)
    # =======================================================================
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='强制禁用 GPU')
    parser.add_argument('--no_DDP', action='store_true', help='禁用分布式并行训练')
    parser.add_argument('--use_assert', action='store_true', help='启用断言检查')

    # 路径与保存
    parser.add_argument('--run_name', default='model', help='运行名称，用于标识实验')
    parser.add_argument('--no_tb', action='store_true', help='禁用 TensorBoard 日志')
    parser.add_argument('--no_saving', action='store_true', help='禁用模型和数据保存')
    parser.add_argument('--show_figs', action='store_true', help='启用图表记录')
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')
    
    parser.add_argument('--log_dir', default=os.path.join(SAVE_DIR, 'log'), help='TensorBoard 日志目录')
    parser.add_argument('--output_modal_dir', default=os.path.join(SAVE_DIR, 'ppo_model'), help='模型保存目录')
    parser.add_argument('--output_data_dir', default=os.path.join(SAVE_DIR, 'running_data'), help='运行数据保存目录')
    parser.add_argument('--output_rollout_dir', default=os.path.join(SAVE_DIR, 'rollout_data'), help='Rollout 数据保存目录')
    parser.add_argument('--test_dir', default=os.path.join(SAVE_DIR, 'test'), help='测试结果保存目录')
    
    parser.add_argument('--log_step', type=int, default=1, help='记录日志的步长间隔')
    parser.add_argument('--checkpoint_epochs', type=int, default=1, help='保存 Checkpoint 的 epoch 间隔')
    
    parser.add_argument('--load_path', default=None, help='加载模型和优化器状态的路径')
    parser.add_argument('--resume', default=None, help='从之前的 Checkpoint 恢复训练')
    parser.add_argument('--auto_latest_resume', action='store_true',
                        help='自动从 resume_dir 或 output_modal_dir 中选择最新的 .pt 作为 --resume')
    parser.add_argument('--resume_dir', type=str, default='',
                        help='自动续训时搜索目录；为空则使用 output_modal_dir')
    parser.add_argument('--resume_keyword', type=str, default='',
                        help='自动续训时可选的 checkpoint 文件名关键字过滤')

    # =======================================================================
    # 后置处理逻辑 (Post-processing)
    # =======================================================================
    raw_args_for_defaults = [str(x) for x in (sys.argv[1:] if args is None else args)]
    def _arg_was_set(flag: str) -> bool:
        return any(x == flag or x.startswith(flag + "=") for x in raw_args_for_defaults)
    opts = parser.parse_args(args)
    if int(getattr(opts, 'objective_split_d5_two_stage_actuator_enable', 0)) or int(
        getattr(opts, 'objective_split_d6_pre_generator_selector_enable', 0)
    ):
        raise ValueError('D5/D6 selector variants are unavailable in this E3aj package')

    # A1 compatibility mode restores the behavior that affected MAPPO training/eval
    # around model_WSNLocation_..._20260520T175948, while still allowing explicit CLI
    # overrides for controlled ablations.
    opts.mappo_param_semantics = str(getattr(opts, "mappo_param_semantics", "current")).lower()
    opts.mappo_action_arch = str(getattr(opts, "mappo_action_arch", "current")).lower()
    if opts.mappo_action_arch == "pre_caf4a62":
        opts.mappo_param_semantics = "current"
        opts.mappo_cfg_param_num = 1
        if not _arg_was_set("--sigma"):
            opts.sigma = 2.0
        if not _arg_was_set("--sigma_candidates"):
            opts.sigma_candidates = "0.01,0.1,1.0"
    if opts.mappo_param_semantics == "a1":
        if not _arg_was_set("--sigma"):
            opts.sigma = 2.0
        if not _arg_was_set("--sigma_candidates"):
            opts.sigma_candidates = "0.01,0.1,1.0"
        if not _arg_was_set("--subopt_cmaes_cov_lr_scale_levels"):
            opts.subopt_cmaes_cov_lr_scale_levels = "0.75,1.0,1.25"
        if not _arg_was_set("--subopt_cmaes_c_s_scale_levels"):
            opts.subopt_cmaes_c_s_scale_levels = "0.75,1.0,1.25"
        if not _arg_was_set("--subopt_sepcmaes_cov_lr_scale_levels"):
            opts.subopt_sepcmaes_cov_lr_scale_levels = "0.75,1.0,1.25"
        if not _arg_was_set("--subopt_sepcmaes_c_s_scale_levels"):
            opts.subopt_sepcmaes_c_s_scale_levels = "0.75,1.0,1.25"
        if not _arg_was_set("--subopt_manual_override"):
            opts.subopt_manual_override = 1

    if opts.auto_latest_resume:
        if opts.load_path:
            raise ValueError("--auto_latest_resume cannot be used with --load_path.")
        if opts.resume:
            raise ValueError("--auto_latest_resume cannot be used with explicit --resume.")
        search_dir = str(opts.resume_dir).strip() if str(opts.resume_dir).strip() else opts.output_modal_dir
        opts.resume = _resolve_latest_checkpoint(search_dir, opts.resume_keyword)

    # 计算总交互次数 ns
    opts.ns = opts.max_fes // opts.subFEs
    opts.sigma_candidates = _parse_float_list(opts.sigma_candidates)
    opts.optimizer_profile_candidates = [x.lower() for x in _parse_str_list(opts.optimizer_profile_candidates)]
    opts.mappo_cfg_param_num = int(max(1, opts.mappo_cfg_param_num))
    if opts.mappo_action_arch == "pre_caf4a62":
        opts.mappo_cfg_param_num = 1
    opts.optimizer_candidates = [x.lower() for x in _parse_str_list(opts.optimizer_candidates)]
    opts.resource_factors = _parse_float_list(opts.resource_factors)
    opts.eval_forced_resource_action = int(
        getattr(opts, "eval_forced_resource_action", -1)
    )
    if opts.eval_forced_resource_action < -1 or (
        opts.eval_forced_resource_action >= len(opts.resource_factors)
    ):
        raise ValueError(
            "eval_forced_resource_action must be -1 or a valid resource action index."
        )
    def _parse_three_int_levels(raw, name: str) -> List[int]:
        vals = _parse_int_list_allow_empty(raw)
        if len(vals) != 3:
            raise ValueError(f"{name} must contain exactly 3 comma-separated values.")
        return [int(x) for x in vals]

    def _parse_three_float_levels(raw, name: str) -> List[float]:
        vals = _parse_float_list(raw)
        if len(vals) != 3:
            raise ValueError(f"{name} must contain exactly 3 comma-separated values.")
        return [float(x) for x in vals]

    opts.subopt_mmes_ms_levels = _parse_three_int_levels(
        opts.subopt_mmes_ms_levels, "--subopt_mmes_ms_levels"
    )
    opts.subopt_vkd_k_init_levels = _parse_three_int_levels(
        opts.subopt_vkd_k_init_levels, "--subopt_vkd_k_init_levels"
    )
    opts.subopt_vkd_kmax_levels = _parse_three_int_levels(
        opts.subopt_vkd_kmax_levels, "--subopt_vkd_kmax_levels"
    )
    opts.subopt_vkd_action_param_mode = str(
        getattr(opts, "subopt_vkd_action_param_mode", "rank")
    ).lower()
    if opts.subopt_vkd_action_param_mode not in {"rank", "tpa_rank"}:
        raise ValueError("--subopt_vkd_action_param_mode must be one of: rank,tpa_rank")
    opts.subopt_vkd_cs_scale_levels = _parse_three_float_levels(
        opts.subopt_vkd_cs_scale_levels, "--subopt_vkd_cs_scale_levels"
    )
    opts.subopt_vkd_k_inc_cond_levels = _parse_three_float_levels(
        opts.subopt_vkd_k_inc_cond_levels, "--subopt_vkd_k_inc_cond_levels"
    )
    opts.subopt_cmaes_cov_lr_scale_levels = _parse_three_float_levels(
        opts.subopt_cmaes_cov_lr_scale_levels, "--subopt_cmaes_cov_lr_scale_levels"
    )
    opts.subopt_cmaes_c_s_scale_levels = _parse_three_float_levels(
        opts.subopt_cmaes_c_s_scale_levels, "--subopt_cmaes_c_s_scale_levels"
    )
    opts.subopt_sepcmaes_cov_lr_scale_levels = _parse_three_float_levels(
        opts.subopt_sepcmaes_cov_lr_scale_levels, "--subopt_sepcmaes_cov_lr_scale_levels"
    )
    opts.subopt_sepcmaes_c_s_scale_levels = _parse_three_float_levels(
        opts.subopt_sepcmaes_c_s_scale_levels, "--subopt_sepcmaes_c_s_scale_levels"
    )
    # train_function_ids alias: allow using --fun_ids for training when train_function_ids is omitted.
    if str(getattr(opts, "fun_ids", "")).strip() and (not str(opts.train_function_ids).strip()):
        opts.train_function_ids = str(opts.fun_ids).strip()
    opts.train_function_ids = _parse_int_list_allow_empty(opts.train_function_ids)
    opts.mappo_env_mode = str(opts.mappo_env_mode).lower()
    opts.wsn_objective_consensus = str(opts.wsn_objective_consensus).lower()
    opts.objective_split_consensus = str(opts.objective_split_consensus).lower()
    if _arg_was_set("--wsn_objective_consensus") and not _arg_was_set(
        "--objective_split_consensus"
    ):
        opts.objective_split_consensus = opts.wsn_objective_consensus
    opts.objective_split_consensus_strength = float(
        min(1.0, max(0.0, opts.objective_split_consensus_strength))
    )
    opts.objective_split_comm_interval = int(max(1, opts.objective_split_comm_interval))
    opts.objective_split_comm_rounds = int(max(1, opts.objective_split_comm_rounds))
    opts.objective_split_comm_force_rounds = int(
        max(0, getattr(opts, "objective_split_comm_force_rounds", 0))
    )
    opts.objective_split_comm_action_enable = int(
        bool(getattr(opts, "objective_split_comm_action_enable", 0))
    )
    opts.objective_split_comm_round_candidates = [
        int(max(1, x))
        for x in _parse_int_list_allow_empty(
            getattr(opts, "objective_split_comm_round_candidates", "1,2,4,8")
        )
    ]
    if len(opts.objective_split_comm_round_candidates) == 0:
        opts.objective_split_comm_round_candidates = [1]
    opts.objective_split_comm_action_reduce = str(
        getattr(opts, "objective_split_comm_action_reduce", "max")
    ).lower()
    opts.objective_split_graph_source = str(opts.objective_split_graph_source).lower()
    opts.objective_split_weight_mode = str(opts.objective_split_weight_mode).lower()
    opts.objective_split_graph_threshold = float(max(0.0, opts.objective_split_graph_threshold))
    opts.objective_split_neighbor_obs = int(bool(opts.objective_split_neighbor_obs))
    opts.objective_split_neighbor_obs_mode = str(
        opts.objective_split_neighbor_obs_mode
    ).lower()
    if opts.objective_split_neighbor_obs_mode == "auto":
        opts.objective_split_neighbor_obs_mode = (
            "full" if opts.objective_split_neighbor_obs else "none"
        )
    opts.objective_split_neighbor_obs = int(
        opts.objective_split_neighbor_obs_mode != "none"
    )
    opts.objective_split_state_comm_mode = str(
        getattr(opts, "objective_split_state_comm_mode", "none")
    ).lower()
    if opts.objective_split_state_comm_mode not in {"none", "full_mean", "graph_mean"}:
        raise ValueError(
            f"Unsupported objective_split_state_comm_mode: {opts.objective_split_state_comm_mode}"
        )
    opts.objective_split_state_comm_include_delta = int(
        bool(getattr(opts, "objective_split_state_comm_include_delta", 1))
    )
    opts.objective_split_state_comm_include_actual_fes = int(
        bool(getattr(opts, "objective_split_state_comm_include_actual_fes", 1))
    )
    opts.objective_split_consensus_reward_weight = float(
        max(0.0, opts.objective_split_consensus_reward_weight)
    )
    opts.objective_split_ccsa_momentum_decay = float(
        min(1.0, max(0.0, opts.objective_split_ccsa_momentum_decay))
    )
    opts.objective_split_ccsa_direction_lr = float(
        max(0.0, opts.objective_split_ccsa_direction_lr)
    )
    opts.objective_split_ccsa_scale_rate = float(
        max(0.0, opts.objective_split_ccsa_scale_rate)
    )
    opts.objective_split_ccsa_scale_min = float(
        max(0.0, opts.objective_split_ccsa_scale_min)
    )
    opts.objective_split_ccsa_scale_max = float(
        max(opts.objective_split_ccsa_scale_min, opts.objective_split_ccsa_scale_max)
    )
    opts.objective_split_ccsa_positive_improve = int(
        bool(opts.objective_split_ccsa_positive_improve)
    )
    opts.objective_split_ccsa_momentum_update_mode = str(
        getattr(opts, "objective_split_ccsa_momentum_update_mode", "lite_only")
    ).lower()
    if opts.objective_split_ccsa_momentum_update_mode not in {
        "off",
        "lite_only",
        "always",
    }:
        raise ValueError(
            "Unsupported objective_split_ccsa_momentum_update_mode: "
            f"{opts.objective_split_ccsa_momentum_update_mode}"
        )
    opts.objective_split_masoie_velocity_decay = float(
        min(1.0, max(0.0, opts.objective_split_masoie_velocity_decay))
    )
    opts.objective_split_masoie_velocity_scale = float(
        max(0.0, opts.objective_split_masoie_velocity_scale)
    )
    opts.objective_split_masoie_velocity_clip_ratio = float(
        max(0.0, opts.objective_split_masoie_velocity_clip_ratio)
    )
    opts.objective_split_rvcpd_arm = str(
        getattr(opts, "objective_split_rvcpd_arm", "off")
    ).lower()
    if opts.objective_split_rvcpd_arm not in {"off", "p0", "p1", "p2"}:
        raise ValueError(
            "Unsupported objective_split_rvcpd_arm: "
            f"{opts.objective_split_rvcpd_arm}"
        )
    opts.objective_split_rvcpd_integration_mode = str(
        getattr(
            opts,
            "objective_split_rvcpd_integration_mode",
            "isolated",
        )
    ).lower()
    if opts.objective_split_rvcpd_integration_mode not in {
        "isolated",
        "d5_post_commit",
    }:
        raise ValueError(
            "Unsupported objective_split_rvcpd_integration_mode: "
            f"{opts.objective_split_rvcpd_integration_mode}"
        )
    opts.objective_split_rvcpd_path_decay = float(
        min(1.0, max(0.0, opts.objective_split_rvcpd_path_decay))
    )
    opts.objective_split_rvcpd_direction_lr = float(
        max(0.0, opts.objective_split_rvcpd_direction_lr)
    )
    opts.objective_split_rvcpd_scale_min = float(
        max(0.0, opts.objective_split_rvcpd_scale_min)
    )
    opts.objective_split_rvcpd_scale_max = float(
        max(
            opts.objective_split_rvcpd_scale_min,
            opts.objective_split_rvcpd_scale_max,
        )
    )
    opts.objective_split_rvcpd_initial_scale = float(
        min(
            opts.objective_split_rvcpd_scale_max,
            max(
                opts.objective_split_rvcpd_scale_min,
                opts.objective_split_rvcpd_initial_scale,
            ),
        )
    )
    opts.objective_split_rvcpd_scale_growth = float(
        max(1.0, opts.objective_split_rvcpd_scale_growth)
    )
    opts.objective_split_rvcpd_scale_decay = float(
        min(1.0, max(0.0, opts.objective_split_rvcpd_scale_decay))
    )
    opts.objective_split_rvcpd_noop_path_decay = float(
        min(1.0, max(0.0, opts.objective_split_rvcpd_noop_path_decay))
    )
    opts.objective_split_rvcpd_ema_decay = float(
        min(1.0, max(0.0, opts.objective_split_rvcpd_ema_decay))
    )
    opts.objective_split_rvcpd_probe_min_ratio = float(
        max(0.0, opts.objective_split_rvcpd_probe_min_ratio)
    )
    opts.objective_split_rvcpd_probe_max_ratio = float(
        max(
            opts.objective_split_rvcpd_probe_min_ratio,
            opts.objective_split_rvcpd_probe_max_ratio,
        )
    )
    opts.objective_split_rvcpd_min_log_improve = float(
        max(0.0, opts.objective_split_rvcpd_min_log_improve)
    )
    opts.objective_split_record_comm_cost = int(bool(opts.objective_split_record_comm_cost))
    opts.objective_split_agent_parallel_workers = int(
        max(1, getattr(opts, "objective_split_agent_parallel_workers", 1))
    )
    opts.objective_split_event_slot_interleaving_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_event_slot_interleaving_enable",
                0,
            )
        )
    )
    opts.objective_split_persistent_sepcmaes_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_persistent_sepcmaes_enable",
                0,
            )
        )
    )
    opts.objective_split_persistent_sepcmaes_recenter_max_ratio = float(
        getattr(
            opts,
            "objective_split_persistent_sepcmaes_recenter_max_ratio",
            0.05,
        )
    )
    if (
        not math.isfinite(
            opts.objective_split_persistent_sepcmaes_recenter_max_ratio
        )
        or opts.objective_split_persistent_sepcmaes_recenter_max_ratio < 0.0
    ):
        raise ValueError(
            "objective_split_persistent_sepcmaes_recenter_max_ratio must "
            "be finite and non-negative."
        )
    opts.objective_split_target_block_field_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_target_block_field_enable",
                0,
            )
        )
    )
    opts.objective_split_target_block_dual_clock_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_target_block_dual_clock_enable",
                0,
            )
        )
    )
    opts.objective_split_target_block_dual_clock_commit_lock_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_target_block_dual_clock_commit_lock_enable",
                0,
            )
        )
    )
    opts.objective_split_target_block_commit_credit_mode = str(
        getattr(
            opts,
            "objective_split_target_block_commit_credit_mode",
            "off",
        )
    ).lower()
    if opts.objective_split_target_block_commit_credit_mode not in {
        "off",
        "path",
        "scale",
        "joint",
    }:
        raise ValueError(
            "Unsupported objective_split_target_block_commit_credit_mode: "
            f"{opts.objective_split_target_block_commit_credit_mode}."
        )
    opts.objective_split_target_block_dormancy_recovery_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_target_block_dormancy_recovery_enable",
                0,
            )
        )
    )
    opts.objective_split_target_block_direction_shadow_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_target_block_direction_shadow_enable",
                0,
            )
        )
    )
    opts.objective_split_target_block_challenge_response_mode = str(
        getattr(
            opts,
            "objective_split_target_block_challenge_response_mode",
            "off",
        )
    ).lower()
    if opts.objective_split_target_block_challenge_response_mode not in {
        "off",
        "shadow",
        "actuate",
    }:
        raise ValueError(
            "Unsupported objective_split_target_block_challenge_response_mode: "
            f"{opts.objective_split_target_block_challenge_response_mode}."
        )
    opts.objective_split_early_stop_mode = str(
        getattr(opts, "objective_split_early_stop_mode", "none")
    ).lower()
    opts.objective_split_early_stop_threshold = float(
        max(0.0, getattr(opts, "objective_split_early_stop_threshold", 1e-10))
    )
    opts.objective_split_early_stop_patience = int(
        max(1, getattr(opts, "objective_split_early_stop_patience", 1))
    )
    opts.objective_split_early_stop_check_interval = int(
        max(1, getattr(opts, "objective_split_early_stop_check_interval", 1))
    )
    opts.objective_split_early_stop_min_steps = int(
        max(0, getattr(opts, "objective_split_early_stop_min_steps", 0))
    )
    opts.objective_split_optimizer_guide_enable = int(
        bool(getattr(opts, "objective_split_optimizer_guide_enable", 0))
    )
    opts.objective_split_optimizer_guide_source = str(
        getattr(opts, "objective_split_optimizer_guide_source", "neighbor_improve_direction")
    ).lower()
    if opts.objective_split_optimizer_guide_source not in {
        "neighbor_improve_direction",
        "ccsa_direction_momentum",
    }:
        raise ValueError(
            f"Unsupported objective_split_optimizer_guide_source: {opts.objective_split_optimizer_guide_source}"
        )
    opts.objective_split_optimizer_guide_strength = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_strength", 0.5))
    )
    opts.objective_split_optimizer_guide_strength_schedule = str(
        getattr(opts, "objective_split_optimizer_guide_strength_schedule", "fixed")
    ).lower()
    if opts.objective_split_optimizer_guide_strength_schedule not in {
        "fixed",
        "disagreement",
        "budget",
    }:
        raise ValueError(
            "Unsupported objective_split_optimizer_guide_strength_schedule: "
            f"{opts.objective_split_optimizer_guide_strength_schedule}"
        )
    opts.objective_split_optimizer_guide_strength_scale_min = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_strength_scale_min", 1.0))
    )
    opts.objective_split_optimizer_guide_strength_scale_max = float(
        max(
            opts.objective_split_optimizer_guide_strength_scale_min,
            getattr(opts, "objective_split_optimizer_guide_strength_scale_max", 1.0),
        )
    )
    opts.objective_split_optimizer_guide_disagreement_low = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_disagreement_low", 0.0))
    )
    opts.objective_split_optimizer_guide_disagreement_high = float(
        max(
            opts.objective_split_optimizer_guide_disagreement_low,
            getattr(opts, "objective_split_optimizer_guide_disagreement_high", 1.0),
        )
    )
    opts.objective_split_optimizer_guide_injection_pairs = int(
        max(0, getattr(opts, "objective_split_optimizer_guide_injection_pairs", 1))
    )
    opts.objective_split_optimizer_guide_use_negative_pair = int(
        bool(getattr(opts, "objective_split_optimizer_guide_use_negative_pair", 1))
    )
    opts.objective_split_optimizer_guide_min_improve = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_min_improve", 0.0))
    )
    opts.objective_split_optimizer_guide_gate_enable = int(
        bool(getattr(opts, "objective_split_optimizer_guide_gate_enable", 0))
    )
    opts.objective_split_optimizer_guide_min_norm = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_min_norm", 0.0))
    )
    opts.objective_split_optimizer_guide_min_alignment = float(
        max(
            -1.0,
            min(1.0, getattr(opts, "objective_split_optimizer_guide_min_alignment", -1.0)),
        )
    )
    raw_guide_apply = str(
        getattr(opts, "objective_split_optimizer_guide_apply_optimizers", "all")
    ).lower()
    if raw_guide_apply in {"all", "*"}:
        opts.objective_split_optimizer_guide_apply_optimizers = ["all"]
    elif raw_guide_apply in {"none", ""}:
        opts.objective_split_optimizer_guide_apply_optimizers = []
    else:
        opts.objective_split_optimizer_guide_apply_optimizers = _parse_str_list(raw_guide_apply)
    opts.objective_split_optimizer_guide_mix_strength = float(
        getattr(opts, "objective_split_optimizer_guide_mix_strength", -1.0)
    )
    if opts.objective_split_optimizer_guide_mix_strength < 0.0:
        opts.objective_split_optimizer_guide_mix_strength = float(
            opts.objective_split_optimizer_guide_strength
        )
    else:
        opts.objective_split_optimizer_guide_mix_strength = float(
            max(0.0, opts.objective_split_optimizer_guide_mix_strength)
        )
    opts.objective_split_guide_replacement_mode = str(
        getattr(opts, "objective_split_guide_replacement_mode", "off")
    ).lower()
    if opts.objective_split_guide_replacement_mode not in {
        "off",
        "shadow",
        "p0",
        "p1",
    }:
        raise ValueError(
            "Unsupported objective_split_guide_replacement_mode: "
            f"{opts.objective_split_guide_replacement_mode}"
        )
    opts.objective_split_guide_replacement_scope = str(
        getattr(
            opts,
            "objective_split_guide_replacement_scope",
            "guide_optimizers",
        )
    ).lower()
    if opts.objective_split_guide_replacement_scope not in {
        "guide_optimizers",
        "all",
    }:
        raise ValueError(
            "Unsupported objective_split_guide_replacement_scope: "
            f"{opts.objective_split_guide_replacement_scope}"
        )
    opts.objective_split_collective_guide_mode = str(
        getattr(opts, "objective_split_collective_guide_mode", "off")
    ).lower()
    if opts.objective_split_collective_guide_mode not in {
        "off",
        "shadow",
        "rate_matched_null",
        "collective_veto",
    }:
        raise ValueError(
            "Unsupported objective_split_collective_guide_mode: "
            f"{opts.objective_split_collective_guide_mode}"
        )
    opts.objective_split_collective_guide_null_veto_rate = float(
        getattr(opts, "objective_split_collective_guide_null_veto_rate", 0.0)
    )
    if not 0.0 <= opts.objective_split_collective_guide_null_veto_rate <= 1.0:
        raise ValueError(
            "objective_split_collective_guide_null_veto_rate must be in [0,1]."
        )
    opts.objective_split_anchor_enable = int(
        bool(getattr(opts, "objective_split_anchor_enable", 0))
    )
    opts.objective_split_anchor_source = str(
        getattr(opts, "objective_split_anchor_source", "consensus")
    ).lower()
    if opts.objective_split_anchor_source not in {"consensus", "mean", "self"}:
        raise ValueError(
            f"Unsupported objective_split_anchor_source: {opts.objective_split_anchor_source}"
        )
    raw_anchor_apply = str(
        getattr(opts, "objective_split_anchor_apply_optimizers", "all")
    ).lower()
    if raw_anchor_apply in {"all", "*"}:
        opts.objective_split_anchor_apply_optimizers = ["all"]
    elif raw_anchor_apply in {"none", ""}:
        opts.objective_split_anchor_apply_optimizers = []
    else:
        opts.objective_split_anchor_apply_optimizers = _parse_str_list(raw_anchor_apply)
    opts.objective_split_anchor_strength = float(
        max(0.0, getattr(opts, "objective_split_anchor_strength", 0.05))
    )
    opts.objective_split_anchor_mix_strength = float(
        getattr(opts, "objective_split_anchor_mix_strength", -1.0)
    )
    if opts.objective_split_anchor_mix_strength < 0.0:
        opts.objective_split_anchor_mix_strength = float(
            opts.objective_split_anchor_strength
        )
    else:
        opts.objective_split_anchor_mix_strength = float(
            max(0.0, opts.objective_split_anchor_mix_strength)
        )
    opts.objective_split_anchor_sample_ratio = float(
        max(0.0, getattr(opts, "objective_split_anchor_sample_ratio", 0.25))
    )
    opts.objective_split_anchor_sample_clip_ratio = float(
        max(0.0, getattr(opts, "objective_split_anchor_sample_clip_ratio", 0.25))
    )
    opts.objective_split_anchor_mean_pull = int(
        bool(getattr(opts, "objective_split_anchor_mean_pull", 1))
    )
    opts.objective_split_anchor_sample_injection = int(
        bool(getattr(opts, "objective_split_anchor_sample_injection", 1))
    )
    opts.objective_split_information_mode = str(
        getattr(opts, "objective_split_information_mode", "legacy_global")
    ).lower()
    if opts.objective_split_information_mode not in {"legacy_global", "local_only"}:
        raise ValueError(
            "Unsupported objective_split_information_mode: "
            f"{opts.objective_split_information_mode}"
        )
    if (
        opts.objective_split_persistent_sepcmaes_enable
        and opts.objective_split_information_mode != "local_only"
    ):
        raise ValueError(
            "objective_split_persistent_sepcmaes_enable requires "
            "objective_split_information_mode=local_only."
        )
    if (
        int(getattr(opts, "eval_save_event_slot_diagnostics", 0))
        and not opts.objective_split_event_slot_interleaving_enable
    ):
        raise ValueError(
            "eval_save_event_slot_diagnostics=1 requires "
            "objective_split_event_slot_interleaving_enable=1."
        )
    if (
        int(getattr(opts, "eval_save_boundary_compare_trace", 0))
        and not opts.objective_split_event_slot_interleaving_enable
    ):
        raise ValueError(
            "eval_save_boundary_compare_trace=1 requires "
            "objective_split_event_slot_interleaving_enable=1."
        )
    if (
        int(getattr(opts, "eval_save_optimizer_forensic_trace", 0))
        and not opts.objective_split_event_slot_interleaving_enable
    ):
        raise ValueError(
            "eval_save_optimizer_forensic_trace=1 requires "
            "objective_split_event_slot_interleaving_enable=1."
        )
    if (
        opts.objective_split_event_slot_interleaving_enable
        and opts.objective_split_information_mode != "local_only"
    ):
        raise ValueError(
            "objective_split_event_slot_interleaving_enable requires "
            "objective_split_information_mode=local_only."
        )
    if (
        opts.objective_split_event_slot_interleaving_enable
        and not int(opts.objective_split_cmaes_numeric_fail_soft)
    ):
        raise ValueError(
            "objective_split_event_slot_interleaving_enable=1 requires "
            "objective_split_cmaes_numeric_fail_soft=1 so a terminated "
            "CMAES session can finish its remaining K slots as "
            "communication-only."
        )
    if (
        opts.objective_split_event_slot_interleaving_enable
        and opts.objective_split_persistent_sepcmaes_enable
    ):
        raise ValueError(
            "objective_split_event_slot_interleaving_enable is event-local "
            "and cannot be combined with persistent SepCMAES before work "
            "package D."
        )
    if (
        opts.objective_split_event_slot_interleaving_enable
        and opts.objective_split_target_block_field_enable
    ):
        raise ValueError(
            "objective_split_event_slot_interleaving_enable cannot be "
            "combined with the target-block field path."
        )
    if (
        opts.objective_split_target_block_dual_clock_enable
        and not opts.objective_split_target_block_field_enable
    ):
        raise ValueError(
            "objective_split_target_block_dual_clock_enable requires "
            "objective_split_target_block_field_enable=1."
        )
    if (
        opts.objective_split_target_block_dual_clock_enable
        and not opts.objective_split_persistent_sepcmaes_enable
    ):
        raise ValueError(
            "objective_split_target_block_dual_clock_enable requires "
            "objective_split_persistent_sepcmaes_enable=1."
        )
    if (
        opts.objective_split_target_block_dual_clock_commit_lock_enable
        and not opts.objective_split_target_block_dual_clock_enable
    ):
        raise ValueError(
            "objective_split_target_block_dual_clock_commit_lock_enable "
            "requires objective_split_target_block_dual_clock_enable=1."
        )
    if (
        opts.objective_split_target_block_commit_credit_mode != "off"
        and not opts.objective_split_target_block_dual_clock_enable
    ):
        raise ValueError(
            "objective_split_target_block_commit_credit_mode requires "
            "objective_split_target_block_dual_clock_enable=1."
        )
    if (
        opts.objective_split_target_block_commit_credit_mode != "off"
        and not opts.objective_split_target_block_dual_clock_commit_lock_enable
    ):
        raise ValueError(
            "objective_split_target_block_commit_credit_mode requires "
            "objective_split_target_block_dual_clock_commit_lock_enable=1."
        )
    if (
        opts.objective_split_target_block_dormancy_recovery_enable
        and opts.objective_split_target_block_commit_credit_mode != "scale"
    ):
        raise ValueError(
            "objective_split_target_block_dormancy_recovery_enable requires "
            "objective_split_target_block_commit_credit_mode=scale."
        )
    if (
        opts.objective_split_target_block_direction_shadow_enable
        and not opts.objective_split_target_block_dormancy_recovery_enable
    ):
        raise ValueError(
            "objective_split_target_block_direction_shadow_enable requires "
            "objective_split_target_block_dormancy_recovery_enable=1."
        )
    if (
        opts.objective_split_target_block_challenge_response_mode != "off"
        and not opts.objective_split_target_block_dormancy_recovery_enable
    ):
        raise ValueError(
            "objective_split_target_block_challenge_response_mode requires "
            "objective_split_target_block_dormancy_recovery_enable=1."
        )
    if (
        opts.objective_split_target_block_challenge_response_mode != "off"
        and opts.objective_split_target_block_direction_shadow_enable
    ):
        raise ValueError(
            "NT080 direction shadow and NT081 challenge response are "
            "mutually exclusive."
        )
    opts.objective_split_global_monitor_enable = int(
        bool(getattr(opts, "objective_split_global_monitor_enable", 1))
    )
    if (
        opts.objective_split_information_mode == "legacy_global"
        and not opts.objective_split_global_monitor_enable
    ):
        raise ValueError(
            "objective_split_information_mode=legacy_global cannot disable "
            "objective_split_global_monitor_enable because exact global F is "
            "part of the historical control path. Use local_only to run with "
            "the detached global monitor disabled."
        )
    opts.objective_split_state_comm_cost_mode = str(
        getattr(opts, "objective_split_state_comm_cost_mode", "auto")
    ).lower()
    if opts.objective_split_state_comm_cost_mode not in {
        "auto",
        "legacy_untracked",
        "piggyback",
        "independent",
    }:
        raise ValueError(
            "Unsupported objective_split_state_comm_cost_mode: "
            f"{opts.objective_split_state_comm_cost_mode}"
        )
    opts.objective_split_committee_mode = str(
        getattr(opts, "objective_split_committee_mode", "off")
    ).lower()
    if opts.objective_split_committee_mode not in {"off", "shadow", "guide"}:
        raise ValueError(
            f"Unsupported objective_split_committee_mode: {opts.objective_split_committee_mode}"
        )
    opts.objective_split_committee_selection = str(
        getattr(opts, "objective_split_committee_selection", "target_block")
    ).lower()
    if opts.objective_split_committee_selection not in {
        "whole",
        "target_block",
        "both",
    }:
        raise ValueError(
            "Unsupported objective_split_committee_selection: "
            f"{opts.objective_split_committee_selection}"
        )
    if (
        opts.objective_split_committee_mode == "guide"
        and opts.objective_split_committee_selection == "both"
    ):
        raise ValueError(
            "objective_split_committee_selection=both is allowed only in shadow mode."
        )
    opts.objective_split_committee_mix_strength = float(
        min(
            1.0,
            max(
                0.0,
                getattr(opts, "objective_split_committee_mix_strength", 1.0),
            ),
        )
    )
    opts.objective_split_committee_acceptance_mode = str(
        getattr(opts, "objective_split_committee_acceptance_mode", "off")
    ).lower()
    if opts.objective_split_committee_acceptance_mode not in {
        "off",
        "report_improve",
    }:
        raise ValueError(
            "Unsupported objective_split_committee_acceptance_mode: "
            f"{opts.objective_split_committee_acceptance_mode}"
        )
    opts.objective_split_committee_acceptance_min_log_improve = float(
        max(
            0.0,
            getattr(
                opts,
                "objective_split_committee_acceptance_min_log_improve",
                0.0,
            ),
        )
    )
    if (
        opts.objective_split_committee_acceptance_mode == "report_improve"
        and opts.objective_split_committee_mode != "guide"
    ):
        raise ValueError(
            "objective_split_committee_acceptance_mode=report_improve requires "
            "objective_split_committee_mode=guide."
        )
    if (
        opts.objective_split_information_mode == "local_only"
        and opts.objective_split_committee_acceptance_mode == "report_improve"
    ):
        raise ValueError(
            "objective_split_information_mode=local_only forbids "
            "objective_split_committee_acceptance_mode=report_improve because "
            "that gate reconstructs exact global F."
        )
    opts.objective_split_committee_shadow_global_eval = int(
        bool(getattr(opts, "objective_split_committee_shadow_global_eval", 1))
    )
    if (
        opts.objective_split_information_mode == "local_only"
        and not opts.objective_split_global_monitor_enable
        and opts.objective_split_committee_mode == "shadow"
        and opts.objective_split_committee_shadow_global_eval
    ):
        raise ValueError(
            "local_only with objective_split_global_monitor_enable=0 cannot "
            "run committee shadow global evaluation. Disable "
            "objective_split_committee_shadow_global_eval or enable the "
            "detached monitor."
        )
    opts.objective_split_candidate_response_mode = str(
        getattr(opts, "objective_split_candidate_response_mode", "off")
    ).lower()
    if opts.objective_split_candidate_response_mode not in {
        "off",
        "shadow",
        "actuate",
    }:
        raise ValueError(
            "Unsupported objective_split_candidate_response_mode: "
            f"{opts.objective_split_candidate_response_mode}"
        )
    opts.objective_split_candidate_generator = str(
        getattr(opts, "objective_split_candidate_generator", "spsa_target")
    ).lower()
    if opts.objective_split_candidate_generator not in {
        "spsa_target",
        "multisecant_target",
        "multisecant_hybrid",
    }:
        raise ValueError(
            "Unsupported objective_split_candidate_generator: "
            f"{opts.objective_split_candidate_generator}"
        )
    opts.objective_split_candidate_multisecant_history_size = int(
        max(
            1,
            getattr(
                opts,
                "objective_split_candidate_multisecant_history_size",
                5,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_min_rank = int(
        max(
            1,
            getattr(
                opts,
                "objective_split_candidate_multisecant_min_rank",
                3,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_rank_tolerance = float(
        max(
            0.0,
            getattr(
                opts,
                "objective_split_candidate_multisecant_rank_tolerance",
                1e-6,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_condition_max = float(
        max(
            1.0,
            getattr(
                opts,
                "objective_split_candidate_multisecant_condition_max",
                1e4,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_center_distance_max_ratio = float(
        max(
            0.0,
            getattr(
                opts,
                "objective_split_candidate_multisecant_center_distance_max_ratio",
                0.10,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_max_age = int(
        max(
            0,
            getattr(
                opts,
                "objective_split_candidate_multisecant_max_age",
                8,
            ),
        )
    )
    opts.objective_split_candidate_multisecant_gradient_min_norm = float(
        max(
            0.0,
            getattr(
                opts,
                "objective_split_candidate_multisecant_gradient_min_norm",
                1e-8,
            ),
        )
    )
    if (
        opts.objective_split_candidate_response_mode != "off"
        and opts.objective_split_information_mode != "local_only"
        and not bool(
            int(
                getattr(
                    opts,
                    "objective_split_d6_pre_generator_selector_enable",
                    0,
                )
            )
        )
    ):
        raise ValueError(
            "Candidate-response control outside local_only is reserved for "
            "the explicit D6 frozen-primary route."
        )
    if (
        opts.objective_split_candidate_response_mode != "off"
        and opts.objective_split_committee_mode != "off"
    ):
        raise ValueError(
            "D3-P candidate-response and the C13 committee cannot be active "
            "together in the first causal implementation."
        )
    for name, default in (
        ("objective_split_candidate_probe_scale", 0.25),
        ("objective_split_candidate_trust_scale", 0.25),
        ("objective_split_candidate_trust_min_ratio", 0.002),
        ("objective_split_candidate_trust_max_ratio", 0.05),
        ("objective_split_candidate_probe_min_ratio", 0.0005),
        ("objective_split_candidate_probe_max_ratio", 0.01),
    ):
        setattr(opts, name, float(max(0.0, getattr(opts, name, default))))
    if (
        opts.objective_split_candidate_trust_max_ratio
        < opts.objective_split_candidate_trust_min_ratio
    ):
        raise ValueError(
            "objective_split_candidate_trust_max_ratio must be >= "
            "objective_split_candidate_trust_min_ratio."
        )
    if (
        opts.objective_split_candidate_probe_max_ratio
        < opts.objective_split_candidate_probe_min_ratio
    ):
        raise ValueError(
            "objective_split_candidate_probe_max_ratio must be >= "
            "objective_split_candidate_probe_min_ratio."
        )
    opts.objective_split_candidate_confidence_min = float(
        min(
            1.0,
            max(
                0.0,
                getattr(opts, "objective_split_candidate_confidence_min", 1e-4),
            ),
        )
    )
    opts.objective_split_candidate_support_min = float(
        min(
            1.0,
            max(0.0, getattr(opts, "objective_split_candidate_support_min", 0.60)),
        )
    )
    opts.objective_split_candidate_actuator_beta = float(
        min(
            1.0,
            max(0.0, getattr(opts, "objective_split_candidate_actuator_beta", 0.50)),
        )
    )
    opts.objective_split_candidate_history_obs_enable = int(
        bool(getattr(opts, "objective_split_candidate_history_obs_enable", 0))
    )
    opts.objective_split_candidate_actuator_action_enable = int(
        bool(getattr(opts, "objective_split_candidate_actuator_action_enable", 0))
    )
    opts.objective_split_candidate_actuator_candidates = [
        float(min(1.0, max(0.0, x)))
        for x in _parse_float_list(
            str(
                getattr(
                    opts,
                    "objective_split_candidate_actuator_candidates",
                    "0.0,0.25,0.5",
                )
            )
        )
    ]
    if len(opts.objective_split_candidate_actuator_candidates) == 0:
        raise ValueError(
            "objective_split_candidate_actuator_candidates must be non-empty."
        )
    opts.objective_split_candidate_actuator_initial_probs = [
        float(max(0.0, x))
        for x in _parse_float_list(
            str(
                getattr(
                    opts,
                    "objective_split_candidate_actuator_initial_probs",
                    "0.2,0.7,0.1",
                )
            )
        )
    ]
    if (
        len(opts.objective_split_candidate_actuator_initial_probs)
        != len(opts.objective_split_candidate_actuator_candidates)
    ):
        raise ValueError(
            "objective_split_candidate_actuator_initial_probs length must match "
            "objective_split_candidate_actuator_candidates."
        )
    prior_sum = float(sum(opts.objective_split_candidate_actuator_initial_probs))
    if prior_sum <= 0.0:
        raise ValueError(
            "objective_split_candidate_actuator_initial_probs must have positive sum."
        )
    opts.objective_split_candidate_actuator_initial_probs = [
        float(x / prior_sum)
        for x in opts.objective_split_candidate_actuator_initial_probs
    ]
    opts.objective_split_candidate_actuator_forced_action = int(
        getattr(opts, "objective_split_candidate_actuator_forced_action", -1)
    )
    opts.objective_split_d5_two_stage_actuator_enable = int(
        bool(getattr(opts, "objective_split_d5_two_stage_actuator_enable", 0))
    )
    opts.objective_split_d5_actuator_reward_mode = str(
        getattr(opts, "objective_split_d5_actuator_reward_mode", "mixed")
    ).lower()
    if opts.objective_split_d5_actuator_reward_mode not in {
        "mixed",
        "local_credit",
    }:
        raise ValueError(
            "objective_split_d5_actuator_reward_mode must be mixed or "
            "local_credit."
        )
    if opts.objective_split_candidate_actuator_forced_action < -1:
        raise ValueError(
            "objective_split_candidate_actuator_forced_action must be -1 "
            "(disabled) or a valid nonnegative action index."
        )
    if opts.objective_split_candidate_history_obs_enable and (
        opts.objective_split_candidate_response_mode == "off"
    ):
        raise ValueError(
            "candidate history observations require candidate_response_mode "
            "shadow or actuate."
        )
    if opts.objective_split_candidate_actuator_action_enable and (
        opts.objective_split_candidate_response_mode != "actuate"
    ):
        raise ValueError(
            "candidate actuator actions require "
            "objective_split_candidate_response_mode=actuate."
        )
    if opts.objective_split_d5_two_stage_actuator_enable:
        if not opts.objective_split_candidate_actuator_action_enable:
            raise ValueError(
                "D5 two-stage actuator requires "
                "objective_split_candidate_actuator_action_enable=1 so the "
                "D4-B primary checkpoint keeps its original architecture."
            )
        if opts.objective_split_candidate_response_mode != "actuate":
            raise ValueError(
                "D5 two-stage actuator requires "
                "objective_split_candidate_response_mode=actuate."
            )
        if opts.objective_split_information_mode != "local_only":
            raise ValueError(
                "D5 two-stage actuator requires "
                "objective_split_information_mode=local_only."
            )
    opts.objective_split_d6_pre_generator_selector_enable = int(
        bool(
            getattr(
                opts,
                "objective_split_d6_pre_generator_selector_enable",
                0,
            )
        )
    )
    if opts.objective_split_d6_pre_generator_selector_enable:
        if opts.objective_split_candidate_response_mode != "actuate":
            raise ValueError(
                "D6 pre-generator selector requires "
                "objective_split_candidate_response_mode=actuate."
            )
        if not opts.objective_split_candidate_actuator_action_enable:
            raise ValueError(
                "D6 pre-generator selector requires "
                "objective_split_candidate_actuator_action_enable=1 for the "
                "separate post-verifier actuator stage."
            )
        if not opts.objective_split_candidate_history_obs_enable:
            raise ValueError(
                "D6 pre-generator selector requires "
                "objective_split_candidate_history_obs_enable=1 so the "
                "runtime observation explicitly contains the five candidate "
                "history fields outside the frozen primary projection."
            )
        if opts.objective_split_candidate_generator not in {
            "spsa_target",
            "multisecant_hybrid",
        }:
            raise ValueError(
                "D6 pre-generator selector supports only SPSA/Hybrid "
                "candidate semantics."
            )
    if opts.objective_split_candidate_actuator_forced_action >= len(
        opts.objective_split_candidate_actuator_candidates
    ):
        raise ValueError(
            "objective_split_candidate_actuator_forced_action is outside the "
            "candidate actuator action range."
        )
    opts.objective_split_candidate_shadow_global_eval_interval = int(
        max(
            0,
            getattr(
                opts,
                "objective_split_candidate_shadow_global_eval_interval",
                20,
            ),
        )
    )
    if (
        opts.objective_split_candidate_response_mode == "shadow"
        and opts.objective_split_candidate_shadow_global_eval_interval > 0
        and not opts.objective_split_global_monitor_enable
    ):
        raise ValueError(
            "D3-P detached shadow global sampling requires "
            "objective_split_global_monitor_enable=1, or set its interval to 0."
        )
    opts.objective_split_optimizer_guide_numeric_guard = int(
        bool(getattr(opts, "objective_split_optimizer_guide_numeric_guard", 0))
    )
    opts.objective_split_cmaes_numeric_fail_soft = int(
        bool(getattr(opts, "objective_split_cmaes_numeric_fail_soft", 0))
    )
    opts.objective_split_optimizer_numeric_telemetry = int(
        bool(getattr(opts, "objective_split_optimizer_numeric_telemetry", 0))
    )
    opts.objective_split_optimizer_numeric_counter = int(
        bool(getattr(opts, "objective_split_optimizer_numeric_counter", 0))
    )
    opts.objective_split_optimizer_numeric_counter_dir = str(
        getattr(opts, "objective_split_optimizer_numeric_counter_dir", "")
    ).strip()
    opts.objective_split_optimizer_numeric_telemetry_dir = str(
        getattr(opts, "objective_split_optimizer_numeric_telemetry_dir", "")
    ).strip()
    opts.objective_split_optimizer_numeric_forensics = int(
        bool(getattr(opts, "objective_split_optimizer_numeric_forensics", 0))
    )
    opts.objective_split_optimizer_numeric_forensics_dir = str(
        getattr(opts, "objective_split_optimizer_numeric_forensics_dir", "")
    ).strip()
    opts.objective_split_optimizer_numeric_forensics_jump_log10 = float(
        max(0.0, getattr(opts, "objective_split_optimizer_numeric_forensics_jump_log10", 2.0))
    )
    opts.objective_split_optimizer_numeric_forensics_jump_max_windows = int(
        max(1, min(32, getattr(opts, "objective_split_optimizer_numeric_forensics_jump_max_windows", 1)))
    )
    opts.objective_split_optimizer_numeric_forensics_jump_alpha_gate = int(
        bool(getattr(opts, "objective_split_optimizer_numeric_forensics_jump_alpha_gate", 0))
    )
    opts.objective_split_optimizer_numeric_forensics_jump_early_write = int(
        bool(getattr(opts, "objective_split_optimizer_numeric_forensics_jump_early_write", 0))
    )
    opts.objective_split_optimizer_numeric_forensics_jump_early_milestone_log10 = float(
        max(0.0, getattr(opts, "objective_split_optimizer_numeric_forensics_jump_early_milestone_log10", 0.5))
    )
    opts.objective_split_optimizer_numeric_forensics_jump_mid_milestone_log10 = float(
        max(0.0, getattr(opts, "objective_split_optimizer_numeric_forensics_jump_mid_milestone_log10", 2.0))
    )
    for _target_name in ("function", "seed", "agent"):
        _target_key = f"objective_split_optimizer_numeric_forensics_target_{_target_name}"
        setattr(opts, _target_key, int(getattr(opts, _target_key, -1)))
    opts.objective_split_optimizer_guide_sigma_exp_clip = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_sigma_exp_clip", 20.0))
    )
    opts.objective_split_optimizer_guide_sigma_clip_ratio = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_sigma_clip_ratio", 0.5))
    )
    opts.objective_split_optimizer_guide_sample_clip_ratio = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_sample_clip_ratio", 0.0))
    )
    opts.objective_split_optimizer_guide_internal_mode = str(
        getattr(opts, "objective_split_optimizer_guide_internal_mode", "off")
    ).lower()
    supported_internal_modes = {"off", "mean", "mean_path", "mean_path_covdiag"}
    if opts.objective_split_optimizer_guide_internal_mode not in supported_internal_modes:
        raise ValueError(
            "Unsupported objective_split_optimizer_guide_internal_mode: "
            f"{opts.objective_split_optimizer_guide_internal_mode}. "
            f"Supported: {sorted(supported_internal_modes)}"
        )
    raw_internal_apply = str(
        getattr(opts, "objective_split_optimizer_guide_internal_apply_optimizers", "cmaes,sepcmaes")
    ).lower()
    if raw_internal_apply in {"all", "*"}:
        opts.objective_split_optimizer_guide_internal_apply_optimizers = ["all"]
    elif raw_internal_apply in {"none", ""}:
        opts.objective_split_optimizer_guide_internal_apply_optimizers = []
    else:
        opts.objective_split_optimizer_guide_internal_apply_optimizers = _parse_str_list(raw_internal_apply)
    opts.objective_split_optimizer_guide_internal_mean_lr = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_mean_lr", 0.0))
    )
    opts.objective_split_optimizer_guide_internal_path_lr = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_path_lr", 0.0))
    )
    opts.objective_split_optimizer_guide_internal_cov_lr = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_cov_lr", 0.0))
    )
    opts.objective_split_optimizer_guide_internal_agree_cos_min = float(
        max(
            -1.0,
            min(1.0, getattr(opts, "objective_split_optimizer_guide_internal_agree_cos_min", -0.25)),
        )
    )
    opts.objective_split_optimizer_guide_internal_max_step_ratio = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_max_step_ratio", 0.05))
    )
    opts.objective_split_optimizer_guide_internal_max_rel_step = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_max_rel_step", 0.5))
    )
    opts.objective_split_optimizer_guide_internal_path_max_rel_norm = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_path_max_rel_norm", 0.5))
    )
    opts.objective_split_optimizer_guide_internal_cov_rank1_clip = float(
        max(0.0, getattr(opts, "objective_split_optimizer_guide_internal_cov_rank1_clip", 0.02))
    )
    opts.objective_split_optimizer_guide_internal_disable_sample_injection = int(
        bool(getattr(opts, "objective_split_optimizer_guide_internal_disable_sample_injection", 0))
    )
    opts.objective_split_collab_action_enable = int(
        bool(getattr(opts, "objective_split_collab_action_enable", 0))
    )
    opts.objective_split_collab_modes = [
        str(x).strip().lower()
        for x in _parse_str_list(
            str(getattr(opts, "objective_split_collab_modes", "consensus,self,leader,soft_diversify"))
        )
        if str(x).strip()
    ]
    if len(opts.objective_split_collab_modes) == 0:
        opts.objective_split_collab_modes = ["consensus"]
    supported_collab_modes = {"consensus", "self", "leader", "soft_diversify"}
    bad_collab_modes = [
        x for x in opts.objective_split_collab_modes if x not in supported_collab_modes
    ]
    if bad_collab_modes:
        raise ValueError(
            "Unsupported objective_split_collab_modes: "
            f"{bad_collab_modes}. Supported: {sorted(supported_collab_modes)}"
        )
    opts.objective_split_collab_soft_diversify_scale = float(
        max(0.0, getattr(opts, "objective_split_collab_soft_diversify_scale", 0.25))
    )
    opts.objective_split_collab_leader_fallback = str(
        getattr(opts, "objective_split_collab_leader_fallback", "consensus")
    ).lower()
    if opts.objective_split_collab_leader_fallback not in {"consensus", "off"}:
        raise ValueError(
            "--objective_split_collab_leader_fallback must be consensus or off"
        )
    opts.objective_split_guide_scale_action_enable = int(
        bool(getattr(opts, "objective_split_guide_scale_action_enable", 0))
    )
    opts.objective_split_guide_scale_candidates = [
        float(max(0.0, x))
        for x in _parse_float_list(
            str(getattr(opts, "objective_split_guide_scale_candidates", "1.0,0.5,0.75,1.25"))
        )
    ]
    if len(opts.objective_split_guide_scale_candidates) == 0:
        opts.objective_split_guide_scale_candidates = [1.0]
    opts.mappo_update_mode = str(getattr(opts, "mappo_update_mode", "joint")).lower()
    if opts.mappo_update_mode not in {"joint", "split_env"}:
        raise ValueError(f"Unsupported mappo_update_mode: {opts.mappo_update_mode}")
    opts.mappo_split_env_adv_norm = int(bool(getattr(opts, "mappo_split_env_adv_norm", 1)))
    opts.feature_num_1 = resolve_mappo_obs_dim(opts)
    opts.wsn_target_num_list = _parse_int_list_allow_empty(opts.wsn_target_num_list)
    opts.dbof1f10_max_fes_list = _parse_budget_int_list_allow_empty(opts.dbof1f10_max_fes_list)
    opts.cdo_max_fes_list = _parse_budget_int_list_allow_empty(opts.cdo_max_fes_list)
    opts.cdo_bench_max_fes_list = _parse_budget_int_list_allow_empty(opts.cdo_bench_max_fes_list)
    opts.wsn_max_fes_list = _parse_budget_int_list_allow_empty(opts.wsn_max_fes_list)
    opts.masoie_wsn_target_num_list = _parse_int_list_allow_empty(opts.masoie_wsn_target_num_list)
    opts.masoie_wsn_max_fes_list = _parse_budget_int_list_allow_empty(opts.masoie_wsn_max_fes_list)
    opts.eval_record_fes_list = _parse_budget_int_list_allow_empty(
        getattr(opts, "eval_record_fes_list", "")
    )
    if any(int(x) <= 0 for x in opts.eval_record_fes_list):
        raise ValueError("--eval_record_fes_list values must all be positive.")
    opts.eval_record_fes_list = sorted(
        set(int(x) for x in opts.eval_record_fes_list)
    )
    # backward-compatible alias: if old cmd used --divide_method WSNLocation
    if opts.divide_method == "WSNLocation" and opts.benchmark_name == "CEC2013LSGO":
        opts.benchmark_name = "WSNLocation"
        opts.divide_method = "CEC2013LSGO"

    # No implicit benchmark-based default fun ids.
    opts.mappo_reward_team_weight = float(min(1.0, max(0.0, opts.mappo_reward_team_weight)))
    opts.mappo_actor_local_beta = float(max(0.0, opts.mappo_actor_local_beta))
    opts.mappo_entropy_coef_opt = float(max(0.0, opts.mappo_entropy_coef_opt))
    opts.mappo_entropy_coef_cfg = float(max(0.0, opts.mappo_entropy_coef_cfg))
    opts.mappo_entropy_coef_res = float(max(0.0, opts.mappo_entropy_coef_res))
    opts.mappo_entropy_coef_comm = float(getattr(opts, "mappo_entropy_coef_comm", -1.0))
    if opts.mappo_entropy_coef_comm < 0.0:
        opts.mappo_entropy_coef_comm = float(max(0.0, opts.entropy_coef))
    else:
        opts.mappo_entropy_coef_comm = float(max(0.0, opts.mappo_entropy_coef_comm))
    opts.mappo_entropy_coef_collab = float(getattr(opts, "mappo_entropy_coef_collab", -1.0))
    if opts.mappo_entropy_coef_collab < 0.0:
        opts.mappo_entropy_coef_collab = float(max(0.0, opts.entropy_coef))
    else:
        opts.mappo_entropy_coef_collab = float(max(0.0, opts.mappo_entropy_coef_collab))
    opts.mappo_entropy_coef_guide_scale = float(getattr(opts, "mappo_entropy_coef_guide_scale", -1.0))
    if opts.mappo_entropy_coef_guide_scale < 0.0:
        opts.mappo_entropy_coef_guide_scale = float(max(0.0, opts.entropy_coef))
    else:
        opts.mappo_entropy_coef_guide_scale = float(max(0.0, opts.mappo_entropy_coef_guide_scale))
    opts.mappo_entropy_coef_actuator = float(getattr(opts, "mappo_entropy_coef_actuator", -1.0))
    if opts.mappo_entropy_coef_actuator < 0.0:
        opts.mappo_entropy_coef_actuator = float(max(0.0, opts.entropy_coef))
    else:
        opts.mappo_entropy_coef_actuator = float(max(0.0, opts.mappo_entropy_coef_actuator))
    opts.mappo_policy_loss_weight_opt = float(max(0.0, opts.mappo_policy_loss_weight_opt))
    opts.mappo_policy_loss_weight_cfg = float(max(0.0, opts.mappo_policy_loss_weight_cfg))
    opts.mappo_policy_loss_weight_res = float(max(0.0, opts.mappo_policy_loss_weight_res))
    opts.mappo_policy_loss_weight_comm = float(
        max(0.0, getattr(opts, "mappo_policy_loss_weight_comm", 1.0))
    )
    opts.mappo_policy_loss_weight_collab = float(
        max(0.0, getattr(opts, "mappo_policy_loss_weight_collab", 1.0))
    )
    opts.mappo_policy_loss_weight_guide_scale = float(
        max(0.0, getattr(opts, "mappo_policy_loss_weight_guide_scale", 1.0))
    )
    opts.mappo_policy_loss_weight_actuator = float(
        max(0.0, getattr(opts, "mappo_policy_loss_weight_actuator", 1.0))
    )
    opts.mappo_value_loss_weight_ref = float(max(0.0, opts.mappo_value_loss_weight_ref))
    opts.mappo_value_loss_weight_opt = float(max(0.0, opts.mappo_value_loss_weight_opt))
    opts.mappo_value_loss_weight_cfg = float(max(0.0, opts.mappo_value_loss_weight_cfg))
    opts.mappo_value_loss_weight_res = float(max(0.0, opts.mappo_value_loss_weight_res))
    opts.mappo_value_loss_weight_comm = float(
        max(0.0, getattr(opts, "mappo_value_loss_weight_comm", 1.0))
    )
    opts.mappo_value_loss_weight_collab = float(
        max(0.0, getattr(opts, "mappo_value_loss_weight_collab", 1.0))
    )
    opts.mappo_value_loss_weight_guide_scale = float(
        max(0.0, getattr(opts, "mappo_value_loss_weight_guide_scale", 1.0))
    )
    opts.mappo_value_loss_weight_actuator = float(
        max(0.0, getattr(opts, "mappo_value_loss_weight_actuator", 1.0))
    )
    opts.d5_actuator_hidden_dim = int(
        max(1, getattr(opts, "d5_actuator_hidden_dim", 64))
    )
    opts.d5_actuator_action_emb_dim = int(
        max(1, getattr(opts, "d5_actuator_action_emb_dim", 8))
    )
    opts.d5_actuator_initial_probs = [
        float(max(0.0, x))
        for x in _parse_float_list(
            str(getattr(opts, "d5_actuator_initial_probs", "1,1,1"))
        )
    ]
    if len(opts.d5_actuator_initial_probs) != len(
        opts.objective_split_candidate_actuator_candidates
    ):
        raise ValueError(
            "d5_actuator_initial_probs length must match "
            "objective_split_candidate_actuator_candidates."
        )
    d5_prior_sum = float(sum(opts.d5_actuator_initial_probs))
    if d5_prior_sum <= 0.0:
        raise ValueError("d5_actuator_initial_probs must have positive sum.")
    opts.d5_actuator_initial_probs = [
        float(x / d5_prior_sum) for x in opts.d5_actuator_initial_probs
    ]
    opts.d5_actuator_lr_actor = float(
        getattr(opts, "d5_actuator_lr_actor", -1.0)
    )
    if opts.d5_actuator_lr_actor < 0.0:
        opts.d5_actuator_lr_actor = float(opts.lr_model)
    opts.d5_actuator_lr_critic = float(
        getattr(opts, "d5_actuator_lr_critic", -1.0)
    )
    if opts.d5_actuator_lr_critic < 0.0:
        opts.d5_actuator_lr_critic = float(opts.lr_critic)
    opts.d5_primary_checkpoint_path = str(
        getattr(opts, "d5_primary_checkpoint_path", "")
    ).strip()
    opts.d5_eval_forced_actuator_action = int(
        getattr(opts, "d5_eval_forced_actuator_action", -1)
    )
    if opts.d5_eval_forced_actuator_action < -1 or (
        opts.d5_eval_forced_actuator_action
        >= len(opts.objective_split_candidate_actuator_candidates)
    ):
        raise ValueError(
            "d5_eval_forced_actuator_action must be -1 or a valid actuator "
            "action index."
        )
    opts.d6_generator_hidden_dim = int(
        max(1, getattr(opts, "d6_generator_hidden_dim", 64))
    )
    opts.d6_generator_action_emb_dim = int(
        max(1, getattr(opts, "d6_generator_action_emb_dim", 8))
    )
    opts.d6_generator_initial_probs = [
        float(max(0.0, x))
        for x in _parse_float_list(
            str(getattr(opts, "d6_generator_initial_probs", "0.5,0.5"))
        )
    ]
    if len(opts.d6_generator_initial_probs) != 2:
        raise ValueError(
            "d6_generator_initial_probs must contain exactly two values "
            "for SPSA and Hybrid."
        )
    d6_prior_sum = float(sum(opts.d6_generator_initial_probs))
    if d6_prior_sum <= 0.0:
        raise ValueError("d6_generator_initial_probs must have positive sum.")
    opts.d6_generator_initial_probs = [
        float(x / d6_prior_sum) for x in opts.d6_generator_initial_probs
    ]
    opts.d6_generator_lr_actor = float(
        getattr(opts, "d6_generator_lr_actor", -1.0)
    )
    if opts.d6_generator_lr_actor < 0.0:
        opts.d6_generator_lr_actor = float(opts.lr_model)
    opts.d6_generator_lr_critic = float(
        getattr(opts, "d6_generator_lr_critic", -1.0)
    )
    if opts.d6_generator_lr_critic < 0.0:
        opts.d6_generator_lr_critic = float(opts.lr_critic)
    opts.d6_primary_checkpoint_path = str(
        getattr(opts, "d6_primary_checkpoint_path", "")
    ).strip()
    opts.d6_primary_source_kind = str(
        getattr(opts, "d6_primary_source_kind", "auto")
    ).lower()
    opts.d6_eval_forced_generator_action = int(
        getattr(opts, "d6_eval_forced_generator_action", -1)
    )
    if opts.d6_eval_forced_generator_action not in {-1, 0, 1}:
        raise ValueError(
            "d6_eval_forced_generator_action must be -1, 0, or 1."
        )
    opts.d6_post_actuator_forced_action = int(
        getattr(opts, "d6_post_actuator_forced_action", 2)
    )
    if opts.d6_post_actuator_forced_action < 0 or (
        opts.d6_post_actuator_forced_action
        >= len(opts.objective_split_candidate_actuator_candidates)
    ):
        raise ValueError(
            "d6_post_actuator_forced_action must be a valid actuator action "
            "index."
        )
    opts.d6_eval_forced_post_actuator_action = int(
        getattr(opts, "d6_eval_forced_post_actuator_action", -1)
    )
    if opts.d6_eval_forced_post_actuator_action < -1 or (
        opts.d6_eval_forced_post_actuator_action
        >= len(opts.objective_split_candidate_actuator_candidates)
    ):
        raise ValueError(
            "d6_eval_forced_post_actuator_action must be -1 or a valid "
            "actuator action index."
        )
    opts.forced_optimizer_enable = int(1 if int(getattr(opts, "forced_optimizer_enable", 0)) else 0)
    opts.forced_optimizer_prob = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_prob", 0.0))))
    opts.forced_optimizer_warmup_ratio = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_warmup_ratio", 0.0))))
    opts.forced_optimizer_mode = str(getattr(opts, "forced_optimizer_mode", "cycle")).lower()
    # 训练执行轴（epoch_start..epoch_end）与日程轴（schedule_start..schedule_end）分离：
    # 日程轴只决定 forced optimizer 的 warmup 比例分母；缺省时 runner 让其跟随执行轴。
    _schedule_start = int(getattr(opts, "schedule_start_epoch", -1))
    _schedule_end = int(getattr(opts, "schedule_end_epoch", -1))
    if _schedule_start >= 0 or _schedule_end >= 0:
        if _schedule_start < 0 or _schedule_end < 0:
            raise ValueError(
                "schedule_start_epoch 与 schedule_end_epoch 必须同时给出。"
            )
        if _schedule_end <= _schedule_start:
            raise ValueError(
                "schedule_end_epoch 必须大于 schedule_start_epoch。"
            )
        if _schedule_end < int(opts.epoch_end):
            raise ValueError(
                "日程轴终点不得短于本次执行终点："
                f"schedule_end_epoch={_schedule_end}, epoch_end={opts.epoch_end}。"
            )
        opts.resume_schedule_start_epoch = _schedule_start
        opts.resume_schedule_end_epoch = _schedule_end
    opts.mappo_critic_agent_emb_dim = int(max(1, opts.mappo_critic_agent_emb_dim))
    opts.mappo_critic_action_emb_dim = int(max(1, opts.mappo_critic_action_emb_dim))
    opts.mappo_diag_reward_frac = float(min(0.5, max(0.05, opts.mappo_diag_reward_frac)))
    opts.mappo_diag_reward_tol_ratio = float(max(0.0, opts.mappo_diag_reward_tol_ratio))
    opts.mappo_diag_kl_low = float(max(0.0, opts.mappo_diag_kl_low))
    opts.mappo_diag_kl_high = float(max(opts.mappo_diag_kl_low, opts.mappo_diag_kl_high))
    opts.mappo_diag_clip_low = float(max(0.0, opts.mappo_diag_clip_low))
    opts.mappo_diag_clip_high = float(max(opts.mappo_diag_clip_low, opts.mappo_diag_clip_high))
    opts.mappo_diag_entropy_low = float(max(0.0, opts.mappo_diag_entropy_low))
    opts.mappo_diag_collapse_ratio = float(min(1.0, max(0.0, opts.mappo_diag_collapse_ratio)))
    opts.mappo_diag_adv_imbalance_factor = float(max(1.0, opts.mappo_diag_adv_imbalance_factor))
    opts.mappo_diag_adv_abs_mean_min = float(max(0.0, opts.mappo_diag_adv_abs_mean_min))
    if opts.objective_split_rvcpd_arm != "off":
        rvcpd_conflicts = []
        if opts.objective_split_information_mode != "local_only":
            rvcpd_conflicts.append("information_mode must be local_only")
        if opts.objective_split_neighbor_obs_mode != "none":
            rvcpd_conflicts.append("neighbor_obs_mode must be none")
        if opts.objective_split_consensus_reward_weight != 0.0:
            rvcpd_conflicts.append("consensus_reward_weight must be 0")
        if opts.objective_split_anchor_enable:
            rvcpd_conflicts.append("anchor_enable must be 0")
        if opts.objective_split_committee_mode != "off":
            rvcpd_conflicts.append("committee_mode must be off")
        if opts.objective_split_collab_action_enable:
            rvcpd_conflicts.append("collab_action_enable must be 0")
        if opts.objective_split_guide_scale_action_enable:
            rvcpd_conflicts.append("guide_scale_action_enable must be 0")
        if opts.objective_split_d6_pre_generator_selector_enable:
            rvcpd_conflicts.append("D6 selector must be disabled")
        integration_mode = opts.objective_split_rvcpd_integration_mode
        if integration_mode == "isolated":
            if opts.objective_split_consensus != "none":
                rvcpd_conflicts.append("consensus must be none")
            if opts.objective_split_state_comm_mode != "none":
                rvcpd_conflicts.append("state_comm_mode must be none")
            if opts.objective_split_comm_action_enable:
                rvcpd_conflicts.append("comm_action_enable must be 0")
            if opts.objective_split_optimizer_guide_enable:
                rvcpd_conflicts.append("optimizer_guide_enable must be 0")
            if opts.objective_split_candidate_response_mode != "off":
                rvcpd_conflicts.append("candidate_response_mode must be off")
            if opts.objective_split_candidate_history_obs_enable:
                rvcpd_conflicts.append(
                    "candidate_history_obs_enable must be 0"
                )
            if opts.objective_split_candidate_actuator_action_enable:
                rvcpd_conflicts.append(
                    "candidate_actuator_action_enable must be 0"
                )
            if opts.objective_split_d5_two_stage_actuator_enable:
                rvcpd_conflicts.append(
                    "D5 two-stage actuator must be disabled"
                )
        else:
            if opts.objective_split_rvcpd_arm not in {"p0", "p1"}:
                rvcpd_conflicts.append(
                    "d5_post_commit supports only P0/P1"
                )
            if not opts.objective_split_d5_two_stage_actuator_enable:
                rvcpd_conflicts.append(
                    "d5_post_commit requires the D5 two-stage actuator"
                )
            if opts.objective_split_consensus != "graph_mean":
                rvcpd_conflicts.append(
                    "d5_post_commit requires graph_mean consensus"
                )
            if opts.objective_split_state_comm_mode != "graph_mean":
                rvcpd_conflicts.append(
                    "d5_post_commit requires graph_mean state communication"
                )
            if not opts.objective_split_comm_action_enable:
                rvcpd_conflicts.append(
                    "d5_post_commit requires the D5 communication action"
                )
            if not opts.objective_split_optimizer_guide_enable:
                rvcpd_conflicts.append(
                    "d5_post_commit requires the D5 optimizer guide"
                )
            if opts.objective_split_candidate_response_mode != "actuate":
                rvcpd_conflicts.append(
                    "d5_post_commit requires candidate_response_mode=actuate"
                )
            if not opts.objective_split_candidate_history_obs_enable:
                rvcpd_conflicts.append(
                    "d5_post_commit requires candidate history observations"
                )
            if not opts.objective_split_candidate_actuator_action_enable:
                rvcpd_conflicts.append(
                    "d5_post_commit requires the candidate actuator action"
                )
        if rvcpd_conflicts:
            contract_name = (
                "isolated strict-local"
                if opts.objective_split_rvcpd_integration_mode == "isolated"
                else "D5 post-commit"
            )
            raise ValueError(
                f"RVCPD {contract_name} integration contract failed: "
                + "; ".join(rvcpd_conflicts)
                + "."
            )
    if opts.objective_split_guide_replacement_mode != "off":
        replacement_conflicts = []
        if opts.objective_split_information_mode != "local_only":
            replacement_conflicts.append("information_mode must be local_only")
        if not opts.objective_split_d5_two_stage_actuator_enable:
            replacement_conflicts.append("D5 two-stage actuator must be enabled")
        if opts.objective_split_consensus != "graph_mean":
            replacement_conflicts.append("consensus must be graph_mean")
        if opts.objective_split_state_comm_mode != "graph_mean":
            replacement_conflicts.append("state_comm_mode must be graph_mean")
        if not opts.objective_split_comm_action_enable:
            replacement_conflicts.append("D5 communication action must be enabled")
        if not opts.objective_split_optimizer_guide_enable:
            replacement_conflicts.append("optimizer guide must be enabled")
        if opts.objective_split_optimizer_guide_source != "ccsa_direction_momentum":
            replacement_conflicts.append(
                "optimizer guide source must be ccsa_direction_momentum"
            )
        if opts.objective_split_ccsa_momentum_update_mode != "always":
            replacement_conflicts.append(
                "ccsa momentum update mode must be always"
            )
        if opts.objective_split_candidate_response_mode != "actuate":
            replacement_conflicts.append("candidate_response_mode must be actuate")
        if not opts.objective_split_candidate_history_obs_enable:
            replacement_conflicts.append("candidate history observations must be enabled")
        if not opts.objective_split_candidate_actuator_action_enable:
            replacement_conflicts.append("candidate actuator action must be enabled")
        if opts.objective_split_anchor_enable:
            replacement_conflicts.append("anchor must be disabled")
        if opts.objective_split_committee_mode != "off":
            replacement_conflicts.append("committee must be disabled")
        if opts.objective_split_d6_pre_generator_selector_enable:
            replacement_conflicts.append("D6 selector must be disabled")
        if opts.objective_split_rvcpd_arm != "off":
            replacement_conflicts.append("RVCPD must be disabled")
        if replacement_conflicts:
            raise ValueError(
                "Guide replacement frozen-evaluation contract failed: "
                + "; ".join(replacement_conflicts)
                + "."
            )
    if opts.objective_split_collective_guide_mode != "off":
        collective_conflicts = []
        if opts.objective_split_guide_replacement_mode != "off":
            collective_conflicts.append("guide replacement must be disabled")
        if opts.objective_split_information_mode != "local_only":
            collective_conflicts.append("information_mode must be local_only")
        if not opts.objective_split_d5_two_stage_actuator_enable:
            collective_conflicts.append("D5 two-stage actuator must be enabled")
        if opts.objective_split_consensus != "graph_mean":
            collective_conflicts.append("consensus must be graph_mean")
        if opts.objective_split_state_comm_mode != "graph_mean":
            collective_conflicts.append("state_comm_mode must be graph_mean")
        if not opts.objective_split_comm_action_enable:
            collective_conflicts.append("D5 communication action must be enabled")
        if not opts.objective_split_record_comm_cost:
            collective_conflicts.append("communication cost recording must be enabled")
        if not opts.objective_split_optimizer_guide_enable:
            collective_conflicts.append("optimizer guide must be enabled")
        if opts.objective_split_optimizer_guide_source != "ccsa_direction_momentum":
            collective_conflicts.append(
                "optimizer guide source must be ccsa_direction_momentum"
            )
        if opts.objective_split_ccsa_momentum_update_mode != "always":
            collective_conflicts.append(
                "ccsa momentum update mode must be always"
            )
        if opts.objective_split_collab_action_enable:
            collective_conflicts.append("collab action must be disabled")
        if opts.objective_split_guide_scale_action_enable:
            collective_conflicts.append("guide scale action must be disabled")
        if opts.objective_split_candidate_response_mode != "actuate":
            collective_conflicts.append("candidate_response_mode must be actuate")
        if not opts.objective_split_candidate_history_obs_enable:
            collective_conflicts.append("candidate history observations must be enabled")
        if not opts.objective_split_candidate_actuator_action_enable:
            collective_conflicts.append("candidate actuator action must be enabled")
        if opts.objective_split_anchor_enable:
            collective_conflicts.append("anchor must be disabled")
        if opts.objective_split_committee_mode != "off":
            collective_conflicts.append("committee must be disabled")
        if opts.objective_split_d6_pre_generator_selector_enable:
            collective_conflicts.append("D6 selector must be disabled")
        if opts.objective_split_rvcpd_arm != "off":
            collective_conflicts.append("RVCPD must be disabled")
        if (
            opts.objective_split_collective_guide_mode == "rate_matched_null"
            and opts.objective_split_collective_guide_null_veto_rate <= 0.0
        ):
            collective_conflicts.append(
                "rate_matched_null requires a positive externally frozen null veto rate"
            )
        if (
            opts.objective_split_collective_guide_mode != "rate_matched_null"
            and opts.objective_split_collective_guide_null_veto_rate != 0.0
        ):
            collective_conflicts.append(
                "null veto rate must remain zero outside rate_matched_null"
            )
        if collective_conflicts:
            raise ValueError(
                "Collective guide frozen-evaluation contract failed: "
                + "; ".join(collective_conflicts)
                + "."
            )
    opts.episode_steps = opts.max_fes // max(1, opts.fixed_agent_num * opts.fixed_subfes_per_agent)
    opts.use_cuda = 1 if (torch.cuda.is_available() and not opts.no_cuda) else 0

    # 分布式设置
    opts.world_size = 1
    opts.distributed = False
    # opts.world_size = torch.cuda.device_count()
    # opts.distributed = (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4869'

    # 生成唯一的运行名称 (如果不是 resume)
    if not opts.resume:
        run_budget = _run_name_budget_value(opts)
        opts.run_name = "{}_{}_{}_{}_{}_{}_{}".format(
            opts.run_name, 
            opts.benchmark_name, 
            "{:.1e}".format(run_budget), 
            opts.m, 
            opts.sub_popsize, 
            opts.lr_model, 
            time.strftime("%Y%m%dT%H%M%S")
        )
    else:
        opts.run_name = Path(str(opts.resume)).expanduser().resolve().parent.name

    # 动态更新各模块的保存子目录
    for dir_attr in ['log_dir', 'output_modal_dir', 'output_data_dir', 'output_rollout_dir', 'test_dir']:
        # 将 output_modal_dir 特殊处理为 modal_save_dir 以符合后续代码逻辑
        if dir_attr == 'output_modal_dir':
            opts.modal_save_dir = os.path.join(opts.output_modal_dir, opts.run_name) if not opts.no_saving else None
        elif dir_attr == 'output_data_dir':
            opts.data_save_dir = os.path.join(opts.output_data_dir, opts.run_name) if not opts.no_saving else None
        elif dir_attr == 'output_rollout_dir':
            opts.rollout_save_dir = os.path.join(opts.output_rollout_dir, opts.run_name) if not opts.no_saving else None
        else:
            setattr(opts, dir_attr, os.path.join(getattr(opts, dir_attr), opts.run_name))

    return opts


def build_options_snapshot(opts, extra: dict = None):
    """
    Build a JSON-serializable options snapshot with `_comment` shown at the top.
    """
    raw = vars(opts).copy()
    if "device" in raw:
        raw["device"] = str(raw["device"])

    ordered = OrderedDict()
    note = str(raw.get("note", ""))
    ordered["_comment"] = note
    ordered["note"] = note
    for k, v in raw.items():
        ordered[k] = v

    if extra:
        for k, v in extra.items():
            ordered[k] = v
    return ordered

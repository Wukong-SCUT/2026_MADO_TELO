import os
import sys
import time
import argparse
import torch
from pathlib import Path
from typing import List
from collections import OrderedDict

RELEASE_ROOT = Path(__file__).resolve().parent
C8C_RELEASE_OUTPUT_ROOT = RELEASE_ROOT / "outputs"
C8C_RELEASE_TRAIN_ROOT = C8C_RELEASE_OUTPUT_ROOT / "train"
C8C_RELEASE_LATEST_TRAINING = C8C_RELEASE_TRAIN_ROOT / "latest_training.json"
C8C_RELEASE_CHECKPOINT_DIR = RELEASE_ROOT / "checkpoints"
C8C_RELEASE_SUMMARY_PRESET = RELEASE_ROOT / "configs" / "summarize_c8c_with_paper_baselines.json"
C8C_RELEASE_TEST_ROOT = C8C_RELEASE_OUTPUT_ROOT / "eval"
C8C_RELEASE_SUMMARY_ROOT = C8C_RELEASE_OUTPUT_ROOT / "summary"

# 获取当前脚本所在路径
# Keep generated files inside the release package regardless of cwd or entry point.
SAVE_DIR = str(C8C_RELEASE_TRAIN_ROOT)


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


def _hide_legacy_release_options(parser: argparse.ArgumentParser) -> None:
    public_flags = {
        "-h", "--help", "--release_profile", "--release_checkpoint_dir",
        "--release_checkpoint_source", "--release_cdo_checkpoint",
        "--release_wsn_checkpoint", "--release_cdo_fun_ids",
        "--release_wsn_fun_ids", "--release_eval_repeat_times",
        "--release_eval_batch_size", "--release_eval_seed0",
        "--release_eval_strategy", "--release_eval_agent_parallel_workers",
        "--release_summary_preset", "--release_summary_test_root",
        "--release_summary_output_dir", "--release_summary_name_prefix",
        "--release_summary_output_prefix", "--note", "--benchmark_name",
        "--mappo_env_mode", "--max_fes", "--cdo_bench_max_fes_list",
        "--wsn_node_num", "--wsn_target_num_list", "--wsn_max_fes_list",
        "--fixed_agent_num", "--fixed_subfes_per_agent",
        "--objective_split_agent_parallel_workers", "--train_function_ids",
        "--epoch_start", "--epoch_end", "--each_question_batch_num",
        "--mappo_update_mode", "--mappo_split_env_adv_norm", "--device",
        "--seed", "--no_cuda", "--run_name", "--no_tb", "--no_saving",
        "--no_progress_bar", "--log_dir", "--output_modal_dir",
        "--output_data_dir", "--output_rollout_dir", "--test_dir",
        "--checkpoint_epochs", "--load_path", "--resume",
        "--auto_latest_resume", "--resume_dir", "--resume_keyword",
    }
    for action in parser._actions:
        flags = tuple(str(x) for x in action.option_strings)
        if flags and not any(flag in public_flags for flag in flags):
            action.help = argparse.SUPPRESS


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
    parser = argparse.ArgumentParser(description="C8c Compact10 MAPPO release")
    # Shared settings for the no-argument train/eval/summary entry points.
    release = parser.add_argument_group('C8c release workflow')
    release.add_argument('--release_profile', type=str, default='c8c_compact10',
                         choices=['c8c_compact10'])
    release.add_argument('--release_checkpoint_dir', type=str,
                         default=str(C8C_RELEASE_CHECKPOINT_DIR))
    release.add_argument('--release_checkpoint_source', type=str,
                         default='latest_or_bundled',
                         choices=['latest_or_bundled', 'latest', 'bundled'],
                         help='checkpoint source for the two-part release evaluation')
    release.add_argument('--release_cdo_checkpoint', type=str, default='mappo-epoch-20.pt')
    release.add_argument('--release_wsn_checkpoint', type=str, default='mappo-epoch-24.pt')
    release.add_argument('--release_cdo_fun_ids', type=str,
                         default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15')
    release.add_argument('--release_wsn_fun_ids', type=str, default='1,2,3,4,5')
    release.add_argument('--release_eval_repeat_times', type=int, default=5)
    release.add_argument('--release_eval_batch_size', type=int, default=1)
    release.add_argument('--release_eval_seed0', type=int, default=42)
    release.add_argument('--release_eval_strategy', type=str, default='mappo_deterministic')
    release.add_argument('--release_eval_agent_parallel_workers', type=int, default=5)
    release.add_argument('--release_summary_preset', type=str,
                         default=str(C8C_RELEASE_SUMMARY_PRESET))
    release.add_argument('--release_summary_test_root', type=str,
                         default=str(C8C_RELEASE_TEST_ROOT))
    release.add_argument('--release_summary_output_dir', type=str,
                         default=str(C8C_RELEASE_SUMMARY_ROOT))
    release.add_argument('--release_summary_name_prefix', type=str, default='c8c_release')
    release.add_argument('--release_summary_output_prefix', type=str,
                         default='summary_eval_c8c_release')
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
                        help='发布包问题集：CDOBenchF1F15 或 WSNLocation')
    parser.add_argument('--divide_method', default="CEC2013LSGO",
                        choices=["CEC2013LSGO", "WSNLocation", "BNS", "OB_nondep"],
                        help='分组方法（兼容旧参数；WSNLocation 仅作为 benchmark_name 的兼容别名）')
    parser.add_argument('--mappo_env_mode', default="cdo_objective",
                        choices=["cdo_objective", "wsn_objective"],
                        help='C8c objective-split 环境语义')
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
    parser.add_argument('--objective_split_masoie_velocity_decay', type=float, default=0.5,
                        help='MASOIE-lite external velocity inertia in [0,1].')
    parser.add_argument('--objective_split_masoie_velocity_scale', type=float, default=1.0,
                        help='MASOIE-lite neighbor-pull scale.')
    parser.add_argument('--objective_split_masoie_velocity_clip_ratio', type=float, default=1.0,
                        help='MASOIE-lite velocity norm clip relative to current mean neighbor-pull norm; <=0 disables clipping.')
    parser.add_argument('--objective_split_record_comm_cost', type=int, default=1, choices=[0, 1],
                        help='Record communication rounds, messages, and transmitted-float estimates.')
    parser.add_argument('--objective_split_agent_parallel_workers', type=int, default=1,
                        help='Objective-split local optimizer workers inside one env step. 1 keeps historical serial agent execution; >1 runs agent local optimizers in a process pool.')
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
    parser.add_argument('--RL_agent', default='mappo', choices=['mappo'], help='强化学习训练算法')
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
                        help='C8c 智能体数量；F1-F15 使用 20，WSN 评估会自动使用 16')
    parser.add_argument('--fixed_subfes_per_agent', type=int, default=2500,
                        help='每个智能体每步的基础局部预算；C8c 默认 1000 FEs，再乘 actor 资源倍率')
    parser.add_argument('--sigma_candidates', type=str, default='0.2,0.3,0.6',
                        help='离散sigma候选，逗号分隔')
    parser.add_argument('--mappo_param_semantics', type=str, default='current', choices=['current'],
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
                        help='训练函数 ID，逗号分隔；C8c 默认使用 2,5,6,8,13,15')
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
    parser.add_argument('--eval_save_actions', type=int, default=1,
                        help='评估时是否保存actions.csv（1开启，0关闭）')
    parser.add_argument('--eval_save_running_data', type=int, default=1,
                        help='评估时是否保存running_data.h5（1开启，0关闭）')
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
    parser.add_argument('--mappo_policy_loss_weight_opt', type=float, default=0.8,
                        help='阶段C optimizer head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_cfg', type=float, default=1.0,
                        help='阶段C config head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_res', type=float, default=1.0,
                        help='阶段C resource head policy loss 权重')
    parser.add_argument('--mappo_policy_loss_weight_comm', type=float, default=1.0,
                        help='Communication-round head policy loss weight.')
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
                        default=None,
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
    parser.add_argument('--test_dir', default=str(C8C_RELEASE_TEST_ROOT), help='测试结果保存目录')
    
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
    # This standalone package intentionally defaults to the finalized C8c
    # configuration. Every value can still be overridden explicitly.
    parser.set_defaults(
        note='C8c-core6-compact10-statecomm-release',
        run_name='C8c_core6_compact10_statecomm_release',
        benchmark_name='CDOBenchF1F15',
        mappo_env_mode='cdo_objective',
        mappo_action_arch='current',
        optimizer_candidates='mmes,vkd,cmaes,sepcmaes',
        mappo_cfg_param_num=4,
        seed=42,
        fixed_agent_num=20,
        train_function_ids='2,5,6,8,13,15',
        max_fes=2600000,
        cdo_bench_max_fes_list=','.join(['3000000'] * 15),
        epoch_end=27,
        fixed_subfes_per_agent=1000,
        each_question_batch_num=5,
        mappo_update_mode='split_env',
        mappo_split_env_adv_norm=1,
        objective_split_consensus='graph_mean',
        objective_split_consensus_strength=1.0,
        objective_split_comm_interval=1,
        objective_split_graph_source='benchmark_w',
        objective_split_weight_mode='metropolis',
        objective_split_neighbor_obs_mode='none',
        objective_split_state_comm_mode='graph_mean',
        objective_split_state_comm_include_delta=0,
        objective_split_state_comm_include_actual_fes=0,
        objective_split_consensus_reward_weight=0.0,
        objective_split_comm_action_enable=1,
        objective_split_comm_round_candidates='2,4,6,8',
        objective_split_comm_action_reduce='mean_round',
        objective_split_agent_parallel_workers=1,
        subopt_vkd_kmax_levels='4,8,32',
        subopt_sepcmaes_cov_lr_scale_levels='0.75,1.5,2.5',
        forced_optimizer_enable=1,
        forced_optimizer_prob=0.25,
        forced_optimizer_warmup_ratio=0.2,
        forced_optimizer_mode='cycle',
        objective_split_early_stop_mode='none',
    )

    _hide_legacy_release_options(parser)

    raw_args_for_defaults = [str(x) for x in (sys.argv[1:] if args is None else args)]
    def _arg_was_set(flag: str) -> bool:
        return any(x == flag or x.startswith(flag + "=") for x in raw_args_for_defaults)
    opts = parser.parse_args(args)

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
    opts.objective_split_masoie_velocity_decay = float(
        min(1.0, max(0.0, opts.objective_split_masoie_velocity_decay))
    )
    opts.objective_split_masoie_velocity_scale = float(
        max(0.0, opts.objective_split_masoie_velocity_scale)
    )
    opts.objective_split_masoie_velocity_clip_ratio = float(
        max(0.0, opts.objective_split_masoie_velocity_clip_ratio)
    )
    opts.objective_split_record_comm_cost = int(bool(opts.objective_split_record_comm_cost))
    opts.objective_split_agent_parallel_workers = int(
        max(1, getattr(opts, "objective_split_agent_parallel_workers", 1))
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
    opts.mappo_policy_loss_weight_opt = float(max(0.0, opts.mappo_policy_loss_weight_opt))
    opts.mappo_policy_loss_weight_cfg = float(max(0.0, opts.mappo_policy_loss_weight_cfg))
    opts.mappo_policy_loss_weight_res = float(max(0.0, opts.mappo_policy_loss_weight_res))
    opts.mappo_policy_loss_weight_comm = float(
        max(0.0, getattr(opts, "mappo_policy_loss_weight_comm", 1.0))
    )
    opts.mappo_value_loss_weight_ref = float(max(0.0, opts.mappo_value_loss_weight_ref))
    opts.mappo_value_loss_weight_opt = float(max(0.0, opts.mappo_value_loss_weight_opt))
    opts.mappo_value_loss_weight_cfg = float(max(0.0, opts.mappo_value_loss_weight_cfg))
    opts.mappo_value_loss_weight_res = float(max(0.0, opts.mappo_value_loss_weight_res))
    opts.mappo_value_loss_weight_comm = float(
        max(0.0, getattr(opts, "mappo_value_loss_weight_comm", 1.0))
    )
    opts.forced_optimizer_enable = int(1 if int(getattr(opts, "forced_optimizer_enable", 0)) else 0)
    opts.forced_optimizer_prob = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_prob", 0.0))))
    opts.forced_optimizer_warmup_ratio = float(min(1.0, max(0.0, getattr(opts, "forced_optimizer_warmup_ratio", 0.0))))
    opts.forced_optimizer_mode = str(getattr(opts, "forced_optimizer_mode", "cycle")).lower()
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

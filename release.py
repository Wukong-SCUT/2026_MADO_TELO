"""Run the packaged E3aj MAPPO algorithm."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "checkpoints" / "mappo-epoch-5.pt"

def command(task, extra):
    common = ["--mappo_action_arch", "current",
              "--optimizer_candidates", "mmes,vkd,cmaes,sepcmaes",
              "--optimizer_profile_candidates", "inherit,numeric_inherit,conservative,balanced,aggressive",
              "--resource_factors", "0.5,1,2", "--mappo_cfg_param_num", "4",
              "--actor1_hidden_dim", "256", "--actor2_hidden_dim", "320",
              "--mappo_opt_emb_dim", "8", "--mappo_cfg_emb_dim", "8",
              "--fixed_subfes_per_agent", "250",
              "--objective_split_information_mode", "local_only",
              "--objective_split_state_comm_mode", "graph_mean",
              "--objective_split_state_comm_include_delta", "0",
              "--objective_split_comm_action_enable", "1",
              "--objective_split_comm_round_candidates", "2,4,6,8",
              "--objective_split_comm_action_reduce", "mean_round",
              "--objective_split_sigma_inherit_enable", "1",
              "--objective_split_sigma_validity_gate_enable", "0",
              "--objective_split_sigma_state_obs_enable", "0",
              "--objective_split_consensus", "graph_mean",
              "--mappo_legacy_comm_old_logp_zero", "1",
              "--objective_split_ccsa_momentum_update_mode", "always",
              "--objective_split_state_comm_cost_mode", "independent",
              "--objective_split_event_slot_interleaving_enable", "1",
              "--objective_split_optimizer_guide_enable", "1",
              "--objective_split_mmes_ratio_success_metric", "full",
              "--objective_split_cma_sep_ratio_path_strength", "0.1",
              "--objective_split_optimizer_guide_apply_optimizers", "cmaes,sepcmaes",
              "--objective_split_committee_shadow_global_eval", "0",
              "--objective_split_cmaes_numeric_fail_soft", "1",
              "--subopt_vkd_kmax_levels", "4,8,32",
              "--subopt_sepcmaes_cov_lr_scale_levels", "0.75,1.5,2.5",
              "--RL_agent", "mappo"]
    if task == "train":
        module = "env.agent.train_mappo"
        flags = common + ["--benchmark_name", "CDOBenchF1F15", "--mappo_env_mode", "cdo_objective",
                 "--seed", "58", "--epoch_start", "0", "--epoch_end", "6",
                 "--schedule_start_epoch", "0", "--schedule_end_epoch", "23",
                 "--train_function_ids", "2,5,6,8,13,15", "--fixed_agent_num", "20",
                 "--cdo_bench_max_fes_list", ",".join(["3000000"] * 15),
                 "--forced_optimizer_enable", "1", "--forced_optimizer_prob", "0.25",
                 "--forced_optimizer_warmup_ratio", "0.2", "--mappo_update_mode", "split_env"]
    elif task == "cdo":
        module = "env.agent.eval_mappo"
        flags = common + ["--benchmark_name", "CDOBenchF1F15", "--mappo_env_mode", "cdo_objective",
                 "--objective_split_auto_agent_num", "--fun_ids", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                 "--max_fes", "3000000", "--cdo_bench_max_fes_list", ",".join(["3000000"] * 15),
                 "--repeat_times", "30", "--batch_size", "5", "--seed0", "42",
                 "--strategies", "mappo_deterministic"]
    else:
        module = "env.agent.eval_mappo"
        flags = common + ["--benchmark_name", "WSNLocation", "--mappo_env_mode", "wsn_objective",
                 "--fixed_agent_num", "16", "--wsn_node_num", "16", "--max_fes", "2600000",
                 "--wsn_max_fes_list", ",".join(["3000000"] * 5),
                 "--actor_only_eval_allow_agent_mismatch", "--fun_ids", "1,2,3,4,5",
                 "--repeat_times", "30", "--batch_size", "5", "--seed0", "42",
                 "--strategies", "mappo_deterministic"]
    flags.extend(["--run_name", f"E3aj_{task}"])
    for flag, folder in (("--log_dir", "log"), ("--test_dir", "eval"),
                         ("--output_modal_dir", "ppo_model"),
                         ("--output_data_dir", "running_data"),
                         ("--output_rollout_dir", "rollout_data")):
        flags.extend([flag, str(ROOT / "outputs" / task / folder)])
    if task != "train": flags.extend(["--model_path", str(MODEL)])
    flags.extend(extra)
    return [sys.executable, "-m", module, *flags]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("train", "cdo", "wsn"))
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    cmd = command(args.task, extra)
    if args.dry_run:
        import shlex; print(shlex.join(cmd)); return
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))
if __name__ == "__main__": main()

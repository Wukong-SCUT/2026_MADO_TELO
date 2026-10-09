import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

"""
Package-mode entry for MAPPO training.
Run from project root:
    python -m env.agent.train_mappo
"""

import json
import os

import torch
try:
    from tensorboard_logger import Logger as TbLogger
except Exception:
    TbLogger = None

from options import get_options, resolve_mappo_obs_dim
from options import build_options_snapshot
from env.agent.mappo import MAPPOPolicy, MAPPOTrainer, MAPPORunner
from env.agent.mappo.checkpoint import (
    checkpoint_rng_state,
    restore_rng_state,
    torch_load_checkpoint,
)
from env.agent.utils.utils import set_random_seed


def main():
    opts = get_options()
    opts.RL_agent = "mappo"
    if len(getattr(opts, "train_function_ids", [])) == 0:
        raise ValueError(
            "Missing training function ids. Please pass --train_function_ids (or --fun_ids alias)."
        )
    if int(opts.episode_steps) < 1:
        raise ValueError(
            "Training budget is too small for one rollout step: max_fes must be at least "
            "fixed_agent_num * fixed_subfes_per_agent."
        )
    opts.use_cuda = 1 if (torch.cuda.is_available() and not opts.no_cuda) else 0
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    set_random_seed(int(opts.seed))
    # opts.train_function_ids = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # opts.train_function_ids = [1,2,3,12,15]

    tb_logger = None
    if (TbLogger is not None) and (not opts.no_tb) and bool(int(getattr(opts, "mappo_log_enable", 1))):
        tb_logger = TbLogger(opts.log_dir)

    obs_dim = resolve_mappo_obs_dim(opts)
    opts.feature_num_1 = obs_dim
    n_agents = int(opts.fixed_agent_num)
    action_dim = int(len(getattr(opts, "optimizer_profile_candidates", ["inherit", "conservative", "balanced", "aggressive"])))

    if int(getattr(opts, "objective_split_d5_two_stage_actuator_enable", 0)) or int(getattr(opts, "objective_split_d6_pre_generator_selector_enable", 0)):
        raise ValueError("This package contains only the selected MAPPO architecture.")

    policy = MAPPOPolicy(opts, obs_dim=obs_dim, n_agents=n_agents, action_dim=action_dim)
    trainer = MAPPOTrainer(opts, policy)
    print(f"[MAPPO Signature] current={getattr(policy, 'policy_signature_str', '')}")

    # Optional checkpoint loading:
    # - --resume: continue training from next epoch
    # - --load_path: load weights/optimizer but keep current epoch_start unless specified
    assert not (opts.resume and opts.load_path), "Only one of --resume and --load_path can be set."
    load_path = opts.resume if opts.resume else opts.load_path
    resume_rng_state = None
    if load_path is not None:
        ckpt = torch_load_checkpoint(load_path, map_location=policy.device)
        ckpt_sig = ckpt.get("policy_signature", None)
        cur_sig = getattr(policy, "policy_signature", None)
        if ckpt_sig is None:
            raise RuntimeError(
                "[MAPPO Signature Mismatch] Loaded checkpoint has no policy_signature. "
                "This usually means an old architecture checkpoint. "
                f"\ncurrent_signature={getattr(policy, 'policy_signature_str', str(cur_sig))}\n"
                "checkpoint_signature=None"
            )
        if ckpt_sig != cur_sig:
            raise RuntimeError(
                "[MAPPO Signature Mismatch] Checkpoint signature does not match current model architecture. "
                f"\ncurrent_signature={getattr(policy, 'policy_signature_str', str(cur_sig))}\n"
                f"checkpoint_signature={ckpt.get('policy_signature_str', str(ckpt_sig))}"
            )
        if "actor" in ckpt:
            policy.actor.load_state_dict(ckpt["actor"])
        if "critic" in ckpt:
            policy.critic.load_state_dict(ckpt["critic"])
        if "actor_opt" in ckpt:
            policy.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        if "critic_opt" in ckpt:
            policy.critic_optimizer.load_state_dict(ckpt["critic_opt"])

        if opts.resume:
            last_epoch = int(ckpt.get("epoch", -1))
            opts.epoch_start = max(int(opts.epoch_start), last_epoch + 1)
            print(f"[MAPPO] Resume from: {load_path}")
            print(f"[MAPPO] last_epoch={last_epoch}, new epoch_start={opts.epoch_start}, epoch_end={opts.epoch_end}")
            resume_rng_state = checkpoint_rng_state(ckpt)
            schedule = ckpt.get("training_schedule", None)
            if isinstance(schedule, dict):
                opts.resume_schedule_start_epoch = int(
                    schedule.get("start_epoch", opts.epoch_start)
                )
                opts.resume_schedule_end_epoch = int(
                    schedule.get("end_epoch", opts.epoch_end)
                )
                print(
                    "[MAPPO Schedule] Restored original schedule axis: "
                    f"start={opts.resume_schedule_start_epoch}, "
                    f"end={opts.resume_schedule_end_epoch}"
                )
            if resume_rng_state is None:
                print(
                    "[MAPPO RNG] Legacy checkpoint has no complete RNG state; "
                    "resume will preserve historical non-bitwise behavior."
                )
            if not isinstance(schedule, dict):
                print(
                    "[MAPPO Schedule] Legacy checkpoint has no original schedule "
                    "axis; forced-optimizer scheduling keeps historical restart "
                    "semantics."
                )
        else:
            print(f"[MAPPO] Loaded checkpoint weights from: {load_path}")

    if not opts.no_saving:
        os.makedirs(opts.modal_save_dir, exist_ok=True)
        args_dict = build_options_snapshot(opts, extra={"stage": "train_mappo"})
        with open(os.path.join(opts.modal_save_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2)
        with open(os.path.join(opts.modal_save_dir, "options_train.json"), "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2)

    runner = MAPPORunner(opts, policy, trainer, tb_logger=tb_logger)
    if resume_rng_state is not None:
        restored = restore_rng_state(resume_rng_state)
        print(f"[MAPPO RNG] Restored checkpoint RNG state: {restored}")
    runner.train()


if __name__ == "__main__":
    main()

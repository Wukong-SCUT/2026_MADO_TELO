import torch
import math
from env.agent.utils.plots import plot_grad_flow, plot_improve_pg
import numpy as np
    
class TrainingLogger:
    def __init__(self, writer):
        self.writer = writer
        self.reset_batch_stats()
        self.reset_update_stats()
        self.reset_mappo_stats()

    def reset_batch_stats(self):
        """重置采样阶段统计"""
        self.batch_rewards = []
        self.batch_best_ys = []
        self.total_steps = 0

    def reset_update_stats(self):
        """重置 PPO 更新阶段（K_epochs）统计"""
        # Actor - Mode 相关
        self.kl_mode = []
        self.clip_frac_mode = []
        self.loss_pi_mode = []
        # Actor - Res 相关
        self.kl_res = []
        self.clip_frac_res = []
        self.loss_pi_res = []
        
        # 公共部分
        self.loss_v = []
        self.loss_ent = []
        self.grad_actor_pre = []    # 裁剪前梯度
        self.grad_actor_post = []   # 裁剪后梯度
        self.grad_critic_pre = []
        self.grad_critic_post = []

    def store_update_stats(self, stats_dict):
        """每次 optimizer.step() 前后调用"""
        for k, v in stats_dict.items():
            if hasattr(self, k):
                getattr(self, k).append(v)

    # ---------------- MAPPO generic logging ----------------
    def reset_mappo_stats(self):
        """重置 MAPPO 指标缓存。"""
        self.mappo_stats = {}

    def store_mappo_update_stats(self, stats_dict):
        """缓存 MAPPO 训练指标（按 key 聚合为列表）。"""
        for k, v in stats_dict.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if k not in self.mappo_stats:
                self.mappo_stats[k] = []
            self.mappo_stats[k].append(fv)

    def summarize_mappo_stats(self):
        """对缓存中的 MAPPO 指标取均值。"""
        out = {}
        for k, vs in self.mappo_stats.items():
            if len(vs) == 0:
                continue
            out[k] = float(np.mean(vs))
        return out

    def write_mappo_to_tb(self, step, lr_actor, lr_critic):
        """把 MAPPO 指标写到 TensorBoard（若 writer 不存在则跳过）。"""
        if self.writer is None:
            return

        summary = self.summarize_mappo_stats()
        for k, v in summary.items():
            self.writer.log_value(f"MAPPO/{k}", v, step)
        self.writer.log_value("MAPPO/lr_actor", float(lr_actor), step)
        self.writer.log_value("MAPPO/lr_critic", float(lr_critic), step)

    def write_to_tb(self, step, lr_actor, lr_critic, explained_var):
        if self.writer is None: return

        # 1. Mode Head 指标
        self.writer.add_scalar('Actor_Mode/KL', np.mean(self.kl_mode), step)
        self.writer.add_scalar('Actor_Mode/Clip_Frac', np.mean(self.clip_frac_mode), step)
        self.writer.add_scalar('Actor_Mode/Loss', np.mean(self.loss_pi_mode), step)

        # 2. Res Head 指标
        self.writer.add_scalar('Actor_Res/KL', np.mean(self.kl_res), step)
        self.writer.add_scalar('Actor_Res/Clip_Frac', np.mean(self.clip_frac_res), step)
        self.writer.add_scalar('Actor_Res/Loss', np.mean(self.loss_pi_res), step)

        # 3. 梯度监控 (最关键的对比)
        self.writer.add_scalar('Grad/Actor_Pre_Clip', np.mean(self.grad_actor_pre), step)
        self.writer.add_scalar('Grad/Actor_Post_Clip', np.mean(self.grad_actor_post), step)
        self.writer.add_scalar('Grad/Critic_Pre_Clip', np.mean(self.grad_critic_pre), step)

        # 4. 整体指标
        self.writer.add_scalar('Train/Loss_Value', np.mean(self.loss_v), step)
        self.writer.add_scalar('Train/Explained_Var', explained_var, step)
        self.writer.add_scalar('Params/LR_Actor', lr_actor, step)

        # 重置缓冲区，准备下一段采样
        self.reset_batch_stats()
        self.reset_update_stats()

def log_train_details(tb_logger, stats, step):
    """
    stats: 包含本 batch 所有训练统计信息的字典
    step: 当前的总步数 (mini_step)
    """
    # --- 1. 训练核心指标 (Performance) ---
    tb_logger.log_value('Train/Reward_Mean', stats['reward_mean'], step)
    tb_logger.log_value('Train/Reward_Max',  stats['reward_max'], step)
    tb_logger.log_value('Train/Return_Target', stats['return_target'], step)
    
    # --- 2. 策略健康度 (Policy Health) - 拆分 Mode 和 Res ---
    
    # [Mode Head] - 离散选择 (选哪个算子/模式)
    tb_logger.log_value('Actor_Mode/Loss',        stats['loss_pi_mode'], step)
    tb_logger.log_value('Actor_Mode/Entropy',     stats['entropy_mode'], step)    # 越高越随机
    tb_logger.log_value('Actor_Mode/Approx_KL',   stats['kl_mode'], step)         # >0.05 说明更新步长过大
    tb_logger.log_value('Actor_Mode/Clip_Frac',   stats['clip_frac_mode'], step)  # 理想在 0.0-0.2
    
    # [Res Head] - 连续控制 (分配多少资源)
    tb_logger.log_value('Actor_Res/Loss',         stats['loss_pi_res'], step)
    tb_logger.log_value('Actor_Res/Entropy',      stats['entropy_res'], step)
    tb_logger.log_value('Actor_Res/Approx_KL',    stats['kl_res'], step)
    tb_logger.log_value('Actor_Res/Clip_Frac',    stats['clip_frac_res'], step)

    # --- 3. 价值函数健康度 (Critic Health) ---
    tb_logger.log_value('Critic/Value_Loss',         stats['loss_v'], step)
    tb_logger.log_value('Critic/Explained_Variance', stats['explained_var'], step) # 越接近 1 越好

    # --- 4. 梯度监控 (Gradients) ---
    # 注：因为 Mode 和 Res 通常共享 Backbone，所以梯度看整体 Actor 即可
    tb_logger.log_value('Grad/Actor',  stats['grad_actor'], step)
    tb_logger.log_value('Grad/Critic', stats['grad_critic'], step)

    # --- 5. 学习率 ---
    tb_logger.log_value('Params/LR_Actor', stats['lr_actor'], step)
    tb_logger.log_value('Params/LR_Critic', stats['lr_critic'], step)


# def log_to_tb_train ( tb_logger, agent, Reward, ratios, bl_val_detached, grad_norms, reward, entropy, approx_kl_divergence,
#                reinforce_loss, baseline_loss, log_likelihood, baseline, show_figs, mini_step, R , state):
    
#     # learning rate
#     tb_logger.log_value('learnrate_pg/actor_lr', agent.optimizer.param_groups[0]['lr'], mini_step)
#     tb_logger.log_value('learnrate_pg/critic_lr', agent.optimizer.param_groups[1]['lr'], mini_step)

#     tb_logger.log_value('train/ratios', ratios.mean().item(), mini_step)

#     tb_logger.log_value('train/Target_Return', Reward.mean().item(), mini_step)
#     tb_logger.log_value('train/ratios', ratios.mean().item(), mini_step)
#     avg_reward = torch.cat(reward).mean()# torch.stack(reward, 0).sum(0).mean().item()
#     max_reward = torch.cat(reward).max()# torch.stack(reward, 0).max(0)[0].mean().item() #reward检查一下
#     tb_logger.log_value('train/avg_reward', avg_reward, mini_step)
#     tb_logger.log_value('train/max_reward', max_reward, mini_step)
#     tb_logger.log_value('train/baseline', np.array(baseline).mean(), mini_step)

#     grad_norms, grad_norms_clipped = grad_norms
#     tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
#     tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
#     tb_logger.log_value('train/entropy', entropy.mean().item(), mini_step)
#     tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)
#     tb_logger.log_value('train/bl_val',bl_val_detached.mean().cpu(),mini_step)

#     tb_logger.log_value('train/R', R.mean().cpu(), mini_step)

#     tb_logger.log_value('train/mean_state', state.mean().cpu(), mini_step)
#     tb_logger.log_value('train/max_state', state.max().cpu(), mini_step)
#     tb_logger.log_value('train/min_state', state.min().cpu(), mini_step)

#     #记录R max state 
    
#     tb_logger.log_value('grad/actor', grad_norms[0], mini_step)
#     tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0], mini_step)
#     tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
            
#     tb_logger.log_value('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)
    
#     tb_logger.log_value('grad/critic', grad_norms[1], mini_step)
#     tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1], mini_step)
    
#     if show_figs and mini_step % 1000 == 0:
#         tb_logger.log_images('grad/actor', [plot_grad_flow(agent.actor)], mini_step)
#         tb_logger.log_images('grad/critic', [plot_grad_flow(agent.critic)], mini_step)

def log_to_tb_test(tb_logger, reward,state):

    avg_reward = reward.mean()#torch.stack(reward, 0).sum(0).mean().item()
    max_reward = reward.max()#torch.stack(reward, 0).max(0)[0].mean().item() #reward检查一下
    tb_logger.log_value('test/avg_reward', avg_reward)
    tb_logger.log_value('test/max_reward', max_reward)

    tb_logger.log_value('test/mean_state', state.mean().cpu())
    tb_logger.log_value('test/max_state', state.max().cpu())
    tb_logger.log_value('test/min_state', state.min().cpu())



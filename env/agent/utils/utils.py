import torch
import math
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
import time


def partition_p_and_s(p_file_path, s_file_path, overlap=0):
    # 读取文件内容
    def read_file(file_path):
        with open(file_path, "r") as file:
            content = file.read().strip()
        return content

    # 解析数据为列表
    def parse_p_values(p_content):
        return list(map(int, p_content.split(",")))

    def parse_s_values(s_content):
        return list(map(int, s_content.split()))

    # 划分维度数据，支持重叠
    def partition_p_by_s(p_values, s_values, overlap):
        partitioned = []
        start_index = 0

        for size in s_values:
            # 如果不是第一个分区，应用重叠
            if partitioned and overlap > 0:
                start_index = start_index - overlap
            end_index = start_index + size
            partitioned.append(p_values[start_index:end_index])
            start_index = end_index

        return partitioned

    # 读取文件内容
    p_content = read_file(p_file_path)
    s_content = read_file(s_file_path)

    # 解析文件数据
    p_values = parse_p_values(p_content)
    p_values = [x - 1 for x in p_values]
    s_values = parse_s_values(s_content)

    # 根据 s 划分 p，支持重叠
    return partition_p_by_s(p_values, s_values, overlap)


def torch_load_cpu(load_path):
    # Load on CPU
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) or isinstance(model, DDP) else model


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def move_to_cuda(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.cuda(device)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    修改后：自动兼容参数列表和优化器参数组
    """
    # --- 新增兼容性处理 ---
    # 如果传入的不是列表，或者列表里的元素不是字典（即直接传了 parameters）
    # 把它包装成函数期望的 [{'params': ...}] 格式
    if not isinstance(param_groups, (list, tuple)):
        # 处理迭代器 (如 generator/filter)
        param_groups = [{'params': list(param_groups)}]
    elif len(param_groups) > 0 and not isinstance(param_groups[0], dict):
        # 处理参数列表 [p1, p2, ...]
        param_groups = [{'params': param_groups}]
    # --------------------

    grad_norms = []
    for group in param_groups:
        # 注意：建议使用带下划线的 clip_grad_norm_，它是官方最新推荐写法
        norm = torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        # 将 tensor 转换为 python 标量 float，方便后面 TensorBoard 记录
        grad_norms.append(norm.item() if hasattr(norm, 'item') else norm)

    grad_norms_clipped = [min(g_norm, max_norm)
                          for g_norm in grad_norms] if max_norm > 0 else grad_norms

    # 保持原有的返回格式：两个列表
    return grad_norms, grad_norms_clipped


def set_random_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    else:
        random.seed(None)
        np.random.seed(None)
        torch.manual_seed(int(time.time()))
        torch.cuda.manual_seed(int(time.time()))
        torch.cuda.manual_seed_all(int(time.time()))


def _gae(delta, dones, gamma, lam):
    """ generalized advantage estimation over time dimension """
    T, B = delta.size()
    adv = torch.zeros_like(delta)
    last = torch.zeros(B, device=delta.device, dtype=delta.dtype)
    for t in reversed(range(T)):
        last = delta[t] + gamma * lam * (1.0 - dones[t]) * last
        adv[t] = last
    return adv

def _normalize(x, eps: float = 1e-8):
    mean = x.mean()
    std  = x.std(unbiased=False)
    return (x - mean) / (std + eps)


# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.entropies = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.entropies[:]

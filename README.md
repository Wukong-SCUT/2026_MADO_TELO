# C8c Compact10

C8c 的独立训练、评估与结果汇总仓库。仓库只支持最终 C8c 所需的 `CDOBenchF1F15` 和 `WSNLocation`，不包含历史 A/B/C 实验、调试 jobs 或无关 benchmark。

## Quick Start

在 WSL/Linux 和 Python 3.11 环境中：

```bash
conda env create -f environment.yml
conda activate meta_
```

从头训练：

```bash
python -m env.agent.train_mappo
```

评估 F1-F15 和 WSN F1-F5：

```bash
python -m env.agent.eval_mappo
```

生成效果、耗时和通信次数汇总：

```bash
python -m env.agent.summarize_eval
```

三个入口都可无参数运行，默认配置集中在 `options.py`。

## Output

所有新产物只会写入仓库根目录的 `outputs/`：

```text
outputs/
  train/
    log/
    ppo_model/
    running_data/
    rollout_data/
    latest_training.json
  eval/
  summary/
```

`outputs/` 已被 Git 忽略，不会把本地运行结果混入源码提交。

## Model Selection

训练正常结束后，`train_mappo` 会写入 `outputs/train/latest_training.json`。无参数评估按以下顺序选择模型：

1. 优先使用最新完成训练的 `mappo-epoch-20.pt` 和 `mappo-epoch-24.pt`。
2. 如果没有可用的训练记录，回退到 `checkpoints/` 中随包提供的模型。
3. 控制台会显示 `latest completed training`、`bundled fallback` 或 `explicit`。

强制评估随包模型：

```bash
python -m env.agent.eval_mappo --release_checkpoint_source bundled
```

指定其他模型目录：

```bash
python -m env.agent.eval_mappo \
  --release_checkpoint_dir outputs/train/ppo_model/<model-directory>
```

## Final Configuration

- 训练问题：CDOBench F2、F5、F6、F8、F13、F15。
- 智能体：20。
- 训练：epoch 0-26，每题 batch=5，`split_env`。
- 基础局部预算：每智能体 1000 FEs。
- actor 输入：16 维自身状态 + 10 维邻居消息 = 26 维。
- 图通信：benchmark `W`、Metropolis 权重、`graph_mean`。
- actor 通信轮数候选：2、4、6、8，`mean_round`。
- 局部优化器：MMES、VKD-CMA、CMA-ES、Sep-CMA-ES。
- forced optimizer：前 20% 训练阶段，概率 0.25，cycle 四个优化器。
- seed：42。

完整复现参数见 `checkpoints/training_config.json`。

## Evaluation

默认评估：

- epoch-20：CDOBench F1-F15，每题 3e6 FEs。
- epoch-24：WSN F1-F5，每题 3e6 FEs。
- 每题重复 5 次，重复实验串行。
- 单环境内使用 5 个智能体局部优化 worker。

快速单次检查：

```bash
python -m env.agent.eval_mappo --release_eval_repeat_times 1
```

## Benchmarks

```text
benchmarks/
  cdo_f1f15/   # F1-F15 实现与 20 智能体数据
  wsn_f1f5/    # WSN F1-F5 实现与 16 节点数据
```

F1-F14 只包含实际使用的 `A_20n100D`、`W_20n`、`R_100D` 和 `xopt_100D`。F15 为固定 seed 的 MASOIE 式 WSN 生成问题；WSN F1-F5 使用随包的 source、target 和 `W_16` 数据。

## Summary References

- 效果基线：MASOIE/CCSA 论文数据。
- 耗时与通信基线：赵懿行的 3e6、5 次重复系统复现数据。
- 通信次数按各算法自身计数定义，适合比较数量级，不等于传输字节数。

## Known Reference Result

随包 checkpoint 的历史 5 次评估中：

- F1-F15 相对 MASOIE 论文基线的 log10 差之和约为 `-28.805`。
- WSN 相对 CCSA 论文基线的 log10 差之和约为 `+0.436`。
- 平均通信轮数约为 F1-F15 `1.80e2`、WSN `1.92e2`。

不同 CPU、BLAS 和进程调度可能影响耗时；从头训练也不保证与历史模型逐位一致。

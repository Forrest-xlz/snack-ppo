# Rust PPO Snake 设计规格

## 目标

构建一个完全使用 Rust 实现的贪吃蛇强化学习项目。项目支持 PPO 训练和带图形窗口的模型推理；所有运行模式和参数均由固定路径 `config/config.yaml` 配置，不使用命令行参数。训练利用 NVIDIA GPU 完成神经网络计算，并使用多个 CPU 核并行推进环境。

## 技术选型

- Rust stable（edition 2021）
- `tch-rs` / CUDA 版 LibTorch：张量、自动微分、模型和 Adam 优化器
- `rayon`：多个环境的 CPU 并行 step/reset
- `macroquad`：推理图形窗口
- `serde`、`serde_yaml`：配置反序列化
- `rand`、`rand_chacha`：可复现的环境随机数和动作采样
- `anyhow`、`thiserror`：带上下文的错误处理

## 运行入口与配置

程序使用单一入口。直接执行 release binary；程序固定读取 `config/config.yaml`，根据顶层 `mode: train | infer` 选择训练或推理。不得从 CLI 覆盖训练参数。

配置结构：

```yaml
mode: train
device: cuda
seed: 42

environment:
  width: 12
  height: 12
  num_envs: 64
  rollout_steps: 256

model:
  hidden_size: 256

ppo:
  learning_rate: 0.0003
  batch_size: 1024
  epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5

training:
  total_iterations: 5000
  checkpoint_interval: 100
  checkpoint_dir: checkpoints

inference:
  model_path: checkpoints/latest.pt
  fps: 10
  deterministic: true
  cell_size: 36
```

启动时必须校验：模式和设备枚举合法；棋盘至少能容纳初始蛇和食物；环境数、rollout、batch、epoch、迭代数、保存间隔、FPS、格子尺寸及隐藏层尺寸为正数；学习率和最大梯度范数为正数；`gamma`、`gae_lambda`、`clip_range`、损失系数位于有效范围。`device: cuda` 而 CUDA 不可用时，程序明确报错并终止，不静默回退 CPU。

## 贪吃蛇环境

环境状态包括棋盘尺寸、蛇身坐标、当前方向、食物坐标、分数、终止标志和独立随机数生成器。动作空间固定为上、下、左、右四个离散动作。

规则如下：

- 与当前方向相反的动作视为无效，蛇继续沿当前方向移动。
- 吃到食物时蛇增长一格、分数增加 1，当前 step 奖励为 `1.0`。
- 未吃到食物时奖励为 `0.0`。
- 撞墙或撞到自身时 episode 终止，奖励仍为 `0.0`。
- 不添加生存奖励、距离 shaping 或死亡惩罚。
- 食物只能生成在空格内；蛇占满棋盘时 episode 终止。

观测编码为 `[3, height, width]` 的 `f32` 张量：通道 0 为蛇头，通道 1 为其余蛇身，通道 2 为食物；对应位置取 `1.0`，其余为 `0.0`。

## 模型

Actor-Critic 模型使用共享特征提取器和两个输出头。为兼容可配置棋盘尺寸，输入首先展平，再经过两层带 ReLU 的全连接层，每层宽度为 `model.hidden_size`。策略头输出四个未归一化 logits；价值头输出单个标量。

训练时使用 categorical 分布按 logits 采样动作，并保存动作 log-probability；推理时由 `inference.deterministic` 控制选择最大概率动作或按分布采样。模型参数由同一个 VarStore 管理并保存为 `.pt` checkpoint。

## 并行采样与数据布局

Trainer 持有 `environment.num_envs` 个长期存在的环境。每次 PPO iteration 执行恰好 `environment.rollout_steps` 个时间步：

1. 将全部环境的观测组成 `[num_envs, 3, height, width]` batch。
2. 在 GPU 上一次完成全部环境的策略和值前向计算。
3. 在策略分布中为每个环境采样一个动作。
4. 将动作传回 CPU，使用 Rayon 并行执行全部环境的 `step`。
5. 记录 observation、action、reward、done、value 和 old log-probability。
6. 终止环境立即 reset，以便下一时间步继续采样；transition 中仍保留 `done=true`。

因此每轮严格获得 `num_envs * rollout_steps` 条 transition，即使中间 episode 结束也不减少样本数。rollout 缓冲区采用时间优先布局 `[rollout_steps, num_envs, ...]`，以便沿每个环境的时间轴计算 GAE；训练前再展平为一个样本维度。

CPU 多核用于并行环境 step/reset；GPU 用于 batched policy inference 和 PPO 优化。每个 rollout 时间步只发生一次批量 GPU 前向，避免各 CPU worker 争用 GPU。

## PPO 更新

rollout 结束后，对 reset 后的当前观测执行一次 value 前向以获得 bootstrap value。沿每个环境的时间维逆序计算：

```text
non_terminal = 1 - done_t
delta_t = reward_t + gamma * next_value * non_terminal - value_t
advantage_t = delta_t + gamma * gae_lambda * non_terminal * advantage_(t+1)
return_t = advantage_t + value_t
```

全部 advantage 在整批样本上标准化。随后在每个 PPO epoch 重新随机打乱全部索引，并按 `batch_size` 切分 mini-batch；末尾不足一个 batch 的样本必须保留。

损失定义：

```text
ratio = exp(new_log_prob - old_log_prob)
policy_loss = -mean(min(ratio * advantage,
                        clamp(ratio, 1-clip_range, 1+clip_range) * advantage))
value_loss = mean((return - value)^2)
entropy_bonus = mean(categorical_entropy)
total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
```

使用 Adam 和 YAML 中的 `learning_rate`，反向传播后按 `max_grad_norm` 裁剪梯度。每轮输出 iteration、总采样步数、平均 episode 分数、最大分数、策略损失、价值损失和熵。

## Checkpoint 与推理

每隔 `checkpoint_interval` iteration 保存一个带 iteration 编号的 checkpoint，并更新 `latest.pt`；训练结束也必须保存。写入时先生成同目录临时文件，再替换最终文件，降低中断导致文件损坏的风险。项目不要求恢复优化器状态；加载 checkpoint 用于推理或作为模型权重加载基础。

推理模式加载 `inference.model_path`，创建单个环境并启动 macroquad 窗口。窗口按 `fps` 推进游戏，显示棋盘网格、蛇头、蛇身、食物、当前分数和本次运行最高分；episode 结束后短暂停留并自动开始下一局。关闭窗口即可退出。

## 模块边界

- `src/config.rs`：配置结构、固定路径加载和完整校验。
- `src/env.rs`：环境状态、动作、观测、step/reset 和规则。
- `src/model.rs`：Actor-Critic 网络、前向、采样和确定性动作。
- `src/rollout.rs`：并行环境集合、固定长度采样及 rollout buffer。
- `src/ppo.rs`：GAE、mini-batch 索引、PPO 损失和更新。
- `src/trainer.rs`：训练循环、统计、日志和 checkpoint 生命周期。
- `src/renderer.rs`：模型推理与 macroquad 绘制。
- `src/main.rs`：读取固定 YAML、初始化设备并分派运行模式。

模块通过明确的数据结构连接，环境模块不依赖 tch 或 GUI，便于快速、确定性测试。

## 错误处理

- 配置读取、YAML 解析和字段校验错误包含字段名与预期范围。
- CUDA 不可用、模型维度不匹配、checkpoint 不存在或无法加载时返回带上下文错误。
- checkpoint 目录不存在时自动创建；创建或原子替换失败时终止训练并保留原 checkpoint。
- 环境内部不使用 panic 表达用户可触发的错误；不可能状态通过明确错误或受测试的断言保护。

## 测试与验收

环境单元测试覆盖：正常移动、反向动作、吃食物奖励恰好为 1、增长、撞墙、自撞、食物不落在蛇身、占满棋盘终止。

算法测试覆盖：终止边界截断 GAE、非终止 bootstrap、rollout 样本数严格等于环境数乘 rollout、mini-batch 不丢尾部、PPO clip 目标的数值行为。

集成烟雾测试使用 CPU、小棋盘、少量环境和短 rollout，完成一次采样及一个 PPO epoch，验证模型参数改变，并验证 checkpoint 保存后可重新加载。GPU 和图形窗口提供 README 中的人工烟雾验证步骤。

完成标准：

1. `cargo test` 全部通过。
2. CPU 烟雾配置可完成至少一次 PPO iteration 并生成 checkpoint。
3. NVIDIA CUDA 配置下训练日志显示 GPU 设备，且一次采样数等于 `num_envs * rollout_steps`。
4. 将 YAML 切换为 `mode: infer` 后，无需 CLI 参数即可打开窗口并由加载的模型控制游戏。
5. 用户要求的学习率、batch、环境数、rollout 数和训练 epoch 均仅通过 YAML 生效。

## 周期评估与 Weights & Biases 补充设计

配置增加 `evaluation.interval`、`evaluation.episodes`、`evaluation.deterministic` 和 `evaluation.max_steps_per_episode`。每完成指定数量的外层 PPO iteration，使用当前模型在独立环境中运行指定局数；评估不进入 rollout、不改变训练环境状态、不执行梯度计算。每局达到最大步数时以当前分数结束，防止循环策略永久阻塞。记录平均、最小和最大评估分数。

配置增加 `wandb.enabled`、`project`、`entity`、`run_name`、`mode` 和 `python_executable`。Rust 主进程启动随项目提供的轻量 Python 日志桥，由官方 W&B Python SDK 创建 run，并通过标准输入接收逐行 JSON。API key 只允许来自 `WANDB_API_KEY` 或已有登录状态，不得写入 YAML 或源码。

W&B 每轮记录 policy/value loss、entropy、训练 episode 平均分和最高分、学习率、iteration、累计样本数；评估轮额外记录 `evaluation/mean_score`、`evaluation/min_score`、`evaluation/max_score`。日志桥无法启动、提前退出或拒绝指标时训练明确失败；`enabled: false` 时完全禁用桥接。

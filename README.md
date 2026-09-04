# PPO Snake (Rust)

基于 PPO 的贪吃蛇强化学习项目。多个环境通过 Rayon 使用 CPU 多核并行推进，Actor-Critic 前向和 PPO 更新通过 `tch-rs` 在 NVIDIA GPU 上执行；推理模式使用 macroquad 图形窗口展示游戏。

## 环境准备（Windows）

1. 安装 Rust stable（MSVC 工具链）。
2. 安装 NVIDIA 驱动和与 LibTorch 匹配的 CUDA 运行库。
3. 下载与 `tch = 0.20` 匹配的 LibTorch 2.7 CUDA 版本，设置 `LIBTORCH` 指向解压目录，并将 `%LIBTORCH%\lib` 加入 `PATH`。项目的 `download-libtorch` feature 只方便 CPU 编译测试；GPU 训练应使用 CUDA LibTorch。
4. 安装 W&B Python SDK：

```powershell
C:\Users\Liuluotu\Anaconda\anaconda3\python.exe -m pip install "wandb>=0.22.3"
```

5. 在线记录前，在用户环境中配置 `WANDB_API_KEY`，不要把密钥写入 YAML 或源码。也可以先运行 `wandb login`。W&B 官方建议使用环境变量保存 API key。

## 配置

程序固定读取 [config/config.yaml](config/config.yaml)，没有命令行训练参数。常用字段：

- `mode`: `train` 或 `infer`
- `device`: `cuda` 或 `cpu`；请求 CUDA 但不可用时直接报错
- `environment.num_envs`: 并行环境数量
- `environment.rollout_steps`: 每个环境每轮采样步数
- `ppo.learning_rate`, `batch_size`, `epochs`: 学习率、mini-batch 和每批 PPO epoch
- `training.total_iterations`: 外层 PPO 循环数
- `evaluation.interval`: 每隔多少个外层循环评估一次
- `evaluation.episodes`: 每次评估完整比赛局数
- `evaluation.max_steps_per_episode`: 单局评估上限，防止循环策略卡住
- `wandb.mode`: `online` 实时同步或 `offline` 本地保存
- `wandb.enabled`: 是否启用 W&B；关闭时不需要 Python SDK

每次采样严格获得：

```text
num_envs * rollout_steps
```

吃到一个食物奖励 `1.0`，其他情况（包括死亡）奖励 `0.0`。

## 训练

将 YAML 设置为 `mode: train` 和 `device: cuda`，然后运行：

```powershell
cargo run --release
```

checkpoint 保存到 `training.checkpoint_dir/run_name/`。例如 `run_name: snake-v1` 会生成 `checkpoints/snake-v1/latest.ot`；`run_name: null` 时自动生成唯一 run 名称。

W&B 保留原有四组指标：`train_samples/*`、`eval_samples/*` 以累计样本数为横轴，`train_iteration/*`、`eval_iteration/*` 以外层 iteration 为横轴。另有 `recent_samples/*` 和 `recent_iteration/*` 两组固定窗口指标，分别使用累计样本数和 iteration 为横轴。窗口大小由 `metrics.score_window_episodes`（最近完成的对局数）和 `metrics.transition_window_samples`（最近唯一 transition 数）配置；不足窗口大小时使用当前已有数据，没有已完成对局时不记录窗口分数。

## 图形推理

把 YAML 改为：

```yaml
mode: infer
inference:
  model_path: checkpoints/latest.ot
  fps: 10
  deterministic: true
  cell_size: 36
```

然后运行 `cargo run --release`。窗口显示蛇、食物、当前分数和最高分；关闭窗口退出。

## 验证

```powershell
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

若提示 CUDA 不可用，确认当前使用的是 CUDA 版 LibTorch、NVIDIA 驱动兼容，并且 LibTorch 的 DLL 目录位于 `PATH`。若 W&B 桥无法启动，检查 `wandb.python_executable` 和该 Python 环境中的 `wandb` 包。

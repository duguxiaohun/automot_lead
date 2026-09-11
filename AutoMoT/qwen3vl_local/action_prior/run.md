# Action prior 使用说明

以下命令在训练机的 `AutoMoT/` 目录执行。准备好 `lead_data`、已有人工标注和本地 Qwen/BEV 权重；脚本自动构建所需索引，不下载模型。

## 开始训练

```bash
# 数据集标定先验：自动准备标签、训练并完成 test/probe。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors

# 开启 UE/特殊 RE 均衡采样，只需增加一个开关。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced

# 默认自动选卡；需要指定四张卡时：
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
```

`--event-balanced`：UE1–UE7、RE2、RE3、RE5 各一份，确认的普通 RE 合计两份。
默认每帧单 epoch 最多重复 8 次，自动确定可行预算，并优先覆盖不同帧。省略此开关就是自然采样；`--no-event-balanced` 可覆盖环境变量中开启的设置。

脚本自动准备 action 索引、Phase1 标签索引、先验标签以及均衡采样所需的候选/full map；已存在的有效结果会复用。
这只构建数据，不训练 Phase1/Phase3。首次准备可能较慢，终端和 pipeline 日志会显示进度。
不需要手写 `DATA_DIR` 或 `EVENT_BALANCE_INDEX`。

```bash
# 使用 Phase1/2 LoRA 现场生成先验（耗时更长，需已有 LoRA 权重）。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --event-balanced

# 缩短本次训练，例如训练 3 个 epoch；其它 train.py 参数同样可直接追加。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --num-epochs 3

# 可选：向标定先验注入 10% 噪声。
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced --prior-noise 0.1
```

均衡采样不改变 Qwen 提示词。独立的 `--event-balanced-scene-priors` 是离线场景文字实验，要求 `--dataset-priors` 且噪声为 0；全流程也会自动准备它的映射，但这种模型不能用于闭环。

## 查看 TensorBoard 和输出

另开一个终端：

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
# 指定端口：
TB_PORT=6006 bash qwen3vl_local/tb_serve.sh checkpoints/action_prior/latest/tb
```

打开终端显示的地址。主要看 `train/loss`、`train/samples_seen`、`val_epoch/route_ade_m` 和 `val_epoch/waypoint_ade_m`；均衡训练还可查看各事件组指标。FM loss 只反映训练拟合，轨迹质量以验证 ADE/FDE 为准。

每次训练使用独立的 `checkpoints/action_prior/run_时间戳/`，`latest` 指向最新 run：

| 文件/目录 | 用途 |
| --- | --- |
| `train.log`、`pipeline.log` | 训练/数据准备日志 |
| `best.pt`、`latest.pt` | 验证最优/最新 checkpoint |
| `tb/` | TensorBoard |
| `sampling/` | 每轮采样配额、唯一帧及重复次数 |
| `test/`、`probe/` | 最终测试指标和案例 |

## 续训

```bash
# 继续训练并完成最终 test/probe；采样开关和索引从原配置恢复。
RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh
GPU_IDS=0,1,2,3 RESUME=checkpoints/action_prior/latest/latest.pt bash qwen3vl_local/action_prior/run_full_pipeline.sh

# 仅续训：
bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/resume.sh checkpoints/action_prior/latest/latest.pt
```

续训使用原代码版本和原配置。更改采样算法或实验条件时开新 run；不要修改旧 checkpoint 的合同。

## 独立测试

```bash
bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/eval.sh --checkpoint checkpoints/action_prior/latest/best.pt

# 小样本可视化：
bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
GPU_IDS=0 bash qwen3vl_local/action_prior/probe.sh --checkpoint checkpoints/action_prior/latest/best.pt
```

默认沿用 checkpoint 的先验设置，test 不进行均衡重采样。闭环、审计、权重迁移等少用操作见 [AUDIT.md](AUDIT.md)，模型与采样合同见 [DESIGN.md](DESIGN.md)。

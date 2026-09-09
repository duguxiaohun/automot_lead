# Action Expert 消融实验

从远端 `AutoMoT/` 目录运行。本目录只做两个不使用 RS/EVENT 标定或 LoRA 先验的 action expert 对照，
轨迹 decoder、Flow Matching、BEV、数据索引和训练循环直接复用
`qwen3vl_local/action_prior/`，不再分别维护主线和消融的训练循环。

## 两个实验

| 子目录 | 条件输入 | 目的 |
| --- | --- | --- |
| `qwen_simple/` | 4 张 stitched RGB + LeadMoT 原简短导航 prompt 的 base Qwen KV + frozen BEV | 对照“没有推理/先验摘要”的 Qwen 视觉语言 encoder |
| `bev_only/` | frozen LEAD BEV(RGB+LiDAR) + speed/target/next target/final goal + route/waypoint query；Qwen prefix KV 长度为 0 | 对照“完全没有 Qwen 图文 KV”的 action expert |

两者都不读取 Phase1/Phase2 adapter，不读取 `prior_labels.jsonl`，不写 RS/EVENT/UNKNOWN 分桶 loss。
`bev_only` 仍沿用 action_prior 的 LEAD BEV encoder 输入与导航状态；这个 BEV backbone
仍融合当前 stitched RGB 与 LiDAR BEV，所以它不是纯 LiDAR/完全无视觉实验。区别是 Qwen
engine 不初始化，decoder 每层拿到的是 zero-length prefix KV。

## 共享代码与实验边界

主线 `action_prior/train.py` 和两个消融入口共同调用
[`action_prior/training_core.py`](../action_prior/training_core.py)：

- 模型与 FM 配置构造、FP32 AdamW、学习率调度、EMA。
- 每轮样本打乱、DDP 分片、梯度累积及不足完整窗口的更新。
- FM loss、固定噪声验证、Euler 采样指标、频繁验证及 epoch/final 验证选优。
- TensorBoard 核心日志、checkpoint 保存、恢复配置校验、待完成验证和预算停止。

`common.py` 只保留消融的输入构造、配置/条件合同、简化审计与入口组装；
主线通过审计接口继续记录 RS/EVENT 指标，消融不产生这些分桶或先验 case 审计。
后续修改上述训练行为应修改 `training_core.py`，三条入口会共同使用新实现。
先验生成、Qwen simple prompt 和无 Qwen 输入仍分别由各自 runtime 管理；修改这些条件分支
不会自动改变其他实验的定义。

`action_prior` 与 `qwen_simple` 的对比同时移除了先验、分析摘要并更换 prompt，
因此衡量的是整条先验推理链的收益，不能单独归因为“生成推理文字”的收益。
`qwen_simple` 与 `bev_only` 衡量 Qwen 图文分支的收益；BEV 的 RGB 融合分支仍然存在。

## 数据索引

默认复用主线索引：

```bash
python qwen3vl_local/action_expert_ablation/build_dataset.py \
  --data-root lead_data \
  --output-dir checkpoints/action_prior_data
```

该 wrapper 直接调用 `action_prior/build_dataset.py`，所以异常时长 route 剔除、4Hz anchor、
物理 route 分 split、route10/waypoint8 监督都与主线一致。若目录已存在不会覆盖。
两个消融的 full pipeline 默认共享该目录；首次启动时会用 `.build.lock` 文件配合
`flock` 做进程级构建锁，拿到锁后再次检查 `train/val/test` 三个 split，避免并发实验
同时写同名临时文件。锁文件残留不会阻塞后续启动，进程退出会自动释放文件锁。

## 快速运行

```bash
# Qwen + simple prompt
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh

# BEV-only
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh

# 只训练
bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh
bash qwen3vl_local/action_expert_ablation/bev_only/train.sh

# smoke：默认 4 个 optimizer update
bash qwen3vl_local/action_expert_ablation/qwen_simple/smoke.sh
bash qwen3vl_local/action_expert_ablation/bev_only/smoke.sh

# 独立 eval
bash qwen3vl_local/action_expert_ablation/qwen_simple/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/qwen_simple/latest/best.pt
bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
```

常用环境变量与 action_prior 对齐：`GPU_IDS`、`DDP_GPU_COUNT`、`DATA_ROOT`、`DATA_DIR`、
`LEAD_BEV_CKPT`、`NUM_EPOCHS`、`LR`、`GRAD_ACCUM`、`VAL_STEPS`、
`SAVE_STEPS`、`LOGGING_STEPS`、`NUM_WORKERS`、`TRAIN_SAMPLED_METRICS`。
`MODEL_DIR` 只对 `qwen_simple` 有效；`bev_only` 不初始化 Qwen。
`OUTPUT_DIR`/`RESUME` 可用环境变量或 CLI `--output-dir`/`--resume` 传入，launcher 会统一
创建/复用真实 run 目录，训练日志和 checkpoint 写在同一个目录。
`run_full_pipeline.sh` 会在训练前固定本次 `RUN_TAG` 和实际 run 路径，最终 eval 直接使用
该路径下的 `best.pt`，不会再通过可能被并发实验改写的 `latest` symlink 找权重。
CLI `--data-root` / `--data-dir` 会同时用于索引构建、训练和最终 eval。
若 `--resume` 指向 `latest/latest.pt` 这类软链接，pipeline 会先解析为真实 checkpoint 路径，
训练与最终 eval 都使用解析后的同一个 run。显式传入的 `MODEL_DIR` / `--model-dir` 和
`LEAD_BEV_CKPT` / `--lead-bev-ckpt` 也会传给最终 eval，再由权重哈希验证同一性。
仅传 `--resume` 时，训练入口会先读取 checkpoint 所在 run 的 `config.json` 恢复原始
LR、epoch、梯度累积、数据索引等非默认参数；launcher 在选 GPU 前读取
`training_plan.json` 恢复原始 `world_size` 默认值。命令行显式传入的参数仍优先生效，
用于数据/Qwen/BEV 路径搬迁等场景。`train.sh --resume` 不会再自动注入脚本默认 LR、
epoch、梯度累积或默认索引；只有用户实际传入的 CLI 参数或环境变量会作为覆盖传给 Python。

## TensorBoard 对比

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/action_expert_ablation/qwen_simple/latest/tb
bash qwen3vl_local/tb_serve.sh checkpoints/action_expert_ablation/bev_only/latest/tb
```

重点看：

- `train/loss`
- `train/route_fm_mse`
- `train/waypoint_fm_mse`
- `train/lr`
- `train/grad_norm`
- `train/samples_seen`：到当前 step 的累计训练 case 呈现数
- `train/step_samples`：当前一次 optimizer update 实际使用的 case 数
- `train/samples_per_second`
- `val/loss`
- `val/route_ade_m`
- `val/waypoint_ade_m`
- `val_epoch/*`

每个完整 optimizer step 是 `world_size * grad_accum_steps` 个 case，默认四卡、`GRAD_ACCUM=16`，
即 64 个 case，与 `action_prior` 一致；epoch 最后一个累积窗口可能不足完整 step，但在同索引、同卡数、
同累积数和同 seed 下尾部也一致。因此 TB 横坐标的 step 代表已经看过的训练样本量可直接对齐。
窗口日志中的 `loss(window_global,N 样本均值)` 也是全 rank 样本均值；`train/samples` 是这个
日志窗口的样本数，不能当作累计样本数。`train/samples_seen` 按训练游标计算，续训后连续，
包括不同 epoch 对同一帧的重复呈现，不含 val/test。

公平比较前核对三组的 `config.json`、`training_plan.json` 和 checkpoint 的 split hash：

| 对比项 | 必须对齐 |
| --- | --- |
| 数据预算 | 同一 train/val/test 索引及内容、GPU 数、梯度累积、seed、epoch/step 上限 |
| 优化设置 | LR、warmup、weight decay、EMA、decoder dropout、dtype |
| FM 定义 | route/waypoint loss 权重、坐标缩放、trajectory layers/heads |
| 评测口径 | 同一 val 子集/全量集、验证 seed、Euler 步数、EMA 或 raw 权重 |
| 曲线显示 | 同一 tag、日志间隔、TensorBoard smoothing，并核对 `samples_seen` |

降低 GPU 数、同时提高累积数即使保持 64 case/step，也可能改变 epoch 尾部与随机数消费，
不能直接声明严格同预算配对。默认训练只算 FM MSE；如需训练 ADE/FDE，三组都统一设置
`TRAIN_SAMPLED_METRICS=1`，同时披露增加的采样开销。

FM MSE 下降快只说明向量场训练误差下降快；最终效果看从纯噪声 Euler 采样的
`val_epoch/route_ade_m`、`val_epoch/waypoint_ade_m`、对应 FDE 与独立 test。
`best.pt` 按加权采样 route/waypoint ADE 选取。完整 epoch 使用全量 val；
`MAX_TRAIN_STEPS` 在 epoch 中途截断时，只做固定小验证集，smoke 的 best 不能与正式全量选优混用。
正式 test 不参与训练选优。

## Checkpoint

checkpoint schema 为 `action_expert_ablation_checkpoint_v1`，会保存：

- `ablation_variant`
- `decoder_config`
- `flow_config`
- EMA/optimizer/scheduler/RNG
- 条件合同 `condition_contract`
- shared dataset hash
- `cursor.validation_pending`，用于在 epoch/final validation 前中断后先补验证和 best 选择

resume 会在打开新的 TensorBoard writer 前归档旧 `tb/` 中 step 大于 checkpoint step 的
event，并保留 checkpoint step 本身的记录，避免续训曲线出现未来 step 或重复 step。
执行指纹绑定消融入口、共享 action_prior/LeadMoT/BEV 执行依赖和关键运行库版本；
未使用的 Phase1/2 prompt 不参与 `bev_only` 或 `qwen_simple` 身份。
`qwen_simple` 与 `bev_only` checkpoint 不能互相 resume/eval；合同会拒绝跨条件加载。


## 版本与验证

本次共享循环重构保持模型结构、FM 定义和 checkpoint 容器 schema；执行源码指纹会改变。
**旧 run 的续训/评测需要其原代码版本**，不能因 schema 相同就绕过合同检查。
新的配对实验应从同一份新代码、相同 seed 分别训练三组 decoder。

源码回归测试（不加载真实 Qwen/BEV、不读取真实 LEAD 数据、不启动 CARLA）：

```bash
python -m pytest -q qwen3vl_local/action_expert_ablation/tests qwen3vl_local/action_prior/tests
```

新增 CPU 小模型测试覆盖三入口的数据/更新/指标一致性，以及两个消融的中途、epoch 验证、
截断预算验证中断后恢复；已完成预算再次续训不执行额外样本。
测试还包含真实 TensorBoard event 裁剪检查，缺少 `tensorboard` 时会明确 skip；
本机未执行这一真实文件路径。CPU 测试不能替代真实 GPU/DDP 或 Qwen/BEV 前向检查。
远端首次运行可先使用上面的 `smoke.sh`，确认 loss 有限、best/latest 保存和独立 eval 可运行，
再按正式预算训练。smoke 是独立短预算 run，不能直接把它改成正式长预算续训。

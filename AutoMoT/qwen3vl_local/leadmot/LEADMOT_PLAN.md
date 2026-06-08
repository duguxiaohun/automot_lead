# LeadMoT 训练计划

> **版本说明**：LeadMoT 目前**只有 v1 一个版本**——单一一套训练/eval/probe 栈。
> 命名风格已经与 GoalGen 对齐（`train.py` / `train.sh` / `eval.py` / `probe.py`，
> 不带 _v1 后缀），未来要做 v2 时按 GoalGen 套路用 `--mode` / `VERSION` env 扩展
> 而**不**新建 `train_v2.py` 之类文件。
>
> **v1 内部当前唯一可控的结构开关**：`use_bev`（config 字段，CLI `--use-bev` /
> `--no-use-bev`，sh `USE_BEV=0/1`，默认 `True`）。开关含义见下面 §6 "use_bev"。

## 目标

训练 `AutoMoT/qwen3vl_local/leadmot/` 里的 planning decoder。训练时只更新 decoder 参数；Qwen3-VL-Instruct 与 `LeadBEVEncoder` 都保持 frozen eval。

输入与快慢推理 demo 完全对齐：

- frozen Qwen3-VL-Instruct prefill 的 HF `past_key_values`
- `_segment_qwen_cache_for_leadmot(...)` 得到的 12 层 prefix K/V
- frozen `LeadBEVEncoder` 的 `(B,512,10,12)` BEV feature
- speed / target_point / next_target_point / final_goal

输出：

- `pred_route`: `(B,10,2)`
- `pred_future_waypoints`: `(B,8,2)`

## 两类轨迹真值

LEAD 里确实有两类 planning 轨迹：

- `route`：空间路线 checkpoints，用于 route lateral controller。
- `future_waypoints`：带时间间隔的未来自车点，用于 waypoint PID；闭环里还用 0.5s 与 1.0s 两个点的距离估 desired speed。

它们的 label 都是当前 ego 坐标系下的累计/绝对点，不是相邻点之间的 delta。

LEAD 的 decoder head 是 `Linear(hidden,2) -> torch.cumsum(dim=1)`：模型内部预测相邻 delta，但输出 `pred_route` / `pred_future_waypoints` 已经 cumsum 成绝对点，然后 loss 直接对 `data["route"]` / `data["future_waypoints"]`。因此 LeadMoT v1 训练也保持同样语义：GT 不做 `diff`，head 内部已经 `cumsum`。

## 数据

`build_dataset.py` 只生成轻量 JSONL 索引，不复制图像、LiDAR 或 meta：

```bash
python qwen3vl_local/leadmot/build_dataset.py \
  --data-root /datashare/IOL4SGH/data/data \
  --samples-per-scenario 0 \
  --output-dir checkpoints/leadmot_v1_data
```

每行记录一个 route 目录与 anchor frame。训练时即时读取：

- `rgb/*.jpg`：沿用快慢推理 demo 的 4 帧 RGB 组。
- `lidar/*.laz` + `metas/*.pkl`：沿用 runner 的 BEV 构造。
- anchor meta 的 `route`：按 LEAD 训练语义先取 `route[:20]`，执行等价 `smooth_path(target_first_distance=2.5)`，再取前 10 点监督 `pred_route`。
- anchor meta 的 `future_positions[[5,10,...,40]]`：监督 8 个未来 waypoint，LEAD 4Hz 下覆盖 2s。

默认 `--samples-per-scenario 0` 与 GoalGen 一致，表示每个 scenario 保留所有合法 anchor；传正整数时按 route-balanced 方式抽样。默认 `--stride 5`，让相邻 anchor 大约间隔 1 秒，减少高度重叠的伪样本；如需全量密集 anchor，可显式 `--stride 1`。train/val 按 route 切分，避免同一路线相邻 anchor 同时进入训练和验证。构建器输出 `train.jsonl` / `val.jsonl` / `stats.json`。`--check-readable` 可选但默认**不推荐**：开了之后每个 anchor 要做 6 次 lzma + 12 次 file stat，几百 route 数据集会变成几小时；train 已经有 DDP-safe 占位 loss 兜底坏样本，不再需要在构建期预校验。

当前不做 Qwen pooled KV 离线缓存。原因是 prompt / RoPE / prefix 组织还在迭代期，prompt 一改缓存就失效；训练先保证范式正确，再考虑缓存工程。

## 优化默认值

Decoder 从头训练，外部 Qwen/BEV 都冻结，所以默认值偏向稳定收敛：

- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `0.01`
- betas: `(0.9, 0.95)`
- warmup ratio: `0.03`
- epochs: `1`
- grad accumulation: `8`
- grad clip: `1.0`
- dtype: decoder/Qwen 均默认 `bfloat16`
- training dropout: `0.1`，只在训练入口的 decoder config 使用；runner 推理仍默认 `0.0`
- Qwen load stagger: DDP 默认每个 rank 按 `LOCAL_RANK * 2s` 错峰调用 runner `_ensure_leadmot_qwen_engine()`
- loss: 默认 `1.0 * waypoint_L1 + 0.5 * (route_ADE_L1 + route_FDE_L1)`；需要更平滑早期梯度时可用 `--loss-type smooth_l1`

Waypoint 直接服务控制，权重更高；route 作为空间路线监督，也额外加末点 FDE 来稳定远端路线。

Route / waypoint head 内部在 cumsum 时临时升到 fp32，再 cast 回原 dtype，避免 bf16 连续累计带来末点误差。

## EMA

默认开 EMA shadow（CLI `--ema`，env `EMA=1`，decay=0.999），结构与 goalgen `DiTEMA` 同：fp32 存储、`mul_`+`add_` 融合更新。

- `ema.update(decoder_module)` 放在 `optimizer.step()` **之后**；放之前 shadow 拿未更新的旧参数会"慢一步"。
- val 用 `with ema.apply_to(decoder): ...` 上下文跑一次，得到 `val_ema/*` 一组 scalar；EMA 关时只有 `val/*`。
- `best.pt` 选 EMA val/loss 优先（更稳）；EMA 关时回退 raw val/loss。
- ckpt 同时持久化 `decoder` 与 `ema_state_dict`；当前 EMA schema 是 `{"decay": ..., "shadow": {...}}`，`eval.py` / `probe.py` 默认 `--use-ema=True` 并 unwrap `shadow` 后 strict load；旧 ckpt 无 EMA 字段时自动回退 raw + print 警告。
- decay 选择：默认 0.999 适配 LeadMoT 默认短 schedule（warmup ~500 step）；长 schedule (≥10 epoch) 可调 0.9999，但 warmup 期前一段 EMA 会拖收敛速度。

## TB 图像 overlay 样例

每 `image-log-every` 步（默认 1000）rank0 从 val 抽 `image-log-samples`（默认 4）条样本，渲染 pred vs gt 的 route + waypoint overlay 拼图（小 PIL 画，比 matplotlib 轻），贴到 TB：

- `samples/planning_overlay_raw`：raw 权重输出
- `samples/planning_overlay_ema`：EMA 权重输出（开 EMA 时多一组）

在 TB 上直接看模型质量进化，比 loss 曲线直观。所有 rank barrier 等 rank0 渲染完再继续训练，不破坏 DDP collective。`image-log-every=0` 关闭。

## TB 标量

| Group | Key | 说明 |
|---|---|---|
| `train/` | `loss` / `route_loss` / `waypoint_loss` / `lr` / `grad_norm` | 训练损失与诊断 |
| `train/` | `route_ade_m` / `route_fde_m` / `waypoint_ade_m` / `waypoint_fde_m` | 米数级 ADE/FDE，与 eval/probe 同口径 |
| `val/` | 同上 5 个 | raw 权重 val |
| `val_ema/` | 同上 5 个 | EMA 权重 val（开 EMA 时） |

`eval.py` 的 `eval_tb/<ckpt>_<ts>/` 子目录会写同名 key `eval/*`，可以在同一 TB 板上叠加对比训练 / 多 ckpt 离线 eval。

## RoPE

训练前向复用 `mot_lead_offline_runner.py` 的推理路径，因此 M-RoPE/MHRoPE 的使用位置与 demo 一致。默认仍建议 `mrope`；`mhrope` 只有在 Qwen prefill 侧也同步 patch 时才打开。

## 产物

默认输出目录：

```text
checkpoints/leadmot_v1_decoder/
```

包含：

- `latest.pt`
- `best.pt`（有 val 时）
- `best.json`（best checkpoint 的 val_loss / epoch / step 元信息）
- `checkpoint-epochXX.pt`（epoch 末池，保留最近 `KEEP_RECENT_CHECKPOINTS` 份）
- `step-checkpoint-NNNNNN.pt`（每 `STEP_SAVE_EVERY` 步独立池，保留最近 `KEEP_RECENT_STEP_CHECKPOINTS` 份，与 epoch 池互不淘汰）
- `tb/`（安装 TensorBoard 时）
- `invocations/`（train/eval/probe argv + env + git_commit）

runner 侧用 `--leadmot-ckpt checkpoints/leadmot_v1_decoder/best.pt` 或 `latest.pt` 加载。

checkpoint 写入使用 `torch.save(tmp)` + `os.replace(tmp, final)`，避免 NFS/中断场景读到半截文件。`--resume` 完整恢复 decoder + optimizer + scheduler；`--init-from-ckpt` 只加载 decoder 权重并重置 optimizer/scheduler。

## Eval / Probe

- `eval.py`：离线汇总 `loss / route ADE/FDE / waypoint ADE/FDE`，支持 torchrun 分片；每个 rank 跑自己的样本，rank0 合并 summary/perline。推荐传 `--save-root checkpoints/leadmot_v1_decoder`，产物落到 `<save-root>/eval/`。
- `probe.py`：随机按 scenario 抽 case，落盘 `planning_overlay.png`、`predictions.json`、`metrics.json`、`overview.md`，用于肉眼检查预测和 GT 是否同向、同尺度、同坐标系。推荐传 `--save-root checkpoints/leadmot_v1_decoder`，case 落到 `<save-root>/eval_cases/`。

## use_bev 开关

`LeadMoTPlanningDecoderConfig.use_bev`（默认 `True`）决定 decoder 是否在 gen 序列里
拼 120 个 BEV token。两档行为：

| `use_bev` | gen 序列长度 | `decoder.bev_projector` 子模块 | BEV encoder forward | 适用场景 |
|---|---|---|---|---|
| `True`（默认） | 142（BEV 120 + 22 status/queries） | 存在 | 跑 | 主训练，对齐 final_goal 路线 |
| `False` | 22（4 status + 18 queries） | **不实例化** | 跳过 | 消融实验、限显存环境 |

**state_dict 在两档之间不兼容**（`bev_projector` 子模块存在性变化），因此：

- 不能跨 `use_bev` 用 `--init-from-ckpt`；切档必须从头训或单独 warm start；
- eval / probe 从 ckpt 里自动读 `use_bev`（保存在 `decoder_config` 字典里），并要求 `use_final_goal=True`；旧 LeadMoT ckpt 缺 `use_final_goal` 字段时直接报错；
- runner（`mot_lead_offline_runner.py`）会先读取 ckpt 的 `decoder_config.use_bev` 再实例化
  decoder；`use_bev` 缺字段时仍按 state_dict 里是否存在 `bev_projector.*` 推断，但 `use_final_goal` 不做兼容推断。
  加载 decoder 权重使用 `strict=True`：`use_bev=True` 必须导入已有 BEV projector 参数，
  `use_bev=False` 则完全不实例化 / 不 forward BEV，绝不把随机 BEV projector 混进推理。

CLI / env 接口：

- `train.py --use-bev` / `--no-use-bev`（argparse `BooleanOptionalAction`）；
- `train.sh USE_BEV=0/1`（env，默认 `1`，转 `--no-use-bev` / `--use-bev` 透传）。

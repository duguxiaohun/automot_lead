# LeadMoT V1 训练计划

## 目标

训练 `AutoMoT/qwen3vl_local/leadmot/` 里的 planning decoder。训练时只更新 decoder 参数；Qwen3-VL-Instruct 与 `LeadBEVEncoder` 都保持 frozen eval。

输入与快慢推理 demo 完全对齐：

- frozen Qwen3-VL-Instruct prefill 的 HF `past_key_values`
- `_segment_qwen_cache_for_leadmot(...)` 得到的 12 层 prefix K/V
- frozen `LeadBEVEncoder` 的 `(B,512,10,12)` BEV feature
- speed / target_point / next_target_point

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

`build_dataset_v1.py` 只生成轻量 JSONL 索引，不复制图像、LiDAR 或 meta：

```bash
python qwen3vl_local/leadmot/build_dataset_v1.py \
  --data-root /datashare/IOL4SGH/data/data \
  --output-dir checkpoints/leadmot_v1_data
```

每行记录一个 route 目录与 anchor frame。训练时即时读取：

- `rgb/*.jpg`：沿用快慢推理 demo 的 4 帧 RGB 组。
- `lidar/*.laz` + `metas/*.pkl`：沿用 runner 的 BEV 构造。
- anchor meta 的 `route`：按 LEAD 训练语义先取 `route[:20]`，执行等价 `smooth_path(target_first_distance=2.5)`，再取前 10 点监督 `pred_route`。
- anchor meta 的 `future_positions[[5,10,...,40]]`：监督 8 个未来 waypoint，LEAD 4Hz 下覆盖 2s。

默认 `--stride 5`，让相邻 anchor 大约间隔 1 秒，减少高度重叠的伪样本；如需全量密集 anchor，可显式 `--stride 1`。建议构建正式训练索引时加 `--check-readable`，提前按训练实际读取集合检查历史 RGB/meta/LAZ、anchor 标签 meta，以及 TP/NTP 未来 meta，过滤缺文件或 meta 解压失败的样本，避免 DDP 训练中单条坏样本造成 collective 错位。

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
- `checkpoint-epochXX.pt`
- `tb/`（安装 TensorBoard 时）

runner 侧用 `--leadmot-ckpt checkpoints/leadmot_v1_decoder/best.pt` 或 `latest.pt` 加载。

checkpoint 写入使用 `torch.save(tmp)` + `os.replace(tmp, final)`，避免 NFS/中断场景读到半截文件。`--resume` 完整恢复 decoder + optimizer + scheduler；`--init-from-ckpt` 只加载 decoder 权重并重置 optimizer/scheduler。

## Eval / Probe

- `eval_v1.py`：离线汇总 `loss / route ADE/FDE / waypoint ADE/FDE`，支持 torchrun 分片；每个 rank 跑自己的样本，rank0 合并 summary/perline。
- `probe_v1.py`：随机按 scenario 抽 case，落盘 `planning_overlay.png`、`predictions.json`、`metrics.json`、`overview.md`，用于肉眼检查预测和 GT 是否同向、同尺度、同坐标系。

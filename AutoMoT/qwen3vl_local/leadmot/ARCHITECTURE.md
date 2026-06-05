# LeadMoTPlanningDecoder 架构说明

`AutoMoT/qwen3vl_local/leadmot/` 是 LeadMoT 快推理 planning decoder 子包。它的职责是接收 frozen Qwen3-VL-Instruct prefill 得到的 prefix K/V、LeadBEVEncoder 的 BEV feature、ego 状态 token，并输出两类 LEAD 风格轨迹：

- `pred_route (B, 10, 2)`：对齐 LEAD `route`，ego-frame 累计/绝对点。
- `pred_future_waypoints (B, 8, 2)`：对齐 LEAD `future_waypoints`，ego-frame 累计/绝对点。

训练时只更新 LeadMoT decoder；Qwen3-VL-Instruct 与 LeadBEVEncoder 必须保持 frozen eval。训练、eval、probe、runner 都应使用同源的 `LocalQwen3VLInstructEngine` prefill，不复用 AutoMoT legacy `InterleaveInferencer` / `gen_context` 路径。

## 1. 设计目标

- **输出语义对齐 LEAD PlanningDecoder**：head 先预测相邻 delta，再在 head 内 `cumsum` 得到累计点，loss 直接对累计/绝对点计算。
- **frozen Qwen prefix K/V 不再过 Linear**：每层读取 `(K, V)`，形状约为 `(B or 1, 8, S, 128)`，保持 Qwen K/V 子空间。
- **gen hidden = 1024 = 8 x 128**：与 Qwen `num_key_value_heads x head_dim` 对齐，方便直接 cross-attend 到 prefix K/V。
- **BEV/status/query 统一成 gen token 序列**：BEV 120 个 token，status 3 个 token，route query 10 个 token，waypoint query 8 个 token。
- **decoder-only 训练**：大模型和 BEV backbone 都不参与梯度，训练成本集中在新初始化的 planning decoder。

## 2. 与 AutoMoT 严格 MoT 的关系

AutoMoT 严格 MoT 是在 Qwen LM 层内分出 gen 路径；LeadMoT 把 gen 路径抽成外部独立 transformer。两者的共同点是：gen Q/K/V 自己投影，prefix K/V 来自 frozen Qwen prefill，attention 时把 gen K/V 与 prefix K/V 拼接。

简化对照：

```text
AutoMoT strict MoT:
  Q_gen, K_gen, V_gen = Qwen layer 内部的 mot_gen projection
  K_prefix, V_prefix  = 同层 past_key_values
  attn(Q_gen, concat(K_prefix, K_gen), concat(V_prefix, V_gen))

LeadMoT:
  Q_gen, K_gen, V_gen = 独立 MoTDecoderBlock projection
  K_prefix, V_prefix  = pooled_kv[i]，来自 Qwen 第若干层 K/V
  attn(Q_gen, concat(K_gen, K_prefix), concat(V_gen, V_prefix))
```

主要工程差异：

| 项 | AutoMoT strict MoT | LeadMoT |
|---|---|---|
| 层数 | 跟随 Qwen LM 层 | 默认 12 层 |
| gen hidden | Qwen hidden 子空间 | 1024 |
| prefix K/V | 同层 Qwen cache | `segment_kv_for_dit(..., mode=select_last)` 后的 pooled K/V |
| 输出 | reasoning + route + waypoint | route + waypoint |
| 参数初始化 | 可从 Qwen 层继承 | 当前从头训练 |

## 3. RoPE 边界

`LeadMoTPlanningDecoderConfig.rope_type` 支持：

| rope_type | 用途 | 说明 |
|---|---|---|
| `mrope` | 默认训练/推理 | gen Q/K 使用 Qwen3-VL 风格 M-RoPE；gen token 作为接在 prefill 后的新文本 token，三轴位置相同。 |
| `mhrope` | 消融或同步 patch 后使用 | head-wise RoPE。只有 Qwen prefill 侧也同步改成 MHRoPE 时才严格一致。 |
| `none` | 对照实验 | gen Q/K 不旋转，attention 合法但缺少连续位置编码。 |

prefix K/V 已经在 Qwen prefill 内带了 Qwen 自己的 RoPE，不要在 LeadMoT 里重复旋转。LeadMoT 只给新生成的 gen Q/K 加 RoPE。

`rope_position_offset` 应来自 runner 侧 Qwen prefill 输出：

```text
input_ids.shape[-1] + outputs.rope_deltas
```

拿不到时才退回 prefix cache 长度。这个 offset 表示 gen token 接在 Qwen prefill 后的位置起点。

## 4. 模块结构

```text
LeadMoTPlanningDecoder
├── LeadBEVProjector        BEV (B, 512, 10, 12) -> (B, 120, 1024)
├── StatusTokenEncoder      speed/tp/ntp -> 3 x (B, 1, 1024)
├── RouteQueryBank          learned queries -> (B, 10, 1024)
├── WaypointQueryBank       learned queries -> (B, 8, 1024)
├── MoTDecoderBlock x 12    prefix K/V attention + FFN
├── final RMSNorm
├── RouteHead               Linear -> fp32 cumsum -> (B, 10, 2)
└── WaypointHead            Linear -> fp32 cumsum -> (B, 8, 2)
```

## 5. Packed Gen 序列布局

`config.slice_layout()` 是下面这张表的代码真值，`decoder._build_gen_sequence()` 的拼接顺序必须与它一致。

```text
index    [0..120)   [120]   [121]   [122]   [123..133)   [133..141)
content  BEV 120    speed   tp      ntp     route_q 10   waypoint_q 8
```

forward 后只取 route query 与 waypoint query 对应位置送入各自 head，BEV/status token 不直接输出轨迹。

## 6. 前向张量流

```text
caller / runner:
  RGB + prompt -> frozen Qwen3-VL-Instruct prefill
  past_key_values -> segment_kv_for_dit(num_segments=decoder.num_layers)
  pooled_kv: list[(K, V)] x num_layers

LeadMoT decoder:
  BEV + speed + target_point + next_target_point
  -> BEV/status/query packed gen sequence
  -> MoTDecoderBlock x num_layers, each attends to pooled_kv[i]
  -> route slice + waypoint slice
  -> Linear + fp32 cumsum heads
```

输入约定：

- `bev`: `(B, 512, 10, 12)`，来自 frozen LeadBEVEncoder。
- `speed`: `(B,)` 或 `(B, 1)`，单位 m/s，不额外归一化。
- `target_point`: `(B, 2)`，当前 ego-frame target point。
- `target_point_next`: `(B, 2)`，下一 lookahead target point。
- `pooled_kv`: 长度必须等于 `config.num_layers`。

## 7. 不在本子包里做的事

- Qwen 图片/文本 prompt 构建与 prefill：由 runner/runtime 调用 `LocalQwen3VLInstructEngine` 完成。
- LEAD route clip 构建、LiDAR/RGB/BEV 对齐：由 `mot_lead_offline_runner.py` 的离线路径完成。
- 训练 loss、optimizer、EMA、checkpoint、eval/probe：由 `train.py`、`eval.py`、`probe.py` 完成；运行细节见 `LEADMOT_PLAN.md` 与 `LEADMOT_RUN.md`。

## 8. 已知边界

- 当前训练/eval/probe 默认每卡单样本前向，prefix padding mask 暂不需要。如果未来 batch 内不同样本 prompt 长度不同，需要给 `PrefixKVAttention.forward()` 增加 `lang_key_padding_mask` 并透传到 attention。
- `mhrope` 不是单改 LeadMoT 就能完整生效；Qwen prefill 侧必须同步 patch，否则 prefix K 与 gen Q 的旋转规则不严格匹配。
- decoder 从头初始化，浅层结构比 Qwen 内部 strict MoT 更轻，收敛质量主要依赖足够的 LEAD 离线数据和稳定的 frozen prefill 分布。

## 9. 最小调用模板

```python
import torch

from AutoMoT.qwen3vl_local.goalgen.qwen_kv import segment_kv_for_dit
from AutoMoT.qwen3vl_local.leadmot import LeadMoTPlanningDecoder

decoder = LeadMoTPlanningDecoder().cuda().eval()

with torch.no_grad():
    qwen_outputs = engine.prefill(qwen_inputs)
    pooled_kv = segment_kv_for_dit(
        qwen_outputs.past_key_values,
        num_segments=decoder.config.num_layers,
        mode=decoder.config.kv_segment_mode,
    )

    out = decoder(
        pooled_kv=pooled_kv,
        bev=bev,
        speed=speed,
        target_point=target_point,
        target_point_next=target_point_next,
        rope_position_offset=rope_position_offset,
    )

pred_route = out["pred_route"]
pred_future_waypoints = out["pred_future_waypoints"]
```

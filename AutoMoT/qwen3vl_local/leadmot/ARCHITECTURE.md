# LeadMoTPlanningDecoder 架构说明

LEAD-MoT 快推理 decoder 设计文档。代码在同目录 `.py` 文件。

## 1. 设计目标

- **输出严格按 LEAD `PlanningDecoder`**：
  - `pred_route (B, 10, 2)` 对齐 `data["route"]`
  - `pred_future_waypoints (B, 8, 2)` 对齐 `data["future_waypoints"]`，4Hz × 2s
- **frozen Qwen prefix K/V，不过 Linear**：`pooled_kv[i] = (K, V)`，每个 `(B or 1, 8, S, 128)`
- **gen 路 hidden=1024 = 8 × 128**，与 Qwen num_kv_heads × head_dim 子空间一致
- **status encoder 严格复刻 AutoMoT 快路**：velocity MLP + 共享 WaypointInputAdaptor

## 2. 与 AutoMoT 严格 MoT 的关系

AutoMoT 严格 MoT 在 Qwen LM 每一层内做 q/k/v_proj_mot_gen 分流（[qwen3vl_navit.py:678-770](../../Automot/mot/modeling/automot/qwen3vl_navit.py#L678)）。本子包把 gen 路放外部独立 12 层 transformer，每层读一段 select_last 池化的 K/V。**attention 数学结构等价**，差别在工程封装。

attention 对照：

```
AutoMoT 严格 MoT (Qwen LM 第 layer_idx 层, 共 36 层):
  Q_gen = q_proj_mot_gen(new_gen);  K/V 同理（+ apply_rotary_pos_emb 给新 K）
  K_cache, V_cache = past_key_values[layer_idx]   # 已带 RoPE，不投影
  attn(Q_gen, merge(K_cache, K_gen), merge(V_cache, V_gen))

leadmot (独立第 i 层, 共 12 层):
  Q_gen = q_proj(gen); K/V 同理; 各自 q_norm/k_norm + 1D RoPE
  K_pooled, V_pooled = pooled_kv[i]                # Qwen 第 (3i+2) 层 K/V
  attn(Q_gen, concat(K_gen, K_pooled), concat(V_gen, V_pooled))
```

关键差异（其余项一致）：

| 维度 | AutoMoT | leadmot | 影响 |
|---|---|---|---|
| transformer 层数 | 36 | 12 | 浅层信息丢失 |
| gen hidden | 2560 | 1024 (=8×128) | FFN 内部更窄 |
| GQA | ✅ | ❌ MHA num_heads=num_kv_heads=8 | Q 表达略少 |
| gen 新 Q/K 加 RoPE | ✅ M-RoPE | ✅ MRoPE / MHRoPE 可选 | 见 §2.1 |
| 参数初始化 | 从 Qwen 复制 | 从头随机 | 训练成本更高 |
| 输出头 | reasoning + route(20) + waypoint(6) | route(10) + waypoint(8) | 明确接受 |

### 2.1 RoPE：MRoPE / MHRoPE 二选一

`PrefixKVAttention` 用 `rope_type` 在两种 freq allocation 间分发（参考
[JJJYmmm/Multimodal-RoPEs](https://github.com/JJJYmmm/Multimodal-RoPEs) ICLR 2026）。
两者共用同一份 3D position（gen token 是接 prefill 末尾的新文本 token，三轴全等）。

| RoPE 类型 | 分配策略 | section 含义 | 默认 |
|---|---|---|---|
| `mrope` | head_dim//2 切 3 段，每段用一个 axis | `mrope_section_dim` 三段长度 (sum = head_dim/2 = 64) | `(16, 24, 24)` Qwen3-VL 标准 |
| `mhrope` | num_heads 切 3 段，每段共享一个 axis | `mrope_section_head` 三段 head 数 (sum ≤ num_heads = 8) | `(3, 3, 2)`，剩余 0 个 head pad 零 |

**MRoPE 与 Qwen3-VL prefix K 的兼容性**：Qwen3-VL prefix K 本身就是按 M-RoPE 旋转
（vision token 三轴 (t,h,w) 互不等，text token (t,t,t) 全等）。LeadMoT gen Q 用
mrope 旋转时，因为 gen token 三轴全等，head_dim 三段都用 t 旋转 — 跟 Qwen 原版
"gen Q · prefix K" attention 数学完全一致：

| head_dim 段 | gen Q 旋转用 | prefix vision K 旋转用 | 相对位置 |
|---|---|---|---|
| 0 | t_Q | t_K | t_Q − t_K |
| 1 | t_Q | h_K | t_Q − h_K |
| 2 | t_Q | w_K | t_Q − w_K |

**MHRoPE 与 prefix K 的兼容性警告**：MHRoPE 把 axis 分配到 **head 维**——
不同 head 用不同 axis。如果 prefix K 是默认 M-RoPE 旋转的（head_dim 切段），
gen Q 改用 head-wise allocation 会有不匹配。要充分利用 MHRoPE 需要 **同时** patch
standalone Qwen3-VL 的 prefill 改用 MHRoPE 旋转 K/V（即 frozen Qwen 也要切到
MHRoPE 模式）。本子包只暴露接口，**真正切到 MHRoPE 时调用方必须自己处理 Qwen
prefill 侧的 RoPE patch**。

**起点对齐**：`rope_position_offset` 优先用 `input_ids.shape[-1] + outputs.rope_deltas`
（Qwen3-VL 增量 decode 的 next-token position），拿不到时回退 prefix `cache.seq_len`。
runner 已自动从 standalone Qwen 输出里拿，传 scalar 给 decoder，内部展开成
`(3, B, L_gen)` 三轴全等的 3D position。

### 2.2 参数从随机起步

如训不动，可考虑从 Qwen 第 (3i+2) 层的 q/k/v/o_proj、norm、mlp 权重初始化 leadmot
第 i 层（仅维度对齐部分）。

## 3. 模块拓扑

```
LeadMoTPlanningDecoder
├── LeadBEVProjector        BEV (B,512,10,12) -> (B,120,1024)
├── StatusTokenEncoder      speed/tp/ntp -> 3 × (B,1,1024)
├── RouteQueryBank          -> (B,10,1024)
├── WaypointQueryBank       -> (B,8,1024)
├── blocks × 12             RMSNorm -> PrefixKVAttention -> +res -> RMSNorm -> SwiGLU -> +res
├── gen_final_norm
├── RouteHead               Linear(1024->2) + cumsum -> (B,10,2)
└── WaypointHead            Linear(1024->2) + cumsum -> (B,8,2)
```

## 4. packed gen 序列 layout

```
索引   [0..120)  [120]  [121]  [122]  [123..133)  [133..141)
内容    BEV 120  speed  tp     ntp    route_q 10  waypoint_q 8
```

`config.slice_layout()` 是这张表的真值，`_build_gen_sequence` 的 cat 顺序必须匹配。

## 5. 前向张量流

```
[调用方完成]
  Qwen3-VL-Instruct prefill -> past_key_values (36 层)
  -> segment_kv_for_dit(num_segments=12, mode='select_last')
  -> pooled_kv: list[(K,V)] × 12, each (B or 1, 8, S, 128)

[本子包]
  bev/speed/tp/ntp -> _build_gen_sequence -> gen_seq (B,141,1024)
  for i in 0..11:
    gen_seq = MoTDecoderBlock_i(gen_seq, pooled_kv[i])
  切片 + heads(Linear+cumsum) -> pred_route (B,10,2), pred_future_waypoints (B,8,2)
```

## 6. 不在本子包做的事

- Qwen prefill / KV 分段：调用方做，可参考 `goalgen/qwen_kv.py:segment_kv_for_dit`
- 真值轨迹提取（LEAD `future_waypoints` / `route`）
- 损失函数 / 优化器 / 数据加载 / 训练循环 / eval 脚本

### 已知待解决问题

> ⚠️ **prefix padding mask 未支持**：`PrefixKVAttention` 不接 attention mask。
> 单样本推理 / 每卡 B=1 训练无影响。多样本同 batch 训练时不同样本 prompt 长度
> 不同会触发 Qwen 左 padding，需为 `PrefixKVAttention.forward` 加
> `lang_key_padding_mask: (B, S)` 参数并透传。

> ⚠️ **训练 / 推理 cache 必须同源**：prefix K/V 必须来自同一个 frozen Qwen 实例。
> 用 standalone `Qwen3-VL-4B-Instruct` 训练就必须用同一份推理。runner 已修复
> 这条：`mot_lead_offline_runner.py` 在 `enable_leadmot_planning` 分支用
> `LocalQwen3VLInstructEngine` 单独跑 prefill，不再复用
> `gen_context["past_key_values"]`（那是 AutoMoT MoT 的产物）。训练侧也必须
> 用同一 engine + 同一对 (system, user) prompt。

## 7. 调用模板

```python
import torch
from AutoMoT.qwen3vl_local.leadmot import LeadMoTPlanningDecoder
from AutoMoT.qwen3vl_local.goalgen.qwen_kv import segment_kv_for_dit

decoder = LeadMoTPlanningDecoder().cuda()

# 1) frozen Qwen prefill (engine.prefill 已 use_cache=True)
with torch.no_grad():
    qwen_outputs = engine.prefill(qwen_inputs)
past_key_values = qwen_outputs.past_key_values

# 2) 池化 num_segments 必须 = decoder.config.num_layers
pooled_kv = segment_kv_for_dit(
    past_key_values,
    num_segments=decoder.config.num_layers,
    mode=decoder.config.kv_segment_mode,
)

# 3) decoder forward
out = decoder(
    pooled_kv=pooled_kv,
    bev=bev,                   # (B, 512, 10, 12)
    speed=speed,               # (B,) 或 (B, 1)
    target_point=tp,           # (B, 2)
    target_point_next=ntp,     # (B, 2)
)
# out["pred_route"]            : (B, 10, 2)
# out["pred_future_waypoints"] : (B,  8, 2)
```

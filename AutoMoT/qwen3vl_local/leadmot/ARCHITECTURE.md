# LeadMoTPlanningDecoder 架构说明

> 这是 LEAD-MoT 快推理 decoder 的设计文档。代码在同目录其它 `.py` 文件里。
> 本文档只讲架构（数据流、张量形状、模块边界），不讲训练（数据集、loss、optimizer）。

---

## 1. 设计目标

- **输出契约严格按 LEAD `PlanningDecoder`**：
  - `pred_route`            : `(B, 10, 2)` ego-local 米，对齐 `data["route"]`
  - `pred_future_waypoints` : `(B,  8, 2)` ego-local 米，对齐 `data["future_waypoints"]`，4Hz x 2s
- **语言 memory 严格走 frozen Qwen prefix K/V**：
  - `pooled_kv[i] = (K, V)`
  - 每个张量形状为 `(B 或 1, 8, S, 128)`
  - K/V 直接进入 attention，不经过任何线性投影
- **gen 路 hidden=1024**：`8 heads * 128 head_dim`，和 Qwen K/V 子空间完全一致。
- **status encoder 严格复刻 AutoMoT 快路**：
  - speed: `Linear(1,256)->ReLU->Linear(256,512)->ReLU->Linear(512,1024)`
  - target points: 共享 `WaypointInputAdaptor`，输入 `(B,2,2)`，输出 `(B,2,1024)`

---

## 2. 与 AutoMoT 严格 MoT 的关系

AutoMoT 原生 MoT 在 Qwen LM **每一层** transformer 内做双路 Q/K/V 投影分流：
gen token 真正混在 frozen Qwen 的 packed sequence 中，每层 attention 同时
看 frozen cache 的 K/V 和当前层新算出的 gen K/V
（[qwen3vl_navit.py:678-770](../../Automot/mot/modeling/automot/qwen3vl_navit.py#L678)）。

本子包不 patch Qwen 源码，而是把 gen 路放到外部的独立 12 层 transformer，每层
读 frozen Qwen 一段池化后的 K/V。**attention 数学结构与严格 MoT 一致**，差别
集中在工程封装。

### 2.1 attention 计算对照

```
AutoMoT 严格 MoT，对第 layer_idx 层（共 36 层）：
  Q_gen = q_proj_mot_gen(new_gen);  K_gen = k_proj_mot_gen(new_gen);  V_gen = v_proj_mot_gen(new_gen)
  Q_gen, K_gen = apply_rotary_pos_emb(Q_gen, K_gen, pos)
  K_cache, V_cache = past_key_values[layer_idx]    # 已带 prefill 时 RoPE，不再投影
  attn(Q_gen, merge(K_cache, K_gen), merge(V_cache, V_gen))
  → o_proj_mot_gen → 进入下一层

本子包 leadmot，对第 i 层（共 12 层）：
  Q_gen = q_proj(gen);    K_gen = k_proj(gen);    V_gen = v_proj(gen)
  Q_gen = q_norm(Q_gen);  K_gen = k_norm(K_gen)   # head_dim 上 RMSNorm
  K_pooled, V_pooled = pooled_kv[i]                # = Qwen 第 (3i+2) 层 K/V, select_last，不投影
  attn(Q_gen, concat(K_gen, K_pooled), concat(V_gen, V_pooled))
  → o_proj → 进入下一层
```

### 2.2 对照表

| 维度 | AutoMoT 严格 MoT | 本子包 leadmot | 实质差异 |
|---|---|---|---|
| gen Q/K/V 独立投影 | ✅ `_mot_gen` 后缀 | ✅ 自己的 `q/k/v_proj` | 🟢 一致 |
| frozen K/V 不过 Linear | ✅ | ✅ | 🟢 一致 |
| q_norm / k_norm | ✅ Qwen3 风格 | ✅ Qwen3 风格 | 🟢 一致 |
| SwiGLU FFN | ✅ `mlp_mot_gen` | ✅ SwiGLU | 🟢 一致 |
| transformer 层数 | 36（Qwen LM 全部） | 12（独立） | 🟡 层数 1/3 |
| 每层 K/V 来源 | 第 N 层 cache | Qwen 第 (3i+2) 层（select_last） | 🟡 浅层信息丢失 |
| gen hidden | 2560（Qwen 主干） | 1024（= 8×128） | 🟡 FFN 内部表达更窄 |
| 是否 GQA | ✅ Qwen GQA | ❌ MHA, num_heads=num_kv_heads=8 | 🟡 Q 表达略少，但 K/V 对齐零成本 |
| gen token 新 K 是否加 RoPE | ✅ apply_rotary_pos_emb | ❌ 不加 | 🔴 相对位置编码不连贯，详见 §2.3 |
| 与 text token 同层分流 | ✅ | — 我们不输出 text | ⚪ 分流概念退化 |
| 参数初始化 | 从 Qwen 复制 + 微调（[qwen3vl_navit.py:1411-1417](../../Automot/mot/modeling/automot/qwen3vl_navit.py#L1411)） | 从头随机 | 🔴 训练成本更高 |
| 输出头 | reasoning + route(20) + waypoint(6) | route(10) + waypoint(8) | ⚪ 明确接受 |

### 2.3 RoPE 缺失的影响

cache K 在 prefill 阶段已加 RoPE（按绝对位置）。如果 gen Q 不加 RoPE，
attention 时 cache 内部 token 的相对位置感会被抹平。**这是与严格 MoT 真正
的功能性差异**，但：

- gen 段自身（BEV / status / query）位置固定且数量少，BEV 已有 2D pos embed，
  不依赖 RoPE
- gen 对 cache 的 attention 只用来读"语义内容"，不依赖严格相对位置
- goalgen v2 的 DiT 也是这么做的（[dit.py JointAttention](../goalgen/dit.py)），跑得起来

如果训练时发现 gen 区分不出 cache 中不同位置 token，下一步可补 RoPE。

### 2.4 参数初始化的潜在补救

AutoMoT 通过 `init_mot()` 把 `_mot_gen` 参数从同名原 Qwen 参数复制起步，相当
于"从训好的 transformer 起步再微调"。本子包当前从头随机初始化，可能需要更多
数据 / 更慢收敛。

如果训不动，可考虑用 Qwen 第 (3i+2) 层的 `q/k/v/o_proj`、`q_norm/k_norm`、
`mlp` 权重初始化 leadmot 第 i 层对应模块（维度对齐的部分，hidden 不同维度的
权重需要降维投影或新初始化）。属于后续优化项。

### 2.5 结论

leadmot **就是 MoT 风格的 KV cache 动作生成**——attention 数学和严格 MoT 同
构。剩下的差异（层数、池化、RoPE、初始化）是工程取舍，与 goalgen v2 一致，
已被那条线证明可行。

---

## 3. 模块拓扑

```
LeadMoTPlanningDecoder
├── LeadBEVProjector            BEV (B,512,10,12) -> (B,120,1024)
├── StatusTokenEncoder
│   ├── encode_speed             (B,) + AutoMoT velocity MLP -> (B,1,1024)
│   └── encode_target_points      tp/ntp (B,2,2) shared adaptor -> (B,2,1024)
├── RouteQueryBank               (B,) -> (B,10,1024)
├── WaypointQueryBank            (B,) -> (B, 8,1024)
├── blocks: ModuleList[MoTDecoderBlock x 12]
│   每层: RMSNorm -> PrefixKVAttention(gen K/V + Qwen K/V) -> residual
│         RMSNorm -> SwiGLU FFN -> residual
├── gen_final_norm               RMSNorm
├── RouteHead                    Linear(1024->2) + cumsum -> (B,10,2)
└── WaypointHead                 Linear(1024->2) + cumsum -> (B, 8,2)
```

---

## 4. packed gen 序列 layout

总长 `L_gen = 141`，固定，没有 padding。

```
索引  [   0 .. 120) | [120] | [121] | [122] | [123 .. 133) | [133 .. 141)
内容    BEV 120       speed   tp      ntp     route_q 10     waypoint_q 8
```

`LeadMoTPlanningDecoderConfig.slice_layout()` 返回这张索引表，`decoder.forward()`
末尾按它切出 `route_hidden` 和 `wp_hidden`。

---

## 5. 前向张量流

```
[调用方完成的慢推理]
RGB 4f + prompt
  -> frozen Qwen prefill
  -> past_key_values
  -> segment_kv_for_dit(..., num_segments=12, mode="select_last")
  -> pooled_kv: list[(K,V)] * 12, each K/V (B or 1, 8, S, 128)

[本子包快推理]
bev (B,512,10,12) ---+
speed (B,) ----------+
tp (B,2) ------------+--> _build_gen_sequence -> gen_seq (B,141,1024)
ntp (B,2) -----------+

for i in 0..11:
  gen_seq = MoTDecoderBlock_i(gen_seq, pooled_kv[i])

slice:
  route_hidden = gen_seq[:, 123:133, :]   (B,10,1024)
  wp_hidden    = gen_seq[:, 133:141, :]   (B, 8,1024)

heads:
  delta_route -> cumsum -> pred_route            (B,10,2)
  delta_wp    -> cumsum -> pred_future_waypoints (B, 8,2)
```

---

## 6. 不在本子包做的事

- **Qwen prefill / KV 分段**：复用 `goalgen/qwen_kv.py` 的 `_to_layer_list`
  / `segment_kv_for_dit` 思路，或由调用方生成等价 `pooled_kv`。
- **真值轨迹提取**：从 LEAD pkl 取 `future_waypoints` / `route`。
- **损失函数**：L1 / Huber / 时间权重。
- **优化器 / 数据加载 / 训练循环 / eval 脚本**。

> ⚠️ **prefix padding mask 未支持**：`PrefixKVAttention` 当前不接受任何
> attention mask。单样本推理 / 每卡 B=1 训练**无影响**（没有 padding）。
> 但**多样本同 batch 训练**时，如果不同样本的 Qwen prompt 长度不同，
> Qwen prefill 会左 padding，K/V 里 padding 位置不会被屏蔽，attention
> 会给它们少量概率质量造成数值偏差。届时需为 `PrefixKVAttention.forward`
> 加一个 `lang_key_padding_mask: (B, S)` 参数，并在 `MoTDecoderBlock` /
> `LeadMoTPlanningDecoder.forward` 透传下去。

---

## 7. 调用模板

完整调用必须包含 **三个阶段**：(1) frozen Qwen prefill；(2) KV 分段池化；
(3) leadmot decoder 前向。

```python
import torch
from AutoMoT.qwen3vl_local.leadmot import LeadMoTPlanningDecoder
from AutoMoT.qwen3vl_local.goalgen.qwen_kv import segment_kv_for_dit

decoder = LeadMoTPlanningDecoder().cuda()

# --- 1) frozen Qwen prefill：调用 engine 拿 past_key_values ---
# engine.prefill 内部已 use_cache=True，无需改 engine.py
with torch.no_grad():
    qwen_outputs = engine.prefill(qwen_inputs)
past_key_values = qwen_outputs.past_key_values    # 36 层 (k, v)

# --- 2) 把 36 层 KV 池化成 decoder.config.num_layers (=12) 段 ---
# num_segments / mode 必须与 decoder 配置一致，否则 _check_pooled_kv 会报错
pooled_kv = segment_kv_for_dit(
    past_key_values,
    num_segments=decoder.config.num_layers,
    mode=decoder.config.kv_segment_mode,         # 默认 'select_last'
)
# pooled_kv: list[(K, V)] * 12, each K/V shape (B or 1, 8, S, 128)

# --- 3) 快路前向 ---
out = decoder(
    pooled_kv=pooled_kv,
    bev=bev,                       # (B, 512, 10, 12)
    speed=speed,                   # (B,) 或 (B, 1)
    target_point=tp,               # (B, 2)
    target_point_next=ntp,         # (B, 2)
)
# out["pred_route"]            : (B, 10, 2)
# out["pred_future_waypoints"] : (B,  8, 2)
```

> 重要：`segment_kv_for_dit` 的 `num_segments` **必须** 等于
> `decoder.config.num_layers`。如果改 decoder 层数，这里也要一起改，
> 否则 `_check_pooled_kv` 会在 forward 入口直接抛错。

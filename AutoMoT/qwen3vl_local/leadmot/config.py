"""LeadMoTPlanningDecoder 的配置类。

这一份配置承担三件事
====================
1. **维度与层数的单一真值**：BEV channels、num_route_queries、num_waypoint_queries、
   num_layers、num_heads、head_dim 等所有可调参数集中在这里；其它模块（projectors、
   query_bank、heads、mot_block、decoder）通过引用 config 字段获取实际数值，避免
   到处硬编码。
2. **与 Qwen3-VL-4B-Instruct K/V 子空间的强对齐约束**：通过 `validate_qwen_kv_shape()`
   在初始化时立即报错，避免 hidden_size / num_heads / head_dim 三者不自洽时把错误
   推到 attention 内部才暴露。
3. **packed gen 序列布局的索引计算**：`total_gen_tokens()` 和 `slice_layout()`
   返回 BEV/speed/tp/ntp/route_q/waypoint_q 在 packed gen 序列中的边界，供 decoder
   切片时使用，保证布局变化时只改一处。

为什么 gen 路 hidden = 1024 而不是 2560
=======================================
- Qwen3-VL-4B-Instruct 的 K/V 子空间形状 = (num_kv_heads=8, head_dim=128)，
  每个 token 的 K/V 实际是 8*128=1024 维。
- 让 gen 路 hidden 也是 1024，gen 自己投影出的 K/V 跟 frozen Qwen K/V 在最后两维
  完全一致，可以 **直接 concat 进 attention，无需任何线性投影**——这是用户的明确偏好。
- 代价：gen 路 hidden 比 Qwen 主干 2560 窄，FFN 内部表达力略低。但在我们这种"只
  做轨迹回归"的轻量任务上，1024 已经够用，goalgen v2 的 DiT 也是这么选的。

为什么数值默认值是 CARLA leaderboard 模式
==========================================
LEAD `config_training.py` 在 CARLA leaderboard 模式下：
- `num_route_points_prediction = 10`
- `num_way_points_prediction = 8`
- `waypoints_spacing = 5`（@ 20Hz CARLA -> 4Hz 输出 -> 0.25s 间隔）
- BEV grid (10, 12)、512 channels（来自 LEAD LeadTransfuserBackbone）

我们直接采纳这些默认值，让输出契约天然与 LEAD 训练分布对齐，免去做时间插值
或重采样。如果切到 navsim / waymo 数据集，需要按 LEAD config 对应模式覆盖。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class LeadMoTPlanningDecoderConfig:
    # ============================================================
    # 公共维度
    # ============================================================
    # gen 路 hidden = num_kv_heads * head_dim = 8 * 128 = 1024
    # 这个等式必须成立（见 validate_qwen_kv_shape），否则 prefix-KV attention
    # 在拼接 frozen K/V 时 reshape 会失败。
    hidden_size: int = 1024

    # Qwen3-VL-4B-Instruct 主干 hidden（仅作元信息，本子包不消费）
    # 留作字段是为了：(a) 文档化我们对接的是哪个 Qwen 版本；(b) 未来若需要做
    # Qwen hidden -> gen hidden 的额外投影（不推荐），方便维度参数化。
    qwen_hidden_size: int = 2560

    # 每个 waypoint 输出维度：只输出 (x, y)，不输出 heading（LEAD CARLA 模式也不输出）
    point_dim: int = 2

    # ============================================================
    # frozen Qwen prefix K/V 维度（必须与运行时 Qwen 一致）
    # ============================================================
    # Qwen3-VL-4B-Instruct GQA 配置：num_attention_heads=32, num_key_value_heads=8
    # 我们只关心 K/V 侧的 num_kv_heads=8（gen 路也用 8 头让维度天然匹配）
    num_kv_heads: int = 8

    # 每个 head 的维度，Qwen3-VL-4B-Instruct = 128
    head_dim: int = 128

    # Qwen 主干 transformer 层数。Qwen3-VL-4B-Instruct = 36 层
    # 用来判断 KV cache 池化时是否能切出足够的段数（见 runner 里 _segment 校验）
    num_qwen_layers: int = 36

    # KV 池化模式：把 num_qwen_layers (36) 池化成 num_layers (12) 段
    # 'select_last' = 每段取该段内最后一层 Qwen 的 K/V（语义最丰富、显存最省）
    # 备选：'mean' 段内平均（旧版本，容易冲淡方向性）；'concat_layers' 段内 token 维拼接（显存翻倍）
    # 默认 select_last，与 goalgen 一致
    kv_segment_mode: str = "select_last"

    # ============================================================
    # BEV 输入维度（LEAD LeadTransfuserBackbone 输出）
    # ============================================================
    # LEAD tfv6_resnet34 backbone 输出 channel = 512
    bev_channels: int = 512

    # LEAD BEV 栅格 (H, W) = (10, 12)
    # 这是 LEAD TransFuser BEV 编码器最终特征图大小，对应物理范围 ±40m x [-32,64]m
    # flatten 后得到 H*W = 120 个 BEV token，是 gen 段最大的一类 token
    bev_grid: Tuple[int, int] = (10, 12)

    # ============================================================
    # LEAD 头部数量（CARLA leaderboard 模式）
    # ============================================================
    # route 点数 = 10，对齐 lead.config_training.num_route_points_prediction
    # route 是"空间路径/横向参考线"，不带时间
    num_route_queries: int = 10

    # waypoint 点数 = 8，对齐 lead.config_training.num_way_points_prediction (CARLA)
    # waypoint 是"时间轨迹"，4Hz × 2s = 8 点
    num_waypoint_queries: int = 8

    # waypoint 时间间隔（仅作元信息，模型不显式消费这个值）
    # 0.25s 是 LEAD waypoints_spacing=5 @ 20Hz 的结果
    waypoint_dt: float = 0.25

    # ============================================================
    # gen 路 decoder 结构
    # ============================================================
    # 12 层 = 36 层 Qwen / 3，每层吃一段 select_last K/V
    # 这与 goalgen segment_kv_for_dit 的默认 num_segments=12 天然对齐
    num_layers: int = 12

    # gen 路 attention 头数 = num_kv_heads = 8（MHA 模式，不做 GQA）
    # 严格相等是 validate_qwen_kv_shape 校验的硬约束
    num_heads: int = 8

    # SwiGLU FFN 中间维 = hidden * mlp_ratio
    # SwiGLU 有 gate + up 两个并行投影，按 LLaMA/Qwen 公式 mlp_ratio = 8/3 让
    # 总参数量 ≈ 4 * hidden^2，与传统 GELU MLP 同规模；得到 ffn_hidden_size ≈ 2730
    mlp_ratio: float = 8.0 / 3.0

    # attention/FFN 的 dropout 概率，默认 0.0（先跑通，过拟合再开）
    dropout: float = 0.0

    # ============================================================
    # status token 编码维度（仅作元信息，实际由 projectors.StatusTokenEncoder 消费）
    # ============================================================
    speed_dim: int = 1                    # 速度标量
    target_point_dim: int = 2             # (x, y)

    # ============================================================
    # 派生量与校验
    # ============================================================
    def total_gen_tokens(self) -> int:
        """gen 段总 token 数 = BEV 数 + status 3 + route + waypoint。

        默认值下 = 120 + 3 + 10 + 8 = 141。
        decoder 内部会用这个值断言拼接出的 gen 序列长度正确，避免布局
        错位静默传播到 attention。
        """
        # BEV 占的 token 数 = H * W
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]
        # status 段固定 3 个 token：speed / target_point / target_point_next
        status_tokens = 3
        return bev_tokens + status_tokens + self.num_route_queries + self.num_waypoint_queries

    def slice_layout(self):
        """返回 gen 段每一类 token 在 packed_gen_sequence 中的 [start, end) 索引。

        约定顺序：BEV | speed | tp | ntp | route_q | waypoint_q
        decoder._build_gen_sequence 按这个顺序 torch.cat，
        decoder.forward 末尾按这个 layout 切片出 route_hidden 和 wp_hidden。

        改这里的同时必须同步改 _build_gen_sequence 的 torch.cat 顺序，
        否则切片就会取到错误的 hidden 段。
        """
        bev_tokens = self.bev_grid[0] * self.bev_grid[1]

        # idx 是滑动游标，每个段加进来后就推进
        idx = 0
        layout = {}

        # BEV 段：120 个 token，从 0 开始
        layout["bev"] = (idx, idx + bev_tokens)
        idx += bev_tokens

        # speed 段：单 token
        layout["speed"] = (idx, idx + 1)
        idx += 1

        # target_point 段：单 token
        layout["tp"] = (idx, idx + 1)
        idx += 1

        # target_point_next 段：单 token
        layout["ntp"] = (idx, idx + 1)
        idx += 1

        # route_query 段：10 个 token，这一段的 hidden 会被 RouteHead 消费
        layout["route"] = (idx, idx + self.num_route_queries)
        idx += self.num_route_queries

        # waypoint_query 段：8 个 token，被 WaypointHead 消费
        layout["waypoint"] = (idx, idx + self.num_waypoint_queries)
        # 这里不再 idx += ...，因为已经到末尾
        return layout

    @property
    def ffn_hidden_size(self) -> int:
        """FFN 中间维度，由 hidden_size 和 mlp_ratio 派生。

        定义为 property 而不是普通字段，是为了：
        - 改 mlp_ratio 时 ffn_hidden_size 自动跟着变
        - 外部直接 cfg.ffn_hidden_size 用，没有"是字段还是方法"的歧义
        """
        # int() 截断到整数，避免 SwiGLU 内部 Linear 维度是浮点数
        return int(self.hidden_size * self.mlp_ratio)

    def validate_qwen_kv_shape(self) -> None:
        """提前校验配置是否能让 gen attention 无投影接 Qwen K/V。

        这两条等式是 prefix-KV attention 工作的硬约束：
        1. hidden_size == num_kv_heads * head_dim
           -> 让 gen 自己投影出来的 K/V reshape 成 (B, num_kv_heads, n, head_dim) 后，
              形状与 frozen Qwen K/V 完全相同，可直接 concat
        2. num_heads == num_kv_heads
           -> 不做 GQA（grouped query attention）。如果 num_heads > num_kv_heads，
              gen 段的 K/V 在 attention 时需要按 head 维 repeat，复杂度增加。
              我们选最简方案：MHA 模式，num_heads = num_kv_heads = 8。

        在 decoder.__init__ 调用，配置错误立即 raise，不让错误延迟到运行时 attention。
        """
        if self.hidden_size != self.num_kv_heads * self.head_dim:
            raise ValueError(
                "hidden_size 必须等于 num_kv_heads * head_dim，"
                f"当前 {self.hidden_size} != {self.num_kv_heads} * {self.head_dim}"
            )
        if self.num_heads != self.num_kv_heads:
            raise ValueError(
                f"num_heads 必须等于 num_kv_heads，当前 {self.num_heads} != {self.num_kv_heads}"
            )

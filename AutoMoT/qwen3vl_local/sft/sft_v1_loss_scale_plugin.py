"""SFT v1 专用的 ms-swift 损失权重插件。

ms-swift 3.12.x **不接受**任意正则 JSON 形式的 ``--loss_scale``，只接受已
注册的损失权重策略名。本插件向 swift 注册一个策略，作用是：

**只让两段 ``<event_name>`` 算 loss，把所有结构性字面 token 全部 mask 为 0**：
- ``ANALYSIS: Observations recorded.\\n`` 占位段
- ``STATUS:`` 关键词 + 后面的空格
- ``\\nSUBGOAL:`` 关键词 + 后面的空格
- 末尾换行 / EOS 占位

理由：Qwen3-VL 基模在 system prompt 强约束下已经会按 ANALYSIS / STATUS /
SUBGOAL 三段格式输出（这也是范式 A runner 能跑通的前提），``STATUS:`` /
``SUBGOAL:`` 这些关键词字面**不需要再训**。把它们也放进 loss 后果有两个：

1. ``STATUS:`` / ``SUBGOAL:`` 在 tokenizer 下通常占 3–5 个 token，而真正的
   事件名（如 ``initial``、``hazard_detect``）只占 1–3 个 token。每条样本里
   关键词字面的 loss token 数 **≫** 事件名 token 数，监督信号被字面 token 主导。
2. 训练过头后，"输出 ``STATUS:`` 这个串"成为最廉价的降 loss 路径，模型会陷入
   ``STATUS: X\\nSTATUS: X`` 循环复读（即 PLAN §11 风险表里那条 ckpt-8100
   失败模式）。

v1 的真正目标是"看到当前帧 RGB → 输出哪个事件名"，所以只保留两段事件名 token
为 loss，其它字面全部 0。

在 AutoMoT/ 目录下的典型用法：

    swift sft ... \\
        --external_plugins qwen3vl_local/sft/sft_v1_loss_scale_plugin.py \\
        --loss_scale sft_v1_analysis_mask
"""

from __future__ import annotations

import re
import sys
from typing import List, Optional, Tuple

from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map


def _disable_swift_matplotlib_image_export() -> None:
    """保留 TensorBoard events，但跳过 ms-swift 结束阶段的 matplotlib 导图。"""

    swift_sft = sys.modules.get("swift.llm.train.sft")
    if swift_sft is None:
        return

    def _noop_swift_plot_images(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        print("[sft_v1_plugin] skip ms-swift matplotlib image export; TensorBoard events are kept.")

    swift_sft.plot_images = _noop_swift_plot_images


_disable_swift_matplotlib_image_export()


# ---------------------------------------------------------------------------
# regex：完整三段匹配 + 仅 ANALYSIS 段 fallback
# ---------------------------------------------------------------------------
#
# 主 regex 一次性匹配完整三段，仅捕获两个事件名。其它字面（ANALYSIS 占位、
# STATUS:、\nSUBGOAL:）靠 match.span("status") / match.span("subgoal") 在
# 主流程里反算 char span 后整段 mask。
#
# 各小段：
#   ANALYSIS:.*?\nSTATUS:[ \t]*    —— 吞掉 "ANALYSIS: ... \nSTATUS: " 整段
#   (?P<status>\S[^\n]*?)          —— 捕获 STATUS 后面的事件名（同一行、非空开头、非贪婪）
#   \s*\nSUBGOAL:[ \t]*            —— 吞掉中间换行 + "SUBGOAL: "
#   (?P<subgoal>\S[^\n]*)          —— 捕获 SUBGOAL 后面的事件名（同一行、非空开头）
#
# 用 `\S[^\n]*` 而不是 `\w+`：兼容多 token 事件名 / 内部空格变体；同时保证
# 不跨行，避免吞到下一段 `<|im_end|>` 等 special token。
_FULL_PATTERN = re.compile(
    r"ANALYSIS:.*?\nSTATUS:[ \t]*"
    r"(?P<status>\S[^\n]*?)"
    r"\s*\nSUBGOAL:[ \t]*"
    r"(?P<subgoal>\S[^\n]*)",
    flags=re.DOTALL,
)

# fallback：当 swift 把 context 按 round / sentence 切碎，传入的片段可能
# 不包含完整三段（例如只看到 ANALYSIS 半截）。这种情况下退回到旧版"仅 mask
# ANALYSIS 占位段"的行为，至少不让占位句污染梯度。STATUS:/SUBGOAL: 字面在
# fallback 路径下仍会算 loss，是已知次优；正常训练 path 一定走 _FULL_PATTERN。
_ANALYSIS_ONLY_REGEX = re.compile(r"ANALYSIS:.*?(?=\nSTATUS:)", flags=re.DOTALL)


class SftV1AnalysisMaskLossScale(LossScale):
    """只让 ``STATUS`` / ``SUBGOAL`` 的事件名算 loss，其它字面全部 mask。"""

    def get_loss_scale(
        self,
        context: str,
        *,
        query: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        if not isinstance(context, str):
            return super().get_loss_scale(context, query=query)

        match = _FULL_PATTERN.search(context)
        if match is not None:
            return self._split_full(context, match)

        # 走 fallback：context 不含完整三段。
        return self._split_analysis_only(context)

    @staticmethod
    def _split_full(
        context: str,
        match: "re.Match[str]",
    ) -> Tuple[List[str], List[float]]:
        """把 context 切成 5 段：[prefix, status, mid, subgoal, tail]。

        权重对应 [0, 1, 0, 1, 0]：
        - prefix：从 context 起点到 STATUS 事件名前（含 ANALYSIS 占位 + STATUS: 字面）
        - status：STATUS 行的 event_name
        - mid：status 末尾到 subgoal 事件名前（含换行 + SUBGOAL: 字面）
        - subgoal：SUBGOAL 行的 event_name
        - tail：subgoal 后面剩余的字符（可能是 \\n、空白、special token 占位）

        切法保证 ``"".join(parts) == context``，与 ms-swift 内部对齐要求一致。
        """
        status_start, status_end = match.span("status")
        subgoal_start, subgoal_end = match.span("subgoal")

        parts: List[str] = []
        scales: List[float] = []

        prefix = context[:status_start]
        if prefix:
            parts.append(prefix)
            scales.append(0.0)

        parts.append(context[status_start:status_end])
        scales.append(1.0)

        mid = context[status_end:subgoal_start]
        if mid:
            parts.append(mid)
            scales.append(0.0)

        parts.append(context[subgoal_start:subgoal_end])
        scales.append(1.0)

        tail = context[subgoal_end:]
        if tail:
            parts.append(tail)
            scales.append(0.0)

        return parts, scales

    @staticmethod
    def _split_analysis_only(
        context: str,
    ) -> Tuple[List[str], List[float]]:
        """fallback：只能识别出 ANALYSIS 占位段时退回的最小保护切法。"""
        match = _ANALYSIS_ONLY_REGEX.search(context)
        if match is None:
            return [context], [1.0]

        parts: List[str] = []
        scales: List[float] = []
        if match.start() > 0:
            parts.append(context[:match.start()])
            scales.append(1.0)
        parts.append(context[match.start():match.end()])
        scales.append(0.0)
        if match.end() < len(context):
            parts.append(context[match.end():])
            scales.append(1.0)
        return parts, scales


# 这里注册的名字与 qwen3vl_local/sft/sft_v1_train.sh 里 --loss_scale sft_v1_analysis_mask 对应。
# 策略名沿用旧名，避免 sft_v1_train.sh / SFT_RUN.md 里命令行不变。
loss_scale_map["sft_v1_analysis_mask"] = SftV1AnalysisMaskLossScale

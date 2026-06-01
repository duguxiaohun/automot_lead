"""SFT v2 专用的 ms-swift 损失权重插件。

与 ``sft_v1_loss_scale_plugin.py`` 的核心区别：v1 把 ``ANALYSIS`` 段权重置 0
（因为 v1 ANALYSIS 是 ``Observations recorded.`` 占位、没监督价值）；v2 ANALYSIS
段是冻结 base Qwen 蒸馏出来的真值，需要参与 loss 约束 LoRA 不漂移。

权重表（与 SFT_V2_PLAN.md §5 完全一致）：

==================  ======  =================================================
段                  权重    理由
==================  ======  =================================================
``ANALYSIS:`` 字面  0       关键词无学习价值
ANALYSIS body       0.3     蒸馏目标 — 0.3 让 student 学到形状但不被 teacher
                            的措辞随机性主导（teacher 是采样输出非真值）
``\\nSTATUS:`` 字面 0       关键词无学习价值
STATUS event_name   1.0     核心监督
``\\nSUBGOAL:`` 字面 0      关键词无学习价值
SUBGOAL event_name  1.0     核心监督
末尾换行 / EOS      0       占位
==================  ======  =================================================

ANALYSIS 权重可通过环境变量 ``SFT_V2_ANALYSIS_WEIGHT`` 在启动训练前 override
（用例：实测 ANALYSIS 还在漂移 → 0.5；过拟合 teacher 措辞 → 0.1）。

策略名 ``sft_v2_analysis_supervised`` 与 ``tools/sft_v2_train.sh`` 里
``--loss_scale`` 对应。

注册路径：``swift sft --external_plugins tools/sft_v2_loss_scale_plugin.py``。
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map


# ---------------------------------------------------------------------------
# 权重配置（启动时一次性读环境变量）
# ---------------------------------------------------------------------------

def _parse_weight(env_name: str, default: float) -> float:
    """从环境变量读 float 权重，非法值兜底为 default。"""
    raw = os.environ.get(env_name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[sft_v2_plugin] ignore non-float {env_name}={raw!r}, fallback to {default}")
        return default


ANALYSIS_WEIGHT = _parse_weight("SFT_V2_ANALYSIS_WEIGHT", 0.3)
STATUS_WEIGHT = _parse_weight("SFT_V2_STATUS_WEIGHT", 1.0)
SUBGOAL_WEIGHT = _parse_weight("SFT_V2_SUBGOAL_WEIGHT", 1.0)


# ---------------------------------------------------------------------------
# regex：完整三段匹配，分别捕获 ANALYSIS body / STATUS event_name / SUBGOAL event_name
# ---------------------------------------------------------------------------
#
# 与 v1 plugin 的关键差别：
# - v1 主 regex 用 ``ANALYSIS:.*?\nSTATUS:`` 跨行吞掉 ANALYSIS 占位段；
# - v2 把 ANALYSIS body 作为命名组 ``analysis`` 单独捕获，限制单行（teacher 后
#   处理已强制单行，没有跨行情况），从而能精确切出 ANALYSIS body 段单独给权重 0.3。
#
# 各段含义：
#   ANALYSIS:[ \t]*               —— "ANALYSIS:" 字面 + 后空格（mask）
#   (?P<analysis>[^\n]*?)         —— ANALYSIS body 正文（单行、非贪婪）
#   \s*\nSTATUS:[ \t]*            —— 换行 + "STATUS: " 字面（mask）
#   (?P<status>\S[^\n]*?)         —— STATUS event_name
#   \s*\nSUBGOAL:[ \t]*           —— 换行 + "SUBGOAL: " 字面（mask）
#   (?P<subgoal>\S[^\n]*)         —— SUBGOAL event_name
#
# 与 v1 完全一致地避免吞到下一段 special token：用 ``[^\n]*`` 限制非换行。
_FULL_PATTERN = re.compile(
    r"ANALYSIS:[ \t]*"
    r"(?P<analysis>[^\n]*?)"
    r"\s*\nSTATUS:[ \t]*"
    r"(?P<status>\S[^\n]*?)"
    r"\s*\nSUBGOAL:[ \t]*"
    r"(?P<subgoal>\S[^\n]*)",
    flags=re.DOTALL,
)

# fallback：context 不含完整三段时（swift 把 context 按 round/sentence 切碎，
# 单 chunk 只看到 ANALYSIS 半截）。退化为"仅 mask ANALYSIS 占位段"的旧行为，
# 至少不让占位句污染梯度，但 STATUS:/SUBGOAL: 字面会被算 loss（次优）。
# 与 v1 plugin 的 fallback 完全一致。
_ANALYSIS_ONLY_REGEX = re.compile(r"ANALYSIS:.*?(?=\nSTATUS:)", flags=re.DOTALL)


class SftV2AnalysisSupervisedLossScale(LossScale):
    """v2 mask 策略：ANALYSIS body 给 0.3 权重，STATUS/SUBGOAL event_name 给 1.0，
    其它字面 / 空白 / EOS 全部 mask 为 0。详见模块 docstring。
    """

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

        # swift 把 context 切碎时走 fallback：只 mask ANALYSIS 占位段。
        return self._split_analysis_only(context)

    @staticmethod
    def _split_full(
        context: str,
        match: "re.Match[str]",
    ) -> Tuple[List[str], List[float]]:
        """把 context 切成最多 7 段：

        [prefix("ANALYSIS: "), analysis, mid1("\\nSTATUS: "), status,
         mid2("\\nSUBGOAL: "), subgoal, tail]

        权重对应 [0, 0.3, 0, 1, 0, 1, 0]。切法保证 ``"".join(parts) == context``
        与 ms-swift 内部对齐要求一致。
        """

        a_start, a_end = match.span("analysis")
        s_start, s_end = match.span("status")
        g_start, g_end = match.span("subgoal")

        parts: List[str] = []
        scales: List[float] = []

        # prefix: 0 .. a_start = "ANALYSIS: " （含末尾空格）
        prefix = context[:a_start]
        if prefix:
            parts.append(prefix)
            scales.append(0.0)

        # ANALYSIS body — v2 与 v1 的关键差别就在这里。
        # 注意：即使 a_start == a_end（ANALYSIS body 为空、teacher fallback 失败）
        # 也要 append 一个空串占位，否则 "".join(parts) != context 会触发 swift assertion。
        parts.append(context[a_start:a_end])
        scales.append(ANALYSIS_WEIGHT)

        # mid1: a_end .. s_start = "\nSTATUS: "
        mid1 = context[a_end:s_start]
        if mid1:
            parts.append(mid1)
            scales.append(0.0)

        # STATUS event_name
        parts.append(context[s_start:s_end])
        scales.append(STATUS_WEIGHT)

        # mid2: s_end .. g_start = "\nSUBGOAL: "
        mid2 = context[s_end:g_start]
        if mid2:
            parts.append(mid2)
            scales.append(0.0)

        # SUBGOAL event_name
        parts.append(context[g_start:g_end])
        scales.append(SUBGOAL_WEIGHT)

        # tail: g_end .. len(context) = 末尾换行 / EOS 占位 / special token
        tail = context[g_end:]
        if tail:
            parts.append(tail)
            scales.append(0.0)

        return parts, scales

    @staticmethod
    def _split_analysis_only(
        context: str,
    ) -> Tuple[List[str], List[float]]:
        """fallback：仅 ANALYSIS 段被识别出时退回的最小保护切法。

        与 v1 plugin 的同名方法完全一致：ANALYSIS 段 mask=0，其余全 1.0。这种 fallback
        下 STATUS:/SUBGOAL: 字面也会进 loss，属于已知次优；正常训练 path 必须走
        ``_FULL_PATTERN``。
        """

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


# 注册到 swift。策略名与 tools/sft_v2_train.sh 里 --loss_scale 对应。
loss_scale_map["sft_v2_analysis_supervised"] = SftV2AnalysisSupervisedLossScale

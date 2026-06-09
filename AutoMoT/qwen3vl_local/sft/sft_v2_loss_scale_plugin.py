"""SFT v2 专用的 ms-swift 损失权重插件。

与 ``sft_v1_loss_scale_plugin.py`` 的核心区别：v1 把 ``ANALYSIS`` 段权重置 0
（因为 v1 ANALYSIS 是 ``Observations recorded.`` 占位、没监督价值）；v2 ANALYSIS
段是冻结 base Qwen 蒸馏出来的真值，需要参与 loss 约束 LoRA 不漂移。

权重表（与 SFT_PLAN.md §5 完全一致）：

==================  ======  =================================================
段                  权重    理由
==================  ======  =================================================
``ANALYSIS:`` 字面  1.0     输出格式起手信号；不能只靠 base/chat template 先验
ANALYSIS body       0.3     蒸馏目标 — 0.3 让 student 学到形状但不被 teacher
                            的措辞随机性主导（teacher 是采样输出非真值）
``\\nSTATUS:`` 字面 **1.0** 段切换信号 — v2 实测必须 ≥1.0；mask=0 会让模型在
                            ANALYSIS body 末尾无梯度学切段，自由生成时陷入
                            "ANALYSIS×N 循环复读"（详见 §"段切换不能 mask"）
STATUS event_name   1.0     核心监督
``\\nSUBGOAL:`` 字面 **1.0** 同 ``\\nSTATUS:`` — 段切换信号必须有监督
SUBGOAL event_name  1.0     核心监督
末尾 tail / EOS     **1.0** 若由 ms-swift 模板放进 plugin context，就属于停止
                            信号；v2 ANALYSIS 长后不能 mask
==================  ======  =================================================

ANALYSIS 权重可通过环境变量 ``SFT_V2_ANALYSIS_WEIGHT`` 在启动训练前 override
（用例：实测 ANALYSIS 还在漂移 → 0.5；过拟合 teacher 措辞 → 0.1）。

**结构字面不能 mask（v2 致命踩坑，2026-06-02）**：
v2.0 (commit ef0eb19 之前) 把 ``ANALYSIS:`` / ``\\nSTATUS:`` / ``\\nSUBGOAL:``
/ 可能进入 context 的 tail/EOS 全部 mask 成 0，理由是"关键词字面无学习价值"——
这条推断在 v1 ANALYSIS 是固定占位
（7 token）时确实没事，但 v2 ANALYSIS 升到 80-150 token 自由文本后，模型在
ANALYSIS body 末尾 token 的 next-token-prediction 没有任何梯度推它去 emit
``\\n``/``STATUS``，自由生成时倾向于继续写 ANALYSIS body 风格的 token，
陷入"ANALYSIS×N 循环"直到 max_gen_tokens 耗尽。修法：所有结构性字面
weight 必须 ≥ 1.0，让模型学到"必须从 ANALYSIS 起手，并在 ANALYSIS body
结束后 emit 段切换 token"。

策略名 ``sft_v2_analysis_supervised`` 与 ``qwen3vl_local/sft/sft_v2_train.sh`` 里
``--loss_scale`` 对应。

注册路径：``swift sft --external_plugins qwen3vl_local/sft/sft_v2_loss_scale_plugin.py``。
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
#   ANALYSIS:[ \t]*               —— "ANALYSIS:" 字面 + 后空格（起手结构信号）
#   (?P<analysis>[^\n]*?)         —— ANALYSIS body 正文（单行、非贪婪）
#   \s*\nSTATUS:[ \t]*            —— 换行 + "STATUS: " 字面（段切换信号）
#   (?P<status>\S[^\n]*?)         —— STATUS event_name
#   \s*\nSUBGOAL:[ \t]*           —— 换行 + "SUBGOAL: " 字面（段切换信号）
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
# 单 chunk 只看到 ANALYSIS 半截）。v2 不再 mask 起手结构字面；能切出 ANALYSIS
# body 时仍给 ANALYSIS_WEIGHT，其余结构文本按 1.0 监督。
_ANALYSIS_ONLY_REGEX = re.compile(
    r"ANALYSIS:[ \t]*(?P<analysis>[^\n]*?)(?=\nSTATUS:|\n?$)",
    flags=re.DOTALL,
)


class SftV2AnalysisSupervisedLossScale(LossScale):
    """v2 mask 策略：ANALYSIS body 给 0.3 权重，STATUS/SUBGOAL event_name 给 1.0，
    起手/段切换结构字面与 tail/EOS 也给 1.0。
    详见模块 docstring。
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

        # swift 把 context 切碎时走 fallback：保留结构字面监督，ANALYSIS body 仍低权重。
        return self._split_analysis_only(context)

    @staticmethod
    def _split_full(
        context: str,
        match: "re.Match[str]",
    ) -> Tuple[List[str], List[float]]:
        """把 context 切成最多 7 段：

        [prefix("ANALYSIS: "), analysis, mid1("\\nSTATUS: "), status,
         mid2("\\nSUBGOAL: "), subgoal, tail]

        权重对应 [1, 0.3, 1, 1, 1, 1, 1]。切法保证 ``"".join(parts) == context``
        与 ms-swift 内部对齐要求一致。
        """

        a_start, a_end = match.span("analysis")
        s_start, s_end = match.span("status")
        g_start, g_end = match.span("subgoal")

        parts: List[str] = []
        scales: List[float] = []

        # prefix: 0 .. a_start = "ANALYSIS: " （含末尾空格）。
        # 这是输出结构的起手信号，也必须学习；不能只靠 base/chat template 先验。
        prefix = context[:a_start]
        if prefix:
            parts.append(prefix)
            scales.append(1.0)

        # ANALYSIS body — v2 与 v1 的关键差别就在这里。
        # 注意：即使 a_start == a_end（ANALYSIS body 为空、teacher fallback 失败）
        # 也要 append 一个空串占位，否则 "".join(parts) != context 会触发 swift assertion。
        parts.append(context[a_start:a_end])
        scales.append(ANALYSIS_WEIGHT)

        # mid1: a_end .. s_start = "\nSTATUS: "
        # 注意 weight=1.0 而不是 0：段切换字面必须被监督，否则模型在 ANALYSIS body
        # 末尾无梯度学切段，自由生成时陷入 "ANALYSIS×N 循环"。详见模块 docstring。
        mid1 = context[a_end:s_start]
        if mid1:
            parts.append(mid1)
            scales.append(1.0)

        # STATUS event_name
        parts.append(context[s_start:s_end])
        scales.append(STATUS_WEIGHT)

        # mid2: s_end .. g_start = "\nSUBGOAL: "
        # 同 mid1：段切换字面必须 weight ≥ 1.0。
        mid2 = context[s_end:g_start]
        if mid2:
            parts.append(mid2)
            scales.append(1.0)

        # SUBGOAL event_name
        parts.append(context[g_start:g_end])
        scales.append(SUBGOAL_WEIGHT)

        # tail: g_end .. len(context) = 末尾换行 / EOS 占位 / special token。
        # 若 ms-swift runtime 把 tail/EOS 放进 plugin context，它就是停止信号，
        # 必须保留梯度；静态 jsonl 通常不含这个 tail。
        tail = context[g_end:]
        if tail:
            parts.append(tail)
            scales.append(1.0)

        return parts, scales

    @staticmethod
    def _split_analysis_only(
        context: str,
    ) -> Tuple[List[str], List[float]]:
        """fallback：仅 ANALYSIS 段被识别出时退回的最小保护切法。

        起手 ``ANALYSIS:`` 与其余结构文本全 1.0；ANALYSIS body 仍给低权重。
        正常训练 path 必须走 ``_FULL_PATTERN``，这个 fallback 只是防止切碎 context
        时把 ANALYSIS 相关监督全部丢掉。
        """

        match = _ANALYSIS_ONLY_REGEX.search(context)
        if match is None:
            return [context], [1.0]

        a_start, a_end = match.span("analysis")
        parts: List[str] = []
        scales: List[float] = []
        if a_start > 0:
            parts.append(context[:a_start])
            scales.append(1.0)
        parts.append(context[a_start:a_end])
        scales.append(ANALYSIS_WEIGHT)
        if a_end < len(context):
            parts.append(context[a_end:])
            scales.append(1.0)
        return parts, scales


# 注册到 swift。策略名与 qwen3vl_local/sft/sft_v2_train.sh 里 --loss_scale 对应。
loss_scale_map["sft_v2_analysis_supervised"] = SftV2AnalysisSupervisedLossScale

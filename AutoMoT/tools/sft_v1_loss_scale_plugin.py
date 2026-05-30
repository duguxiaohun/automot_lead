"""SFT v1 专用的 ms-swift 损失权重插件。

ms-swift 3.12.x **不接受** 任意正则 JSON 形式的 ``--loss_scale``，只接受已
注册的损失权重策略名。本插件向 swift 注册一个策略，作用是：把每条 assistant
回复里 ANALYSIS 占位段的损失权重置 0，而 STATUS/SUBGOAL 段保持正常训练。

在 AutoMoT/ 目录下的典型用法：

    swift sft ... \
        --external_plugins tools/sft_v1_loss_scale_plugin.py \
        --loss_scale sft_v1_analysis_mask
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map


_ANALYSIS_REGEX = re.compile(r"ANALYSIS:.*?(?=\nSTATUS:)", flags=re.DOTALL)


class SftV1AnalysisMaskLossScale(LossScale):
    """把 ANALYSIS 占位段的损失屏蔽掉，STATUS/SUBGOAL 段正常计算损失。"""

    def get_loss_scale(
        self,
        context: str,
        *,
        query: Optional[str] = None,
    ) -> Tuple[List[str], List[float]]:
        if not isinstance(context, str):
            return super().get_loss_scale(context, query=query)

        match = _ANALYSIS_REGEX.search(context)
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


# 这里注册的名字与 tools/sft_v1_train.sh 里 --loss_scale sft_v1_analysis_mask 对应
loss_scale_map["sft_v1_analysis_mask"] = SftV1AnalysisMaskLossScale

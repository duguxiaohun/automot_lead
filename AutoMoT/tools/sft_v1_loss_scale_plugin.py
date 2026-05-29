"""ms-swift loss_scale plugin for SFT v1.

ms-swift 3.12.x does not accept arbitrary regex JSON via ``--loss_scale``.
It expects a registered loss_scale strategy name.  This plugin registers one
strategy that masks only the placeholder ANALYSIS span in each assistant
response while keeping STATUS/SUBGOAL trainable.

Usage from AutoMoT/:

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
    """Mask placeholder ANALYSIS text and train STATUS/SUBGOAL normally."""

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


# Name used by tools/sft_v1_train.sh: --loss_scale sft_v1_analysis_mask
loss_scale_map["sft_v1_analysis_mask"] = SftV1AnalysisMaskLossScale

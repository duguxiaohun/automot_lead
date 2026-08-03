"""SFT baseline: 高速/非高速 + RE/UE 单问直接监督训练包。

它从 sft_baseline 降维而来：仍复用 collection_output 的 RS/EVENT 标注解析，
但学生只回答当前帧是否属于高速场景，以及当前是否为 RE/UE。
"""

DATASET_VERSION = "sft_baseline_highway_reue_joint_v1"



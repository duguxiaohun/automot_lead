"""SFT base: RS / EVENT 两问直接选项监督训练包。

它复用 sft_v5 的数据、候选池和 memory 状态机，但训练目标是普通
teacher-forced CE：没有 OPSD、没有 CoT、没有 privileged teacher。
"""

DATASET_VERSION = "sft_base_rs_event_direct"

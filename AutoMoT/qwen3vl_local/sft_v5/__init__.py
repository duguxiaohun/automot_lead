"""SFT v5: RS / EVENT 两问串行 OPSD 训练包。

只在这里暴露数据集版本常量，避免 build/train/eval/probe 各自硬编码版本号。
版本号参与 Q2 选项随机 seed；协议变化时应同步 bump，防止旧缓存和新候选表混用。
"""

DATASET_VERSION = "sft_v5_rs_event_sequence"

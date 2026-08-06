"""SFT base simple: 高速/非高速 + RE/UE 四格随机均衡训练包。

它从 sft_baseline 继续简化而来：撤掉显式 transition 采样，训练/测试都按
当前帧 GT 的 HIGHWAY:UE、HIGHWAY:RE、NON_HIGHWAY:UE、NON_HIGHWAY:RE 四格随机均衡。
"""

DATASET_VERSION = "sft_base_simple_highway_reue_fourbin_v1"




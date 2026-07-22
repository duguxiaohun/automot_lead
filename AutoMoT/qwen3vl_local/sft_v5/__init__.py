"""SFT v5：RS_SLOW / EVENT_FAST 双频串行 OPSD 训练包。

Q1 低频判断道路结构 RS；稳定快帧不生成 Q1，而是复用上一帧 RS hypothesis。
Q2 在 RS gate 正确的每个真实帧重新读取当前 RGB，并从显式标注 REGULAR/UNUSUAL
的混合候选中选择 EVENT。两个 hypothesis 都是不可信先验：训练会在线构造
aligned、UNKNOWN/no-prior omission 与 contradiction/stale 三类关系，并分别携带
从该 label 最近一次变化开始累计的 4Hz frame age。稳定 RS 的下一次慢问默认从
3/4/5 帧中可复现抽取；错误、UNKNOWN 或 recovery 状态仍逐帧慢问。

本模块只暴露数据集版本常量，避免 build/train/eval/probe 各自硬编码版本号。
版本号参与 Q2 选项随机 seed；候选协议或 index schema 发生不兼容变化时应同步
bump，防止旧缓存和新候选表混用。训练行为入口见 ``SFT_V5_RUN.md``，完整设计
约束见 ``SFT_V5_PLAN.md``。
"""

DATASET_VERSION = "sft_v5_rs_event_sequence"

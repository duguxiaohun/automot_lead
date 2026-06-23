"""SFT v4：子场景时间序列训练包。

v4 把训练单元从单帧升级为一个 sub-scenario 的完整时间序列，引入：
  - 学生自维护的文本 Memory（BELIEVED_SCENE / STATUS / SUBGOAL + EGO_TO_GOAL_XY）
  - Hindsight Oracle / OPD 蒸馏（Frozen Qwen teacher 以学生口吻纠错，学生 token-CE 对齐）
  - Phase A（memory 自更新）+ Phase B（GT scene 强制注入）双阶段训练窗口

v4 不替代 v2，v2 仍保留作为单帧串行选择题基线。
"""


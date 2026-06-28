"""SFT v3：子场景时间序列训练包。

v3 把训练单元从单帧升级为一个 sub-scenario 的完整时间序列，引入：
  - 学生自维护的文本 Memory（BELIEVED_SCENE / STATUS / SUBGOAL + EGO_TO_GOAL_XY）
  - offline on-policy OPSD 蒸馏（学生自由 rollout 更新 memory，disable_adapter teacher
    对同一批 student step token 给 full-vocabulary forward-KL 分布监督）
  - Phase A（memory 自更新）+ Phase B（GT scene 强制注入）双阶段训练窗口

v3 不维护独立 prompt；`sft_v3.prompts` 只 re-export v4 prompt / Memory / 状态机。
"""

"""无 memory 的 Qwen agent-loop 第一轮四问。

本包刻意不复用 ``sft_baseline`` 的二分类 checkpoint：第一轮问题的监督对象是
当前可见交通事实，而不是 RS/EVENT 的折叠标签。后续 loop 可以把这里的四个离散
观察结果作为下一轮问题的输入。
"""

DATASET_NAME = "sft_loop_phase1_static_obstacle_visible_facts"

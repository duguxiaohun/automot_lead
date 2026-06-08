"""LeadMoT closed-loop CARLA evaluation harness.

子包概览（详见 EVAL_CARLA_PLAN.md / EVAL_CARLA_RUN.md）：

- agent.py            : leaderboard 实时 agent（LEAD 3 摄像头；use_bev=True 时启用双 LiDAR）
- video_recorder.py   : input / debug / demo / grid 四路视频写入 + ffmpeg 压缩
- visualizer.py       : ego-frame 点投影到相机 / BEV 顶视
- scenario_picker.py  : 220 routes 与 LEAD scenario 反向映射 + 子集选择
- aggregate.py        : eval json 按 scenario 聚合
- run_eval.sh         : 一键启动 220 routes 评测，支持自动空闲 GPU + 单卡/多卡 worker
- webapp/             : Flask 浏览器查看（路线列表 + 视频 + leaderboard 指标）

维护约定：
- 新增在线输入、坐标变换、输出目录字段时，优先在对应函数旁补中文注释；
- 同步更新 EVAL_CARLA_PLAN.md / EVAL_CARLA_RUN.md，避免代码和操作说明分叉。
"""

__all__ = []

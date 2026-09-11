# Action prior 实现与合同

操作入口见 [run.md](run.md)，可选复核和评测解释见 [AUDIT.md](AUDIT.md)。

## 模型

四张 4 Hz stitched RGB、当前速度与导航进入冻结 Qwen。默认用 Phase1/2 LoRA 提供先验，再禁用所有 LoRA 由 base 生成短摘要。
`dataset-priors` 直接读取标定标签，默认关闭摘要模型复核；冷启动为一次 base 生成和一次完整 KV prefill。
最终 KV 保留四图、自然 RS/EVENT 描述、导航和摘要，不注入类别 JSON、NO/UNKNOWN 或逐帧动作标签。

冻结 LEAD BEV，与 KV 共同条件化联合 Flow Matching 轨迹 decoder：route `(10,2)`、waypoint `(8,2)`。
训练默认仅向量场 MSE；验证从高斯噪声默认 10 步 Euler 采样，按样本身份固定噪声。
可训练参数、AdamW、EMA 为 FP32，decoder 可用 BF16 autocast。best 按采样 ADE 选取。
旧 Linear+cumsum 和逐点 FM checkpoint 不兼容当前 `action_prior_checkpoint_v4`。

主线与两个 action-expert 消融共用 `training_core.py`；统一 DDP、梯度累积、EMA、验证、checkpoint 与恢复。
中断只在 optimizer 安全点保存，SIGKILL/掉电仍依赖周期 checkpoint。源码身份变化需新 run。

## 先验边界

- RS_HIGHWAY 是独立事实，R3 不能推出高速；缺失标签不当 NO。
- Phase1 全问加 RS 分层复核；Phase2 使用已训练的双域问法，不伪造 EVENT hierarchical 接口。
- 复核一致只表示接受条件，摘要模型复核也不保证语义正确；失败可 fallback。
- `event-balanced-scene-priors` 仅适用于无噪声 dataset 条件，是额外的离线实验。
- RE2 当前可描述导航变道或早先障碍记录；没有可靠证据时不声称“UE2 刚结束且恢复待完成”。

## 均衡采样

全帧 map 将 `special_eligible`、`special_filtered`、`confirmed_regular`、`unconfirmed` 分开。
Phase3 candidate 只标记特殊帧 eligibility，candidate 缺席不等于普通背景；只有确认常规进入两份背景池。
UE1–7、RE2/3/5 十桶各一份，普通背景两份；开发路线强制 train-only，val/test 保留自然分布。

相同事件归属的帧压缩为小型最小费用流网络。首次使用费用 0，额外使用费用 1，固定总配额下精确最大化全局唯一帧覆盖。
共享帧可跨桶重分配，展开时跨桶共用游标，组内路线轮转；路线轮转是顺序偏好，不是硬性路线配额。
全局 presentation 构建完成后再按 DDP rank 分片。默认每帧单轮上限 8，预算须满足 `lcm(12, world_size)`。

## 文件与恢复

自动准备只构建缺失的中间产物，使用现有原始数据、人工标注和本地权重，不下载或伪造标签。
候选/full map 使用锁和临时目录原子发布；全帧 map 绑定候选内容、规则版本和 action 三 split 的 SHA256。
v1 map 必须重建。候选规则或构建代码变化产生新的自动缓存目录，旧产物保留。

模型/adapter 字节指纹、prompt、导航、RGB 和执行源码共同约束 checkpoint 与文本缓存。
LoRA 选择会固定来源并复制到 run/lora；摘要缓存可共享，但最终 KV 每次完整 prefill。
审计来源身份与实际生成身份分开，来源不可访问记 unknown；同内容路径迁移不应改变生成条件。
详细历史事实以仓库根目录 PROJECT_CONTEXT.md 为准。

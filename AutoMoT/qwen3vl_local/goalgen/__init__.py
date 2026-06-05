"""goalgen：基于 Qwen KV cache 的子目标 latent 生成路线。

整体路线见 PROJECT_CONTEXT.md §15。本子包提供：

- vae：冻结的 VAE 编码/解码封装（依赖 AutoMoT/vae_standalone 的源码与权重）。
- prompt：teacher-forced 的 system / user prompt（告诉 Qwen 当前 STATUS 和 SUBGOAL）。
- qwen_kv：调用本地 Qwen3-VL-Instruct 跑 prefill，并把 36 层 KV 分段给 DiT。
- keyframes：从 keyframes_all_scenarios.json 查子目标对应关键帧。
- dit：MoT 风格 joint-attention 的 12 层 DiT，用 Qwen KV 作为冻结语言上下文。
- flow：flow matching 训练目标与 Euler 推理积分。
- build_dataset：从 keyframes 时间链条生成 GoalGen jsonl 数据集。
- train / train.sh：冻结 Qwen/VAE、只训练 DiT-MoT 的单卡/DDP 入口。

只在 runner 里组合，不在 __init__ 引入重型依赖（torch、PIL、cv2 等），避免本地静态检查时
直接触发导入失败。
"""

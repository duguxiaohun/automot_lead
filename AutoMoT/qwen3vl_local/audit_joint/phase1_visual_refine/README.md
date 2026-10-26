# Phase1 RGB 标定与提示词候选（2026-10-26）

2026-10-27 发布入口：在当前服务器 AutoMoT 根激活原环境后，执行 `bash qwen3vl_local/audit_joint/phase1_visual_refine/run.sh all`。默认按用户已完成的 `judgment_release_20261025_2rgb_endpoints_4rgb` pipeline，分别复用两图/四图 `eval_review/lora_production/evaluation_inputs.json` 及其中记载的原 adapter。支持 `--pipeline-root` 指定当前机器的另一份完整运行，不猜 latest，不跨机拷贝权重。缺任何所选模式的原产物，加载模型前报错。

该命令只做同权重旧/新提示词配对验证，不训练、不重建数据、不应用测试标签修正。默认自动选一张显存占用较低的 GPU；显式指定示例：`GPU_IDS=0 bash qwen3vl_local/audit_joint/phase1_visual_refine/run.sh all`。`2rgb_endpoints`、`4rgb` 也可以独立运行。继承激活环境（可设置 `PYTHON`），资产软链接兼容；在前台输出进度，禁止 core dump。

结果保存到 `checkpoints/phase1_visual_refine_remote/run_<时间>_<PID>/`。每个模式完成后自动生成 `<mode>_audit_bundle.tar.gz`（30 MiB 上限、完整raw输出不截断）。发生指标退化时先报 `[REVIEW_REQUIRED]`、保存包并继续另一模式，最后退出2；推理/资源/完整性异常直接失败。只有全部完成且未观察到退化才打印 `[DONE]`。两机布局仅本地模拟验证，尚未真实远程/GPU验收。不要在推理期间同步或更改其源码。

这是显式启用的候选实验，默认训练入口仍是原版本。Phase2 不受影响；两阶段继续独立。没有新模型推理结果，不能把 CPU 检查、重标测试标签或新提示词文字当作性能提升。

标定只作用于逐帧审阅、绑定四帧有序 RGB SHA256、旧 RS／完整答案／split 的明确记录。`retain`、`uncertain` 不改标签；不按场景或整条路线传播。原数据与旧回执不覆盖。`derive` 输出独立父库派生版本；非修改行保留原字节。默认采样方法、配额、划分、重复上限均不变。标签变化可能自然改变候选桶成员，不能称逐条训练样本不变。

本机证据位于 `checkpoints/phase1_visual_review_20261026`，派生数据与检查位于 `checkpoints/phase1_visual_refine_v1_20261026`。完整远端审阅是 `reviews.json`；本机适用的是 `reviews_local.json`，显式记录一条缺失成员及两条本机已为 R4 的差异。不要将本机回执当作另一台服务器的准入；两机标签与成员已实证不同，任何不匹配都应直接报错。

从当前 AutoMoT 根、已激活 Python 环境使用；不要求另一台服务器绝对目录。`lead_data`、`checkpoints` 资产目录可为软链接，源码不复制或热更新。

```bash
python -B -m qwen3vl_local.audit_joint.phase1_visual_refine derive --parent-index <父库/frame_index.jsonl> --reviews <适用的逐帧审阅.json> --data-root lead_data --output <新的候选数据目录>
python -B -m qwen3vl_local.audit_joint.phase1_visual_refine prepare --inputs <原审计包/lora_production/evaluation_inputs.json> --data-root lead_data --output <新的CPU准备.json>
python -B -m qwen3vl_local.audit_joint.phase1_visual_refine replay --inputs <原审计包/lora_production/evaluation_inputs.json> --data-root lead_data --model-dir checkpoints/Qwen3.5-4B --adapter-dir <此包对应的本机adapter目录> --output <新的配对实验目录>
python -B -m qwen3vl_local.audit_joint.phase1_visual_refine compare --inputs <原审计包/lora_production/evaluation_inputs.json> --baseline <配对实验/baseline.jsonl> --candidate <配对实验/candidate.jsonl> --output <新的比较.json>
```

尖括号必须替换成当前机器存在的路径。直接 `replay` 同样自动选单卡；显式 pin 在上述命令前加 `GPU_IDS=0`。`prepare` 只读 CPU 校验全部冻结成员和所用 RGB；`replay` 明确需要原模型与 adapter，保持原始题目、顺序、标签、图像数、生成上限，同一模型交替先后运行两种提示词，每次原生 fresh KV。保留完整 raw 输出、逐题耗时、输入／输出 token、显存测量和来源 SHA。旧 adapter 不改名、不伪造新训练身份。此命令用单进程，应使用空闲 GPU，不与旧任务共享进程热更新。

`compare` 原生解析 raw 输出，核验全部成员／身份／题目／GT，分别统计主问及全部答案的 accuracy、precision、recall、F1、format；任何原正确题或原正确答案损失、任何分项下降均输出 `review_required` 并退出 2。总分净涨不能覆盖损失。无观察到的退化不保证未知样本不退化。比较器验证保存记录；真实推理与资产身份须同时核验 `replay` 的完成回执，不能靠手工伪造 case 文件声称实测。

改过标签的测试记录属于开发审计，单独统计标签诊断，不能混进冻结原标签的性能提升；新模型最终还需要未参与本轮开发的验证材料。深度审阅为 Codex 视觉判读，不冒称独立人工真值认证。

当前本地派生全池两图／四图原生三轮规划均为每轮 12,288 呈现、3,072 更新，总 9,216 更新；这是 CPU 投影，不是 GPU 训练。新数据需要自己的正式准备与完整性回执，不复用原库 ready。候选提示词先做配对重放再决定是否进入新训练，本包不自动触发训练或修改 Phase2。

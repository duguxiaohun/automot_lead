# Phase1/2：适中预算、跨轮覆盖、旧 action 辅助与固定方法目录

> 2026-10-25：采样保持；用户已确认Phase2默认配额1024；新运行增加严格完成验收、完整审计打包和比较工具，使用步骤见 [INTEGRITY_DELIVERY_20261025.md](INTEGRITY_DELIVERY_20261025.md)。

2026-10-20，coverage_v2。本页更新同日 bounded_v1 的过小默认配额；沿用历史任务均衡，增加跨轮未见优先。覆盖旧运行文档中的默认配额、目录与 RUN_BUILD 说明；历史指标保留。
本轮为本地代码与 CPU 验证；用户已授权白名单发布。没有服务器 GPU 训练、SSH 或重新标定。

## 采用哪套方法

用户要求沿用历史审计最优结果的方法，不把 action 当作新的 Phase1/2 监督目标。

- Phase1：沿用融合 v5。八个 focus 各自 YES:NO=1:1，Phase1/道路结构各占一半，训练问法增强 4:1:1，原 loss 权重和提示词不变。历史最优使用 2026-08-31 的 step-40000；缩短后的新预算尚未证明达到该效果。
- Phase2：保留 UE1/UE3/UE5/UE6 四正类 1:1，RE 两倍配额、INVALID 一倍，以及现有 route-diverse、highway UE3 与 INVALID 子类处理。历史严格可比 v2/v3/v4 中 v3 总体最优；当前代码已有 v5 highway UE3 合同，本次不回滚其标签/提示词，更不把历史 v3 分数当成当前 v5 已验收分数。
- action：已有旧标定仅用于温和选样，不进入 prompt、答案或 loss；不增加样本总量。只在全部 Phase1/2 标签、原 event 和原子类一致的候选间替换。不再锁死某一条 route 的原抽样配额，允许换到低曝光路线；保留 highway/INVALID 等原子类配额。缺标注、冲突或被旧 action 数据分到 val/test 物理路线的帧，不使用其 action 标签，兼容标签组内已知/未知标定覆盖数量不变。部分匹配不能称为全池动作均衡。
- 评测采样函数、val/test 标签和提示词未改；旧指标不会重算成新预算成绩。

历史依据：
[Fusion v5 冻结审计](FUSION_V5_REPEAT_RUN_AUDIT_AND_FREEZE_20260902.md)、
[Phase2 方法及 v3/v5 边界](../sft_new_loop_phase2/SFT_NEW_LOOP_PHASE2_RUN.md)。

## 新默认预算

以下按 4 卡、每卡每步 1 个训练项、梯度累积 1、3 轮计算。两种 RGB 模式相同；不是这么多个独立物理帧。

| 方法 | 每类基础配额 | 每轮全局呈现 | 每轮训练步 | 三轮训练步/更新 | warmup 更新 |
|---|---:|---:|---:|---:|---:|
| Phase1 | 每 focus×YES/NO 768 | 12288 | 3072 | 9216 | 167 |
| Phase2 | 每 UE/INVALID 1024；RE 2048 | 7168 | 1792 | 5376 | 1000 |

Phase1 原默认 9216/桶、三轮 110592 步已关闭。新 768/桶是原采样预算的 1/12，比上一版 576/桶增加 1/3。
不能简单使用 512：原算法按整数保持精确 4:1:1，512 存在配额冲突；新训练入口要求配额为 6 的倍数。
Phase2 在2026-10-25真实全池审计后，经用户选择统一 Python / train.sh / run_full_pipeline.sh / prepare_data.sh 默认配额为1024。原1536因稀缺高速UE3与INVALID共享帧容量不足拒绝；类别比例、算法和重复上限均保留。
默认 val/generation 每 1000 步、adapter 保存每 2000 步；评测成员配额不变。
自动 warmup 按各阶段历史预算比例缩放；显式 WARMUP_STEPS 可覆盖，不能吞掉整个正式训练预算。

启动前打印本机 world、每轮/全程步数、optimizer 更新数和 warmup；默认每轮最多 4096、全程最多 12288 个训练循环步。超限在数据扫描/构建（流水线）和模型加载前拒绝。
单卡不会假装获得四卡速度：同配额单卡超限会明确退出。梯度累积不会缩短 forward/backward 步数。
`MAX_STEPS` 是显式截断，可能没有完整消费一个均衡 epoch；不能拿计划的整轮配额宣称截断 run 已实际达到 1:1。

默认开始后每约 100 步在已应用 optimizer 更新的位置测算 ETA（包含此前评测时间），预计单轮超过 10 小时即提前停止。所有 rank 统一决定，rank0 保存 `budget_stop/` 后一起退出；失败仅停止当前阶段；两阶段独立启动，不建立成功依赖。
`budget_stop` 只有 adapter，不含 optimizer/RNG 恢复状态，只能按明确路径评测，不能声称无损续训。
模型加载后还会打印已加载 Qwen3.5 实现的 fast-path 可用状态；缺 FLA/causal-conv1d 时明确提示，不自动安装软件。

**步数缩短不保证一天完成。** 按此次约 9.5 秒/步，仅训练算术估计 Phase1 每轮约 8.1 小时、三轮 24.3 小时；若 Phase2 也为同样速度，每轮约 4.7 小时、三轮 14.2 小时。Phase2 的真实单步耗时未测，两个阶段加总也不能承诺一天完成。每阶段还包含构建（仅缺库时）、加载和最终评测；服务器速度问题仍需实际启动日志核对。

## 跨 epoch 覆盖与真实消费

原生采样器仍决定任务/正负类/完整答案/event/highway/INVALID 子类/问法配额；训练专用选择器在这些条件以及标定已知/未知覆盖不变的候选组内选择帧。它按“历史成功反向传播次数 + 本轮计划呈现次数”最少者优先，然后兼顾 action 软配额、低曝光路线和固定随机种子。相同物理帧跨 focus/问法共享曝光计数，防止换问法就冒充新画面；另报新 frame×task/variant 组合。

两阶段每轮计划同帧最多 10 次，且不为扩预算放宽上限。超过一半呈现复用本轮帧时，计划输出 repetition_warning；兼容桶容量不足直接给出 task/需求/独立候选数并停止，不补标签、不无限循环。可用容量按兼容桶计算，不用整个数据池总行数推断稀缺正例足够。自定义 world 不整除预算时，原 DDP 尾部 padding 仍单独报告；默认四卡预算整除，不需要 padding。

跨轮历史只累计本 run **成功反向传播**的输入，不把初始计划算成已训练，不计超长输入的零损失占位。每卡保存实际记录，轮末汇总所有 rank 的历史再选下一轮。梯度累计尚未应用的输入单列 pending_backward；它们已参与反向传播，可能影响下一轮选样，但不能称为已写入 optimizer 更新。正常结束的 summary 在最后不足累积窗口提交后生成。

新增输出：

- `epoch_preflight.json`：全部计划轮次的投影容量/新帧数；不代表实际消费。
- `balance/action_seed_*.json` / `action_sampling.json`：计划覆盖、每桶容量/剩余未见量、独立帧与路线、重复与 action 分布，不是实际消费。
- `exposure/rank_*.json`：每约 100 步及轮末/结束保存当前 rank 的累计实际帧/任务计数、跳过和 padding；文件记录最后持久化 step。突然进程终止可能丢失最后未持久化尾段，不能当精确续训断点。
- `exposure/epoch_*.json` 与顶层 `exposure_epochs.json`：各轮所有 rank 实际消费、首次见到的帧/任务/路线、实际 task/action 计数。
- `exposure/summary.json` 与顶层 `exposure_summary.json`：正常结束后的全 rank 总结。原审计打包器会收集顶层轻量 JSON；大 rank 明细保留本机。

旧的 8000 步 run 没有导入曝光历史，也不自动恢复 optimizer。这里的“未见”指本次训练前面各轮尚未成功反向传播过，不能声称相对所有历史实验全新；独立帧也不等于独立场景。每次从头新训练建立自己的历史，不跨服务器混用。启动前会预演全部计划轮次，并缓存原生任务配额，提前检查后续 seed 的容量/配额错误；投影假设所有计划输入都成功反向传播，不能当作实际覆盖。真正下一轮仍按各卡汇总的实际曝光重新选帧。PLAN_ONLY 输出各轮投影和第一轮详细计划，不写实际消费记录。

验证集/测试集保持原固定成员和均衡口径，不使用训练曝光历史或 action 重选。P1八 focus×YES/NO均衡；P2评测六类等额，训练仍四UE等额/RE两倍/INVALID一倍。

## 数据不按时间戳命名

两个服务器独立使用本机固定方法路径：

```text
checkpoints/sft_new_loop_phase1_data/frame_index.jsonl
checkpoints/sft_new_loop_phase2_data/frame_index.jsonl
```

2026-10-25 独立运行更正：训练入口默认 `RUN_BUILD=0`，只核验并复用已有数据；缺失立即退出，不隐式开始全池建库。各阶段独立 `prepare_data.sh` 默认 `RUN_BUILD=auto`，先完成 CPU 数据检查和原生三轮计划，再单独启动该阶段训练。显式 `RUN_BUILD=1` 仍表示“确保有完整数据”：已有完整 index/manifest 就核验并复用，坏数据、残缺目录不自动重建。详见[独立准备与训练说明](INTEGRITY_DELIVERY_20261025.md)。
新采样预算和 action 辅助不需要重建这两个索引，更不需要运行 Phase3/4 teacher。
直接 build_dataset.py 也拒绝覆盖既有 index/manifest；并发构建使用目录 flock，不删除锁文件。

如果旧服务器数据确实位于不同时间戳目录，保持原目录不动，各机只显式绑定一次：

```bash
# 在本机 AutoMoT 根目录执行；将 /本机/已有目录 替换为本机真实路径。
python -m qwen3vl_local.sft_new_loop_phase1.local_assets bind \
  --phase 1 --source /本机/已有Phase1数据目录 \
  --target checkpoints/sft_new_loop_phase1_data
python -m qwen3vl_local.sft_new_loop_phase1.local_assets bind \
  --phase 2 --source /本机/已有Phase2数据目录 \
  --target checkpoints/sft_new_loop_phase2_data
```

固定目标已存在时拒绝替换；不需要重复绑定。也可通过 `DATA_OUTPUT_DIR` 或 `INDEX` 指定本机旧目录，完全不迁移数据。
绑定器建立本地符号链接，不复制、不联网、不猜最新，不把一台服务器的绝对路径用于另一台。每次复用记录 hostname、实际解析路径、index/manifest SHA256 和 split 计数。相同方法名不代表两机数据相同。

## 旧 action 标定怎样使用

默认 `ACTION_INDEX=auto` 只检查本机固定路径：

```text
checkpoints/sft_new_loop_phase3_data_v23/frame_index.jsonl
```

如果存在则只读匹配；不存在则打印缺失，继续原有 task 均衡，不新建任何标定。
若旧 action 索引在别处，给 `ACTION_INDEX=/本机/已有action目录/frame_index.jsonl` 即可；不同机器可各自设置一次，无需名字或时间戳一致。显式路径不存在/JSON 损坏会退出，不能静默回退。
`ACTION_INDEX=off` 可关闭辅助采样。支持已有 `primary_action` 或完整五个布尔动作字段；未知不当作 KEEP，多 context 冲突不猜。

默认 `ACTION_BALANCE_STRENGTH=0.5` 混合自然频率与平方根频率。在历史加本轮计划曝光相同的候选中，用动作配额缺口做软优先；不为动作比例选择高曝光帧去挤掉兼容的未见帧。受匹配率、同标签候选和重复上限限制，不强凑动作 1:1。
`action_sampling.json` 和 `balance/action_seed_*.json` 记录源 SHA、匹配率、未知/冲突、原事件、task 配额及动作曝光前后数；这些才是是否实际改善动作分布的证据。

## 固定训练目录与执行

默认流水线目录为 `checkpoints/sft_new_loop_phase{1,2}_pipeline/coverage_v2_2rgb_endpoints/`；四图用对应 `4rgb` 后缀。
默认训练输出也按阶段、方法和 RGB 模式命名，使用固定 `coverage_v2` 实验名，不自动附当前时间。
已有结果一律拒绝覆盖。需要重复实验时显式给语义名，如 `PIPELINE_TIMESTAMP=coverage_v2_repeat2`（保留旧变量名兼容，值无需是时间）。也可以明确 `PIPELINE_ROOT` 和 `TRAIN_OUTPUT_DIR`。
旧的 8000 步 run、旧最优权重及旧时间戳目录全部保留；重新执行本入口不是续训。

源码同步之后，任选有本机资产的服务器，在本机虚拟环境和 AutoMoT 根目录使用原来的两个入口即可：

```bash
HISTORY_RGB_MODES=2rgb_endpoints RUN_BUILD=1 \
  bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh && \
HISTORY_RGB_MODES=2rgb_endpoints RUN_BUILD=1 \
  bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

若要复用位于其它位置的 action 标定，在每条命令前加同机 `ACTION_INDEX=...`。不从另一台机器拷贝权重或题库。
可选 `PLAN_ONLY=1` 只读采样并输出 CPU 计划，不加载模型；使用独立 `_plan` 输出名，不修改训练 latest 链接，也不占用正式训练目录。不是必需的额外 readiness 任务。

新增启动前检查：本地模型/runtime/processor、三个 split 的索引标签及所选 RGB 文件头、最终评测桶/排除条件；重复 GPU 编号、重复 RGB 模式、多个模式共用训练目录或已有输出都会提前拒绝。图像检查只读已有文件，不生成标定，会增加开训前扫描耗时。它检查文件头而非完整像素解码，不能承诺未来文件变化、GPU OOM 或硬件故障不会发生。最终评测采样算法保持原版；三个 launcher 的 torchrun 调用均显式加 `--` 分隔训练脚本参数。

关注 `pipeline-preflight` / `epoch-preflight` / `launch-budget` / `training-budget` / `coverage-plan` / `exposure-actual` / `action-balance` / `model-runtime` / `training-eta`。
最终仍按原 eval.sh 生成审计包；超时或前置失败交 pipeline.log、train.log（若已创建）、training_budget.json、training_eta.json 和 action_sampling.json（实际存在的文件），不重跑去凑成功包。

## 本地验证边界

已验证真实 Phase1/2 采样器的默认配额、action 替换不改原答案/问法/子类、原 highway/INVALID/答案配额不变、路线低曝光优先、来源 heldout 路线排除、未知不补 KEEP、预算超限、数据复用/绑定/覆盖拒绝。
真实 shell 流水线在隔离 fixture 下以 RUN_BUILD=1 / PLAN_ONLY=1 复用数据，重复启动拒绝；两阶段原生训练循环以小 CPU 线性模型完成三轮/截断/超长跳过/最后累积提交，真实两个 Gloo rank 验证历史汇总、下一轮一致及停止前保存。新专项验证三轮未见覆盖、action不覆盖未见优先、稀缺桶重复上限。
小模型用于执行控制测试，不代表 Qwen3.5 4B 模型或 GPU 性能验收。
现有本地旧 action 索引只读兼容探针为 384 行、317 个独立 train 身份，其中 287 帧可解析为六类动作提示；不代表服务器匹配率，也不是重标定。
未改 Phase3/4 生产代码、老师、提示词及历史数据。无新 GPU 实验，不声称保留了旧最优准确率。

本轮相关回归 88 项通过（73.74 秒）；最后补齐 Phase1 自定义 world 尾部 padding 的曝光标记后，4 项原生训练循环测试再次通过（与 88 项重叠，不相加）。完整测试回执留 `checkpoints/phase12_coverage_verification_20261020/`；轻量源码/边界回执为 [coverage_training_verification_20261020.json](coverage_training_verification_20261020.json)。旧 `bounded_training_verification_20261020.json` 只代表上一版，不是当前代码认证。

## 发布前串联检查

新增两阶段微型原生 Qwen3.5 双图 loss/backward、LoRA 保存和合同校验重载、重载后生成测试；未下载权重，使用本地随机小模型，不代表 4B 精度或 GPU 性能。Phase1 使用短英文 fixture prompt，避免 byte-level 测试 tokenizer 把中文生产提示词展开成超长序列，生产 prompt 保持原版。
真实 bash 入口执行 Phase1 && Phase2、eval.sh 与审计包生成；只有昂贵模型任务使用隔离测试替身，两个真实打包器均生成小于 30MiB 的包并含曝光摘要、不含权重。故障测试验证 Phase1 退出 17 后不启动 Phase2、不修改旧 latest。另验证缺失/坏图像、非布尔标签、后续轮采样失败在训练前拒绝。
最新发布验证见 [pipeline_release_verification_20261020.json](pipeline_release_verification_20261020.json)；上面 bounded_v1 和 coverage_v2 初版回执属于历史快照，不是新增前置检查后源码的认证。

本次完整回归 95 项通过（103.40 秒），另以真实 torchrun 启动两个 Gloo CPU worker，验证显式 `--` 后的脚本 `--run` 参数不再被 launcher 误解析；6 个 shell 逐一通过 bash -n。未执行服务器 4B/GPU 训练。

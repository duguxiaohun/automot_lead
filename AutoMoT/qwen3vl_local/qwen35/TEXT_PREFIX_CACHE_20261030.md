# 跨图片的文本前缀缓存（2026-10-30）

新入口开关为 `AUTOMOT_TEXT_PREFIX=1`。默认仍为 0，以保护现有训练、权重和运行中的冻结源码。新模式将**同一份问题文字**移到图片前；训练与推理同时采用 `system → 问题文字 → 当前图片 → assistant`。它是一个新的输入合同 `text_before_images_v1`，需要独立新训练验证质量，不能把旧 adapter 换顺序后当作正式成绩。

## 可行性与边界

因果模型只能复用输入起始位置完全相同的连续前缀。图片后的问题状态已经依赖图片，不能抽出来和其它图片拼接。新顺序中的 system/问题不依赖后面的图片，因此可以跨不同图片复用。图片的视觉编码、语言模型中的图片状态和回答仍须重新计算。

Qwen3.5 是 hybrid：完整注意力 K/V、卷积状态、recurrent state 一起复制。每张卡的每个 bundle 默认最多保留 **2 项、256 MiB、每项 4096 个文本 token**，不落盘、不跨卡传输。每次新图从独立副本开始；保留缓存中没有图片、答案或上一帧的动态状态。

精确命中按实际 token 判断。两种固定问题交替时可各保留一项；题序或动态速度变化时，在容量已满后可以重新计算并保存真正相同的较短前缀，绝不截断旧 recurrent state 冒充短前缀。缓存键不使用标签、文件名或问题类别来代替 token 核验。每次从**当前完整输入**重算 M-RoPE 位置和 delta，允许图片数、网格和分辨率改变。

权重正常原位更新、load_state_dict、LoRA 启停/切换/合并、model 实例、dtype/device 变化触发失效。训练模式拒绝持久缓存；训练反传仍是完整图文前向，采样、问题顺序、loss、更新预算均未改变。外部代码若绕过 PyTorch 版本机制直接修改 `.data`，须显式调用 `bank(bundle).clear()`；同一 bundle 不允许并发 forward。

闭环按当前帧立即回答，不等待凑批、不重排实时帧。调用方可以在一段时间内使用固定的**问题及顺序**，直接获得完整命中；这里不改原生随机题序或冻结评测成员来人为提高命中率。随机问法的实测命中率及复用 token 数必须报告。

**1+2 的取舍：** 同图分叉旧模式可以共用图片语言状态；新模式的两种问题位于图片前，图片状态受各自问题影响，必须分别计算。两个专家若使用不同 LoRA，也不能相互复用缓存。新方案适合较长固定文本、连续图片流；不保证比旧同图分叉更快。Phase3 的速度、导航、场景等动态文本也会缩短可复用前缀。

## 已接入的入口

| 路径 | 新模式行为 |
|---|---|
| Phase1 原生 train / generation / eval | 训练、推理同布局；逐卡文本缓存；adapter 指纹不匹配拒绝 |
| Phase2 原生 train / generation / eval，包括 evidence 修订源码 | 同上，保留问题域、INVALID、标定及采样 |
| Phase3 原生 train / eval、候选 prompt、独立 base worker | 同布局；base 仍为四图 choice、三轮 7680 更新；新 release 与历史清单分离 |
| Phase3 批量生成入口 | 新模式按原顺序逐请求使用缓存，报告 serial execution；不声称保留原并行批处理吞吐 |
| 新 Phase1+2 shared LoRA | 每题从纯文本分叉，缓存跨图片保留；保留完整 token/EOS 对照与独立评分 |

Phase3 历史 dense/stream/window/frame-bank 实验有自己的时间读出合同，不将读出 query 搬到图片前。新 release 仅准入当前独立 base；其它实验在这个开关下明确拒绝。旧冻结源码、旧 release、旧恢复包、默认模型及 Phase4 不改。

当前目录分别发布支持该功能的旧布局清单与新文本布局清单，默认旧布局也可正常新启动。旧 release 会拒绝本轮已变化的源码；恢复历史运行应使用原冻结源码，不重写旧清单来绕过检查。

## 运行与数据

在 cr02 或 cee 的**当前 AutoMoT 根目录、已激活正确模型环境**运行；先 `bash ./sync_automot.sh`。不使用另一台服务器的绝对路径或旧 ready。如下命令是独立任务，不串联两个阶段，也不自动提交 GPU 作业。

```bash
# Phase1：保留选定修订与预算；新 run 绑定新布局。
AUTOMOT_TEXT_PREFIX=1 bash qwen3vl_local/audit_joint/phase12_retrain/run.sh cpu 1 --revision visual_20261029
# 正式训练将 cpu 改为 all；若先 CPU 再 train，使用该次确切 --run-dir，且仍带开关。

# Phase2：保留 evidence 修订与完整六轮预算。
AUTOMOT_TEXT_PREFIX=1 bash qwen3vl_local/audit_joint/phase12_retrain/run.sh cpu 2 --revision evidence_20261030 --budget-profile phase2_10k

# Phase1+2：独立共享模型新运行。
AUTOMOT_TEXT_PREFIX=1 bash qwen3vl_local/sft_new_loop_phase12/run.sh cpu

# Phase3：仍需该服务器已有、且通过原始身份检查的历史 index；新输出目录。
AUTOMOT_TEXT_PREFIX=1 bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
  --index /path/to/existing/frame_index.jsonl --output checkpoints/p3_text_prefix_new \
  --stage complete --prepare-only
# 完整训练用另一个新的 --output，去掉 --prepare-only。
```

Phase1/2 标定修订、预算参数继续按原任务选择；上述示例不额外授权改标签。文本顺序在 processor 编码时生成，**不需要为本功能重新标定或全池生成标签 JSONL**，原始图片与索引可复用。Phase3 原索引 manifest 绑定旧 prompt，因此入口自动在新运行的 `input_view/` **逐字节复制索引、另建新 manifest**，核验仅变化输入布局，绑定并保留父 manifest；不原地改旧回执。恢复新模式运行时传该次 `input_view/frame_index.jsonl`，不会重复派生。需要新 CPU 准入、独立训练运行及新 adapter。已有 GPU 服务器标签池是否与示例修订匹配，仍由原生检查决定，不允许跳过或倒写旧 SHA。

独立评测或闭环加载新 adapter 时同样设置 `AUTOMOT_TEXT_PREFIX=1`。adapter/run 保存 `input_layout`，prompt SHA 绑定该布局；Phase1/2 和 1+2 的新 CPU ready 也绑定布局。忘记开关或把旧 adapter 用于新布局会被正式评测入口拒绝。

同一新布局下的无缓存对照：

```bash
AUTOMOT_TEXT_PREFIX=1 AUTOMOT_TEXT_PREFIX_CACHE=0 python -m qwen3vl_local.sft_new_loop_phase2.eval --help
```

将 `--help` 换成该次正式评测的同一组参数，另选输出目录。只切第二个开关；不能用 `AUTOMOT_TEXT_PREFIX=0` 充当同输入对照，因为那会改变图文顺序。

## 闭环集成接口与性能验收

生产调用复用同一个 bundle，使用各阶段原生 messages builder 后调用 `_kv_start_state`、`_student_generate_kv`；1+2 继续调用其 `cache.generate`。新缓存附着于 bundle，不能每帧重新加载模型或重建 bundle。`text_prefix.report(bundle)` 提供 hits/misses、实际复用 token、容量、evictions/invalidations；各阶段原生 eval metrics 已接入。DDP 指标中的 cache 计数是报告所在 rank 的本地计数，不冒称全卡聚合。

独立计时工具接受有序 JSONL，每行仅 `system`、`text`、`images`。必须从实际调用方取原生文本；images 路径相对 JSONL 所在目录或使用本机绝对路径。用 2–32 条包含不同图片的真实请求，覆盖 2/4 图、两类问题和动态文本，不能为高命中率重排。

```bash
AUTOMOT_TEXT_PREFIX=1 python -m qwen3vl_local.qwen35.text_prefix_benchmark \
  --model-dir checkpoints/Qwen3.5-4B --requests /path/to/ordered_requests.jsonl \
  --adapter-dir /path/to/new_text_first_adapter
```

工具自动选一张空闲 GPU；`GPU_IDS=0` 可明确指定一张空闲卡。CPU 读取 adapter 后按设备放置，1800 秒超时，新目录保存 worker 日志与 report；不下载模型、不训练、不修改原运行。省略 adapter 可测 base，但不代表训练后 adapter 通过。

验收比较完整生成 token、EOS、截断及文本，差异直接失败。先冷启动对照，再交替执行两种路径；每次整段回放从空缓存开始，保留真实 miss、缓存复制与失效检查开销。单独记录 processor/读图耗时，模型计时不冒充闭环总延迟。报告还须结合主任务指标、旧正确损失、控制器端到端 p50/p95 和截止时间；平均文本 prefill 变快不能代替这些指标。

当前仅 CPU 小型真实 Qwen3.5 验证；本机没有完整 4B 模型，没有运行 GPU、新训练或 CARLA 闭环，尚无真实加速比例或质量结论。

本轮最终回归：486 passed、1 skipped，42.03 秒；跳过项是 Phase2 不依赖 Phase1 历史隔离的不适用参数组合。覆盖小型真实 Qwen3.5 的 1/2/4 图与不同网格、短/长文本、连续解码、两题轮换、LoRA 更新/切换、autocast 精度切换、缓存不可变、容量、四阶段消息构造、Phase3 索引视图字节保持与篡改拒绝，以及既有生成/采样/合同回归。小模型结果不外推 4B 精度、吞吐或闭环质量。

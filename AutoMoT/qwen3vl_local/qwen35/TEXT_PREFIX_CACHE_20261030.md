# 跨图片的文本前缀缓存（2026-10-30）

新入口开关为 `AUTOMOT_TEXT_PREFIX=1`。默认仍为 0，以保护现有训练、权重和运行中的冻结源码。新模式将**同一份问题文字**移到图片前；训练与推理同时采用 `system → 问题文字 → 当前图片 → assistant`。它是一个新的输入合同 `text_before_images_v1`，需要独立新训练验证质量，不能把旧 adapter 换顺序后当作正式成绩。

## 可行性与边界

因果模型只能复用输入起始位置完全相同的连续前缀。图片后的问题状态已经依赖图片，不能抽出来和其它图片拼接。新顺序中的 system/问题不依赖后面的图片，因此可以跨不同图片复用。图片的视觉编码、语言模型中的图片状态和回答仍须重新计算。

Qwen3.5 是 hybrid：完整注意力 K/V、卷积状态、recurrent state 一起复制。每张卡的每个 bundle 默认最多保留 **2 项、256 MiB、每项 4096 个文本 token**，不落盘、不跨卡传输。每次新图从独立副本开始；保留缓存中没有图片、答案或上一帧的动态状态。

精确命中按实际 token 判断。两种固定问题交替时可各保留一项；题序或动态速度变化时，直接计算并保存新问题的完整可缓存前缀，按 LRU 淘汰旧项；已有较短文本状态时，从其副本续算剩余文字并升级缓存。绝不截断旧 recurrent state 冒充短前缀。缓存键不使用标签、文件名或问题类别来代替 token 核验。每次从**当前完整输入**重算 M-RoPE 位置和 delta，允许图片数、网格和分辨率改变。

权重正常原位更新、load_state_dict、LoRA 启停/切换/合并、model 实例、dtype/device 变化触发失效。训练模式拒绝持久缓存；训练反传仍是完整图文前向，采样、问题顺序、loss、更新预算均未改变。外部代码若绕过 PyTorch 版本机制直接修改 `.data`，须显式调用 `bank(bundle).clear()`；同一 bundle 不允许并发 forward。

闭环按当前帧立即回答，不等待凑批、不重排实时帧。在线入口自动在一段时间内保持固定题序并轮换；正式训练与评测仍保留原生随机题序及冻结成员，不通过重排提高命中率。随机问法的实测命中率及复用 token 数必须报告。

**1+2 的取舍：** 同图分叉旧模式可以共用图片语言状态；新模式的两种问题位于图片前，图片状态受各自问题影响，必须分别计算。两个专家若使用不同 LoRA，也不能相互复用缓存。新方案适合较长固定文本、连续图片流；不保证比旧同图分叉更快。Phase3 的速度、导航、场景等动态文本变化会触发缓存重建，不为提高命中率冻结它们。

## 已接入的入口

| 路径 | 新模式行为 |
|---|---|
| Phase1 原生 train / generation / eval | 训练、推理同布局；逐卡文本缓存；adapter 指纹不匹配拒绝 |
| Phase2 原生 train / generation / eval，包括 evidence 修订源码 | 同上，保留问题域、INVALID、标定及采样 |
| Phase3 原生 train / eval、候选 prompt、独立 base worker | 同布局；base 仍为四图 choice、三轮 7680 更新；新 release 与历史清单分离 |
| Phase3 批量生成入口 | 新模式按原顺序逐请求使用缓存，报告 serial execution；不声称保留原并行批处理吞吐 |
| 新 Phase1+2 shared LoRA | 每题从纯文本分叉，缓存跨图片保留；保留完整 token/EOS 对照与独立评分 |

Phase3 历史 dense/stream/window/frame-bank 实验有自己的时间读出合同，不将读出 query 搬到图片前。新 release 仅准入当前独立 base；其它实验在这个开关下明确拒绝。旧冻结源码、旧 release、旧恢复包和默认模型不改。action_prior 的 Phase1/2 先验生成增加显式接入，最终 base 条件与 Phase4 规划计算不改。

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

生产调用复用同一个 bundle。Phase1/2/3 使用 `qwen35.online.predict_phase`，1+2 使用 `sft_new_loop_phase12.infer.predict`，自动执行固定题序窗口；底层 messages / `_kv_start_state` / `cache.generate` 仍可供需要自行控制题序的调用方使用。新缓存附着于 bundle，不能每帧重新加载模型或重建 bundle。`text_prefix.report(bundle)` 提供 hits/misses、full_hits/partial_hits、promotions、实际复用和新算 token、容量、evictions/invalidations；各阶段原生 eval metrics 已接入。DDP 指标中的 cache 计数是报告所在 rank 的本地计数，不冒称全卡聚合。

独立计时工具接受有序 JSONL，每行仅 `system`、`text`、`images`。必须从实际调用方取原生文本；images 路径相对 JSONL 所在目录或使用本机绝对路径。用 2–32 条包含不同图片的真实请求，覆盖 2/4 图、两类问题和动态文本，不能为高命中率重排。

```bash
AUTOMOT_TEXT_PREFIX=1 python -m qwen3vl_local.qwen35.text_prefix_benchmark \
  --model-dir checkpoints/Qwen3.5-4B --requests /path/to/ordered_requests.jsonl \
  --adapter-dir /path/to/new_text_first_adapter
```

工具自动选一张空闲 GPU；`GPU_IDS=0` 可明确指定一张空闲卡。CPU 读取 adapter 后按设备放置，1800 秒超时，新目录保存 worker 日志与 report；不下载模型、不训练、不修改原运行。省略 adapter 可测 base，但不代表训练后 adapter 通过。

验收比较完整生成 token、EOS、截断及文本，差异直接失败。先冷启动对照，再交替执行两种路径；每次整段回放从空缓存开始，保留真实 miss、缓存复制与失效检查开销。单独记录 processor/读图耗时，模型计时不冒充闭环总延迟。报告还须结合主任务指标、旧正确损失、控制器端到端 p50/p95 和截止时间；平均文本 prefill 变快不能代替这些指标。

当前仅 CPU 小型真实 Qwen3.5 验证；本机没有完整 4B 模型，没有运行 GPU、新训练或 CARLA 闭环，尚无真实加速比例或质量结论。

上一版提交 8599cb3e12 记录的回归：486 passed、1 skipped，42.03 秒；跳过项是 Phase2 不依赖 Phase1 历史隔离的不适用参数组合。覆盖小型真实 Qwen3.5 的 1/2/4 图与不同网格、短/长文本、连续解码、两题轮换、LoRA 更新/切换、autocast 精度切换、缓存不可变、容量、四阶段消息构造、Phase3 索引视图字节保持与篡改拒绝，以及既有生成/采样/合同回归。小模型结果不外推 4B 精度、吞吐或闭环质量。


## 审计修复 r2：在线调度与计时口径

`AUTOMOT_QUESTION_WINDOW=32` 表示连续 32 次在线输入组使用同一组 seed，之后轮换；每次 `predict` 消费一组 RGB，不缓存图像答案。`AUTOMOT_QUESTION_SLOTS=1` 为默认；独立 Phase1/2/3 可以设为 2，窗口内交替两种题序。种子包含 rank、窗口、槽位，进程间独立且可重复。`schedule=False` 保留调用方传入的固定 seed；也可以显式传 `QuestionSchedule(window_frames=32, slots=1, seed='drive', rank=0)`。窗口不是时间等待，不会积攒图片或打乱帧序。

1+2 的两种任务已占用两个缓存槽，因此 `task='both'` 只接受一个题序槽；`PriorEngine` 同样有两个 EVENT 域，只接受一个题序槽。问题域/上下文若继续变化，精确匹配仍会出现 miss，不能靠调度 seed 冒称完整命中。`max_tokens=4096` 截断的文本缓存命中记为 partial，不是 full。

```python
# 环境提前设置 AUTOMOT_TEXT_PREFIX=1；bundle 已通过对应原生 adapter 合同检查。
# 不要在循环内重新加载模型或重建 bundle。
from qwen3vl_local.qwen35.online import predict_phase
from qwen3vl_local.sft_new_loop_phase12.infer import predict

for images in rgb_stream:  # 当前按时间顺序选好的 4 张 RGB
    result = predict_phase(phase1_bundle, 1, images, history_rgb_mode='4rgb')
    # 独立 Phase2：predict_phase(phase2_bundle, 2, images, domain=phase2_prompts.ROAD_DOMAIN)
    # domain 使用该版本 prompts.QUESTION_DOMAINS 中的正式值。
    # Phase3 还需传当前 context_id、road_structure、速度/导航等实际观测；
    # 此 API 不提供自动场景识别，也不替代 CARLA 的上游上下文提供者。
    # shared 1+2：predict(shared_bundle, images, '4rgb', domain)
```

`action_prior.runtime.PriorEngine(..., online_text_prefix=True)` 已接入；`make_runtime` 对应环境开关为 `AUTOMOT_ONLINE_TEXT_PREFIX=1`，并且须同时启用 `AUTOMOT_TEXT_PREFIX=1`。要求真实 Phase1/2 adapter 的 `input_layout`、prompt SHA 与当前源码匹配，只支持四帧历史入口（内部继续按各 adapter 的 2/4 图模式选帧）。旧权重、单帧 prior、dataset labels 模式不能误走新缓存。每个 LoRA 保留独立缓存；多轮复核的旧答案和后续问题仍在当前图片后面，绝不存入跨图片前缀。默认 history 复核保留同一首问前缀；independent/compare 的不同复核问题可能超过容量，如实计 miss，不删除复核以制造加速。base 分析和最终规划条件保持 image-first 路径。

原生 1+2 评测的主指标现在是 `performance.stream_benchmark`：整个 split 先清空缓存，按原始题序每题只执行一次生产推理；累计包含首次建缓存、问题切换、失效检查、复制和生成。记录每组生产前后计数，交包审计要求计数连续且每组只增加两次请求。`saving_fraction` 比较同一新布局下的独立完整前向；另报加入 processor 编码耗时的比例。基线总在生产后执行，因此仍存在执行顺序/设备热身偏差，不等于闭环端到端收益；独立 benchmark 工具提供交替整流对照。

`paired_benchmark` 降为辅助：新协议 `paired_text_prefix_cold_pair_v2` 的每次回放从隔离空缓存开始，明确包括重建成本，不再把首次 miss 排除后称为总体加速。诊断 logits 和回放都不能写入生产缓存，异常退出也恢复原缓存。旧 image-prefix 的历史计时协议保留。

本次修复没有改变 `text_before_images_v1` 布局、提示词定义、标定、采样、数据成员或训练预算；不需要再生成一套标签库。若已有该布局的新权重，缓存修复本身不要求重新训练；旧 image-first 权重仍必须按新布局独立训练验收。Phase3 的新源码发布清单使用 `*_sources_r2_20261030.json`，以原清单 SHA 为父来源，原清单和旧回执保持原字节。


r2 验证记录：主组合 **512 passed、1 skipped，43.18 秒**；包含原 486 项范围、10 项在线/计时/多 adapter 缓存回归、新增短前缀升级与 token 上限测试、14 项 action_prior 张量运行测试。此前单独的 41 项和 23 项包含于本轮组合，不能相加。TinyBase LoRA 代数测试显式使用夹具限定的 PEFT 加载器，不冒称合成线性模型通过 Qwen3.5 权重合同；真实小 Qwen3.5 的多 LoRA 缓存命中另有测试。

扩展检查独立列报：加入历史 `test_pipeline_candidate.py` 的组合为 504 passed、1 skipped、3 failed；三个失败来自历史 shell 夹具未选择 legacy 入口、单 GPU 配置不符合当前四卡要求。进一步检查旧 `test_prefill_only.py` 时有两项因 `v23_io1` 冻结 release 的外部 `highway_ue3_audit.py` SHA 不匹配被拒绝（到第二项失败即停止）；使用提交前 runtime 在内存中重跑其中一项，得到相同拒绝。没有改写旧冻结 SHA、放宽生产准入或把这些失败称为通过。主组合和扩展组合均未复现审计所述 `No CUDA GPUs are available`，不能宣称已经定位/修复其根因。

本轮未运行 4B GPU 训练/评测或 CARLA。Phase3 新模式仍逐请求生成并绕过原 `reuse_rgb`，1+2 仍分别计算两题对应的图像语言状态；这些取舍需要远程实测。旧 Phase3/Phase4 冻结依赖问题仍须独立新 release 验收，不能把此次在线 prior 接线称为整条 CARLA 已准入。

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
# 环境提前设置 AUTOMOT_TEXT_PREFIX=1；显式选择已获准使用的设备。
# 校验对应原生 adapter / 基座 / prompt 合同，并一次性加载和绑定。
from qwen3vl_local.qwen35.online import load_online_bundle, bind_prompt_contract, predict_phase
phase1_bundle = load_online_bundle(base_dir, phase1_adapter_dir, 1, device='cuda:0')
# 已由原生加载器校验并加载的 bundle，则显式执行：
# bind_prompt_contract(bundle, phase, saved_adapter_config)
# 不要在循环内重新加载模型或重建 bundle。
from qwen3vl_local.sft_new_loop_phase12.infer import predict

for images in rgb_stream:  # 当前按时间顺序选好的 4 张 RGB
    result = predict_phase(phase1_bundle, 1, images, history_rgb_mode='4rgb')
    # 独立 Phase2：predict_phase(phase2_bundle, 2, images, domain=phase2_prompts.ROAD_DOMAIN)
    # domain 使用该版本 prompts.QUESTION_DOMAINS 中的正式值。
    # Phase3 从绑定合同恢复 candidate/baseline、choice/binary 和 RGB 模式；
    # 还需传当前 context_id、road_structure、速度/导航等实际观测；
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


## 审计修复 r3：实际提示词合同与导入隔离

`qwen35.online.predict_phase` 不再默认选择 Phase3 baseline/binary/4rgb。必须先通过 `load_online_bundle` 加载并绑定，或对已经验证、正确加载的权重调用 `bind_prompt_contract`。后者只验证并绑定提示词/RGB/布局元数据，不能替代权重来源校验。每次推理由绑定合同决定实际 renderer、输出模式和图片数；显式覆盖只能与合同相同，缺失绑定、模型/活动 adapter 切换、布局变化或覆盖冲突都会拒绝。Phase3 消息经过原生 `prompt_contract.build_action_messages`，候选版本仍为训练使用的 `v23_rgb_stage_candidate_20261006`；预测回执记录 prompt SHA 与元数据 SHA。

Phase1/2 恢复器仅执行仓库内已审查的训练转换函数，支持 Phase1 RS2/RS4 legacy 与 direct8 visual、Phase2 visual v6 与 evidence v7。它不执行权重包里的 Python；prompt 名称、RGB 模式、输入布局与完整 surface SHA 必须匹配。严格权重检查和 available 检查均使用同一恢复器；LoRA 打包时归档实际恢复出的源码，避免把原生旧模板误写成新版本来源。

Phase1 direct8 使用首问 + 一次重新排序的八问复核，覆盖全部八个输出；不再发送该权重拒绝的 GROUP/DETAIL/hierarchical 问题。RS_HIGHWAY 未被该版本训练，明确记录为 unavailable/None，不拿 HIGHWAY 代填。格式失败、两次不一致、多 RS 同时 YES 仍拒绝相应条件；Phase2 保留两个问题域及 INVALID 保护。history/independent 通常共 6 次模型问答，compare 共 9 次；旧层级权重仍走原来的 9/15 次流程。新 direct8 复核协议单独写入 action contract 身份，不能复用旧协议的条件缓存或把调用次数变化当作 KV 缓存收益。此处改变的是新权重的在线复核协议，训练题库、标签、采样、预算及最终规划计算不变；复核质量需要实测。

Phase2 eval 的自动 GPU 选择现在仅在 `__main__` 执行，仍位于 torch import 前。完整组合测试进一步定位到 `goalgen.eval` 的同类导入写入，亦作相同修复。导入二者不会调用 nvidia-smi、等待 torchrun 对端、改写 CUDA_VISIBLE_DEVICES 或 GOALGEN_LOCAL_CUDA_INDEX；Phase2 CLI 仍保留 --device cpu 和自动选卡；两者均保留既有 GPU_IDS pin 行为，GoalGen 没有新增 CPU CLI 选项。未改环境回执收集函数来吞掉 CUDA 错误。

Phase3 新启动使用 `*_sources_r3_20261030.json`，父来源为 r2 清单；旧清单、旧权重、旧恢复包不改。在线合同修复没有改变训练提示词定义，因此无需为此重建标签库或重训已匹配的新布局权重；旧 image-first 权重仍不能直接使用 text-first 缓存。


r3 验证：在 `CUDA_VISIBLE_DEVICES=''` 下完整组合 **606 passed、1 skipped，57.56 秒**。覆盖原 r2 主组合、在线合同绑定/恢复/实际发送消息、未知与篡改合同拒绝、新 Phase1/2 direct8/event 调用链，以及 Phase2/GoalGen 的 12 组独立进程导入顺序/rank 组合和两个 CLI 选卡保护测试。恢复打包模块在该组合中通过；没有通过吞异常、清空环境报告或更换测试顺序规避错误。专项 111 项与本组合有重叠，不相加。额外 available/LoRA 打包检查仍在旧 `v23_io1` 外部 `highway_ue3_audit.py` SHA 不匹配处拒绝（42 passed、2 failed 后停止），该独立历史冻结依赖未放宽；不能称所有历史流水线已通过。

回归使用真实小型 Qwen3.5 验证缓存数值/状态；在线提示词与复核协议测试使用合成回答，验证的是消息、解析和调用合同，不是 4B 判断能力。未 GPU、未 CARLA、未重训、未更改标定或新建标签库。日志与源码 SHA 保存在本机 `checkpoints/text_prefix_contract_r3_20261030/`，不随公开代码上传。

## r4：原生在线输入流的成对验收入口

新增 `qwen35.online_benchmark`，适用于独立 Phase1/2/3 与 shared Phase1+2。它复用在线接口的 `prepare_request`，按已验证 adapter 的提示词合同、RGB 模式和题序窗口构造输入；不接受手写提示词覆盖。shared 推理新增 `infer.load_online_bundle`，保留原生内容身份和 backend 校验，CPU 读 adapter 后放置到显式设备。此次提取请求构造函数未改变在线题目、训练、正式评测、复核或规划逻辑。

请求是 JSONL，每行仅有 `images` 和 `context`。图片为**当前 adapter 已选好的 2 或 4 张**，按时间排列；相对路径以 JSONL 所在目录为基准，输入组保持原始流顺序，不按问题重排。允许 2–4096 组；测轮换时至少覆盖两个完整窗口，建议使用现场连续输入，不能仅反复播放同一张图。Phase1 的 context 为 `{}`；Phase2/12 必须传实际 `domain`，例如 `{"domain":"ROAD_CORRIDOR"}`；Phase3 必须传实际 `context_id` 和 `road_structure`，并按观测传 `context_detail`、`goal_xy`、`current_speed_mps`。不包含标签、不固定动态信息、不把场景名猜成 context。context 的值沿用相应原生提示词定义。

```json
{"images":["rgb/frame_000.png","rgb/frame_003.png"],"context":{}}
{"images":["rgb/frame_004.png","rgb/frame_007.png"],"context":{}}
```

上面是 Phase1 两图格式示意。先在实际服务器的 AutoMoT 根目录、已激活环境下做 CPU 输入检查（不加载模型、不选 GPU）：

```bash
AUTOMOT_TEXT_PREFIX=1 python -m qwen3vl_local.qwen35.online_benchmark \
  --phase 1 --requests "$REQUESTS_JSONL" \
  --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" --check-only
```

输入检查仅表示原生提示词合同、字段和图片数量匹配，不表示权重通过验收或图像质量达标。模型运行仍要通过原生加载器的完整来源校验。远程 GPU 实测命令去掉 `--check-only`，例如：

```bash
AUTOMOT_TEXT_PREFIX=1 python -m qwen3vl_local.qwen35.online_benchmark \
  --phase 1 --requests "$REQUESTS_JSONL" \
  --model-dir "$MODEL_DIR" --adapter-dir "$ADAPTER_DIR" \
  --window-frames 32 --slots 1 --repeats 2 --max-new-tokens 160 \
  --output "$NEW_OUTPUT_DIR"
```

`--phase 12` 为 shared 两题；需要对应的 `fusion_config.json` 权重，只允许 `--slots 1`（两题已占两槽）。独立阶段可 `--slots 2`。`--rank` 记录要模拟的卡号，单进程单卡顺序执行；并非 torchrun。默认选择一张空闲 GPU，也可用 `GPU_IDS` 指定一张空闲物理卡。所有路径由当前机器提供，不依赖 cr02/cee 的固定根路径或另一台的权重；输出目录必须新建。`--device cpu` 仅供小模型机制验证，本机 codon/wang 不跑 GPU。

每条路径在每轮开始时清空**自己的**缓存和调度器，包含首次建缓存及换题重建。先执行一组完整冷缓存对照，再交替 `full→cached` / `cached→full`；主比例只来自成对轮次，每一轮仍包含全部输入，重复次数要求为偶数。这里的“冷”仅指文本缓存，不表示清空 OS 文件缓存或重置 GPU。整个实验保持同一权重、原生题目、生成预算、图片顺序与模型调用次数，逐条核验图片 SHA、完整编码输入 SHA、完整 token、EOS、截断状态和文本；权重、输入或输出变化立即失败。测试缓存与生产缓存、生产调度计数分离。

`report.json` 和每轮 `trial_XX.json` 保存逐组、逐题证据。`saving_fraction` 计入读图/解码、提示词构造、processor、设备传输、完整 prefill/decode、缓存复制/失效检查和审计散列；`model_saving_fraction` 仅计模型与缓存操作。另报各部分耗时、全流 p50/p95、full/partial/miss 分类，以及首组/换窗口/变 context 的延迟（这几个分组可重叠）。`--deadline-ms` 可统计**该测量链路**超限次数，但含审计开销，不能当成 CARLA 控制截止时间。输入散列开销单列，主比例不会悄悄扣掉它。worker 捕获异常时落盘失败回执；外部超时由 `worker.json` 记录，不能将仍为 running 的报告视为完成。`--timeout` 默认 7200 秒，可按流长度显式调整；已有报告不覆盖。

这不是 PriorEngine 全复核或 CARLA 控制器 benchmark：每组独立阶段调用一次，shared 调用两次；不与旧 direct8/层级复核的不同调用次数做加速比较，不认证复核质量。它也不与 Phase3 的历史 batch/reuse_rgb 路径比较，不能由此断言新串行布局优于原路径。正式报告仍需远端 4B 质量评测、旧正确损失和 CARLA 端到端控制延迟；结果中的 `quality_certified`、`carla_certified` 始终为 false。

本轮不改标定、标签库、数据成员、采样或训练预算，不需要重建标签库；旧 image-first 权重仍不能直接用于 text-first。Phase3 新启动清单为 `*_sources_r4_20261030.json`，绑定本轮源码并以 r3 的 SHA 为父来源；旧清单不改写。历史 `v23_io1` 的 `highway_ue3_audit.py` 外部依赖仍与冻结 SHA 不同（期望 `9cddc3b0…`，当前 `f3b7f8a1…`），原有两项拒绝的根因未消除，需独立的新 release 验收，不能称旧流水线通过。


r4 验证：CPU 完整组合 **630 passed、1 skipped，60.71 秒**；新增 24 项包含真实小型 Qwen3.5 四阶段流式输出等价、冷启动/换题/动态速度、两槽与部分命中、生产状态隔离、像素/权重漂移拒绝、合同篡改、失败回执及 CPU-only 输入检查。图像 processor 为合成夹具，不能把这些测试当作真实 4B 图像理解或 GPU 性能验收。后补 CLI 超时与 DDP 环境隔离检查后，流式 benchmark + Phase3 新清单专项 **35 passed，5.41 秒**，与组合范围重叠，不相加。未运行 GPU、CARLA 或重训，未生成新标签库。

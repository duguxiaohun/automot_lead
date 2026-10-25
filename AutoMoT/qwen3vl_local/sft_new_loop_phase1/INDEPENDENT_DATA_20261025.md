# Phase1 / Phase2 独立运行与真实本地全池审计

用户要求两个阶段独立训练、互不依赖，并在本地提前排查建库和启动阻断。两阶段都从 Qwen3.5 基座独立加载；不传递 adapter、不等待另一阶段、不共用 run/latest/完成状态。旧文档的 `&&` 顺序启动建议撤回。Phase1 原有八问里的道路结构仍是其自身任务，不改已确定的采样或 prompt。

## 已完成的检查与修正

- 源码回执按阶段绑定：另一阶段专用 train/eval/prompt/建库文件的修改不再使本阶段回执失效。确实共用的基础工具仍绑定。
- 训练默认 `RUN_BUILD=0`，数据缺失时在占用实验目录前失败；两阶段各自新增 `prepare_data.sh`，独立进行 CPU 建库、图片检查和原生三轮采样投影，无需模型/GPU。
- 显式从训练入口建库时，先检查 Qwen3.5 资产。CPU 建库、视觉审计和 preflight 有完整日志、10秒心跳、阶段状态及默认7200秒超时；失败不进入后续步骤，超时只终止自己启动的进程组。保留失败目录，不删除锁或悄悄重建。
- 独立 shell 测试只提供当前阶段数据，另一阶段留有失败状态，仍能完成当前阶段训练、测试与交包；自身失败不改另一阶段状态。
- 177项不同CPU测试通过（146.85秒）；最后两个提前拒绝入口修改后，相关16项再通过（41.82秒），不与177相加。两个独立测试包解包后CLI验收通过，VLM输出替身不作为真实精度。27个采样函数AST与10个关键文件逐字/SHA保持。

## 真实原生建库及 CPU 结果

两份数据分别由未经修改的原生 builder 扫描本机完整 collection/RGB 生成，未使用小样本替身、未改标签或划分。

| 阶段 | train | val | test | 已检查的不同RGB文件头 | 当前默认三轮计划 |
|---|---:|---:|---:|---:|---|
| Phase1 | 569269 | 37450 | 65525 | 683245 | 768配额，9216更新，ready |
| Phase2 | 5430 | 234 | 798 | 14871 | 1024配额，5376更新；原1536拒绝保留 |

Phase1 val/test 均按原生八focus×YES/NO×64生成1024题；Phase2 val/test 均按原生六类×64生成384题。这是成员/标签/原生选择器检查，没有模型推理。图像文件头检查不等于全像素解码、RGB人工复审或未来文件不变保证。

Phase1 每轮12288呈现，三轮新帧12288/12288/11862，单轮最多1次；全部是计划投影，不是实际训练曝光。

Phase2 高速 UE3 与 INVALID 共享28个训练物理帧。在1536配额下，两组要求192+154=346次呈现，原单帧上限10只允许280；精确最大流/最小割确认缺66次容量，不是贪心选择误拒绝。1280仍要求160+128=288，缺8次。诊断算法仅证明容量，没有替代原生采样器。

1024与768都通过原生三轮计划。1024方案保持四UE等额、RE两倍、INVALID一倍、高速/INVALID子组、路线规则及原重复上限，每轮7168呈现/1792更新，三轮5376；三轮物理帧最大重复均9，首轮覆盖5027个物理帧，后两轮新增0，不能宣称增加了独立数据支持。用户随后明确选择1024作为默认配额，现已统一到Python训练器、训练shell、完整流水线和独立CPU准备入口；1536/1280拒绝证据保留，未改采样方法。

本机没有旧 `sft_new_loop_phase3_data_v23/frame_index.jsonl`，action旁路状态如实为 unavailable，使用原生已支持的任务均衡分支；未补造标定。远程若存在不同action来源或不同索引，应重新核验其实际计划，不把两机当作同一数据条件。

## 本机产物与分别启动

原生数据保存在 `checkpoints/phase12_independent_20261025/phase1_data` 和 `phase2_data`，本机标准 `checkpoints/sft_new_loop_phase1_data` / `sft_new_loop_phase2_data` 分别绑定各自目录，不覆盖旧数据。绑定本身不代表任何配额均可训练。

- Phase1当前源码复核准入：`checkpoints/sft_new_loop_phase1_data_cpu/coverage_v2_after_phase2_default1024/status.json`，原`coverage_v2`回执保留。
- Phase2原默认拒绝：`checkpoints/sft_new_loop_phase2_data_cpu/coverage_v2/status.json`，完整失败证据保留。
- Phase2新默认准入：`checkpoints/sft_new_loop_phase2_data_cpu/coverage_v2_default1024/status.json`，原候选`coverage_v2_1024`回执保留。
- 容量证据：`checkpoints/phase12_independent_20261025/phase2_capacity_diagnostic.json`。

Phase1可在目标机器具备模型并通过原生检查后单独运行：

```bash
PIPELINE_TIMESTAMP=independent_phase1_v1 bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

Phase2现默认采用已验收的1024方案，可单独运行：

```bash
PIPELINE_TIMESTAMP=independent_phase2_1024_v1 bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

不要把两个命令接成成功依赖。共享一组四卡时分别安排；需要同时运行则明确分配不重叠GPU，且按实际卡数重新检查步数预算。本轮没有4B GPU训练、远程操作、push或真实新旧模型性能对照。

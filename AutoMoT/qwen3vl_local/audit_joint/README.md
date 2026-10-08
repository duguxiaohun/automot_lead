# 联合审计第一批：G0 本机冻结与跨任务隔离

落实 [总方案 §5、§14](../sft_new_loop_phase3/PHASE3_PHASE4_AUDIT_ROADMAP_20261008.md)。这是独立审计入口，不接入默认训练、修改标签或重新划分旧 split。G0 总体验收仍为 **in_progress**；E0/E1/G1/G2 尚未执行。本工具成功产出报告不代表基线可复现或独立性通过。

服务器从更新、全量建库到 G0、现有基线训练/验证的完整顺序见 [服务器执行手册](SERVER_RUN_20261008.md)，命令统一从 `AutoMoT/` 执行。

## 2026-10-09 第二轮边界补强

Action 有效池必须与本次 `--action-data` 的 train/val/test 内容 SHA 逐一相符。原始索引可搬迁，只要字节相同；旧请求也从 action_raw 来源绑定，不仅检查有效池自身。RGB 根目录和其它筛选依赖仍遵循原合同；不把“索引可搬迁”解释为可任意改变这些依赖。本次导出源码变化后，旧 v2 有效池同样需在新目录重新导出小索引，不修改旧合同。

外部引用记录 `logical_path`、`resolved_path`、大小和 SHA。`verify-inputs` 分别报告旧目标的 `target_status` 与当前请求的 `binding_status`；即使旧目标未变或新目标字节相同，改指也不能通过。旧捕获缺少逻辑路径时为 incomplete，需重新捕获，不向旧回执补字段。此检查针对外部引用；快照本体完整性仍由 `verify` 核验。

来源认证与观察结果分开：依赖 manifest 不合格或预期 SHA 不匹配时，完整可读索引保持 `status=invalid`、`observation_status=observed`。路线用途进入账本，冲突带 `evidence_status=suspected` 和逐来源状态；相关候选排除，但这些来源不贡献完整池或人工验证支持。格式损坏的索引仍不提交部分记录，不把未读部分当已审核。

## 2026-10-09 外部审计边界修复

同一 resolved route 的逻辑别名做传递归并，参与用途矩阵、冲突与候选排除；图像 SHA 相同仍只作待审证据，不合并路线。请求给 Phase3/Action/候选接入实际数据根目录，支持分别指定。

guard 在主进程正常或异常退出后都清理自身进程组残留，保留退出码。verify-inputs 重新核验已登记基座权重/配置/tokenizer 等资产及集合变化，报告逐文件结果/数量；捕获时缺模型为 incomplete，漂移为 failed。

Action 默认/显式有效池统一要求经过 v2 manifest 实际验证，未验证依赖的来源不计完整池。v2 绑定原生读取代码、稳定 Phase3 声明的外部依赖、异常时长过滤器，以及全部原始路线（含被过滤路线）的真实路径、帧数及判定；复用时重扫核对。旧 v1 只保留，不补写 hash，需重新导出小索引。

prepare 将截断或错误类型的 manifest/metadata 写为诊断；捕获保留可取得 SHA、invalid 和阻塞项。新增 Phase4 模式/不同目录/绑定及全训练 ID、语义、共享 RGB 的流式配对检查。规则/标签/生产格式与旧数据不改，缺少 data4 的服务器先交缺口报告，不重复生成大回放。详见服务器手册；新检查要求重新 prepare 请求。

## 2026-10-09 磁盘占用修订

服务器发生磁盘满后，先按手册第 0 步只读盘点，不重复全量建库，不自动删除 core/旧实验。第 2 步改为优先复用已有题库；缺失项可先交报告，完整基线验收仍保持未通过。

G0 默认 `references`：源码及小型 JSON 元数据仍冻结；adapter 权重、完整索引和历史逐题输入仅保存原路径/大小/SHA，不再复制一遍。需要额外冻结输入字节时显式选择 `--storage-mode full`；基座仍只记录身份。轻量捕获不是完整输入快照，必须保留服务器原件，`verify-inputs` 会报告缺失/漂移；`verify` 仅核验捕获自身。

`storage-plan --config 请求 --output 新捕获目录` 预估 G0 复制量及空间余量；`storage-inventory --root checkpoints --core-dir .` 只读列目录/大文件及 ELF core，不删除、不读取转储正文。`run_guarded.sh --space-path 输出所在路径 -- python ...` 每次设置 core 软/硬限制为 0，拒绝不受该限制约束的管道收集器，并每秒检查空闲空间；低于预留停止本次进程组，不影响其它运行。周期检查不是硬配额，不能保证其它程序不会写满磁盘。

本机同一批输入完整快照296575294字节→轻量捕获28697111字节；66项外部引用SHA通过，除存储字段外隔离报告相同、账本逐字节相同、原合同不变；ZIP8296154字节及旧/新回执通过。新增10项存储/保护测试，相关回归87通过。本机管道core收集器确实在启动前被拒绝，未制造崩溃或修改系统设置。验证记录见 [storage_verification_20261009.json](storage_verification_20261009.json)。

全量老师逐帧回放与两图四图题库仍是大产物；本次不修改 Phase4 生产格式、标签或冻结合同，也不宣称完整流程只有 30 MB。ZIP 上限仍为 30,000,000 字节。

## 已实现

- 对实际工作区（含未提交/未跟踪源码）生成逐文件 SHA 和本机内容快照；另存 Git HEAD/差异摘要。固定 Phase3 明确 prompt variant、Phase4 v42/老师 v10、本地 Qwen 后端合同。
- 校验 Phase4 原 manifest/监督文件 SHA，Phase3 历史 adapter 的 variant、prompt、mapping 与训练索引身份。模型存在时流式哈希全部本地模型资产；不存在则记 missing，不下载、不改旧 hash。
- 读取完整索引/路线预约/曝光清单，复用 Phase3 与 Action 的物理组函数；二者不一致即拒绝。保留数据源、用途、记录数和首个记录位置。
- 检查训练/开发曝光/历史回归与 val/test/独立审核的交集；软链接同目录和跨组文件/像素重复单列。相同黑帧不自动证明同一路线，但相关新预约候选须待核。
- 新开发 val 候选列逐组排除原因；缺完整训练池时只列 `eligible_pending_complete_audit`，不自动预约。旧退休预约保留 quarantine。
- 人工 val 支持按输入来源、事件、边、答案、原生静止/运动阈值统计；弱老师参考和不明来源不补人工缺额。
- Action 用原生 `read_rows` 导出运行时实际池，包含开发隔离和支持重分配；保留采样前全部行（含 sampler 不合格行，是保守隔离上界），不以 epoch 抽样替代。原始 split 只记 raw_inventory，避免把已重分配的旧 val 冒认为当前 val。
- `run` 完成捕获后自动生成同级 `OUTPUT.audit.zip`，压缩后硬限 30,000,000 字节；`pack` 可对旧捕获单独打包或汇总指定训练/评测目录，`verify-package` 检查 ZIP 清单及 SHA。完整报告和逐题结果不截断，预算不足仅可显式省略源码正文；仍超限不发布包。
- 输出目录必须新建。读取失败/中途源码漂移不写成功回执；`verify` 检查文件全集及 SHA，只证明包字节完整。

## 本机执行

在项目根目录运行，使用含 numpy/Pillow 的现有环境。报告捕获不需要加载模型：

```bash
PYTHONPATH=AutoMoT python -m qwen3vl_local.audit_joint prepare \
  --phase3-prompt-variant v23_rgb_stage_candidate_20261006 \
  --phase4-data2 /tmp/phase4_v42_final/build2 \
  --phase4-data4 /tmp/phase4_v42_final/build4 \
  --verify-images --output /tmp/joint_g0_request.json

PYTHONPATH=AutoMoT python -m qwen3vl_local.audit_joint run \
  --config /tmp/joint_g0_request.json \
  --output AutoMoT/checkpoints/joint_g0_NEW_RUN

PYTHONPATH=AutoMoT python -m qwen3vl_local.audit_joint verify \
  AutoMoT/checkpoints/joint_g0_NEW_RUN
```

`run` 当前捕获阶段存在未完成验收时返回 **2**，并保留完整报告及自动交接 ZIP；不能据报告文件存在就继续训练。`verify` 返回 0 只表示回执通过。重复运行必须换输出路径。`prepare` 请求可读可改，所有路径/用途显式登记；未知格式或缺失字段不会当作空池成功。

模型/数据文件只作本机快照或哈希；不上传，不修改数据，不读取 Codex 聊天/认证/历史目录。原始 RGB 仅在显式 `--verify-images` 时由 Phase4 原 loader 程序校验，不生成新的人工视觉标签。

## 本轮本机结果

最终产物位置：`AutoMoT/checkpoints/joint_g0_20261008_local_final`。第一次真实探针保留在同级 `joint_g0_20261008_local_probe`，二者各有自己的源码快照。

已核验 v42 两图/四图各 train=8058、val=273、test=186 的开发产物，逐输入执行源/像素 SHA 检查；不是全量 v42 生产复现。Phase3 四个历史包的 variant/prompt/mapping 校验通过，权重与原训练索引缺失，未复跑其准确率。

本地可用清单发现 **13 个不同物理组、5 类用途交集**：Phase3 开发曝光→Phase4 val 2组/test 6组/独立审核2组，Phase3 历史回归→Phase4 test 2组/独立审核1组。这是新联合实验的独立性问题清单，不倒写旧实验分数。逐组及来源见 `split_cross_audit.json` 和 `exposure_ledger.jsonl`。完整 Phase3/Action 池尚缺，不能声称已找齐全部冲突。

人工 val 覆盖 U-E1/U-E5/U-E6/U-E7；缺 R-E2/R-E3/R-E5/U-E2/U-E3/U-E4。未预约新路线，未看独立 RGB，未新增人工标签或严格批准。

## 服务器需要执行的内容

先在具有原始数据和匹配环境的服务器补齐 G0。本机缺 Phase3 完整训练索引、选定 adapter 权重、Qwen3.5-4B、本次 Action 三 split/full map、全量 v42 产物。当前本机 pvi 环境只有 torch 2.5.1+cu124，缺 Transformers 5.3.0/PEFT/FLA/causal-conv1d；不能据有 CUDA 就开始 E1 性能验收。

1. 为 Action 准备 JSON 数组文件，内容为该实验实际使用的 CLI 参数，例如 `['--data-root', ...]` 的合法 JSON（双引号）。必须包含真实 `--data-root`、`--data-dir`、`--event-balance-index` 与 `--sampling-mode`；action_balanced 或 high-level token 模式还要提供其原 token 索引参数，不以默认值替代旧实验。导出实际池：

   ```bash
   PYTHONPATH=AutoMoT python -m qwen3vl_local.audit_joint export-action-pool \
     --argv-json /path/to/action_actual_argv.json --output /path/to/new_action_pool_audit
   ```

2. 用真实路径准备并运行新 G0 请求：

   ```bash
   PYTHONPATH=AutoMoT python -m qwen3vl_local.audit_joint prepare \
     --phase3-prompt-variant v23_rgb_stage_candidate_20261006 \
     --phase3-index /path/to/phase3/frame_index.jsonl \
     --phase3-adapter /path/to/selected_phase3_adapter \
     --phase4-data2 /path/to/matching_v42/data2 \
     --phase4-data4 /path/to/matching_v42/data4 \
     --action-data /path/to/action_data \
     --action-effective-index /path/to/new_action_pool_audit/effective_pool_audit.jsonl \
     --data-root /path/to/lead_data --model-dir /path/to/Qwen3.5-4B \
     --verify-images --output /path/to/new_g0_request.json
   ```

   再用上述 `run --config ... --output 新目录` 和 `verify`。路径占位符必须替换为服务器真实路径。若只冻结 baseline prompt，显式改 variant；历史 candidate 包仍作为历史证据，不强行变成 baseline checkpoint。

3. 根据完整交叉矩阵建立新联合实验准入清单，再补六事件开发 val。当前工具只过滤候选，不解决物理路线不足，不替人批准预约。

4. 后续单独实现/执行 E0：相同 checkpoint 完整评测、固定更新题集的数值/计时基线、四个 H20 rank 的实际内核绑定及 BF16 前反向/生成 profiler。E0 通过后再进入 E1。当前没有 E0/E1 运行命令，不用猜测的命令或预计吞吐冒充已实现接口。

完整快照与大产物留在执行主机。将自动生成的 `OUTPUT.audit.zip` 手工交回即可继续审计；工具不自动上传，ZIP 不进 Git。包内含完整 G0 报告、SHA/合同、配置及可选源码，不包含模型/原始 RGB/完整数据索引；因此不替代训练恢复快照或人工目视证据。训练/评测结果合包方式见服务器手册第 6 步。

## 测试

同日推送前，使用已有 `/tmp/automot-qwen35-env`（Transformers 5.3.0/PEFT 0.18.1）扩大到完整 Phase4 测试目录，**1138 passed**。原 pvi 的 3 项缺依赖失败在兼容环境复验通过；未下载权重或启动真实训练。

2026-10-08 本机执行以下范围：**66 passed**，其中新增 G0 测试 25 项；其余覆盖原 Phase3 prompt、Phase4 路径身份和 Action split support。

```bash
PYTHONPATH=AutoMoT python -m pytest -q \
  AutoMoT/qwen3vl_local/audit_joint/tests \
  AutoMoT/qwen3vl_local/sft_new_loop_phase3/test_prompt_candidate.py \
  AutoMoT/qwen3vl_local/sft_new_loop_phase4/tests/test_data_paths.py \
  AutoMoT/qwen3vl_local/action_prior/tests/test_split_support.py
```

测试覆盖程序合同，不是老师准确率、模型等价性或驾驶能力认证。Action exporter 本机仅原生接口模拟测试；全数据运行留待服务器。

2026-10-08 交接包增量：11 项打包测试及原 G0 25 项共 **36 passed**；实际旧 G0 捕获生成 **8,271,720 字节** ZIP，含冻结源码，回执与 ZIP 核验通过。旧快照/旧合同未修改；此为打包实测，不是新增服务器训练或 E0/E1 验收。

扩展 prompt/path/Action split 相关回归共 **77 passed**。最终打包代码对同一 G0 加四套历史 Phase3 结果实测 **12,486,935 字节**（1,229 个清单文件），逐题记录完整纳入，ZIP 核验通过；没有新推理结果。

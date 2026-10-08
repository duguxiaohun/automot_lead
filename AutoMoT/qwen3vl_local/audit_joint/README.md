# 联合审计第一批：G0 本机冻结与跨任务隔离

落实 [总方案 §5、§14](../sft_new_loop_phase3/PHASE3_PHASE4_AUDIT_ROADMAP_20261008.md)。这是独立审计入口，不接入默认训练、修改标签或重新划分旧 split。G0 总体验收仍为 **in_progress**；E0/E1/G1/G2 尚未执行。本工具成功产出报告不代表基线可复现或独立性通过。

服务器从更新、全量建库到 G0、现有基线训练/验证的完整顺序见 [服务器执行手册](SERVER_RUN_20261008.md)，命令统一从 `AutoMoT/` 执行。

## 已实现

- 对实际工作区（含未提交/未跟踪源码）生成逐文件 SHA 和本机内容快照；另存 Git HEAD/差异摘要。固定 Phase3 明确 prompt variant、Phase4 v42/老师 v10、本地 Qwen 后端合同。
- 校验 Phase4 原 manifest/监督文件 SHA，Phase3 历史 adapter 的 variant、prompt、mapping 与训练索引身份。模型存在时流式哈希全部本地模型资产；不存在则记 missing，不下载、不改旧 hash。
- 读取完整索引/路线预约/曝光清单，复用 Phase3 与 Action 的物理组函数；二者不一致即拒绝。保留数据源、用途、记录数和首个记录位置。
- 检查训练/开发曝光/历史回归与 val/test/独立审核的交集；软链接同目录和跨组文件/像素重复单列。相同黑帧不自动证明同一路线，但相关新预约候选须待核。
- 新开发 val 候选列逐组排除原因；缺完整训练池时只列 `eligible_pending_complete_audit`，不自动预约。旧退休预约保留 quarantine。
- 人工 val 支持按输入来源、事件、边、答案、原生静止/运动阈值统计；弱老师参考和不明来源不补人工缺额。
- Action 用原生 `read_rows` 导出运行时实际池，包含开发隔离和支持重分配；保留采样前全部行（含 sampler 不合格行，是保守隔离上界），不以 epoch 抽样替代。原始 split 只记 raw_inventory，避免把已重分配的旧 val 冒认为当前 val。
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

`run` 当前捕获阶段存在未完成验收时返回 **2**，并保留完整报告；不能据报告文件存在就继续训练。`verify` 返回 0 只表示回执通过。重复运行必须换输出路径。`prepare` 请求可读可改，所有路径/用途显式登记；未知格式或缺失字段不会当作空池成功。

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

所有结果留在执行主机。只需反馈缺失项、冲突摘要和计时/数值结论，不要求上传原始数据、模型或审计包。

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

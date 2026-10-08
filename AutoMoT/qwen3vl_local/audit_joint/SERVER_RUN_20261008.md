# 服务器执行顺序：先 G0，再基线训练与性能审计

本次已实现 G0 本机快照/隔离检查。E0 计时、四 rank 实际内核探针、E1 稀疏 logits、G1 时间线和 G2 盲标尚未实现；当前七轮流水线仍是 v42 弱监督基线，不会自动执行上述新实验。

本轮先执行第 1–4 步，将完整服务器资产与交叉隔离结果落实；第 4 步结束自动生成 **不超过 30,000,000 字节（30 MB）的 ZIP**，交回该文件即可审计。第 5 步说明现有训练/评测入口，暂不作为新联合方案的验收；已有训练/评测结果可按第 6 步合包。

## 1. 更新源码，核对原训练环境

在服务器仓库根目录执行。若有本地代码修改使快进失败，保留修改并处理冲突；不用 reset/clean 删除数据或强行覆盖。

```bash
git pull --ff-only origin main
cd AutoMoT
ulimit -S -c 0
```

激活服务器原来的 Qwen3.5 训练环境。后续命令均从 `AutoMoT/` 运行，`python` 必须是该环境的解释器；不要用另一环境的 `pip` 升级整个依赖栈。

```bash
python -m pytest -q \
  qwen3vl_local/audit_joint/tests \
  qwen3vl_local/sft_new_loop_phase3/test_prompt_candidate.py \
  qwen3vl_local/sft_new_loop_phase4/tests \
  qwen3vl_local/action_prior/tests/test_split_support.py

python -m qwen3vl_local.qwen35.preflight \
  --model-dir checkpoints/Qwen3.5-4B
```

模型另存时替换 `--model-dir`。预检只核验本地资产/vendor 与 Transformers 5.3.0，**不是四卡 FLA/conv/attention 实际调用验收**；不自动下载模型。缺依赖先记录原版本及报错，再对照 `qwen3vl_local/qwen35/requirements.txt` 处理。保持原可复现环境，不能将盲目升级后的结果当旧基线。

## 2. 只准备全量 v42 配对题库，不训练

新建本轮目录；后续保持同一终端的这些变量。若中断续跑，重新设为原路径，不重新生成时间戳。

```bash
G0_ROOT="checkpoints/joint_remote_$(date +%Y%m%d_%H%M%S)"
P4_DATA="$G0_ROOT/phase4_data"
mkdir -p "$G0_ROOT"

python -m qwen3vl_local.sft_new_loop_phase4.full_pipeline \
  --data-root lead_data --data-dir "$P4_DATA" --workers 16
```

这是全合格路线扫描/规则回放/两图四图编译，会读取整个数据集，可能耗时较长；16 是路线 CPU worker 数，不是 GPU 数。该模块不启动训练、评测或下载。原始数据另存时替换 `lead_data`；支持既有软链接。

完成后应有 `data2/manifest.json`、`data4/manifest.json`、`pipeline_ready.json`。首次全量 v42 的题数由实际结果决定，不能要求等于本机 140 路线开发库的 8058。

## 3. 定位 Phase3 与 Action 原资产

Phase3 选择具体 adapter 目录（含 `adapter_model.safetensors` 与 `sft_new_loop_phase3_adapter_config.json`），不要传整个 audit_bundle 或仅含元数据的目录。选择原 run 的 best/final 规则，不按 test 成绩重新选。

```bash
P3_INDEX=checkpoints/sft_new_loop_phase3_data_rgb_stage_20261006_isolated/frame_index.jsonl
P3_ADAPTER=/替换为所选Phase3适配器目录
MODEL_DIR=checkpoints/Qwen3.5-4B
ACTION_DATA=/替换为Action三split目录
```

`P3_INDEX` 必须匹配 adapter 记录的 SHA。若找不到旧索引或权重，记 missing；不要用新索引/当前源码替旧 checkpoint 放行。上述 candidate 的 prompt variant 是 `v23_rgb_stage_candidate_20261006`，若选择 baseline 必须同步改 variant 和匹配索引。

为 Action 新建 `"$G0_ROOT/action_argv.json"`，内容是该实验的**实际 CLI 参数数组**，例如：

```json
[
  "--data-root", "lead_data",
  "--data-dir", "checkpoints/action_prior_data",
  "--event-balance-index", "/替换为原Action完整事件map文件.jsonl",
  "--sampling-mode", "event_balanced"
]
```

必须将示例替换为原实验路径/模式/相关参数。若原来为 `action_balanced` 或启用了 high-level action token，还要填写原 token 索引参数。这个文件不是新训练配置；它用于调用原生读取器恢复实际数据池。

```bash
python -m qwen3vl_local.audit_joint export-action-pool \
  --argv-json "$G0_ROOT/action_argv.json" \
  --output "$G0_ROOT/action_pool"
```

导出不加载模型、不采样 epoch、不改旧 split。缺 Action 数据时先跳过此导出，并从下一命令移除 `--action-effective-index`；G0 会明确记缺口，不能宣称跨 Action 的隔离通过。

## 4. 跑服务器 G0，自动打包并核验

```bash
python -m qwen3vl_local.audit_joint prepare \
  --phase3-prompt-variant v23_rgb_stage_candidate_20261006 \
  --phase3-index "$P3_INDEX" --phase3-adapter "$P3_ADAPTER" \
  --phase4-data2 "$P4_DATA/data2" --phase4-data4 "$P4_DATA/data4" \
  --action-data "$ACTION_DATA" \
  --action-effective-index "$G0_ROOT/action_pool/effective_pool_audit.jsonl" \
  --data-root lead_data --model-dir "$MODEL_DIR" --verify-images \
  --output "$G0_ROOT/request.json"

python -m qwen3vl_local.audit_joint run \
  --config "$G0_ROOT/request.json" --output "$G0_ROOT/capture"
```

**当前 `run` 会返回 2**：即使资产齐全，仍明确保留“同 checkpoint 完整复现、固定更新数值/性能、四 rank 内核”未执行项。它是捕获阶段的未通过状态，不代表一定崩溃，也不应改为 0 伪装验收。不要用 `&&` 把它与下一步串联；另行执行：

```bash
python -m qwen3vl_local.audit_joint verify "$G0_ROOT/capture"

python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/capture.audit.zip"
```

**交给我的是 `$G0_ROOT/capture.audit.zip`**，无需手工复制摘要。`run` 在报告/回执生成后自动打包并核验 ZIP，再返回 G0 状态码；即使有缺资产、交叉冲突或未执行阶段，也先保留包。终端打印 ZIP 绝对路径、实际字节数和 SHA256。不能用返回 2 代表打包失败；包不存在或核验失败才需查异常。若终端启用了 `set -e`，用 `python ... || g0_status=$?` 接住状态后检查，不能直接忽略所有错误。

包内包含完整 G0 报告、用途账本、冲突/缺额明细、原合同和环境、逐文件 SHA、配置/manifest 元数据及预算允许的冻结源码。完整快照中的数据索引和权重 blob 不复制到交接包，原始 RGB/视频也不打包；这不是可直接恢复训练的完整快照，更不是 RGB 人工目视审计包。

大小按压缩后的实际 ZIP 计算，硬上限 30,000,000 字节。默认带冻结源码；超限先仅移除源码正文，保留完整源码 SHA 清单并显式记录原因。若完整报告/逐题结果仍超限，则拒绝发布 ZIP，保留服务器原件；不截断指标/错例，不把多个包的总大小说成一个 30 MB 包。

旧 G0 已完成、或第一次打包失败修复后，可只打包，**无需重复全量回放**。输出必须使用未存在的路径：

```bash
python -m qwen3vl_local.audit_joint pack \
  --capture "$G0_ROOT/capture" --output "$G0_ROOT/g0_handoff_retry.audit.zip"
python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/g0_handoff_retry.audit.zip"
```

完整快照和模型数据保留服务器，不进 Git；工具不自动上传。没有 `receipt.json` 或 `verify` 失败时，先反馈报错，不能将残缺目录打成完成包。

下一步以完整交叉矩阵处理新实验准入/六事件 val 缺额，落实 E0 的固定题序、环境和四卡测量代码；再进行 E1 等价性。当前本机已发现的 13 个交叉用途物理组不能靠改旧 split 或修改 hash 消除。

## 5. 现有训练/评测入口是什么

以下是已有 **v42 弱基线**入口，供需要复现原任务时使用；它不解决 G0 交叉隔离，也不会实施 E0/E1。按总方案做新联合实验时，先处理第 4 步结果。

四卡采样/资产预检（不训练，不证明快速内核实际被调用）：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.preflight \
  --dataset "$P4_DATA/data4" --paired-with "$P4_DATA/data2" \
  --data-root lead_data --model-dir "$MODEL_DIR" \
  --sampling-policy phase3_balanced --epochs 7 --epoch-samples 10240 \
  --accumulation 8 --world-size 4 --require-ready
```

独立训练两套 adapter，四图完成后再两图。此处直接调用 train，避免在开发阶段自动反复查看最终 test：

```bash
GPU_IDS=0,1,2,3 DATASET="$P4_DATA/data4" MODEL_DIR="$MODEL_DIR" \
  OUTPUT_DIR="$G0_ROOT/train_rgb4" \
  bash qwen3vl_local/sft_new_loop_phase4/train.sh ddp \
  --paired-with "$P4_DATA/data2" --sampling-policy phase3_balanced \
  --epochs 7 --epoch-samples 10240 --accumulation 8

GPU_IDS=0,1,2,3 DATASET="$P4_DATA/data2" MODEL_DIR="$MODEL_DIR" \
  OUTPUT_DIR="$G0_ROOT/train_rgb2" \
  bash qwen3vl_local/sft_new_loop_phase4/train.sh ddp \
  --paired-with "$P4_DATA/data4" --sampling-policy phase3_balanced \
  --epochs 7 --epoch-samples 10240 --accumulation 8
```

训练期间已有人工 val 选优；仅四事件覆盖，不能宣称十事件改进。Phase3 原模型优先做相同 checkpoint 的开发验证复现（不必先重训）：

```bash
GPU_IDS=0,1,2,3 INDEX="$P3_INDEX" MODEL_DIR="$MODEL_DIR" \
  SPLIT=val CASES_PER_BIN=0 RUN_AUDITS=0 \
  bash qwen3vl_local/sft_new_loop_phase3/eval.sh "$P3_ADAPTER"
```

`RUN_AUDITS=0` 关闭额外审计/RGB 导出，不代表跳过逐题评估。此命令是生成式验证，不是训练/反向/内核性能验收。最终 test 只在候选和协议冻结后执行；CARLA 真闭环也另立验收。

若明确要跑已有四图→两图、训练→历史 test→打包的整套弱基线，旧入口仍为 `GPU_IDS=0,1,2,3 bash qwen3vl_local/sft_new_loop_phase4/run.sh`；它会执行完整七轮，不能把这条命令当成新方案的一键审计。

## 6. 后续训练/评测完成后，汇总结果再交接

完成并停止写入后，显式选择本轮实际 run 目录。当前 Phase4 原生 `train.sh` 直接写入指定 `OUTPUT_DIR`（目录已存在会拒绝）；下例与第 5 步的路径一致。Phase3 使用 eval 命令实际打印的输出目录。若其它入口使用 `latest` 链接，先解析成固定 run 路径，避免选到下一次运行。不要传整个 checkpoints 或原始数据目录。

例如两套 Phase4 训练均已完成：

```bash
P4_RUN4="$G0_ROOT/train_rgb4"
P4_RUN2="$G0_ROOT/train_rgb2"
python -m qwen3vl_local.audit_joint pack \
  --capture "$G0_ROOT/capture" \
  --result-dir "phase4_rgb4=$P4_RUN4" \
  --result-dir "phase4_rgb2=$P4_RUN2" \
  --output "$G0_ROOT/training_handoff.audit.zip"
python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/training_handoff.audit.zip"
```

需要合入已完成的 Phase3 评估时，在 `pack` 上另加 `--result-dir "phase3_val=/替换为实际eval输出目录"`。可把这条打包命令放在原训练/评测脚本最后，仅在任务成功完成时执行；G0 的 `run` 则已默认自动打包。

结果白名单收集配置/选优记录、采样与训练指标、逐轮验证及完整 `cases*.jsonl`/逐轮 cases；逐题结果不抽样。`handoff_manifest.json` 列出每个目录实际纳入和排除的文件、每个入包文件的字节数/SHA。训练日志中已有数值指标会保留在对应 JSON/JSONL；任意文本日志、权重、RGB、大型 profiler trace 不自动纳入。尚未实现的 E0/E1 profiler 采集与打包规范需随实现接入，不能靠找到同名结果文件就宣称已验收。

额外结果只作为附件，保留其原状态/覆盖率；工具不会替你判断是不是完整七轮、完整 test，也不会自动启动训练或打开最终 test。合包不会将原 G0 中的 `not_run` 改成通过；收到包后应按实际结果进一步审计。若多 run 合包超限，分别指定单个 run 生成各自不超过 30 MB 的包；单 run 核心结果仍超限时先反馈，不自行裁掉逐题记录。

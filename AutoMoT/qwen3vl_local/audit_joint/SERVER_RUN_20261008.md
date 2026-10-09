# 服务器执行顺序：先低磁盘占用 G0，再基线训练与性能审计

2026-10-09 已收到两组服务器短跑交接；接下来按[完整一轮与自动重载步骤](../sft_new_loop_phase4/rgb_short/NEXT_RUN_20261009.md)执行。新版源码须重生成小视图，已完成四图库/回放继续复用。

**2026-10-09 最新执行入口：用户已确认 RGB4 保持原样、RGB2 缩短为 `[-2,0]`。后续基线准备改用[短两图准备与四卡短跑](../sft_new_loop_phase4/rgb_short/README.md)。不要再为该实验续建旧跨度 data2。新入口保留 v42 生产合同，只复用/编译一份四图库，再生成两种输入共用的小视图；已收到 recovery-check 的8614路线/1062401帧完整回执。下文旧两库命令仅供历史基线，不是当前默认下一步。**

本次已实现 G0 本机快照/隔离检查。E0 计时、四 rank 实际内核探针、E1 稀疏 logits、G1 时间线和 G2 盲标尚未实现；当前七轮流水线仍是 v42 弱监督基线，不会自动执行上述新实验。

**首份服务器交接已审计：** `capture.audit/` 的1216个文件及包内回执绑定通过，G0仍阻塞。3对冲突涉及10个不同物理路线组，不能只补文件就宣告隔离通过。当前下一步是[首份交接审计报告中的只读 recovery-check](REMOTE_CAPTURE_REVIEW_20261009.md)，先确认18 GB回放能否复用及续建空间下限；不用再次提交同样的缺失报告。解压目录也可直接 `python -m qwen3vl_local.audit_joint verify-package checkpoints/capture.audit`，但无法据此验证原 ZIP 大小和服务器实时外部引用。

**2026-10-09 容量修订：旧手册第 2 步的新目录全量建库不再是默认前置步骤。先盘点并使用已有资产，缺失项如实报告。30 MB 仅是交接 ZIP 上限，不是服务器全过程的磁盘占用上限。**

本轮执行第 1–4 步，检查现有服务器资产与交叉隔离；第 4 步结束自动生成 **不超过 30,000,000 字节（30 MB）的 ZIP**，交回该文件即可审计。第 5 步说明现有训练/评测入口，暂不作为新联合方案的验收；已有训练/评测结果可按第 6 步合包。

**2026-10-09 外部审计修订：** 同一真实路线目录的不同别名现在计入跨任务冲突；Action 有效池当前使用 v3 合同，默认路径与显式路径都必须核验 manifest、源码/外部筛选依赖、输入 SHA 和原始路线帧数。损坏的 manifest/metadata 会记录 invalid 并生成阻塞态 G0，不能获得完整池资格。RGB2/RGB4 要求不同目录、正确模式和一致绑定，并流式核对全部训练配对；额外内存只存 ID/摘要，不复制大题库。

已运行的旧 Action 导出不改 hash，也不覆盖原文件：如需纳入本轮完整池，重新导出到 `action_pool_v3_candidates`（仅有效池索引和依赖清单），再重新 `prepare` 生成新请求，不能继续用旧请求绕过新增门禁。旧 Phase4 生产合同未改，已有 `full_production`/题库可以继续核验或原生续跑，**不需要因本次修复重新生成 30 GB**。

**第二轮补强：** `--action-data` 三 split 现在必须与导出的原始输入逐一 SHA 相同；索引搬迁允许，误配另一题库会阻塞。旧 v2 导出因源码合同变化也需另目录导出（例如 `action_pool_v3_candidates`），不要覆盖旧导出。重新 prepare/run 时使用新捕获目录；已有 Phase4 30 GB 产物继续复用。

新引用同时保存逻辑路径和目标。`verify-inputs` 会检测索引/adapter 软链接改指，即使旧文件仍在；旧捕获缺逻辑路径时返回 incomplete。来源 manifest 不合格时，可读索引中的路线重叠仍作为带来源状态的疑似冲突保留，相关路线不能直接成为新 val，不计完整覆盖或人工支持。

**图片核验与路径重放补强：** 缺图或图片 SHA 不符时保留完整可读索引的路线用途，作为未认证证据参与冲突和新 val 排除；图片失败不会使路线静默消失。索引本身损坏仍不提交部分来源。Action v3 同时记录原 argv、导出 cwd 及绝对依赖路径，复验可换工作目录，原始三 split 也可在字节不变时搬迁。旧 v2 缺少可靠路径基准，需按第 3 步重新导出小索引并重新 prepare；不需要重建原始题库。旧产物保留，不补写 hash。缺 Action 数据时仍跳过导出，使用第 4 步缺失资产分支。

**Action 隐式候选依赖补强：** `action_balanced` 即使未显式开启 token，也会读取动作候选。新版审计同时核验候选文件和同目录的 manifest.json、candidate_counts.json、frame_index.jsonl。纯 event_balanced 且 token 关闭时不强加未读取的候选依赖。v3 格式保持，但源码合同已更新；此前导出的 v3 小索引也需按第 3 步另目录重导，不修改旧 manifest/hash。无需重建原始数据、Phase4 或训练；Action 数据仍缺失时继续第 4 步缺失分支。

当前服务器回报：`full_production` 约 18 GB、`data2` 约 13 GB，尚未看到 `data4`。这只能证明这些目录存在，不能证明 `data2` 完整。先检查 `data2/manifest.json`、`data4/manifest.json`、`pipeline_ready.json` 是否存在；缺少四图库时先交缺口报告，不用复制四图库冒充两图库，也不删除原回放。

## 0. 磁盘已满时，先停止本次仍在写盘的任务并盘点

不要重新启动全量建库，也不要重复运行时间戳命令创建另一份目录。不要使用 `rm -rf checkpoints` 或直接删除所有 `core.*`：目录内可能混有模型、旧实验、可续跑回放和正常源码。

在服务器 `AutoMoT/` 下，只读查看（此时还没有新版代码也可执行）：

```bash
df -h .
du -h --max-depth=1 checkpoints | sort -h
cat /proc/sys/kernel/core_pattern
```

把输出发回即可先定位占用。能更新源码后，可进一步列最大的 20 个文件、各目录实际占用，并在 checkpoints 及当前目录检查 ELF core 文件头，不删除、不复制文件：

```bash
python -m qwen3vl_local.audit_joint storage-inventory --root checkpoints --core-dir .
```

该盘点不跟随目录软链接，不搜索整个服务器；仍被进程打开但已删除的文件不在目录清单中。需要清理时先按实际路径确认范围，不由工具猜测并删除。

## 1. 更新源码，核对原训练环境

先进入实际包含 `qwen3vl_local/` 的应用目录。本机是仓库下的 `AutoMoT/`，服务器也可能将这个目录本身作为 Git 根目录；以下命令兼容两者，不要再盲目追加一次 `cd AutoMoT`。若有本地代码修改使快进失败，保留修改并处理冲突；不用 reset/clean 删除数据或强行覆盖。

```bash
git pull --ff-only origin main
ulimit -S -c 0
```

激活服务器原来的 Qwen3.5 训练环境。后续命令均从 `AutoMoT/` 运行，`python` 必须是该环境的解释器；不要用另一环境的 `pip` 升级整个依赖栈。

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path checkpoints -- \
  python -m pytest -q \
  qwen3vl_local/audit_joint/tests \
  qwen3vl_local/sft_new_loop_phase3/test_prompt_candidate.py \
  qwen3vl_local/sft_new_loop_phase4/tests \
  qwen3vl_local/action_prior/tests/test_split_support.py

bash qwen3vl_local/audit_joint/run_guarded.sh --space-path checkpoints -- \
  python -m qwen3vl_local.qwen35.preflight \
  --model-dir checkpoints/Qwen3.5-4B
```

模型另存时替换 `--model-dir`。预检只核验本地资产/vendor 与 Transformers 5.3.0，**不是四卡 FLA/conv/attention 实际调用验收**；不自动下载模型。缺依赖先记录原版本及报错，再对照 `qwen3vl_local/qwen35/requirements.txt` 处理。保持原可复现环境，不能将盲目升级后的结果当旧基线。

## 2. 复用已有 Phase4 题库，不默认新建全量回放

先把 `P4_DATA` 设置为之前实际生成的、同时含 `data2/` 和 `data4/` 的目录。以下默认路径只是常规位置；如果此前输出在 `joint_remote_*/phase4_data`，就直接使用那一份，不要复制或移动。`G0_ROOT` 只存这次轻量审计，失败重试优先复用已完成产物，不自动生成一串时间戳目录。

```bash
G0_ROOT=checkpoints/joint_audit_20261009_candidate_deps
P4_DATA=checkpoints/joint_remote_20261008_211924/phase4_data
mkdir -p "$G0_ROOT"
```

已有 `pipeline_ready.json` 时可只验证原全量回执，**不建库**：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$P4_DATA" -- \
  python -m qwen3vl_local.sft_new_loop_phase4.full_pipeline \
  --data-root lead_data --data-dir "$P4_DATA" --skip-build
```

没有完整题库时，本轮可直接进入第 3、4 步收集现状，G0 会记录 missing/invalid/incomplete；这份报告可用于决定后续补跑，不代表全量审计通过。原题库源码合同不匹配时不能修改其 manifest/hash 伪装兼容，也不要仅为消除 missing 又立刻启动全量建库。

无完整回执、但保留 `full_production` 时，先只读检查（输出名须未使用）：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$P4_DATA" --min-free-gib 10 -- \
  python -m qwen3vl_local.audit_joint recovery-check \
  --data-dir "$P4_DATA" --data-root lead_data \
  --output checkpoints/phase4_recovery_check_20261009.json
```

工具完整核验回放索引及逐路线/逐帧回执，报告源码合同、候选池、registry、双库回执及空间下限。未通过时仍保留诊断 JSON；没有新建库。缺 manifest 的残缺 data2 会在原生续建时被重命名保留，随后重新编译，每个新库仍复制完整 production；18 GB回放对应两套复制约36 GB额外写入，另加题目/元数据等。`lower_bound_and_reserve_fit=true` **不表示总空间足够**，`sufficient_for_build` 仍未知。原目录保留，不自动删除残片。

只有检查报告已审查、确实需要补全、已处理容量和 core 问题后，才单独执行以下**可选的重任务**，沿用原部分完成目录以尝试原生续跑：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path "$P4_DATA" --min-free-gib 10 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.full_pipeline \
  --data-root lead_data --data-dir "$P4_DATA" --workers 16
```

全量规则回放会保存逐路线/逐帧证据，两套训练题库也会写盘；当前并未将它们改成压缩格式或删减版。10 GiB 是停止前的空闲预留，不是预测总需求或总产物上限。源码/配置匹配时原生入口复用已完成结果；如果合同不匹配先报告，保留原目录，不不断换新路径重跑。

`run_guarded.sh` 每次启动重新设置 core 软/硬限制为 0；发现 `core_pattern` 以 `|` 开头会在任务启动前拒绝，因为系统收集器可能绕过 core 限制，需要先核对服务器收集器策略。它不改系统全局配置。运行期间每秒检查指定文件系统的剩余空间，低于预留则停止本次创建的进程组，退出 75；主进程结束时也会清理同组残留 worker，并保留主进程退出码；只管本次任务，不影响其它训练，也不自动删除文件。检查存在时间间隔，不能代替文件系统配额或保证其它任务不会写满磁盘。空间路径要指向实际输出所在磁盘；审计 capture 和 ZIP 建议放在同一盘。

## 3. 定位 Phase3 与 Action 原资产

Phase3 选择具体 adapter 目录（含 `adapter_model.safetensors` 与 `sft_new_loop_phase3_adapter_config.json`），不要传整个 audit_bundle 或仅含元数据的目录。选择原 run 的 best/final 规则，不按 test 成绩重新选。

```bash
P3_INDEX=checkpoints/sft_new_loop_phase3_data_rgb_stage_20261006_isolated/frame_index.jsonl
P3_ADAPTER=/替换为所选Phase3适配器目录
MODEL_DIR=checkpoints/Qwen3.5-4B
ACTION_DATA=/替换为Action三split目录
```

各任务确实使用不同 RGB 数据根目录时，给第 4 步 prepare 添加 `--phase3-data-root`、`--action-data-root`、`--candidate-data-root`。Phase3 和候选默认使用 `--data-root`；Action 优先使用 v2 导出记录的实际 data_root。必须填写实际读取路径，不能用不存在的目录跳过别名检查。

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
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$G0_ROOT" -- \
  python -m qwen3vl_local.audit_joint export-action-pool \
  --argv-json "$G0_ROOT/action_argv.json" \
  --output "$G0_ROOT/action_pool_v3_candidates"
```

已有本轮匹配的 v3 `action_pool_v3_candidates` 时直接沿用；G0 会核对其源合同及输入 SHA，不重复导出。导出不加载模型、不采样 epoch、不改旧 split。缺 Action 数据时先跳过此导出，并从下一命令移除 `--action-effective-index`；G0 会明确记缺口，不能宣称跨 Action 的隔离通过。

## 4. 跑服务器 G0，自动打包并核验

**目录布局修复（2026-10-09）：** 默认从当前导入模块所在位置查找 Git 根目录，并分别识别 `仓库/AutoMoT/qwen3vl_local/` 和 `仓库/qwen3vl_local/`，不依赖启动 shell 的 cwd。`project_root` 与 `application_root` 分别记录，源码/模型/索引默认路径都以应用目录构建。不要仅把旧代码的 `parents[3]` 改成 `parents[2]`，否则旧版 `make_config` 仍可能多追加一个 `AutoMoT/`。Git 信息不可取得时记录 `git_identity` 阻塞，不伪造 HEAD，不因此在生成诊断回执前崩溃。

如果上一轮因 `not a git repository` 失败，更新代码后重新 prepare 到新目录，保留旧 request 和失败 capture。不能只修改旧 request 的 project_root，因为其中源码和资产路径也可能错误。目录布局修复本身不改变生产合同；本轮图片/路径补强升级了 Action 导出合同，已有旧有效池需按第 3 步重导小索引。无需重建原始题库。

**当前 Phase3 索引/adapter 和 Action 数据缺失时，使用以下完整分支替代本节后面的完整资产示例。** 这些资产参数不是必填项；无需虚构路径、编造 Action argv 或先重训。下面的缺失报告目录必须尚未使用；若已有失败输出，保留并改用另一个新名称。

```bash
G0_ROOT=checkpoints/joint_missing_assets_20261009_images_paths
P4_DATA=checkpoints/joint_remote_20261008_211924/phase4_data
mkdir -p "$G0_ROOT"

python -m qwen3vl_local.audit_joint prepare \
  --phase3-prompt-variant v23_rgb_stage_candidate_20261006 \
  --phase4-data2 "$P4_DATA/data2" --phase4-data4 "$P4_DATA/data4" \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --storage-mode references --output "$G0_ROOT/request.json"

python -m qwen3vl_local.audit_joint storage-plan \
  --config "$G0_ROOT/request.json" --output "$G0_ROOT/capture"
```

确认 storage-plan 退出 0 且 sufficient=true，再执行。此处只收集现有文件，不补建 Phase4 残缺题库，也不下载或训练模型：

```bash
g0_status=0
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$G0_ROOT" -- \
  python -m qwen3vl_local.audit_joint run \
  --config "$G0_ROOT/request.json" --output "$G0_ROOT/capture" \
  --storage-mode references || g0_status=$?
echo "G0 exit code: $g0_status"
python -m qwen3vl_local.audit_joint verify "$G0_ROOT/capture"
python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/capture.audit.zip"
```

退出 2 时仍须确认两项核验通过；其它退出码或缺少 ZIP 时反馈原始错误。交回 `$G0_ROOT/capture.audit.zip`。以下原命令适用于已经定位完整 Phase3/Action 资产的情况，不与缺失分支重复执行。

```bash
python -m qwen3vl_local.audit_joint prepare \
  --phase3-prompt-variant v23_rgb_stage_candidate_20261006 \
  --phase3-index "$P3_INDEX" --phase3-adapter "$P3_ADAPTER" \
  --phase4-data2 "$P4_DATA/data2" --phase4-data4 "$P4_DATA/data4" \
  --action-data "$ACTION_DATA" \
  --action-effective-index "$G0_ROOT/action_pool_v3_candidates/effective_pool_audit.jsonl" \
  --data-root lead_data --model-dir "$MODEL_DIR" --verify-images \
  --storage-mode references \
  --output "$G0_ROOT/request.json"

python -m qwen3vl_local.audit_joint storage-plan \
  --config "$G0_ROOT/request.json" --output "$G0_ROOT/capture"

bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$G0_ROOT" -- \
  python -m qwen3vl_local.audit_joint run \
  --config "$G0_ROOT/request.json" --output "$G0_ROOT/capture" \
  --storage-mode references
```

默认 `references` 模式只复制源码及不超过 2 MiB 的 JSON 元数据；adapter 权重、训练索引、历史逐题大文件等只记录原路径/大小/SHA，不再复制进 capture/blobs。完整 G0 报告和用途账本仍保留，图像核验不减少题数。`storage-plan` 显示本次预计复制字节数、引用输入字节数和空间预留；这是 G0 估算，不包含可选全量建库。确实需要额外冻结输入字节时才显式选 `full`，基座模型仍只记录身份。

轻量捕获依赖服务器原件保留；`verify` 只验证捕获自身，另用以下命令检查外部引用，以及已登记的基座权重、配置、tokenizer 等资产 SHA 和资产集合是否变化。基座在捕获时缺失则返回 incomplete；漂移/删除返回 failed，均退出 2，不再以空检查列表返回 verified：

```bash
python -m qwen3vl_local.audit_joint verify-inputs "$G0_ROOT/capture"
```

**当前 `run` 会返回 2**：即使资产齐全，仍明确保留“同 checkpoint 完整复现、固定更新数值/性能、四 rank 内核”未执行项。它是捕获阶段的未通过状态，不代表一定崩溃，也不应改为 0 伪装验收。不要用 `&&` 把它与下一步串联；另行执行：

```bash
python -m qwen3vl_local.audit_joint verify "$G0_ROOT/capture"

python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/capture.audit.zip"
```

**交给我的是 `$G0_ROOT/capture.audit.zip`**，无需手工复制摘要。`run` 在报告/回执生成后自动打包并核验 ZIP，再返回 G0 状态码；即使有缺资产、交叉冲突或未执行阶段，也先保留包。终端打印 ZIP 绝对路径、实际字节数和 SHA256。不能用返回 2 代表打包失败；包不存在或核验失败才需查异常。若终端启用了 `set -e`，用 `python ... || g0_status=$?` 接住状态后检查，不能直接忽略所有错误。

包内包含完整 G0 报告、用途账本、冲突/缺额明细、原合同和环境、逐文件 SHA、配置/manifest 元数据及预算允许的冻结源码。默认轻量捕获只引用原数据索引和权重，既不复制进 capture，也不复制到交接包，原始 RGB/视频也不打包；这不是可直接恢复训练的完整快照，更不是 RGB 人工目视审计包。

大小按压缩后的实际 ZIP 计算，硬上限 30,000,000 字节。默认带冻结源码；超限先仅移除源码正文，保留完整源码 SHA 清单并显式记录原因。若完整报告/逐题结果仍超限，则拒绝发布 ZIP，保留服务器原件；不截断指标/错例，不把多个包的总大小说成一个 30 MB 包。

旧 G0 已完成、或第一次打包失败修复后，可只打包，**无需重复全量回放**。输出必须使用未存在的路径：

```bash
python -m qwen3vl_local.audit_joint pack \
  --capture "$G0_ROOT/capture" --output "$G0_ROOT/g0_handoff_retry.audit.zip"
python -m qwen3vl_local.audit_joint verify-package "$G0_ROOT/g0_handoff_retry.audit.zip"
```

捕获目录及其引用的原始输入、模型数据保留服务器，不进 Git；工具不自动上传。没有 `receipt.json` 或 `verify` 失败时，先反馈报错，不能将残缺目录打成完成包。

下一步以完整交叉矩阵处理新实验准入/六事件 val 缺额，落实 E0 的固定题序、环境和四卡测量代码；再进行 E1 等价性。当前本机已发现的 13 个交叉用途物理组不能靠改旧 split 或修改 hash 消除。

## 5. 现有训练/评测入口是什么（当前容量问题解决前不要执行）

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

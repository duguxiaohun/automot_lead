# Phase3 A：固定当前 event 题域的逐点未来意图

用户最终明确：**始终沿用当前 event 的题域预测未来**。所有查询保留锚点 event 的动作范围；未来 event、RS 或原始 road_id 变化不会切换题域，也不要求旧索引在该时点仍有同事件行。原生 base 标签、动作规则、阈值及 B 的窗口协议不改。

## 时间、题域与标签

A 从 `LeadMoTPlanningDecoderConfig` 源码默认配置读取 `num_waypoint_queries=8`、`waypoint_dt=0.25`，绑定配置 SHA，预测 +0.25…+2 秒八个位置；另保留 T=0 原生动作读出，与历史固定题库比较。输出时间间隔与规则取证窗口分开；不是 +3/+6 秒。每个未来 T 重新计算速度变化和首跨，观测速度/状态不被假定保持。

- 输入仍为当前四图，当前 event、速度、导航等原生前提。未来图像、真实速度、真实动作及未来事件不进入输入。
- `LEAD_BRAKE`、`SIGNAL_FAILURE` 等纵向题域，未来继续只判断 DECELERATE/STOP/RESUME/KEEP；横向证据不足不妨碍已知纵向标签。
- `STATIC_BLOCKAGE` 等完整操纵题域，未来继续要求原生横向证据完整，包括 STOP 的原生题域准入要求。调用原生 `choice_annotation`，没有增加 STOP 绕过条件或更改规则。
- KEEP 始终相对于锚点 event 的动作范围；它不是把上一标签延续到未来。即使两个事件请求同一物理未来帧，题域目标身份也分开绑定，不能混用纵向 KEEP 与包含变道的操纵答案。
- 直接从冻结路线相应原始 meta 生成未来标签。缺帧、不可用速度、题域要求的横向证据不足、quarantine 命中等位置屏蔽，不填 KEEP、不复制最近动作。锚点到目标之间的 meta 连续性单独检查。源数据风险使用原生题域对应的取证范围。
- 预测的是各未来时刻“接下来主要动作意图”，不是每段 0.25 秒正在执行的控制状态。末个 +2 秒标签的完整横向取证可能延伸到约 +5.25 秒。未将这些动作接入 waypoint decoder。

A 的目标保存在 `a_future/trajectory_targets.json`，原始 meta、RGB、历史索引及 manifest 不重写。同一路线只加载一次原始轨迹，同一物理帧的信号只算一次，再按需要的锚点题域投影。标注产物绑定原索引、题域协议、配置、源码发布、原始 meta SHA/成员/stat、逻辑挂载和物理目标；summary 记录规则版本、时间、hostname、AUTOMOT_SERVER 及可取得的 git。纯白名单部署无 Git 时以核验的发布 SHA 为准。

同发布重启可用 `--trajectory-targets <明确旧 a_future/trajectory_targets.json>` 复用目标。校验输入及源文件变化，拒绝改 SHA 绕过；完整证明只留本机，不打入反馈包。

## 采样、loss 与覆盖验收

A 使用和 base 完全相同的独立锚点计划：event 平衡、原 action 配额、B1/累积1、三轮7680更新，无少见题或连续片段优先。所有方法均从同一 Qwen 初始权重开始，A/B 不加载训练后的 base。

每事件 loss 总质量维持原预算：一半给当前动作，保留原 action 配额比例；一半均分到该事件内不同物理未来目标，再按重复曝光次数分摊。同一物理目标在相邻锚点/重复采样中出现多次，不会反复获得完整权重。未来题域允许的动作即使没出现在本轮当前动作配额中，仍可获得未来监督。报告记录计划及实际 batch 的位置、动作、目标身份和权重，消费必须与计划匹配。

此 1:1 权重是明确的实验设计，不是已验证最优；三方法改变了提示词、监督分布或架构，不能称为纯监督密度消融。新的 trajectory 协议暂拒绝 single，原 legacy single/all 保持原对照。

默认 `--trajectory-coverage-policy masked` 明确接受未知位置屏蔽，准入条件是：

1. 当前动作标签完整，固定锚点不更换。
2. 每个事件平均每锚点至少新增 1 个有效未来监督。
3. 每个事件、每个未来位置至少有 2 个不同物理目标；同路线相邻帧仍相关，不宣称统计独立。

三轮训练计划和固定 val/test 各自检查，缺失原因在失败前写入包。此准入**放宽了旧的逐时点80%完整率要求**，保证实际有新增监督，不代表八个未来位置全部齐全或统计支持充分。`--trajectory-coverage-policy strict` 保留旧80%要求；`min-future-coverage`/`min-eval-coverage` 比例参数仅用于 strict。独立旧 dense 实验默认 legacy，三方法 suite 默认 trajectory。

B 仍为八图、五个严格四图窗口，末端间隔0.25秒，使用已实现的独立帧语言 KV 共享实验；本次没有改其输入、标签、架构和更新预算。原生 base 提示词、标签及训练算法不改。

## 本机真实数据复核

历史最优四图 val 的320条记录逐项匹配冻结320个ID。59条本机路线、7606个meta源SHA；在用户确认的固定当前event题域下，2560个未来位置中2079个可监督（81.21%），未知481个：速度窗口不完整326、完整操纵题域跨道路横向窗口115、横向缺meta40。

每事件平均新增未来监督4.531–7.875个/锚点，这批本机固定val通过masked准入。`SIGNAL_FAILURE` +2秒为9/32：其余23个速度证据不完整，不再被横向证据额外限制。低覆盖位置仍必须单独报告，不能把总体81.21%理解为每个时点都达80%，更不是模型准确率。

原统一六动作草案在同批位置有1787个已知目标；它不是最终题域，也不是服务器已经执行的模型结果。完整本地证据保留在 `checkpoints/phase3_trajectory_intent_20261020/local_val_anchor_domain_coverage.json`，轻量统计见 `qwen3vl_local/audit_joint/phase3_trajectory_intent_verification_20261020.json`。此次属于已观察val标签覆盖后的协议开发，不称盲测；未看test预测调参、未筛换固定成员。

这不是cr02源文件等同认证，也不是train/test全池或H20验收；服务器仍执行自己的准入与四卡探针。不承诺恢复历史分数。

## 服务器一次执行

保留本次已有 `phase3_base_ab_v1/result/shared_data` 及所有旧运行目录。下面命令复用旧共享数据，只生成一次新的 A 目标，不再运行原来约18分钟的共享建库过程，不复制RGB。

在本次cr02的 `/home/cruser1/lda/AutoMoT`、automot环境执行。换机器必须改用该机已有资产，不迁移或猜测目录。

```bash
bash ./sync_automot.sh && (
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  mkdir phase3_base_ab_anchor_event_v1
  nohup env GPU_IDS=0,1,2,3 SKIP_TRAIN=0 PIPELINE_CHECK_ONLY=0 \
    PYTHON="$(command -v python)" \
    bash qwen3vl_local/audit_joint/phase3_suite.sh \
    --index phase3_base_ab_v1/result/shared_data/base/frame_index.jsonl \
    --prepared-data phase3_base_ab_v1/result/shared_data \
    --forecast-protocol trajectory --trajectory-coverage-policy masked \
    --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
    --output phase3_base_ab_anchor_event_v1/result \
    > phase3_base_ab_anchor_event_v1/launch.log 2>&1 < /dev/null &
  printf 'PID=%s\n日志=%s/phase3_base_ab_anchor_event_v1/launch.log\n' "$!" "$PWD"
)
```

顺序为三组CPU准入（A生成目标）→A/B四卡显存探针→base、A、B各自三轮/固定val/test→三个独立审计包及汇总。首次失败停止，不自动重试或重跑旧模型。`--prepare-only` 或既有只检查开关仅做CPU；检查与正式分开运行时用新输出并显式引用检查阶段的 `--trajectory-targets`，避免重算。

用 `tail -f phase3_base_ab_anchor_event_v1/launch.log` 查看进度；A详细日志为 `result/a_future/preflight.log`。所有日志位于当前目录的新运行目录。结束标志是 `[suite] exit=0` 和 workflow 状态 `three_methods_completed_quality_pending`，不代表质量通过。

成功或失败交 `phase3_base_ab_anchor_event_v1/result/handoff/` 内八个文件：base/a_future/b_windows_bank/suite 各ZIP+JSON；没有handoff则交launch.log。原始证明/RGB/权重不入包。逐rank batch审计文件上限8MiB、未压缩合计128MiB、ZIP仍最多30MiB；超限不会冒报打包成功。

## 发布兼容与验证

精确兼容原 `1103ece79` 数据生产者：原六个生成函数源码摘要和其余生成依赖不变；七个已改变的训练/提示词/评测消费者从字节不变集合分离，原SHA留在兼容说明中。未知生产者仍拒绝，旧receipt不修改。

CPU回归包括原生题域逐项一致、跨道路纵向有效/操纵未知、同一未来帧不同题域不串标、九读出真实小Qwen反向传播、原生训练消费与checkpoint、三轮计划及固定评测形状、重复目标权重、四rank共享报告写入保护及旧数据兼容。最终数量和源码绑定见轻量回执。未进行H20训练或性能恢复验收。

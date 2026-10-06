# v30：过滤后的全量老师、窗口风险准入和加权采样

合同 v30/data_v30，老师来源版本 v5。释放/停止义务的数值阈值未调；严格批准政策 v2 保持，真实批准表仍为空。所有产物重新生成，未替换旧产物的哈希。

## 修复范围

候选扫描复用项目统一的异常时长过滤器：4 Hz、严格大于 360 张 JPG 时排除，保留 BlockedIntersection / ControlLoss 例外。过滤器源码纳入依赖摘要。另排除整条路线缺 metas 的情况；排除路线、原物理组、原 split、原因和帧清单保留在候选账本，不能把排除解释为无事件或 NO。

扫描 9,712 条，异常时长 963 条；其中 848 条也缺 metas。另外 135 条时长合规但完全缺 metas。合计排除 1,098 条，纳入 8,614 条、1,062,401 帧，训练 7,268／验证 658／测试 688。保留路线 split 全部与 v29 一致。独立预约从 109 条过滤为 93 条（13 条异常时长、3 条缺 metas）；被排除的 16 条仍隔离在原 val/test，不回流训练。旧抽检扩展文件作为历史筛选证据保留，新活动预约由 v10 清单给出。

风险对老师监督改为按登记证据的帧范围否决。检查包括完整七帧因果几何/RGB，以及更早的 STOP 履行证据，不只检查模型收到的两张或四张图。无可定位帧的登记仍按未解决处理；窗口外仍必须通过所有路线共用的自动几何、可见性和不连续检查。登记帧的包络不是“其余时间都安全”的证明，自动异常检测也未经过独立认证。人工标注的原逐帧审核合同保持。

弱监督默认 `event_weighted`：事件等额，proceed/depart/enter/return/complete 权重 8，recover_follow 权重 4，其他边权重 1；同边同时有 YES/NO 时等额，缺答案不伪造。共享物理帧 cap8，单题每轮 cap4，可用 `--max-question-repeat` 显式配置。默认预算按题池和容量计算，显式不可行预算仍报错。实际曝光次数用于后续轮次轮转，恢复绑定题池、权重、重复上限、seed、world 和预算。原纯人工库默认采样不变。

新增独立 `teacher_evaluation` 参考包和 `evaluate --teacher-references`。它只接受冻结的 val/test，拒绝作为训练数据加载；与老师的一致率单独报告，人工准确率和模型选优不包含这批答案。所有 UNKNOWN 排除并记账，产量不等于正确率。

建库遇到相同可见输入的人工/弱标签重叠，保留人工题并丢弃重复弱题；弱题内部相反答案全部隔离。没有自动把冲突改成 NO。

## 本轮审计性质

本轮做代码、源文件身份、全量程序回放和数据构建验证，没有新增 RGB 目视或独立人工参考。用户报告的三条可精确定位的新曝光已显式 train-only；OppositeVehicleRunningRedLight/Town01 缺完整路线 ID，记录为待补充，未猜测某条路线，也未把整个 Town 改成训练。独立图像只由程序打包，没有打开查看或用其逐题结果调整规则。

完整逐帧 RGB 审计的旧缺口保持。没有执行 GPU 梯度训练、DDP 实训、CARLA 闭环或独立老师准确率测量。

## 全量验收结果

最终回放 8,614 条／1,062,401 帧，生成耗时 1244.7 秒，逐帧记账核对通过。`complete_causal_production=true` 表示所有合格帧均有处置，不表示每帧有可靠标签或十类老师均已实现。

| 图数 | 训练题 | 弱监督 | 训练物理路线 | 弱监督物理路线 | 已核验 RGB 的题数 | 七轮实际覆盖（world1 / world4） |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 81,377 | 80,934 | 1,914 | 1,888 | 82,375 | 711 / 711 |
| 4 | 81,377 | 80,934 | 1,914 | 1,888 | 82,375 | 711 / 711 |

原 443 道人工训练题逐行保持，val273/test186/review539 逐字节保持。RGB 检查为文件及解码像素身份核验，不是人工视觉审核。弱题风险窗口否决、可见输入冲突、重复与路线限额见各数据目录 `teacher_compilation.json` 和 `manifest.json`。

795 项专项测试通过，2/4 图真实 `run_full_pipeline.sh MODE=check`、各七轮采样通过；另核验 world4 的七轮覆盖。严格 `formal_data_ready=false` 保持，弱监督 `trainable=true`。

留出老师参考每模式 val 8,736、test 8,369，已通过独立参考包加载校验；仅用于老师一致率。93 条活动预约生成 933 张 2 图盲审卡，实际来自 57 条有抽中题目的路线，12 个规则类，最佳情形可达严格批准仍为 0 类。卡片全部未填写。

一次建库在因果源文件校验处拒绝。独立重查 81,000 条候选涉及的 316,794 个文件未发现持续性差异；使用显式绝对数据根目录和只记录诊断的包装器重新构建，所有原校验保持。首次失败没有具体文件定位，原因未复现，不宣称已证明来源从未变化；失败日志与重查结果保留在产物目录。最终验收指重建后的产物。

## 采样检查发现与修正

首次分配器虽然限制单题重复为 4，仍会在有大量未用题时提前重复。已改为先开放不同题目的容量，只有缺额时逐级开放重复，保留共享帧容量与历史曝光轮转。新增充足题池不重复、仅稀缺格重复的回归测试。在首次同一题池上，预算 400 不变，七轮覆盖从 424 提升至 711。该对照保存在 `phase4_v30_full_sampling_draft`，包含合同绑定源码快照；最终产物按修正源码重新全量回放、建库，没有改写旧 manifest 的 SHA。两遍全量回放逐路线答案数量和处置数量一致。

最终默认预算为每轮 400 次，主要五类边占 378 次。R-E5 仅有 5 道 YES，单题 cap4、同边 YES/NO 等额使其最多提供 40 次；十事件等额因此最多 400 次。题池扩到八万道并不能消除这个数学容量限制。七轮仍有 80,666 / 80,666 道（2/4 图）未被抽到。扩大有效训练规模需要补齐稀缺事件，或另行明确采用事件子集/不同配比；本版没有偷偷放松 1:1 或重复上限。

## 使用已验收产物

从项目根目录运行，二图示例；四图把 `data2` 改为 `data4`，并使用新的输出目录。下例开始实际 GPU 训练，需主机的完整模型与 GPU 预检通过：

```bash
PYTHON=/tmp/automot-qwen35-env/bin/python \
DATA_ROOT=/home/codon/automot_lead/AutoMoT/lead_data \
DATASET=/home/codon/automot_lead/AutoMoT/checkpoints/phase4_v30_full/data2 \
OUTPUT_DIR=/home/codon/automot_lead/AutoMoT/checkpoints/phase4_v30_full/run2 \
bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh single
```

默认已选择 `event_weighted`，不再固定传 `--epoch-samples 100`；重复上限可用 `--max-question-repeat` 显式设定。保持现有约束时，不要把更大的显式预算当作增加数据，超出容量会报出诊断。

训练后在 `evaluate` 命令加 `--teacher-references /home/codon/automot_lead/AutoMoT/checkpoints/phase4_v30_full/references2`（四图用 `references4`），将老师一致率附加到人工指标报告；人工选优不受影响。盲审入口：`/home/codon/automot_lead/AutoMoT/checkpoints/phase4_v30_full/independent_review2/index.html`，审核者只使用公开目录，私有答案留在同级 `independent_private2.json`。

完整计数与验证摘要：[thirtieth_audit_verification_20261005.json](thirtieth_audit_verification_20261005.json)。本轮未启动 GPU 训练，也没有新增人工标签或独立准确率结论。

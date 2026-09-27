# Phase3 / Action 稳定版本约定（2026-09-27）

## 当前规则：效果基线与工程修订分别管理

**语义/效果基线仍为v23，当前工程发布为 `v23_io1`。**
不应把“保持已验证的模型行为”误解为“所有工程代码必须回到历史字节”。

| 改动类型 | 接入稳定版的要求 |
| --- | --- |
| ESTALE重试、原子发布、缓存完成状态保留、同合同恢复等纯工程修复 | 代码审查、故障回归、正常路径输出等价；可回移，不要求GPU分数提升 |
| 提示词、标定、动作定义/阈值/窗口、标签、过滤/split、采样分布、优化/模型条件 | 保持历史验证版本；有相应效果证据并正式晋升后才换默认 |
| 混合改动或无法证明正常路径语义不变 | 拆分提交/修订；语义部分按效果改动处理，不能因为名称叫bugfix就放行 |

本轮补回Phase3内部文件发布修复，Action外层的ready标记/哈希复核/持锁续发继续保留。
新增工程快照 `phase3_releases/v23_io1`，不修改原 `v23`。提示词、轨迹规则、采样和所有标注资源均未改。
[工程维护证据](V23_IO_MAINTENANCE_20260927.md)记录来源、配对输出和训练机同步边界。

`phase3_release promote` 根据review的 `decision` 分流：`promote` 要求效果提升证据；
`engineering_fix` 要求 `fix_kind=estale_publication`、`semantic_outputs_unchanged=true`、
实际 `regression_runs`、逐项 `changed_files`、当前baseline、candidate/report SHA及审查确认，
同时校验语义基线相同，保护prompt/规则/采样/标注资源。仅允许已审查的builder/扫描器/来源哈希接线差异；
自动检查不能证明builder内部没有语义变化，仍需审查具体diff和配对产物。
其他工程修复应先定义对应审查范围与回归，不能滥用ESTALE分类。
工程revision不提高模型版本号；三个Action入口统一跟随登记的新revision，运行中进程仍固定原选择。

当前mapping hash为 `6beab3826605f2c10324b789a187fbb1955528197a66d662c4f970c36b17fe6a`，
由v23源码加I/O修复及filesystem依赖得到；Phase3和Action稳定构建器一致。
源码身份变化仍需复核/重建索引、新run，不伪装旧hash。后续效果晋升的baseline填写实际登记 `v23_io1`。
本次工程修订908项相关CPU回归通过，包含15项原v23/修订产物等价与ESTALE故障注入；未真实挂载/GPU验收。

下文保留初次回退记录；其中精确恢复80个文件、原始hash与1064项测试均描述工程修订之前的状态。


当前稳定基线为 **v23**，来源提交 `b433aa605251130e755967c53d89f98966a5525f`。
按用户要求，先记录 v24，再恢复 Phase3，并让 Action 始终跟随经过验证、正式晋升的稳定版本。
“代码最新”“完成审计”“CPU 回归通过”均不等于“模型效果最好”，不能据此更新 Action 默认。

## 本次记录与回退

- [四包完整审计](../sft_new_loop_phase3/AUDIT_COMPARISON_20260927.md)：v24 四组 exact 为
  64.22/63.73/63.24/64.12%，v23 为70.24/71.32/68.48/67.39%；macro F1也全部下降。
  各版本测试题及路线不同，这些是历史表观差异，不是同题配对因果结论。
  v23是用户指定的稳定基线，并非宣称它在所有单项指标、所有旧合同下均为冠军。
- 本机回退前完整已追踪源码在仓库根目录
  `checkpoints/phase3_version_archive/v24_20260927_before_v23_restore/tracked_source.tar`，
  同目录 `record.json` 记录来源提交、归档SHA和审计SHA。原权重、审计包与数据仍保留；没有上传。
  来源为 `0cf7ac19dc9b2ea6936263105e0df1bd4b63fb69`，归档SHA为
  `ebbd3dc2a2b873f655e244b4a920a5c19d89ba44e49e614b7880a1734a96d291`。
- Phase3 的 Python/shell 恢复到上述v23提交，包括 prompt、构建、训练、评估和采样。
  默认4rgb+choice、data_v23、`v23_grounded_stage`；原始mapping hash为
  `1d3feee7fc45ae69760b3fcc6b01361cfa80e57afe688b43d2f542fa728f5188`。
  不混入后续smooth_cap、完整训练池或INVALID来源内回流；Phase3内部ESTALE修复已按上方工程规则补回。
- Action 对应采样接线恢复v23：event维持事件1:…:1:2；action-balanced采用事件内主要动作容量回流。
  保留此前独立指定的每轮116256次/cap11；显式预算0按容量计算，容量不足必须报错。
  保留优化器/7轮计划、token/单图开关和Action发布缓存的ESTALE恢复。
  这不是把整个仓库或所有Action文件按日期回滚，也不代表116256/cap11已获得新的GPU验证。

## 必须遵守的规则

1. `sft_new_loop_phase3` 是可继续研究的工作目录。Action生产代码禁止直接导入它，
   也不能直接读取它的builder、规则、prompt、共享采样源码来决定默认行为或执行指纹。
2. `phase3_releases/stable.json` 是唯一稳定版登记；目前选择 `phase3_releases/v23_io1`，语义基线为v23。
   `phase3_stable` 命名空间只暴露该快照。主线、qwen_simple、bev_only全部共用。
   缺失、损坏或哈希不符必须失败，不能回退到实验目录或“编号最大”的版本。
3. 已发布快照不可原地修改。对效果/语义候选，v24/v25即使改完并通过单元测试，也只能保持候选状态。
   必须有真实训练/评估证据、同题配对比较、预先确定的主要指标和关键分组回归复核，
   证明相对当前稳定版改善并记录审查结论后才能晋升。Phase3准确率不能替代Action轨迹指标：
   影响动作条件时应同时检查Action支持量、条件合同及相应轨迹回归。
4. 晋升一次登记后，三个Action入口的**新进程**自动使用新稳定版；无需分别改版本号。
   已启动进程及其builder子进程固定原选择。旧checkpoint仍受原合同约束，必须用对应源码恢复，
   不能把晋升理解为给运行中的训练热换标签、切断点版本或绕过校验。
5. 分模块挑历史“最好”也必须测试组合；禁止把不兼容版本拼接后当作已验证稳定版。

## 查看与晋升

在 `AutoMoT/` 下查看实际选择：

```bash
python -m qwen3vl_local.action_prior.phase3_release show
```

未来候选先在独立 `phase3_releases/vN/` 中冻结完整运行依赖和标注资源，并生成
`release.json`；结构参考v23，必须记录原提交、原始文件SHA、迁移后文件SHA、外部依赖SHA、
mapping合同和数据集名。内部导入改为 `phase3_stable`，检查文件深度相关路径。
不能只复制prompt，不能机械复制v23的hash；新增依赖也必须进入快照/manifest。
封装本身必须做与已评估源码的行为等价检查，并跑三入口回归。

v23快照仅适配导入命名空间和目录深度；mapping/rule身份保留原值，但加载器首先校验快照
全部实际字节及外部依赖，另以release manifest SHA进入缓存/训练合同。此做法不是关闭hash检查。
未来修改语义必须产生新的原始身份和新release。

审查文件示例（数值只是格式示例，不是可用晋升证据）：

```json
{
  "decision": "promote",
  "release": "v25",
  "baseline_release": "v23_io1",
  "manifest_sha256": "候选release.json的SHA256",
  "same_evaluation_cases": true,
  "regressions_reviewed": true,
  "approved_for_action": true,
  "reviewer": "实际审查人及记录",
  "evaluation_runs": ["实际baseline运行", "实际candidate运行"],
  "acceptance_metric": {"name": "预先确定的同题主要指标", "baseline": 0.60, "candidate": 0.65},
  "report_path": "相对AutoMoT的实际评估报告路径或绝对路径",
  "report_sha256": "评估报告SHA256"
}
```

指标按越高越好录入，ADE等越低越好指标使用明确命名的负值转换，并在报告保留原值。
报告需提供同题身份/预算/条件、各组及拒绝率、路线支持、Action影响、回归结论和选择理由。
程序检查必要字段、报告/候选哈希、当前baseline和主指标严格提升；它不能代替人核实报告真实性，
也不能用随意指定的一个改善分数掩盖关键分组退化。当前v23是用户明确接受的历史基线例外，
不追造不存在的同题实验；新版本不得复用该例外。

```bash
python -m qwen3vl_local.action_prior.phase3_release promote \
  --release v25 --review /实际路径/review.json
```

命令校验后加锁、原子替换登记。拒绝时保留原稳定版。晋升不修改实验目录、旧权重或现存数据；
缓存按稳定release身份选新路径，显式索引继续严格检查。同步其他服务器时须同步完整release、
登记及所引用的评估报告，不能只同步stable.json。

## 验证边界

历史v23的1779个开发路线隔离组保留原合同。后续v24审计曝光清单/报告仍留在原目录，
回退不会把看过的路线变回盲测；新版本晋升必须采用固定可比的评估方案，并对未曝光路线另报结果。
本轮未运行新的GPU训练、全生产数据重建或真实多卡验收，不能宣称恢复代码后分数已回升。
Phase3内部ESTALE重试已恢复；有限重试耗尽或原始输入读取故障仍可能中断构建。

本轮相关CPU回归 **1064项通过**，含真实v23 builder小样本发布/复用、三入口采样/恢复、
稳定来源隔离、快照损坏拒绝、无证据/退化候选不得晋升；80个Phase3 Python/shell文件与基线字节一致，
原mapping/rule身份一致，变更shell语法及diff空白检查通过。
更广测试首次1516通过、41失败、1跳过；修正本次相关问题后，这41项复查为4通过、37未通过。
剩余包括缺只读runner、peft、matplotlib，以及4项预检fixture/可视化参数预期问题；
后4项已用回退前归档源码独立复现。未补造runner、安装模型依赖或放宽生产校验。
日志及精确记录见上述归档目录的 `restore_validation/` 与 `restore_record.json`。

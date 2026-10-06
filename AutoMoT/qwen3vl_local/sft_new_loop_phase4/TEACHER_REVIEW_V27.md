# v27：规则证据修复与可执行盲审流程（2026-10-04）

本轮完成运动证据修复，以及 `prepare → import → assemble → approved teacher dataset` 的真实工具连接。没有签发真实规则批准，也没有启动正式训练。

## 具体修复

- UE1 的间距增长还必须有前车沿自车前向运动证据。自车倒退造成的间距增长不能解释为前车驶离。
- 纵向执行回执使用 ego_matrix 将当前/历史世界位移投影到起始车体坐标，要求前进位移、侧向位移上限和朝向一致。倒退、侧移、明显转弯不能仅凭位移长度确认 proceed 已执行。仍是追踪器证据，不是 RGB 可见后继证明。
- UE1 缺道路/车道身份不能靠 None==None 建立前车关系。
- 条件值只接受 bool/None，避免 Python 把 0 当 False；条件集合和来源 SHA 格式也严格校验。
- 完成的实例经过连续三次可见、未触发同类冲突的观察后，同一 ID 可以建立新的事件实例。缺图、断帧、消失、异常及待复核不触发重开；终止原因另行记录。
- 抽检/编译去重键使用有序 RGB 内容文件摘要和状态对，不再包含源文件路径、帧号。跨文件复制的同一可见题不能增加批准样本数。训练最终仍用既有解码像素身份检查。
- 逐规则批准要求 YES、NO 各自覆盖至少20条不同物理路线，防止总路线数达标但某一答案实际上只来自单一路线。原100总判断、30转移附近判断及Wilson下界保持；样本间相关性仍意味着逐题Wilson不是路线级安全保证。

## 审核包与审批

`teacher_review.py` 按规则类、答案、场景、Town、不同路线选题，优先因果已观察到的答案变化。默认每规则类200题、每物理路线10题；预算不足、未选数量和路线缺额保存在私有统计中。这是分层多样性抽样，不是无偏总体准确率估计。

公开目录仅有实际2/4张RGB、当前/下一状态文字、待填写答案模板和本地HTML。它不包含老师答案、规则事实、候选参与者ID、场景名称或转折建议。私有快照包含这些字段，**不得把私有快照或生产JSONL一并交给盲审者**。

每张卡都必须填写 YES/NO/UNKNOWN、证据说明，以及事件身份、当前状态、当前输入可判性的确认。任何一项未确认时只能填 UNKNOWN。不能删掉难题；导入要求完整覆盖原样本，重复、漏答、空证据、错包、改变转折分层均拒绝。审核者独立声明仍是声明，代码无法证明其现实身份或真的看过图。

导入后按规则类分开开发和独立参考，再组装批准清单。缺少任一侧的规则类会列为未配对，置信度不足的类不会获批。独立参考仍必须来自冻结的老师抽检路线；这些路线不能流入开发或模型选优。

## 本轮实际产物

- 全目录清点仍为9712路线、1,637,680帧，train/val/test为8166/765/781。
- 开发实跑17路线/1615帧，212张审核卡。
- 冻结独立实跑40路线/9759帧，152张审核卡。**只有程序处理和材料生成，未查看这些RGB、未填写参考、未用其答案调规则。**152是跨多规则类、含2/4图的卡数，不能视为任何单类已达到100判断/20路线。
- 新增人工标签0，真实规则批准0，新增RGB目视审计0，新增完整序列0。
- 736项专项通过。2/4图真实pipeline check及每种902监督+539待审RGB解码核验通过。
- train443/45路线、val273/test186，train/val/test/review四JSONL与v26逐字节相同。有限数据训练 `trainable=true`；`formal_data_ready=false`。

本机审核包：

- 开发：`/tmp/phase4_v27_work/development_review/index.html`
- 独立：`/tmp/phase4_v27_work/independent_review/index.html`
- 填写同目录 `decisions.json`；提交给独立审核者的是该公开目录。
- 私有快照分别是 `/tmp/phase4_v27_work/development_private.json`、`independent_private.json`，由组织审核的人保管。

`/tmp` 产物不是持久存储；迁移前按下列命令在指定新目录重建。源码内保留轻量验证报告 `twenty_seventh_audit_verification_20261004.json`。

## 重建与导入命令

从 `AutoMoT` 目录运行。先按 `RULE_TEACHER_V26.md` 的生产命令重建**当前v27**候选和生产索引，旧源码产物不能改hash复用。使用项目依赖环境的Python。

```bash
python -m qwen3vl_local.sft_new_loop_phase4.teacher_review prepare \
  --candidate-pool /path/v27_candidates.json \
  --production-index /path/v27_production/index.json \
  --purpose independent --private-snapshot /path/private/independent.json \
  --public-dir /path/reviewer/independent --data-root lead_data
```

开发包使用 `--purpose development` 和另一套目录。独立准备只选冻结的老师抽检路线；开发准备只选train，不会混入留出路线。

审核者完成公开目录中的 `decisions.json` 后：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.teacher_review import \
  --private-snapshot /path/private/independent.json \
  --decisions /path/reviewer/independent/decisions.json \
  --output /path/imported_independent.json

# 开发参考按相同命令导入为 imported_development.json，然后组装。
python -m qwen3vl_local.sft_new_loop_phase4.teacher_review assemble \
  --development /path/imported_development.json \
  --independent /path/imported_independent.json \
  --rule-author actual_rule_author \
  --output /path/reviewed_registry.json
```

组装会生成批准清单和 `.report.json`，不会强行批准不达标的类。接到数据构建：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --candidate-pool /path/v27_candidates.json \
  --production-index /path/v27_production/index.json \
  --teacher-registry /path/reviewed_registry.json \
  --annotations qwen3vl_local/sft_new_loop_phase4/reviewed_state_pairs_v9.json \
  --data-root lead_data --output-dir /path/new_data2 --rgb-mode 2
```

4图使用另一新目录和 `--rgb-mode 4`。原pipeline也接受 `PRODUCTION_INDEX`、`TEACHER_REGISTRY`；`MODE=check`可做工程检查。

## 保持的边界

合同v27/data_v27、老师v2、批准政策v2、抽检冻结v7；40条路线及split保持。旧真实标注v9、冻结taxonomy/prompts/calibration/observation、控制器、安全许可、采样、训练及选优源码保持。
自动老师仍只支持UE1/UE4；其余八类、自动catchup视觉证据和真实recover_follow正例支持尚未补齐。没有进行全9712路线生产、GPU/完整4B/DDP/CARLA验收。

下一步是由独立审核者完成已生成的材料；不应靠增加测试数量或直接修改formal标志替代规则误差测量。

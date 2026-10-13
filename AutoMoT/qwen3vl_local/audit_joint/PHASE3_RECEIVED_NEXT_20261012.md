# Phase3 交接核验与自动建库下一步

> 2026-10-13：正式建库交接已收到并通过，**不要重复本页建库命令**。下一步直接复用正式索引进行首轮，见 [正式准入与首轮步骤](PHASE3_ADMISSION_FIRST_EPOCH_20261013.md)。

## 收到的文件说明了什么

Downloads 中只有 `phase3_index_report.zip`、对应 package JSON 和一份同内容的报告 JSON。本轮 ZIP 为 **25,977 字节、2 个成员**；外层 SHA、成员 SHA、CRC 通过。468 个源码 SHA 与本机及 Git `4e59862ed01ef9dae8266024e1e210a2fc074f36` 逐项一致。服务器同步正常，不再运行 readiness 或重复本次路径盘点。

报告是 `index_not_selected`，`paths=[]`、`truncated=true`，没有训练。它是限定深度、排除部分目录且不跟随目录软链接的盘点；不能解释成“全服务器确实没有索引”，也不是正式索引准入成功。结合此前原索引误删且没有可用副本的背景，下一步直接在新目录生成正式索引，不需要恢复旧 adapter 才能从基座新训。

## 目前的开训判断

| 项目 | 当前结论及下一步 |
| --- | --- |
| 代码同步、环境 CPU 验证 | 已关闭，不重跑 |
| Phase3 自动标定/入口 | 现有构建器、原始动作复算和正式 train/eval 准入可复用；缺实际正式索引结果 |
| 训练事件间 1∶1 | 原生十事件每事件 1024 次；重复数、独立题数、物理组数分开报告；binary 的 INVALID 另报 |
| val/test 事件间 1∶1 | 实际过滤后的十事件固定无放回清单，缺事件拒绝；不承诺沿用旧池配额 |
| Phase4 正式池 | v52 的 301588 train、val 每事件37/test每事件18是历史有效证据；不能替代 v53 正式池准入 |
| U-E5 | 已覆盖本机89条合格训练路线；现有 v53 proceed 711 NO、290 UNKNOWN、0 YES。需要可观察的真实正例来源，重复旧回放不能补齐 |
| U-E2/U-E7 | 历史正式池 return/complete 正例有支持但物理组稀疏；不把题数等量称为泛化已通过 |
| 提示词 | 本轮无新明确缺陷，不改文本。新报告绑定 candidate 提示词及四图/短两图 SHA |
| 联合验收 | Phase3/Action 正式资产及跨任务隔离仍未完成；不阻止已满足独立条件的 Phase3 开发基线 |

无需人工逐题标注全库。规则生成、异常隔离、保留 UNKNOWN、原始信号复算与覆盖门继续自动执行。自动检查和既有抽样 RGB 证据不能保证十事件全部真实场景都无误。尤其事件题数等量不能代替 U-E5/proceed 的正反条件支持。

## 新增 CPU 建库交接入口

`server_phase3_prepare.sh` / `.py` 只做 CPU 数据准备，不加载权重、不启动训练：

1. 检查471文件发布清单，确认 `lead_data` 和正式 `*_result.json` 存在；不用局部审核缓存代替正式来源。
2. 只允许新输出目录，原生 `build_dataset` 自动生成一份索引和 manifest，保留冻结开发路线隔离；不复制 RGB、Phase4 回放或权重。
3. 用正式 candidate 准入、binary/choice 的原始 meta 与标签重算、RGB路径存在检查、时间边界统计验收。
4. 使用原生采样器检查 train 每事件1024、val loss最多16/生成最多32、test按最小可用事件容量等量；固定 seed=20260904、无放回评测、单卡/四 rank 全局题序。保存固定评测 ID 清单。报告四图与 **`2rgb_short`** 的输入及提示词合同；不走历史 `four_runs run` 的首尾两图/旧实验复验流程。
5. 核对索引、manifest、collection内容和路径绑定前后不漂移。完成后是 `cpu_data_prepared`，不是 GPU 或联合 G0 验收通过。
6. 保留20GiB可用空间，轮询限制本次净写盘约8GiB；这是停止保护，不是容量保证。单线程数值库、默认2个CPU worker。失败和中断仍打包固定白名单诊断、日志尾部，ZIP严格≤30,000,000字节。

**本入口随本次白名单发布。通过既有 `sync_automot.sh` 同步后执行下面命令。** 不依赖服务器遗留 Bosch Git。

在服务器 AutoMoT 目录、现有 automot 环境中，执行一次：

```bash
OUTPUT_DIR="checkpoints/phase3_prepare_$(date +%Y%m%d_%H%M%S_%N)" \
BUILD_WORKERS=2 \
bash qwen3vl_local/audit_joint/server_phase3_prepare.sh
```

如果正式原始数据或 collection 位于别处，在同一命令前加实际的 `DATA_ROOT=...`、`COLLECTION_DIR=...`。默认分别为 `lead_data`、`keyframe_filter/collection_output`。缺失时交错误包，不复制局部缓存冒充全源，也不要覆盖旧目录。

交回终端最后打印路径中的两个文件：

```text
phase3_prepare_report.zip
phase3_prepare_package.json
```

无需上传完整索引、RGB 或权重；完整索引保留在该输出的 `data/frame_index.jsonl`。成功包提供自动标定复算、正式隔离、条件分布及实际均衡计划。验收后按 Phase3 保存策略核对首轮空间预算，再从基座分别运行短两图/四图一轮并做保存重载和固定 test；不等待 U-E5，也不直接使用历史四组三轮总入口。Phase4 待真实支持缺口解决后再一次生成 v53 正式池并验收，不重复旧 v52 五小时建库。

## 本机验证范围

- 214项相关回归通过，29.16秒，包括9项新增入口测试；原生索引读取与采样的合成集成测试没有替换读取器，覆盖缺事件、四 rank 题序及评测去重。
- 原生新CLI：471源清单 matched；缺正式 collection 退出2，生成26,279字节、3成员诊断ZIP，外层及成员SHA/CRC通过。未冒称服务器正式建库已成功。
- shell语法检查通过。没有新增生产标签、阈值或提示词修改；审计阶段没有新RGB目视、全源重放或GPU/远程训练；随后按用户授权发布白名单。
- 完整本机回执在 `checkpoints/phase3_received_20261012/`；轻量收件核验为 `phase3_received_verification_20261012.json`。本轮未重复已有2368项全套回归。

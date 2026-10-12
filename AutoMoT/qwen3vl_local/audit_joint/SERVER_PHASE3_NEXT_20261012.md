# 本轮发布后：Phase3正式索引交接

沿用已经验证的`sync_automot.sh`（源仓库`duguxiaohun/automot_lead`）覆盖同步；不要通过服务器遗留Bosch `.git`判断版本。本轮发布清单`release_sources_v53_20261012.json`含468文件，以下入口自动核验源码。旧Phase4 v52题库保留，当前不重复readiness、旧fit、五小时建库或GPU训练。

## 1. 同步后在AutoMoT目录运行

激活现有`automot`环境。**如果还没有确认Phase3正式索引路径，直接运行：**

```bash
ulimit -S -c 0
P3_REPORT="checkpoints/phase3_index_handoff_$(date +%Y%m%d_%H%M%S_%N)"
python -m qwen3vl_local.audit_joint.server_phase3_index \
  --output "$P3_REPORT"
```

这是默认推荐路径：仅核验源码及查找checkpoints内深度≤4的`frame_index.jsonl`路径，不读取权重/RGB、不重建数据。不会自动选中某份历史诊断索引。`index_not_selected`、退出2表示尚未指定正式索引，不是训练失败。外部挂载/更深目录/目录软链接未遍历，不能据此认定文件已丢失。

**如果已经明确正式索引路径，用以下命令替代上面的Python命令：**

```bash
# 先将P3_INDEX设置为实际完整frame_index.jsonl路径；同目录必须保留manifest.json。
python -m qwen3vl_local.audit_joint.server_phase3_index \
  --index "$P3_INDEX" \
  --prompt-variant v23_rgb_stage_candidate_20261006 \
  --action-output-mode choice \
  --output "$P3_REPORT"
```

每次使用新的P3_REPORT目录。只运行适用的一种，不需要先盘点再立即猜路径核验。若索引明确不存在，交缺失态包即可，不必先重训或恢复旧adapter。

## 2. 交回两个文件

```text
$P3_REPORT/phase3_index_report.zip
$P3_REPORT/phase3_index_package.json
```

两者都位于上一步终端打印的output路径；请上传实际文件，不只贴终端摘要。ZIP严格不超过30,000,000字节、生成时核验外层及成员SHA。包含源码核验、路径盘点或索引准入结果；指定索引且manifest可读时还包含manifest。权重、RGB和完整索引不打包。

- `index_admitted`、退出0：这份索引的合同和用途隔离通过；仍需审阅正式来源、实际采样与训练预算后安排首轮，不等于联合G0已通过。
- `blocked`、退出2：交包，其中列出缺manifest、开发曝光、跨split重叠、提示词/来源漂移等具体原因；不要删manifest或改SHA。
- `index_not_selected`、退出2：交包，由候选路径确定正式资产，或确认需生成新正式索引。
- `source_mismatch`、退出2：交包中的文件差异，用既有同步方式补齐；不要求重跑环境readiness。

入口不触发模型、训练、全量题库构建；有正式索引时主要耗时是全索引CPU检查及前后SHA，未实测服务器耗时，不给固定分钟保证。

## 本地并行标定工作

本轮继续查看Town13两条训练路线各七帧：1148_0 f58–64、1159_1 f77–83，共14不同RGB，42份原始源SHA核验。黑色来车均从前方驶过左下并离开画面；截至原生实例截止，当前走廊clearance为True，但native_visible_departure仍False。它们没有提供足够证据把UNKNOWN改为YES。保留原标定/阈值/提示词，非G2盲审、不冒称全源标签准确率。

29项交接工具及正式入口专项通过（含3项新工具测试）；已有完整2368项回归结果保留，不重复全套。真实CLI合成合法/缺失两类均生成并核验小包。完整本机证据`checkpoints/phase3_publish_rgb_20261012/`；本地标定可与服务器数据交接并行，Phase3正式数据验收后可独立推进，不等待U-E5。

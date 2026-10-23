# 四图独立 base 增步版与新版单帧 six

**当前入口已更新为允许有限重复的 base bounded_v3，正式全池 CPU 已通过。** 请使用 [新版运行说明](PHASE3_BASE_BOUNDED_RUN_20261023.md) 中的 `phase3_compare.sh` 入口。本文的 coverage_v2 保留严格去重合同，不自动改成 v3。

**2026-10-23 复核：下列旧后台命令不能作为当前可开训保证。** 用户真实日志和本机全池均复现 base 动态切入容量不足；新版完整训练池仍只有 925 个有效 train 帧，不能满足 1,280 个不同帧。旧 study six 也不代表当前 single v3。当前状态、修复与证据见 [启动复核](PHASE3_BASE_STARTUP_REVIEW_20261023.md)。下文保留历史预算与发布记录。

用户已明确放弃 sequence，要求补齐四图 base 的增步、未见数据优先与重复限制。新入口为 `phase3_base_coverage.sh`，显式选择 `phase3_launch --method base --base-budget coverage_v2`。旧 base/单帧入口默认配置保留，不把它们冒称新版本。

## 合同与预算

- 四图 base：四张历史 RGB、原生六动作单题、原提示词/标签/单题 loss、每轮重新产生原有选项增广。独立锚点选题、打乱顺序，无 sequence 连续片段优先。
- 十事件每轮各 1280，四卡 B1、累积1、三轮：每轮 3200 更新，总计 9600；warmup2500、lr1e-5。每轮12800次呈现，合计38400。
- 优先未曝光物理帧，再考虑路线/4秒片段覆盖和曝光次数。同一物理帧跨事件共享容量，每轮最多一次、三轮最多三次。类别偏好为一半自然分布/一半平方根分布，权重上限4，只在新颖性相同层内起作用；不保证动作严格等额。容量不足拒绝，不重复补齐、不自动减配额。
- CPU 在训练前投影三轮计划，检查原生索引、固定val/test成员和全部实际计划输入形状。原生完整epoch只有全rank优化和消费核验完成后才推进曝光；本入口不提供中途或旧实验续训。
- 四图真实GPU probe后从基座独立训练，周期loss/生成验证间隔3200。保留原base“最终第三轮模型test”的政策，不以test选模。训练完成核验9600、38400呈现、三轮事件/逐rank ID/曝光与CPU计划一致；保存最终adapter文件SHA与增步合同，test前重验。
- 新版单帧使用 `geometry_study` 的 `six`：保留动作含义、已确认变道统一当前方向参考、离线历史允许后续证据；同为9600更新，但使用study共同池及验证选模。**base与six并非同一训练成员、同一预测任务或同一选模政策**；此处是同更新预算的开发对照，不能宣称完全等信息消融或直接按准确率排名。
- 原生261源通过已有发布兼容文件在独立缓存中核验，服务器不需要Git历史；不更改原训练器/旧release SHA或geometry冻结源。新工作进程显式装载增步适配器，合同写入job、原生checkpoint配置与最终adapter回执。缓存损坏拒绝复用。

## 两组全流程

在任意当前服务器的已有 AutoMoT 根目录和可用Python环境执行。不再次拼接AutoMoT、不填写资产占位符。默认项目 `.venv/bin/python`（存在时）或当前 `python`，数据 `lead_data`、模型 `checkpoints/Qwen3.5-4B`、标注 `keyframe_filter/collection_output`。

```bash
bash ./sync_automot.sh
```

四图base从兼容公共数据准备/校验开始，随后CPU输入准入、四卡probe、三轮训练、验证汇总、最终test、自动报告打包：

```bash
(set -o noclobber
 nohup env GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_base_coverage.sh \
   > phase3_base_coverage_v2.log 2>&1 < /dev/null &
)
```

若只先检查CPU数据/容量/输入，追加 `--prepare-only`。计划输出为独立 `run_base_coverage_v2_plan`，不启动GPU、不占用正式run。正式命令仍执行自己的必要核验；已有计划/正式输出均不覆盖。无需为开训重复跑计划。

新版单帧six须等待当前正在运行的geometry study CPU任务成功产生 `ready_1280.json`；训练入口会重新校验其完整准入、源与三轮计划。已有CPU任务不要重复启动。

```bash
(set -o noclobber
 nohup env GPU_IDS=0,1,2,3 STUDY_PER_EVENT=1280 \
   bash qwen3vl_local/audit_joint/geometry_study/remote.sh train six \
   > phase3_six_1280.log 2>&1 < /dev/null &
)
```

six命令自动完成三轮训练、验证选模和所选checkpoint的test。若另一台尚未准备study数据且没有CPU任务，才运行 `bash qwen3vl_local/audit_joint/geometry_study_abc.sh cpu`，成功后运行six；不能绕过准入。两条任务在同一台四卡机器应顺序执行，或在两台机器分别执行。各9600步、总19200步；四图耗时不能套用单图历史单步数。base保留每阶段48小时/20GiB磁盘余量守护；six保留启动1h与单轮10h实测ETA门。没有本次4B GPU耗时实测或精度改善保证。

## 进度与交回文件

```bash
tail -n 40 phase3_base_coverage_v2.log
tail -n 40 checkpoints/phase3_base_coverage_v2/run_base_coverage_v2/train.log
tail -n 40 phase3_six_1280.log
```

base父日志每约20秒报告当前阶段和子进程；实际训练步数/loss在train.log。完整状态见 `checkpoints/phase3_base_coverage_v2/run_base_coverage_v2/launch_workflow.json`。只看到DDP初始化不代表完成。

base交回同目录 `handoff/phase3_report.zip` 与 `phase3_package.json`；前置未形成包时交父日志。six交回 `checkpoints/geometry_study_v1/runs/six_1280/` 下 `startup.json`、`epoch_1/2/3.json`、`validation_1/2/3.json`、`training_complete.json`、`test/metrics.json` 和父日志，不需要上传权重或RGB。旧目录、失败现场和原始数据均保留。

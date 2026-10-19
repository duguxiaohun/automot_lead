# Phase3：一次数据准备，base / A / B 各三轮

本轮用户明确选择三组均从同一份Qwen3.5初始权重开始，各训练3个epoch，不先训练base再让A/B追加。统一入口为 `phase3_suite.sh`。原有单实验入口仍保留显式base adapter初始化，新增 `--dense-initialization qwen` 供直接初始化；不会自动把旧adapter用于本实验。

## 历史参照与三组定义

历史已收三轮审计：4RGB choice val=242/320（75.625%）、test=274/380（72.105%）；2RGB short choice对应240/320（75%）和270/380（71.053%）。因此采用4RGB choice。来源为 `phase3_three_epoch_acceptance_verification_20261014.json` 和冻结 `phase3_sequence_reference.json`，不重新推理历史模型。这是动作准确率，不是闭环驾驶成功率。

| 组 | 目标 | 每次呈现的监督 |
|---|---|---|
| base | 恢复的独立event/action配额训练，原提示词及动作词组输出 | 一题当前动作 |
| A：a_future | `future`，现有A–F选择token，当前四图 | 默认T=0、+3、+6秒对应的原生未来窗口动作；只用当前观察 |
| B：b_windows_bank | `windows_bank`，逐帧独立语言编码＋严格四图KV融合 | 八帧共享语言编码，五个严格四帧窗口读出 |

三组均为4卡、每卡B1、累积1、十事件各1024/轮、三轮30720次锚点呈现、7680次更新、warmup2000、lr=1e-5、同seed。都从同一个本地Qwen目录重新初始化LoRA，A/B不会加载base结果。每阶段核对初始模型文件成员、SHA及元数据；显存探针是独立进程，不将探针更新带入正式训练。

B遵循v5的新架构，LoRA dropout=0并关闭gradient checkpointing；base/A维持原配置。A/B输出编码、提示词、训练候选资格和B架构都有差异。这次是三种方法的整体效果比较，不能单独证明收益来自loss密度。没有额外安排single对照或旧模型推理。

## 数据只准备一次

保留历史base索引原字节/SHA及固定val/test锚点，只复制小型索引和manifest；RGB、原始meta、模型不复制。一次生成两条密集路线共用的辅助索引，输出在 `shared_data/`：

- 只扫描冻结题库已有路线，仍调用原生collection注释映射、轨迹标定、RGB历史、风险过滤和choice语义，保留原split。
- 只补所需过去四帧及未来目标帧；实际帧没有同context/road映射时不造标签，不把原帧答案复制过去，不扩大训练锚点或替换评测题。
- 记录可用时的本机git版本（纯白名单部署没有Git也可运行）、实际校验的源码清单SHA、hostname、可选AUTOMOT_SERVER、原始collection SHA、缺失支持与题库SHA。A/B各自准入时再执行原始meta复算，这是验证同一份标签，不是重新生成两份数据。
- 旧base题库已有的标签不被新规则覆盖。冻结index必须是SHA `925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99`。

三组CPU准入全部通过，再执行A/B四卡探针，均通过后才开始训练。未来/窗口覆盖不足、某个原有动作被过滤掉、标签来源不符或显存不足，会在训练前停止并打包原因，不自动降低门槛。一次生成不保证原始数据具备全部时点支持。

## 训练、评测与反馈

顺序执行base、A、B，均直接完整三轮，不在首轮停下等待。保留每轮完整断点、固定160题loss验证；base沿原生每轮320题生成验证，A/B最终从保存状态重载做320题读出验证。每组完整7680更新后测试固定380题；不按test选择checkpoint。A/B原生中间loss是迁移诊断，主要比较最后完整三轮的动作结果。

每组保存消费核对、逐题预测、资源探针、准入、训练和测试回执。`comparison.json`包括分事件、分动作准确率、混淆、缺失窗口覆盖率，并分别报告完整固定题数（缺失预测计错）与三组共同支持锚点；原各模式支持子集的主分数也保留，不混用分母。不同输出协议的影响仍需结合错误案例判断。

所有日志与数据均位于当前AutoMoT下的显式输出目录，不写到/tmp。目录已存在时拒绝覆盖；源码升级后旧断点仍须用其原发布版本恢复。不清理旧资产、不自动重试失败GPU任务。失败时已完成的组不会被删除。

## 运行命令

在本次实际服务器的AutoMoT根目录和automot环境执行。下列index是之前提供的cr02资产路径；若换机器，必须改为该机同SHA索引，以及该机数据、模型路径，不假设两机互通。四张卡按实际空闲情况填写。

```bash
bash ./sync_automot.sh && (
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  mkdir phase3_base_ab_v1
  nohup env GPU_IDS=0,1,2,3 PYTHON="$(command -v python)" \
    bash qwen3vl_local/audit_joint/phase3_suite.sh \
    --index checkpoints/phase3_four_data_20261017_223254_120427458/frame_index.jsonl \
    --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
    --collection-dir keyframe_filter/collection_output \
    --output phase3_base_ab_v1/result \
    > phase3_base_ab_v1/launch.log 2>&1 < /dev/null &
  printf 'PID=%s\n日志=%s/phase3_base_ab_v1/launch.log\n' "$!" "$PWD"
)
```

`tail -f phase3_base_ab_v1/launch.log` 查看总进度，详细阶段日志位于 `result/{base,a_future,b_windows_bank}/`，数据生成日志为 `result/data_build.log`。结束以总日志 `[suite] exit=0` 且 `result/suite_workflow.json` 的 `status=three_methods_completed_quality_pending` 为准；这表示执行完成，不自动表示效果恢复。

将 `phase3_base_ab_v1/result/handoff/` 整个目录交回，内含：

- `base_report.zip` / `base_package.json`
- `a_future_report.zip` / `a_future_package.json`
- `b_windows_bank_report.zip` / `b_windows_bank_package.json`
- `suite_report.zip` / `suite_package.json`（包含总流程和可用时的比较报告）

非零退出也交该目录；若尚未生成handoff，交launch.log。旧训练不用重跑。

如果只想先生成并CPU准入，添加 `--prepare-only`。后续正式运行使用新输出目录，并加 `--prepared-data /本机原输出/shared_data`，就不会重新生成数据；仍重新核验共享文件、覆盖和输入形状。这个开关也兼容 `SKIP_TRAIN=1` 或 `PIPELINE_CHECK_ONLY=1`。不要对已完成组直接重跑整套入口；中断后先交审计包定位具体checkpoint，再接续剩余工作。

## 本地验证

相关回归186项通过（180.77秒，包含首批15项新流程测试）；最终新流程17项复跑通过（2.59秒，与前一批重叠），共188项不同测试。发布清单255项源码SHA匹配。未执行服务器全池构建、H20训练或新模型效果验收。[验证回执](phase3_suite_verification_20261019.json)。

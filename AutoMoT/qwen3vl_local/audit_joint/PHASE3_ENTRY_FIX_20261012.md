# Phase3正式索引准入统一修复

已修复[入口续审](PHASE3_ENTRY_REVIEW_20261012.md)中的两项缺口：正式训练（含sampling-only）不再接受缺少manifest的裸索引；独立评测在选题前检查完整索引，不允许开发路线进入holdout或物理路线跨split混用。

## 实现

- `preflight.check_index`强制同目录`manifest.json`，核验原生prompt合同及全索引动作/mapping/choice合同、完整类别与物理路线隔离。新增`prompt_variant`参数；候选提示词还必须匹配冻结的额外开发清单及来源SHA。未新增绕过正式准入的开关。
- `train.train`、`eval.evaluate`共同调用该函数。独立eval先解析adapter元数据和实际prompt/output模式，然后准入，再初始化分布式、创建输出与选题；无效数据不会进入模型加载。
- `train.sh`前置检查传入实际prompt variant，避免等到创建输出后才发现候选准入不符。`four_runs`共用相同检查，仍额外要求所有对照组使用候选开发隔离清单，包括显式baseline组，没有放宽原保护。
- 原有行级测试夹具补上合法manifest，继续验证原来的行级规则；无manifest场景另有专门拒绝回归。低层读行函数并不冒充正式入口。

本次没有改变Phase3标定规则、动作语义、提示词、训练/评测事件等量策略。Phase4 v53生产及学生合同与既有本机view逐项一致；不需要因此修改Phase4合同或重放U-E5。旧实验是否曾泄漏不能从这个入口缺陷推出。

## 验证结果

完整Phase3/Phase4/audit_joint **2368项通过，105.36秒**。新增26项回归覆盖两个入口、baseline/candidate、二元/choice准入、正常采样及模型前拒绝。合法eval正向测试仅在模型加载边界用哨兵停止；读取、索引校验及等量选题均使用原生实现。

另外启动6次原生CLI子进程，使用小型合成全索引，不替换校验器或采样器：

| 场景 | 结果 |
| --- | --- |
| baseline train缺manifest | 退出1，未创建运行目录 |
| candidate eval开发组落入test | 退出1，未创建运行目录 |
| baseline eval物理组跨train/test | 退出1，未创建运行目录 |
| baseline eval提示词SHA不符 | 退出1，未创建运行目录 |
| candidate eval额外开发清单SHA不符 | 退出1，未创建运行目录 |
| 合法candidate train sampling-only | 退出0，40呈现=十事件各4；loss/generation val均十事件各1；未创建训练目录 |

这些夹具的INVALID覆盖要求显式关闭，仅用于验证入口及有效事件均衡，不证明真实正式池支持完备。没有加载权重、GPU训练或真实模型推理。CLI导入会执行现有GPU自动选址逻辑，但本次均使用CPU采样或在准入阶段拒绝。

新发布清单466文件、源码核验matched。轻量校验：[phase3_entry_fix_verification_20261012.json](phase3_entry_fix_verification_20261012.json)；完整证据：`checkpoints/phase3_entry_fix_20261012/`。

## 下一步：Phase3正式索引可独立验收

本轮没有取得服务器正式索引，不能宣布Phase3已具备开训资格。代码发布同步后，只需要对实际索引执行数据准入，不重复服务器readiness。以下`P3_INDEX`必须指向实际完整索引，保留原manifest，不能用历史诊断夹具替代：

```bash
python -m qwen3vl_local.sft_new_loop_phase3.preflight \
  --index "$P3_INDEX" \
  --prompt-variant v23_rgb_stage_candidate_20261006 \
  --action-output-mode choice \
  --output checkpoints/phase3_index_admission_20261012.json
```

交回该JSON和索引旁的`manifest.json`，同时说明实际索引路径及计划的二元/choice、两图/四图组合；输出文件已存在则使用新名字保留原报告。若正式索引还没有生成，直接报告缺失，由现有完整建库入口生成，不删除manifest或取消隔离检查。正式索引、实际采样和训练预算通过后可以从基座新训，不要求先恢复旧adapter，也不必等待U-E5。

U-E5真实proceed YES仍为0；本轮没有重复89路线重放、RGB目视或改标签。Phase3正式数据与跨任务隔离尚未验收，联合G0继续阻塞。本轮未远程执行、训练或push。

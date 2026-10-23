# Base 外层完成验收修复

后续发布复核补充了 single／abc 的保存产物验收，当前实现与发布结果见 [发布复核](PHASE3_RELEASE_REVIEW_20261023.md)。以下为 base 修复的历史记录，108项测试与源码范围不包含后续修改。

五类正式 CPU 数据准入已通过。本次修复的是 `phase3_compare` 在 base 子进程成功后，直接从主目录导入验收函数导致旧源码指纹不匹配的问题。

外层现调用训练入口同一个 `phase3_native_runtime.prepare`，校验并复用内容寻址的兼容源码快照，再用独立 Python 子进程运行真实 `phase3_base_bounded.verify_training`。子进程固定快照工作目录和 PYTHONPATH，禁止写入字节码。原生源码 SHA、训练合同、三轮计划、38,400 次逐 rank 消费、实际曝光、最终 adapter 文件散列及9,600步检查均保留；验收通过才写外层回执并继续下一组。快照来源或产物不匹配仍拒绝。

生产代码仅修改外层 `phase3_compare.py`，没有修改 base／single／abc 绑定的训练模块、已有 ready、原数据及历史回执，不触发正式重建。外层回执额外记录验收使用的快照路径及原生核验结果。

## 已完成训练但外层收尾失败

本轮尚未 push。发布后，在服务器已有 AutoMoT 根目录同步并仅补做验收：

```bash
bash ./sync_automot.sh
bash qwen3vl_local/audit_joint/phase3_compare.sh verify base
```

此命令要求 base 内层 `launch_workflow.json` 已是 `completed_quality_pending`，且所有原生完成证据通过。只核验已有产物并写 `checkpoints/phase3_compare/base_completed.json`，不检查 GPU、不启动建库、训练、评测或打包，不改实验目录。缺失或损坏证据会明确失败，不能借此将未完成训练标成完成。

已经完成 base 时，不要再执行 `train all`：其输出碰撞保护仍拒绝重训。若剩余四组均未运行，可依次执行 `train single`、`train abc`；已有其他完成组时按实际缺少的方法单独启动。新任务的 `train all` 会自动使用修复后的收尾链路。

## 验证范围

新增回归使用真实兼容源码快照、真实验收子进程和合成的三轮完整消费记录。它复现主目录调用失败、快照内验收成功，覆盖外层写回执并顺序继续 single／a／b／c，以及独立补验收。源码／合同篡改、权重字节变化、缺 rank 消费、未完成状态、job 指向其他输出均拒绝；失败不写成功回执、不启动下一组。

训练命令和权重使用测试替身；这些测试不代表已经执行4B模型或真实GPU训练。本机仍缺完整4B模型，下一项硬件验收仍是服务器前台四卡探针。

测试结果和当前文件散列见 `phase3_completion_verification_20261023.json`。旧回执保留其当时的源码散列，不因本次修改重写。

最终相关回归 **108项通过（59.44秒）**，其中新增9项收尾测试；包含此前99项，不将两批相加。真实兼容缓存仍为此前 `a1fd2101…` 快照，快照内计算的合同与正式 base ready 完全一致。此前 base 回执绑定的13份证据全部未变，single／abc 182项源码证明中仅公共外层 `phase3_compare.py` 改变，其余181项一致。没有重建正式数据、运行GPU或push。

评测限制保持：single 动态切入 test 的12个查询包含 STOP／RESUME，目标质量合计3／9，并非全部STOP；仍只来自一个物理组。single 跟车制动58锚点／219查询也仅一个测试物理组，信号灯故障仅一个查询。验证缺类和4／13米候选边界尚未视觉审查的问题，不由本次收尾修复或训练重复解决。

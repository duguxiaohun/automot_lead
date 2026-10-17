# v74：可选评估 batch 与冻结 RGB 特征复用

> 历史 v74 说明。当前 v75 已默认启用评估 B2 与冻结 RGB 复用，GPU 验证已改四卡；请使用 [四卡全流程命令](FULL_FOUR_GPU_20261017.md)，不要执行本页旧单卡命令。

已选训练配置保持：Phase3两/四图每卡B2、累积2；Phase4两/四图每卡B2、累积4。此次不重新测容量或搜索batch，也不把新优化的速度说成已经验证。

新增两项显式开关，默认仍为评估B1、视觉复用关闭，先验证目标模型再启用新实验：

- Phase3训练：`GENERATION_EVAL_BATCH_SIZE=2 REUSE_FROZEN_RGB=1`；独立eval.sh：`EVAL_BATCH_SIZE=2 REUSE_FROZEN_RGB=1`。Python参数分别是`--generation-eval-batch-size 2` / `--batch-size 2`及`--reuse-frozen-rgb`。
- Phase4首轮和多轮入口：`EVAL_BATCH_SIZE=2 REUSE_FROZEN_RGB=1`。正式train、逐轮val、reload和test记录/沿用相同配置；resume时改动配置拒绝。底层train参数`--eval-batch-size 2 --reuse-frozen-rgb`。

评估使用左padding合批，一次生成多个独立答案，保留题目顺序和尾批，不改变事件等量或采样成员。Phase3保留原EOS及1.05重复惩罚，padding不进入重复惩罚集合。不同batch形状可能产生浮点差异，服务器命令要求测试样本生成完全一致，不能据此承诺全池所有题逐位一致。

视觉复用只支持本地Qwen3.5冻结且无随机计算的视觉塔。比较本次前向的真实预处理像素和grid，唯一图像编码一次，再展开到原来的每个图像槽；不缓存到下次前向，不跨GPU传特征。视觉参数可训练、输入需要梯度、视频或额外隐藏状态输出时走原路径并记录原因。四卡8片段对应9/11个唯一源时刻，并不意味着四卡合计只做9/11次视觉编码。

## 先执行的短验证

在服务器AutoMoT目录，按原同步流程更新源码后执行。CPU阶段最多180秒；通过后GPU阶段最多240秒（超时另需数秒清理子进程）。首错停止，不自动重复。GPU只用GPU_IDS指定的一张卡、重复一个真实样本组成B2、两图和四图，检查逐题/批量生成以及有/无视觉复用的loss和梯度。LoRA dropout设0以隔离随机性；不保存权重，不是训练质量、四卡DDP、全池或吞吐验收。这里不需要学生view或全池加载。

```bash
(
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
export BATCH_RUNTIME_CHECK="checkpoints/v74_runtime_check_$(date +%Y%m%d_%H%M%S_%N)"
printf '测试输出：%s\n' "$BATCH_RUNTIME_CHECK"
python -m qwen3vl_local.audit_joint.batch_runtime_check \
  --check cpu --timeout 180 --output "$BATCH_RUNTIME_CHECK/cpu"
GPU_IDS=0 python -m qwen3vl_local.audit_joint.batch_runtime_check \
  --check gpu --timeout 240 \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --output "$BATCH_RUNTIME_CHECK/gpu"
cat "$BATCH_RUNTIME_CHECK/gpu/result.json"
)
```

交回两份`status.json`和GPU的`result.json`；失败时交对应`check.log`末尾80行。124是硬超时，未通过不能自动放宽容差或排队重跑。

## 版本和恢复边界

此次新增运行路径改变了当前学生源码合同及Phase3恢复源码绑定。**v73或更早checkpoint须留在原版代码续训，不能在v74上绕过SHA恢复。** 目前没有要求重新训练已有实验。

新实验需要当前view；已知v72/v73旧view已加入显式`phase4_view_upgrade`允许列表，仍完整检查父库/成员/图像sidecar/曝光清单，输出新目录，不覆盖旧view。原生producer/teacher、`phase4_load.py`和`phase4_production_cache.py`未改，已有v69路线缓存继续有效，无需重建原生库或重放老师。短验证本身不需要迁移。

通过短验证后，新实验的启用配置如下；沿原正式入口使用已核验的当前view，不用该片段替代数据准入：

```bash
export REUSE_FROZEN_RGB=1
export GENERATION_EVAL_BATCH_SIZE=2  # Phase3训练内val
export EVAL_BATCH_SIZE=2            # Phase3独立eval / Phase4 train、reload、test
```

本机验证包含真实小型Qwen3.5两/四图、不同长度左padding、旧Phase3 KV循环对照、梯度checkpointing、视觉复用前后语言参数梯度、尾批顺序与异常恢复。目标4B/GPU结果尚待服务器交回。

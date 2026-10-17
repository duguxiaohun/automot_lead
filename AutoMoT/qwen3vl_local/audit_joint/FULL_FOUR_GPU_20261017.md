# v75：Phase3／4 正式模型流程四卡

> 2026-10-17 已收到 v75 四卡成功回执：CPU 41 项通过，GPU correctness_passed/exit0，两模式每卡 B2。详见 four_gpu_received_20261017.json。该短检查无需重跑，以下命令仅保留用于复现。

当前推荐入口保持训练每卡B2，四卡每次前向8个独立片段。Phase3累积2（16样本/update），Phase4累积4（32样本/update），GC开启。评估B2和冻结RGB复用现在是普通参数的默认值，无需额外export启用；显式关闭参数保留用于诊断。旧checkpoint继续遵守保存时的配置和源码绑定。

| 阶段 | 执行方式 |
| --- | --- |
| Phase3训练／Phase4两四图训练 | 四进程DDP，梯度同步 |
| Phase3 loss验证／生成验证／最终生成检查 | 四卡完整batch分片，汇总指标或答案 |
| Phase3独立val/test | eval.sh默认四卡，完整batch分片 |
| Phase4逐轮val／reload／固定test | 四卡完整batch分片，rank0写最终报告 |
| 原生库校验／view／运行快照准备 | GPU启动前的CPU阶段；无需DDP |

评估没有反向传播，因此采用四卡分布式推理，不进行DDP梯度同步。直接CUDA评估CLI也启动四卡；显式CPU诊断保留。卡不足就拒绝，不静默降到单卡。生成前核对成员，生成后核对每题仅出现一次；尾批不补重复题。某rank推理报错会汇总失败，不发布成功报告。

训练sequence采样未变：两图[t-2,t]、锚点推进2；四图[t-3,t-2,t-1,t]、推进1。事件等量，少见题／锚点优先、连续段优先，action软补偿；短段拼接独立样本，不伪造跨路线历史。完整全局batch再分给四rank。train/val/test的事件等量默认、transition历史重叠和epoch曝光恢复保持。

视觉复用只在单rank单次前向的冻结确定性视觉塔内进行，按实际像素和grid去重；不跨rank缓存，不承诺四卡合计只编码9/11图。两四图评估成员仍可不同，不声称同题配对。

## 同步后一次限时验证

在服务器AutoMoT目录，沿现有同步方法更新到v75。下列命令CPU最多180秒、GPU最多240秒，另有少量进程清理时间，首错停止。只读首个真实原生样本，两模式重复组成B2，检查四卡DDP反向与分片生成。无完整库加载、view重建、容量搜索或权重保存。不是全池质量或吞吐验收。

```bash
(
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
export FOUR_GPU_CHECK="checkpoints/v75_four_gpu_$(date +%Y%m%d_%H%M%S_%N)"
printf '测试目录：%s\n' "$FOUR_GPU_CHECK"
python -m qwen3vl_local.audit_joint.batch_runtime_check \
  --check cpu --timeout 180 --output "$FOUR_GPU_CHECK/cpu"
GPU_IDS=0,1,2,3 python -m qwen3vl_local.audit_joint.batch_runtime_check \
  --check gpu --timeout 240 \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --output "$FOUR_GPU_CHECK/gpu"
cat "$FOUR_GPU_CHECK/gpu/result.json"
)
```

成功要求两份status.json的completed=true、returncode=0；GPU result为correctness_passed、world_size=4、ranks含0/1/2/3，两RGB模式的生成per_rank_counts均[2,2,2,2]。交回这三份JSON即可。失败交对应check.log末尾80行；124表示硬超时，不自动重跑或放宽容差。

服务器此前交回v74单卡两模式生成一致、梯度对照通过，是已有证据；不能替代此次四卡检查。本机四CPU/Gloo回归包含真实Phase4两epoch DDP循环的每rank验证、P3/P4生成分片、尾批和失败传播，不冒称本机已完成4B GPU验收。

## 正式入口及版本

Phase3使用audit_joint/train_phase3_batch.sh及sft_new_loop_phase3/eval.sh；Phase4使用audit_joint/train_phase4_batch.sh或audit_joint.phase4_training，内层rgb_short/run.sh、evaluate.sh依次四卡运行。无需新增batch优化开关。已有正式数据准入仍须通过；短诊断不代替它。

新源码改变学生运行合同及Phase3恢复绑定。已有v74及更早checkpoint留原版继续，禁止绕过SHA跨版本恢复。新实验可用phase4_view_upgrade将已知旧view写入新目录（现支持已知v74运行差异），仍核验父库、成员、sidecar与曝光。原生producer/teacher、phase4_load和production_cache未改，既有路线缓存可用，无需重建原生库或重放老师。短检查本身不需要view迁移。

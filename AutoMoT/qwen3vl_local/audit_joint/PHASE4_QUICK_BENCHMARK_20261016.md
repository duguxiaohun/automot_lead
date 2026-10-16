# Phase4 已通过 GPU 单步后的限时小池对照

> 当前决定：用户已接受Phase3／Phase4两图、四图均每卡B2，batch选型已结束，不再要求新增benchmark或全库检查。正式配置与入口见 [最终B2配置](BATCH_SELECTION_FINAL_20261016.md)。下文保留历史流程。

服务器交回：NVIDIA H20-3e 四卡 RGB2/RGB4 均 model_step_passed / exit0。
单题B1两图 allocated 12.43 GiB、最大reserved18.23 GiB；四图15.00/23.40 GiB。
这些不是完整池容量上界或 batch 推荐。旧整库准备在CPU解析/逐题校验，不能据此称GPU不可用；后续模型加载SHA/PEFT导入也在CPU。

同步 v67_quick_benchmark 后，从服务器 AutoMoT 应用目录执行。无需生成学生view、重新建原生库或回放老师。

```bash
export P4_QUICK="checkpoints/p4_quick_rgb2_$(date +%Y%m%d_%H%M%S_%N)"
NCCL_DEBUG=WARN python -m qwen3vl_local.audit_joint.phase4_quick_benchmark \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B \
  --gpus 0,1,2,3 \
  --rgb-mode 2 --batches 1 2 \
  --samples-per-event 8 --update-samples 8 \
  --prepare-timeout 60 --trial-timeout 180 \
  --output "$P4_QUICK"
```

CPU一次最多60秒；B1/B2各最多180秒，共约7分钟预算，另有少量启动/检查/退出清理时间。默认首错停止，不运行其余模式、重复或正式epoch。需要更短的首次检查可以使用 `--batches 2`，但只有B2结果不能比较吞吐。不把30秒faulthandler调用栈当任务退出；summary/进程返回码才是结束结果。

小池最多读取20000行/256MiB，原生大文件从32个确定性分散位置读取局部连续窗口，避免只读前几个scenario。每事件最多保留32个候选；两模式转换及候选内冲突排除后，每事件至少8个独立题才继续。未凑齐就在CPU停止，日志报告缺少的事件，不静默重复稀少题、不扩大预算。只核对原生源码合同、读取文件stat前后和选中图内容身份，**不声称完整train文件SHA、全库老师证明或跨split隔离已校验**。

每组十事件各8题，计时80个不同题；另有十事件各8次呈现作为预热，共160次呈现。每update固定8题：四卡B1累积2，B2累积1。B1/B2使用相同问题和update成员，按模式步长排顺序，复用正式encode_batch/LoRA16/32/GC/DDP/AdamW及实际消费回执核验。正式训练32/update未修改。只一次重复且小池有窗口选择偏差，因此不用于正式batch选型或能力评估。

输出：`prepared.json`、`selection_binding.json`、`request.json`、`summary.json`，以及每组worker.log/rank报告/实际消费step记录；不保存权重。summary固定 `full_pool_admitted=false`，候选字段为 `provisional_subset_candidates`，没有正式 `fastest_measured_candidates`。

简短查看结果（不启动GPU）：

```bash
python - "$P4_QUICK" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads((p/'summary.json').read_text())
print('output:',p,'status:',r.get('status'),'full_pool_admitted:',r['full_pool_admitted'])
for t in r.get('trials',[]):
    print('RGB',t['rgb_mode'],'B',t['batch_size'],'status',t['status'],
          'samples/s',t.get('samples_per_second'),'headroom',t.get('fits_with_headroom'),
          'reserved_GiB',t.get('max_peak_reserved_bytes',0)/2**30)
PY
```

两图通过后，再把命令 `--rgb-mode 2` 改为4，输出变量改成新的 `p4_quick_rgb4_...`。模式间不能复用同一个prepared文件。暂不一口气跑旧12组矩阵。

同一模式后续试验可在新输出目录显式加 `--selection 原目录/prepared.json`，跳过CPU窗口读取及图身份准备。复用时核验当前学生/诊断源码、父manifest、原生train stat、数据根、模式/配额/update/seed/world及选中题SHA；worker继续逐图核验实际内容。源码/输入改变则拒绝。`--prepare-only` 只做CPU准备；后续可用该文件启动试验。完整数据池准入和正式训练入口性能仍是后续单独工作，不因小池运行通过关闭。

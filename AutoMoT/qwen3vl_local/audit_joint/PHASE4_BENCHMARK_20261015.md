# Phase4 四卡实测：从现有原生库到结果

先用既有 `sync_automot.sh` 流程同步本轮白名单源码。下面在服务器 **AutoMoT目录、automot环境** 一次完整执行；使用已验收v57原生库，不重复老师回放或约20小时的建库。四张卡须空闲。先生成当前采样v2对应的学生view，再比较两图／四图各B1、B2、B4；这一步不启动正式整轮训练。

```bash
(
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  export DATA_ROOT=lead_data
  export MODEL_DIR=checkpoints/Qwen3.5-4B
  export GPU_IDS=0,1,2,3
  P4_NATIVE=checkpoints/phase4_v57_20261014_232157_762998141/data4
  P4_MEASURE_ROOT="checkpoints/p4_batch_measure_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir "$P4_MEASURE_ROOT"
  export P4_NATIVE P4_MEASURE_ROOT
  printf '本次输出目录：%s\n' "$P4_MEASURE_ROOT"

  python - <<'PY'
import json, os
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
root = Path(os.environ['P4_MEASURE_ROOT'])
report = source_check(Path('.'), Path('qwen3vl_local/audit_joint/release_sources_v61_batch_audit_20261015.json'))
(root/'source_check.json').write_text(json.dumps(report, indent=2)+'\n')
if report['status'] != 'matched':
    raise SystemExit('源码不匹配，查看source_check.json；同步本轮代码后使用新输出目录')
native = Path(os.environ['P4_NATIVE'])
if not (native/'manifest.json').is_file():
    raise SystemExit(f'原生v57数据不存在：{native}')
for name in ('DATA_ROOT', 'MODEL_DIR'):
    if not Path(os.environ[name]).is_dir():
        raise SystemExit(f'{name}目录不存在：{os.environ[name]}')
if not (Path(os.environ['MODEL_DIR'])/'config.json').is_file():
    raise SystemExit('模型目录缺少config.json')
print('源码及输入路径检查通过', flush=True)
PY

  python -m qwen3vl_local.audit_joint.guard \
    --space-path "$P4_MEASURE_ROOT" --min-free-gib 20 -- \
    python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
    --dataset4 "$P4_NATIVE" --data-root "$DATA_ROOT" \
    --output "$P4_MEASURE_ROOT/student" \
    2>&1 | tee "$P4_MEASURE_ROOT/view_build.log"

  DATASET="$P4_MEASURE_ROOT/student/view" \
  OUTPUT_DIR="$P4_MEASURE_ROOT/benchmark" \
  bash qwen3vl_local/audit_joint/benchmark_phase4_batch.sh \
    2>&1 | tee "$P4_MEASURE_ROOT/benchmark.log"

  printf '\n实测完成，交回这两个文件：\n%s\n%s\n' \
    "$P4_MEASURE_ROOT/benchmark/summary.json" \
    "$P4_MEASURE_ROOT/benchmark/request.json"
)
```

默认checkpointing开启，两模式各3种batch、每种两次重复，共12个trial，顺序B1→B2→B4再B4→B2→B1。每trial十事件各32个计时呈现，共320；每次更新固定32样本、共10次更新，四卡B1／2／4分别累积8／4／2次。每event额外预热一次更新，显存计入预热峰值、吞吐不计预热。每个trial独立加载基座，真实forward/backward/clip/AdamW及DDP，使用正式encode_batch设备路径，不保存权重。

`end_to_end`输出的mean_rank_stage_seconds为空、largest_measured_stage为null是正常的，不能把host_submission_seconds解释为完整GPU阶段耗时。每个trial的worker.log打印更新进度；父进程在trial结束时输出结果。源码和路径错误在GPU前停止，view失败不会启动benchmark；benchmark的OOM／超时保留诊断，全部完成条件不满足则退出2。

完成后交 `benchmark/summary.json` 和 `benchmark/request.json`。失败时另交相应trial的worker.log及rank*.json；若卡在生成视图阶段，则交view_build.log。输出目录由命令开始时打印，不要重新执行date猜目录。无需交RGB、基座或权重。选型依据是所有rank／重复的显存余量和吞吐，不按最大可容纳batch直接选型。

生成的 `$P4_MEASURE_ROOT/student/view` 可供之后相同源码下的正式首轮复用：`phase4_first_epoch --view <该目录>`；正式训练仍需独立准入，不能把小样本benchmark当作完整数据池／模型质量验收。既有U-E5支持缺格保持。

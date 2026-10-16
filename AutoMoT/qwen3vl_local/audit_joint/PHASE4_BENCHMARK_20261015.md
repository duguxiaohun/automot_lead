# Phase4 四卡实测：从现有原生库到结果

> 更新：已定位逐题重复协议校验，旧服务器view绑定v61。请先按 [v64限时迁移与验证](PHASE4_LOAD_V64_20261016.md) 执行，以下保留为历史命令。

> 2026-10-16服务器连续超时后，请先按 [短时定位与单trial](PHASE4_SHORT_DIAGNOSIS_20261016.md) 执行，勿直接重跑下面12组。

> 当前v63保留v62的采样v3及可见空闲GPU筛选，最终同步后新建view；详细变更见 [尾部拼段与模式配置](BATCH_TAIL_MODES_20261015.md)。显式GPU_IDS表示由你选择卡，本例仍明确pin四卡。实测选型后，首轮支持分别传 `--batch-size-rgb2` 和 `--batch-size-rgb4`。

先用既有 `sync_automot.sh` 流程同步本轮白名单源码。下面在服务器 **AutoMoT目录、automot环境** 一次完整执行；使用已验收v57原生库，不重复老师回放或约20小时的建库。四张卡须空闲。先生成当前采样v3对应的学生view，再比较两图／四图各B1、B2、B4；这一步不启动正式整轮训练。

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
report = source_check(Path('.'), Path('qwen3vl_local/audit_joint/release_sources_v63_bounded_benchmark_20261016.json'))
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

完整矩阵仅在短时排查通过后运行。默认checkpointing开启，两模式各3种batch、每种两次重复，共12个trial，顺序B1→B2→B4再B4→B2→B1。每trial十事件各32个计时呈现，共320；每次更新固定32样本、共10次更新，四卡B1／2／4分别累积8／4／2次。每event额外预热一次更新，显存计入预热峰值、吞吐不计预热。每个trial独立加载基座，真实forward/backward/clip/AdamW及DDP，使用正式encode_batch设备路径，不保存权重。

`end_to_end`输出的mean_rank_stage_seconds为空、largest_measured_stage为null是正常的，不能把host_submission_seconds解释为完整GPU阶段耗时。每个trial的worker.log打印更新进度；父进程实时转发阶段日志，并在trial结束时输出结果。源码和路径错误在GPU前停止，view失败不会启动benchmark；benchmark的OOM／超时保留诊断，默认首个失败即停止；CPU准备每模式一次、默认限时120秒。

完成后交 `benchmark/summary.json` 和 `benchmark/request.json`。失败时另交相应trial的worker.log及rank*.json；若卡在生成视图阶段，则交view_build.log。输出目录由命令开始时打印，不要重新执行date猜目录。无需交RGB、基座或权重。选型依据是所有rank／重复的显存余量和吞吐，不按最大可容纳batch直接选型。

生成的 `$P4_MEASURE_ROOT/student/view` 可供之后相同源码下的正式首轮复用：`phase4_first_epoch --view <该目录>`；正式训练仍需独立准入，不能把小样本benchmark当作完整数据池／模型质量验收。既有U-E5支持缺格保持。

实测通过后，复用上面已经生成的view即可，无需再建一次。以下命令在同一个AutoMoT目录执行；填入实测输出目录，以及 `summary.json` 支持的两种模式batch。这里不预设Phase4的B2结论。

```bash
(
  set -euo pipefail
  : "${P4_MEASURE_ROOT:?请先设置为实测开始时打印的输出目录}"
  : "${P4_BATCH_RGB2:?请根据两图实测结果设置为1、2或4}"
  : "${P4_BATCH_RGB4:?请根据四图实测结果设置为1、2或4}"
  GPU_IDS=0,1,2,3 python -m qwen3vl_local.audit_joint.phase4_first_epoch \
    --prepared checkpoints/phase4_v57_20261014_232157_762998141 \
    --view "$P4_MEASURE_ROOT/student/view" \
    --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
    --batch-size-rgb2 "$P4_BATCH_RGB2" --batch-size-rgb4 "$P4_BATCH_RGB4" \
    --output "checkpoints/phase4_modes_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
)
```

前面的实测命令在子shell内运行，退出后需按打印路径重新设置 `P4_MEASURE_ROOT`，不能重新执行date生成另一个目录名。首轮入口按两模式各自batch计算累积，均保持32样本/update；该命令仍执行正式准入、各一轮、重载和test。若实测之后又改变学生合同，应先重新核对view兼容性。

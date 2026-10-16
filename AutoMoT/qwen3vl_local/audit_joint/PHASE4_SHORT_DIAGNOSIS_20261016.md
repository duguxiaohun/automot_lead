# Phase4超时后的短时排查

服务器快照：四worker各约14GiB RSS，GPU 0%／3–327MiB，首trial在30分钟后收到父进程SIGTERM。支持尚未进入正常模型训练，不证明具体卡在哪个函数；NCCL弃用warning不是已证实根因。当前迁到H20-3e环境，不沿用此前H20性能结论。用户已确认该trial的1个torchrun及4个worker全部退出。

本次只修改benchmark调度及诊断，不改原生producer、学生输入或采样合同。有效v62学生view可复用；不重复view生成、原生建库或老师回放。若服务器view早于v62，仍应按实际合同检查结果处理，不能改SHA绕过。

## 现在即可执行：旧代码兼容的120秒定位

在服务器AutoMoT目录、当前venv执行。这条命令不依赖新参数，不创建GPU进程，不改数据；30秒一次调用栈，120秒终止、额外最多5秒后强制结束。返回124表示限时触发；日志中的`Timeout (0:00:30)!`是faulthandler的定时栈，不是NCCL报错。

```bash
(
  set -uo pipefail
  ulimit -S -c 0 2>/dev/null || true
  OUT="checkpoints/p4_cpu_trace_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir "$OUT"
  printf '诊断输出：%s\n' "$OUT"
  timeout --signal=TERM --kill-after=5s 120s python -u - <<'PY' 2>&1 | tee "$OUT/prepare.log"
import os, faulthandler, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
from pathlib import Path
from types import SimpleNamespace
from qwen3vl_local.audit_joint.batch_benchmark import load_pool, schedule
a = SimpleNamespace(phase='4', dataset=Path('checkpoints/p4_batch_measure_20261016_093447_958074352/student/view'))
start = time.monotonic()
print('START load_pool; CPU only', flush=True)
items, records, manifest = load_pool(a, 2)
print('DONE load_pool', len(items), 'seconds', time.monotonic()-start, flush=True)
updates, selection = schedule(records, per_event=32, update_samples=32, world=4, batch=1, mode=2, seed=20260929)
print('DONE sampling', selection, 'total_seconds', time.monotonic()-start, flush=True)
faulthandler.cancel_dump_traceback_later()
PY
  code=${PIPESTATUS[0]}
  printf 'exit=%s；交回 %s/prepare.log\n' "$code" "$OUT"
  exit "$code"
)
```

此结果用于定位正式全池具体热点，不将一次超时解释为GPU容量失败。不要直接把限时加到数小时；交回prepare.log即可继续定向优化。

## 同步本补丁后的短时入口

新增 `--prepare-only`、`--prepare-timeout`；CPU准备在GPU前独立执行，每个RGB模式一次，通过完整加载校验后保存选定小集合，四rank和不同batch／repeat复用相同SHA绑定。保留每更新成员、事件等量、模式顺序、正式encode_batch路径。默认任一失败即退出；只有显式 `--continue-on-error` 才继续矩阵。

```bash
python -m qwen3vl_local.audit_joint.batch_benchmark \
  --phase 4 \
  --dataset checkpoints/p4_batch_measure_20261016_093447_958074352/student/view \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --gpus 0,1,2,3 --modes 2 --batches 1 \
  --samples-per-event 32 --update-samples 32 --seed 20260929 \
  --prepare-only --prepare-timeout 120 \
  --output "checkpoints/p4_cpu_check_$(date +%Y%m%d_%H%M%S_%N)"
```

若CPU准备通过，下一步只测一个trial：

```bash
python -m qwen3vl_local.audit_joint.batch_benchmark \
  --phase 4 \
  --dataset checkpoints/p4_batch_measure_20261016_093447_958074352/student/view \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --gpus 0,1,2,3 --modes 2 --batches 1 \
  --samples-per-event 32 --update-samples 32 --seed 20260929 \
  --gradient-checkpointing --repeats 1 --timing end_to_end \
  --prepare-timeout 120 --trial-timeout 300 \
  --output "checkpoints/p4_one_trial_$(date +%Y%m%d_%H%M%S_%N)"
```

这条命令CPU最多120秒、GPU trial最多300秒，另有少量进程清理时间；不承诺一定在此时间内完成测量。新调用仍重新校验一次完整池，不跨调用复用未经重新校验的准备快照。准备未通过不会启动GPU。模型加载、DDP建立、预热和计时均实时显示，等待时保留30秒调用栈。Ctrl+C／SIGTERM／超时会按PID及启动时间清理本次已追踪后代，包括独立session的worker，不扫描终止其它训练。

交回新输出下的 `summary.json`、`request.json`，若CPU失败再交 `prepare_rgb2.log`，若GPU阶段失败再交 `rgb2_batch1_repeat0/worker.log`。不需要上传prepared样本JSON或RGB。一次trial只证明基本路径和当次资源表现，不能决定最佳batch；两图通过后以 `--modes 4` 同预算做四图检查，之后再按需要扩大对照矩阵。

## 本地验证范围

118相关测试通过，含CPU/Gloo更新一致性、固定样本SHA／更新组、父进程每模式只准备一次、准备失败零GPU启动、默认首错停止、真实独立session子进程超时和SIGTERM清理。真实既有1457train小池CPU CLI通过：两图选278不同题、320呈现，未加载目标4B模型。正式29.9万题热点尚需服务器短时调用栈；没有宣称服务器瓶颈已全部消除或GPU验收完成。

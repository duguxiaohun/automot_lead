# 小池 GPU 已通过后的下一步：限时完整原生库校验

已收到四H20-3e两/四图B1/B2全部measured及headroom通过。相同小池8/update，B2吞吐分别+7.89%/+6.20%；单重复、小池结论，不是正式32/update或全库验收。不要重跑通信、单题smoke或同配置小池对照。

早期无GPU训练的调用栈停在JSON解析、逐题协议校验；诊断入口软链接误拒绝已修复。v68进一步复用一次CPU读取事务内的46个老师源码SHA：visible_scope调用validate_question时原先反复rules.identity读取这些源码。每道题完整校验照常执行，事务结束恢复原函数并重读源码/参数比对，变化拒绝；不改冻结producer/teacher/学生合同，不持久缓存，不跳过teacher证明。增加read_rows/model_input/coverage/training_report阶段日志。完整库耗时尚待服务器实测，不保证240秒完成。

同步v68后只执行下面一次CPU校验；不生成view、不建原生库、不回放老师、不启动GPU。原生全量验证成功只是当前库读取与原生内部一致性通过，空val/test等后续训练准入仍单独判断。

```bash
export P4_FULL_CHECK="checkpoints/p4_full_load_v68_$(date +%Y%m%d_%H%M%S_%N)"
python -m qwen3vl_local.audit_joint.phase4_load \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --data-root lead_data \
  --timeout 240 \
  --output "$P4_FULL_CHECK"
```

总CPU子进程上限240秒，另加少量进程启动/清理。30秒调用栈仅诊断；最终status.json returncode0/completedtrue且result.json status=native_validated才算成功。124表示本次240秒预算不足；不要自动重复或增加超时，交日志看具体阶段。输出读取（几秒，无新训练）：

```bash
python - "$P4_FULL_CHECK" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
for name in ('status.json','result.json'):
    f=p/name
    if f.is_file(): print(name,json.dumps(json.loads(f.read_text()),ensure_ascii=False))
f=p/'load.log'
if f.is_file():
    with f.open('rb') as stream:
        stream.seek(max(0,f.stat().st_size-16000))
        tail=stream.read().decode(errors='replace')
    print('\n'.join(tail.splitlines()[-60:]))
PY
```

成功后才推进旧view的显式迁移及正式32/update的B2入口验证。该步骤没有生成持久化全库验证缓存；未来读取仍需要验证，不宣称所有正式入口加速已接入。CPU无需空闲GPU。无需修改或重装CUDA/NCCL/模型依赖。

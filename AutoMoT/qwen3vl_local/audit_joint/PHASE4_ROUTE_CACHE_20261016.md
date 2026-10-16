# v69：完整校验拆成可续接的路线步骤

> 当前决定：用户已接受Phase3／Phase4两图、四图均每卡B2，batch选型已结束，不再要求新增benchmark或全库检查。正式配置与入口见 [最终B2配置](BATCH_SELECTION_FINAL_20261016.md)。下文保留历史流程。

v68服务器returncode124/completedfalse，无result，表示240秒预算耗尽。日志证明JSON读取及逐题校验持续推进，model_inputs最终完成15.08秒；没有证明整库通过，也没有报告数据错误。30秒Timeout调用栈是周期性诊断，最终124才是进程限时退出。两/四图四卡小池B1/B2已通过，不重跑GPU诊断。

v69在一次加载内复用内容相同题目的首次完整证明检查，每次复用仍重新计算整个question内容SHA并核对teacher身份。生产路线检查拆成独立入口：直接读取已有生产记录，逐条调用原校验器，不先解析298787条train；每条通过即原子保存回执。超时只丢弃尚未完成的路线。启动及复用仍检查输入SHA，源文件/索引/候选池/路线或回执变化拒绝。缓存仅供本机可信工作区续接，不是签名证明，不要从不可信来源下载缓存。源码升级可能要求新缓存目录。

## 现在只执行这一步（CPU最多240秒，另加启动清理）

先沿既有sync_automot.sh同步代码。下面命令在服务器AutoMoT目录、既有.venv中运行；不用模型路径或GPU_IDS，不重建库、不回放老师。每次output新建，cache路径保持不变。

```bash
(
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  python - <<'PY'
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
r=source_check(Path('.'),Path('qwen3vl_local/audit_joint/release_sources_v69_route_cache_20261016.json'))
print('v69 source:',r['status'])
if r['status']!='matched':
    print([x for x in r['files'] if x['status']!='matched'])
    raise SystemExit('先同步完整v69源码，再执行')
PY
  export P4_ROUTE_CACHE=checkpoints/p4_verified_routes_v69
  export P4_ROUTE_RUN="checkpoints/p4_route_check_v69_$(date +%Y%m%d_%H%M%S_%N)"
  rc=0
  python -m qwen3vl_local.audit_joint.phase4_production_cache \
    --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
    --cache "$P4_ROUTE_CACHE" \
    --timeout 240 \
    --output "$P4_ROUTE_RUN" || rc=$?
  python - <<'PY'
import json,os
from pathlib import Path
run=Path(os.environ['P4_ROUTE_RUN'])
cache=Path(os.environ['P4_ROUTE_CACHE'])
print('本次日志:',run/'load.log')
for f in (run/'status.json',cache/'progress.json'):
    print(f, json.loads(f.read_text()) if f.is_file() else '尚未生成')
PY
  exit "$rc"
)
```

- returncode0且progress.status=production_routes_validated：该索引中的生产路线校验完成。full_pool_admitted仍为false；不直接启动整轮训练。
- returncode124：本轮到时停止。progress中newly_validated是本次新增路线，reused是本次已复用路线，checked_this_invocation是本次走到的位置，并非跨运行累加。若有新增，下次原命令保留cache、换新output即可续接，无须从头重查证明；启动SHA检查仍会重复。不要自动while循环。
- 其它错误、没有新增路线，或尚未生成progress：交status.json和load.log末60行，停止盲目重试。初始绑定阶段仍可能耗时，需要日志判断，不能保证4分钟完成全库。
- 中途Ctrl+C也保留已落盘路线回执；同一cache不允许同时跑两个进程。日志输出每100条路线更新一次，progress.json每条更新。

## 路线步骤成功后的完整原生校验（单独执行）

```bash
python -m qwen3vl_local.audit_joint.phase4_load \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --production-cache checkpoints/p4_verified_routes_v69 \
  --timeout 240 \
  --output "checkpoints/p4_full_load_v69_$(date +%Y%m%d_%H%M%S_%N)"
```

该步骤仍完整解析train、检查每题/跨split/覆盖及原生准入摘要，路线证明可复用；不生成view，不代替正式训练准入。需要status.json completedtrue和result.json native_validated。超时仍交日志，不承诺全量JSON和其余检查都能在240秒内完成。`phase4_load --view ... --production-cache ...`亦支持复用，但先确认native校验结果，不同时排队迁移或正式训练。

本地验证：85相关测试通过；真实子进程超时保存首条路线，下一次只补剩余条目；17路线1457train数据原生返回与无缓存完全一致，路线首查1.13秒、复用0.019秒，完整适配读取1.64→0.56秒。仅小池CPU结果，不外推服务器速度或模型质量。冻结producer/teacher/学生源码合同字节未改；正式训练器没有自动接入此缓存，当前只作用于上述审计/迁移入口。

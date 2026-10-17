# v82：接续启动保护、RGB复核与服务器唯一执行步骤

本次只完善外围接续入口：同一输出目录的并发接续立即拒绝；错误GPU_IDS在预检/写输出之前拒绝；悬空输出软链接也视作已有输出，防止覆盖。目录锁从预检持有至打包结束，进程退出由内核释放，无锁文件需要删除，软链接别名共享同一目录锁。只协调新版phase4_continue，不保证拦截旧脚本或任意外部写入；Linux本地真实进程验收通过，服务器挂载的锁支持仍需实际接续确认，失败直接报错不降级。

**不改学生/teacher/提示词/采样/读取适配器，不重建view或快照。现有RGB2训练、成功reload、权重及optimizer继续复用。** v81哈希边界修正保留，服务器最初失败根因仍未证实。正式训练依旧四卡、每卡B2；Phase4累积4，十事件每轮各1024。test固定18/event。

## 本地验收

- 新增8个测试实例（含5种错误卡号），覆盖参数拒绝不写输出、悬空链接拒绝、真实子进程与目录别名互斥、SIGTERM退出后锁自动释放、整个执行/打包期间持锁。其中最初7个实例在旧实现失败，再补等价数字重复卡号反例；最终28个接续专项通过于下述105项内。
- 105项相关测试15.25秒通过：接续、首轮编排、交接打包、哈希边界、真实CPU torchrun、运行快照；不是服务器GPU或文件系统锁验收。早期测试调用的相对PYTHONPATH在子进程切cwd后失效，改为绝对路径/AutoMoT cwd后完整通过，未修改生产导入或测试断言绕过。
- v82发布清单564源码；仅外围接续/既有测试和三个发布指针变化。已完成的Phase3三轮及Phase4 RGB2不重复运行。

## 本轮逐帧RGB

用户要求并行启动，后台审计任务完成原生重放后因调用额度中止；主任务接手检查原始证据并实际看完16帧RGB，未把后台未完成的视觉部分计作已审。两条均为既有train路线，未用val/test错例选择素材；和紧邻上一轮40帧不重叠，但不是全新独立场景或盲审。

原生从起点重放93帧，279份源SHA前后核对，16帧对应48份RGB/meta/bbox，13道RGB4提议。完整证据留本机`checkpoints/v82_release_rgb_20261017/`，不入Git。

| 路线/帧 | RGB与规则复核 | 问题结论 |
| --- | --- | --- |
| VehicleTurningRoute / Town06_Rep0_Town06_Scenario4_17_route0_01_08_22_49_38，53–60 | 骑行者向左离开路面，57帧左侧可见两骑行者，58帧后消失。proceed为53–54 NO，55–57 UNKNOWN；57的release/clear为True，但priority仍未知。58以instance_rgb_discontinuity结束，completed=false，无后续题 | 没有把离开/消失当完成或放行。雾天与优先权信息不足仍是可观察性限制，无确定改标依据 |
| HardBreakRoute / Town12_Rep0_1902_0_route0_01_09_03_48_13，24–31 | STOP标志与横向车辆可见。25短暂停车记录被26/27速度回升重置；28–30连续停稳建立义务完成，30/31 satisfied=true，但全部8题仍NO | 原始stop_sign_hazard全段True，是priority_satisfied=False的明确上游否决；停车完成并不覆盖其它限制。RGB不能单独证明全部通行权，无新增放行YES或改标依据 |

本轮没有改阈值、标签或prompt，不因这次小样本审核重新回放全池。U-E5缺proceed YES、U-E4/6模型YES召回偏低和评测物理组稀疏仍保留；先完成两模式对照。上轮十事件审计及284项回归见[开训收敛报告](TRAINING_CLOSURE_AUDIT_20261017.md)，不与本次105项累加为独立测试数。

## 服务器步骤

在服务器**已有AutoMoT根目录**、之前成功RGB2训练的环境中运行。通常根目录是`/mnt/workspace/cee3sgh/AutoMoT`；若该任务实际位于其他路径，使用包含当前checkpoints目录的那个根。已有任务运行期间不要同步覆盖源码。以下是一段顺序执行命令，任一步失败停止：

```bash
set -euo pipefail
bash ./sync_automot.sh
python - <<'PY'
from pathlib import Path
from qwen3vl_local.audit_joint.server_readiness import source_check
r=source_check(Path('.'),Path('qwen3vl_local/audit_joint/release_sources_v82_continuation_guard_20261017.json'))
print('v82 source:',r['status'],'files:',len(r['files']),flush=True)
for f in r['files']:
    if f['status']!='matched': print(f,flush=True)
if r['scan_errors']: print(r['scan_errors'],flush=True)
raise SystemExit(0 if r['status']=='matched' else 2)
PY
P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69 \
python -m qwen3vl_local.audit_joint.phase4_continue \
  --root checkpoints/phase4_restart_20261017_162511 \
  --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B
```

默认自动选空闲四卡；如果需要明确固定卡，仅将上面的最后一条替换为下面这条，不执行两遍：

```bash
GPU_IDS=0,1,2,3 P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69 \
python -m qwen3vl_local.audit_joint.phase4_continue \
  --root checkpoints/phase4_restart_20261017_162511 \
  --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B
```

实际顺序为：CPU检查现有RGB2快照→RGB2固定test→RGB4首轮/val→RGB4重载→RGB4固定test→打包。已经通过的RGB2训练和reload不重跑。不要运行旧first_epoch新建入口，不删除当前输出，不重建原生库/迁移view，不追加readiness/batch测试，不自动续七轮。

如果此目录已经有continuation.json或handoff_continued，直接反馈已有输出，不删后重跑。锁冲突保留现有进程/输出，不强制结束其它任务。前置错误只交终端报错；执行阶段首错停止，优先交下面两份文件。现有权重不随交接ZIP上传，整个run目录必须保留。

## 反馈什么

成功或执行阶段失败，均交：

```text
checkpoints/phase4_restart_20261017_162511/handoff_continued/phase4_epoch_report.zip
checkpoints/phase4_restart_20261017_162511/handoff_continued/phase4_epoch_package.json
```

若没有生成ZIP，交完整终端错误以及已经存在的`continuation.json`、`continuation_snapshot_check.log`末尾80行；若已经进入test或RGB4，再交相应`continuation_rgb2_test.log`或`rgb4.log`末尾80行。不要求传权重、原始RGB或全库。日志尚未生成就明确说明，不为补文件重复启动。

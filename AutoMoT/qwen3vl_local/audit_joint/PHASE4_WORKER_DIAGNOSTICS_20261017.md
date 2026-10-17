# Phase4 升级139日志分析与最小诊断修补

## 已知事实

服务器交回两层日志尾和status，均为同一个worker PID 1662801，不是运行了两次。8614路线输入绑定完成，298787题JSON读取62.13秒，协议复用25万次在173.4秒有进度，随后进入validate_model_inputs。最后一次30秒定时栈打印停在JSON encoder后的半行`File`，status returncode=139、completed=false；先前workflow记录总210.27秒。

`Timeout (0:00:30)!`来自`faulthandler.dump_traceback_later(30, repeat=True)`，不是900秒任务超时，不是每30秒重启。当前证据没有GPU模型加载、训练或CUDA OOM。原生val/test为0符合独立references4留出设计，不是此次崩溃原因。

## 判断边界

定时栈打印中途截断并随即139，提示应先排除诊断器本身。CPython官方[issue 116008](https://github.com/python/cpython/issues/116008)记录过Python3.11在打印到`File`时于dump_frame崩溃；[issue 131580](https://github.com/python/cpython/issues/131580)也涉及线程环境下faulthandler崩溃。前者使用信号触发，不与本次watchdog线程路径完全相同。仅凭相似日志不能确认同一根因，更不能把最后显示的json.dumps当作已确定故障函数。未取得本次C栈/解释器补丁版本，不要求用户生成巨大core或立即重装环境。

## 已实现

benchmark_runtime.trace_waits取消周期性watchdog栈打印，保留fatal faulthandler，启动输出Python版本及诊断配置。父进程每30秒输出PID、已用时间和上限；明确存活不代表阶段已完成，不向worker发诊断信号、不引入worker Python后台线程。保留原有完整数据校验、超时124及139等返回码传播、SIGINT/TERM和独立session后代清理。退出时打印实际返回码，原子成功回执要求不变。

两项新行为测试旧代码先失败；补丁后98项相关回归19.65秒通过（含真实CPU worker执行JSON循环、父进程存活日志、超时/中断清理、读取/升级/路线缓存/交接/首轮编排）。没有复现服务器原始SIGSEGV，测试通过不能当作139已解决；未再跑GPU/建库/老师回放。

v78清单558源。除benchmark_runtime、其测试及三个发布指针外，v77所有绑定源码字节未变；学生/原生/老师/标定/提示词/采样、phase4_load及phase4_production_cache保持原SHA。已有有效学生view、准备快照及v69缓存不因本次诊断补丁失效。旧v61 view仍需之前已经确定的升级，不需额外重建原生数据。用户已授权本轮白名单发布，远程提交以推送核验回执为准。

## 发布同步后的唯一下一步

先同步v78；不再跑readiness、smoke、batch比较或独立全池测试。确认没有本实验的活动进程后，以新输出目录执行既定B2首轮入口。它完成升级后直接继续两模式一轮、四卡验证/重载/test及交接，不是只做诊断然后另跑一遍。

```bash
(
set -euo pipefail
export PREPARED=checkpoints/phase4_v57_20261014_232157_762998141
export DATASET=checkpoints/phase4_epoch1_20261015_214227_800375064/student/view
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32
export GRADIENT_CHECKPOINTING=1 EPOCHS=1
unset RESUME_RGB2 RESUME_RGB4 P4_VALIDATED_DATA
export OUTPUT_DIR="checkpoints/phase4_b2_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
test -d "$PREPARED/data4"
test -f "$DATASET/student_view.json"
test -d "$P4_PRODUCTION_CACHE"
echo "本次输出：$OUTPUT_DIR"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
)
```

缓存目录使用此前已完成的实际路径；若不在上述位置，只替换该变量，不重建缓存。默认自动选空闲四卡，不覆盖其它任务。保持这次训练源码固定，失败不自动重试或扩大超时。若仍失败，新交接包已包含升级日志，交回handoff/phase4_epoch_report.zip与phase4_epoch_package.json即可，不再手工收集同一批日志。成功也交这两个文件。

预算与规则维持：每轮十事件各1024，验证/test固定十事件等量，不要求YES:NO等量；U-E5放行正例缺失仍是开发模式限制。本补丁不改变标签质量或证明全能力通过，也不承诺整轮耗时；本次仅省去可疑的周期性诊断干扰。

## 本轮同步本地RGB复核

逐帧实际复看HardBreakRoute/Town12_Rep0_1902_0_route0_01_09_03_48_13与ParkedObstacleTwoWays/Town13_Rep0_1748_1_route0_01_08_18_54_13各40–46帧，共14张RGB。均为此前已看过的train开发窗口，不是新独立样本或盲审。42份目视源SHA通过；4题7帧因果几何事实重算的54源SHA通过，两类证据去重共60份。完整记录checkpoints/v78_release_rgb_20261017，仅轻量摘要发布。

两路线43帧stable YES、HardBreak46帧/Parked44帧re_yield YES原生事实一致。车在路口推进时侧向车辆逐渐接近，与后续限制重新出现并不构成直接矛盾；短RGB本身不能证明先前停车时长、隐藏优先权或几何走廊关系。重算沿用已绑定生产证据中的STOP历史，未完整重放两条路线或新增标签，没有发现足以更改标定/提示词的确定依据。该复核不阻塞服务器下一次执行，不认证十事件全源质量。

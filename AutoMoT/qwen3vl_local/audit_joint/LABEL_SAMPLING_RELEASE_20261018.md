# v84 标定与关键采样发布（2026-10-18）

本页替代旧的 cr02 开训建议和 cee 接续命令。cee 两模式首轮、重载、固定 test 已收到并核验，不重复执行。cr02 只有旧正式题库、没有训练和缓存。两台服务器资产不互通。

发布复核：生产修复已在远端 main 的 `c5cba7c41`；随后补充审计记录不改变此版本的571项源码清单。再次执行82项入口/采样/接续回归全部通过，真实本地只读诊断完成并生成校验通过的ZIP。此前4路线40RGB及1534项回归的详细证据见[关键采样与RGB复核](CRITICAL_SAMPLING_RGB_20261018.md)；这些测试有重叠，不累计为独立数量。新增相邻16RGB结果见[发布复核回执](label_release_followup_verification_20261018.json)。

## 本轮落地

- 发布此前本地验证的教师 v23/v24：U-E1/U-E4 的明确否决不被 UNKNOWN 覆盖；U-E3/U-E5 目标隐藏时仍保留独立阻挡或优先权否决。新 plan39/registry34，旧合同保持原样；不放宽阈值，不修改提示词。
- Phase4 连续采样升为 v5：在每事件1024预算内，为每个有原生自动 train/readiness 支持的关键边 YES/NO 各预留至少1道不同题。余量继续采用原连续/低曝光优先；逐轮轮换稀少题。缺格如 U-E5/proceed/YES 只报告，不伪造。不是 YES:NO 等量，也不保证一次曝光足以学会。
- 默认关键边与 `critical_support()` 使用相同清单，由回归测试保证一致：U-E2 return、U-E3/5/6 proceed、U-E7 complete、R-E5 proceed。catchup、人工复核题不冒充原生关键支持。历史绑定包含新策略和最低曝光；旧 v4 history/checkpoint 不得跨策略恢复。
- Phase3 未请求最低曝光时保持 v4 选择算法；共享源码 SHA 已变化，已有 Phase3 checkpoint 继续保留原版环境，不因此重训或直接跨合同续训。其测试夹具补齐既有 batch 评估所需 tokenizer/device，不改生产训练。
- 新增 `label_release_check`：只读 manifest 和一次 train.jsonl，验证原始 SHA/数量/读取中漂移，列出当前合同差异，模拟两模式B2/四卡各1024事件预算。自动输出 ZIP 和回执，实际 ZIP 上限30MB；坏输入也产生阻塞报告。它不读取图像、不加载模型、不建立缓存、不改数据、不证明全部标注可靠。

## 本机验证和 RGB

- 本轮重新原生重放3完整路线266帧、核对798份源文件SHA；编译/两模式重载各130训练题、无val/test。与上一轮 v24 的130题逐字段一致。
- 实际复看27张既有RGB：1176_0的71–79、4591_0的54–62、1378_0的71–79。属于开发复核，不是27个新增独立样本或盲审。黑车消失后前方黄车仍可见；另两路线红车出画并不能证明可观察放行。未发现新增足以改变阈值或提示词的证据。
- 57路线的3187道历史训练题仅做CPU采样模拟：两模式十事件均1024。该局部池仍缺若干关键格，不外推服务器全池。旧题库被正确报告为合同不同，不给新版准入。
- 1457项相关回归通过，2项因旧生产池不存在跳过；另127项入口/分布式/采样测试通过，和前组有重叠，不相加。包括真正四CPU worker 的消费验证；不是本轮服务器GPU测试。

现有 U-E5 proceed YES 仍为0。已修的是负证据丢失，不能宣称完整放行能力已解决。十事件训练等量与固定val/test等量机制保持，但新版正式池和实际消费仍需各自验收；本轮不重选固定test，也不据其结果调标签或采样。

## 服务器现在只执行这一项

先确认本机没有训练/建库任务正在使用源码，再按已有 `sync_automot.sh` 同步。不要使用服务器遗留 `.git` 来判断版本，不删除现有目录。此处命令从**该服务器 AutoMoT 根目录**、既有 Python 环境执行。

### cr02（当前需要交包的机器）

```bash
(
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
bash ./sync_automot.sh
export AUTOMOT_SERVER=cr02
P4_DATA=checkpoints/phase4_labels_20261017_125230_304771113/data4
P4_CHECK="checkpoints/label_release_cr02_$(date +%Y%m%d_%H%M%S_%N)"
echo "交付目录：$P4_CHECK"
python -m qwen3vl_local.audit_joint.label_release_check \
  --dataset "$P4_DATA" --output "$P4_CHECK"
)
```

交回打印目录下两个文件：`label_release_report.zip`、`label_release_package.json`。成功或失败都交；退出2先看包内errors，不自行重建或重训。报告的 `diagnostic_complete` 只表示诊断完成，`training_authorized` 始终为 false。旧合同的 `different` 是预期，不是同步失败。

只顺序读取一次训练JSONL和两次CPU采样，不是20小时回放。不承诺服务器固定分钟数；每25000行和每种模式完成都会打印进度。

### cee（本轮无需重复执行）

保留 `checkpoints/phase4_restart_20261017_162511` 及其权重、view和缓存。其结果已经核验，不再运行 phase4_continue。若之后需要在cee做相同诊断，先同步，显式 `export AUTOMOT_SERVER=cee`，从该机已有 workflow.json 的 `prepared` 定位该机 data4，再调用同一模块；不能复制cr02路径。缺workflow或prepared时先报告，不猜数据位置。

本轮不发新的GPU命令。收包后只决定新版生产标签更新范围和新实验所需资产；不通过修改旧SHA、旧view或旧checkpoint来跨过版本检查。旧服务器题库与新标定的正式转换尚未验收。

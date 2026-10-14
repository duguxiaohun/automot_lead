# v55 白名单发布、服务器只读盘点与本地 RGB 续审

2026-10-14。用户授权本轮白名单提交和 push。发布包含 STOP 义务保留修复、R-E5 放行正反支持硬门、只读 STOP 盘点/代理工具、首轮交接健壮性及相应测试和轻量记录。使用 `release_sources_v55_re5_admission_20261014.json` 的489项清单；生产仍为v55/teacher20。已有 Phase3 三轮结果保留，不重复。

## 服务器现在执行

在原服务器 `AutoMoT/` 目录，使用用户现有 `sync_automot.sh` 覆盖同步。这里接受既有同步方式，不要求迁移实验目录。如下步骤不启动建库、不加载模型、不训练；只读取既有正式回放。

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
bash sync_automot.sh
conda activate automot

# 仅校验本次489份源码；不是重新运行readiness。
python - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('qwen3vl_local/audit_joint/release_sources_v55_re5_admission_20261014.json')
files = json.loads(p.read_text())['files']
bad = [name for name, sha in files.items()
       if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha]
if bad:
    raise SystemExit('SOURCE_MISMATCH: ' + ', '.join(bad))
print('SOURCE_MATCHED', len(files))
PY
```

只有打印 `SOURCE_MATCHED 489` 后，执行：

```bash
P4_PREPARED=checkpoints/phase4_existing_20261013_184628_526027234
REPORT=checkpoints/phase4_stop_inventory_v54_20261014.json
if [ -f "$REPORT" ]; then
  echo "已有报告，直接交回：$REPORT"
else
  python -m qwen3vl_local.audit_joint.stop_inventory \
    --index "$P4_PREPARED/inputs/production/index.json" \
    --output "$REPORT"
fi
```

交回 **`checkpoints/phase4_stop_inventory_v54_20261014.json`** 即可，放到本机 `/home/codon/Downloads`。若校验失败，交终端错误；若盘点退出2但生成JSON，照样交JSON，不重试覆盖。该工具成功状态是 `inventoried_not_relabelled`，失败是 `incomplete`，两者都不表示允许训练。

此扫描要顺序解压旧回放并前后校验SHA，因此耗时由回放大小和磁盘决定，尚无全服务器实测时长；它不执行上次约20小时的原始数据建库。不要同时再运行 `server_phase4_existing.sh`、旧池首轮或旧优化器续训。已有结果保留。

收到报告后结合本地位置校准确定正式池更新范围，再准入当前合同下的训练/固定val/test。没有STOP问题记录不能证明路线没有STOP；盘点不授权直接复用旧合同路线。R-E5缺正例不能被“十事件都有题”或U-E5开发豁免掩盖。

## 本轮独立复核

- 101项STOP、R-E5原始/过滤支持、开发策略、等量采样和交接测试通过，2.74秒。两个Phase4 launcher的shell语法通过；489份发布源码逐项SHA一致。未重复完整回归及GPU测试。
- 本机实际CLI扫描旧/新17路线回放，各2205帧：旧回放找到8条未完成记录消失后的proceed YES候选，v55为0；均无扫描错误。不是全池标签认证。
- 实际查看3段连续窗口，共21不同RGB帧，并在查看后核验63份RGB/meta/bbox源SHA。两条既有开发路线，不是盲审。未新建完整题库；诊断逐帧加载原始源，题目展示来自已绑定的v55回放。
- HardBreakRoute / Town12 1902_0，第二处STOP528：f160–166仍在前车后等待，代理gap约7.17m，NO有当前等待依据，不应该仅因停车就计完成。f253–259移到路口后连续停车，速度绝对值均小于0.011m/s，但代理gap约−1.954至−1.957m，完成记录仍False。当前窗口没有该事件题目，不能凭空补题，也不能把这一记录直接计作错标。
- ParkedObstacleTwoWays / Town13 1748_1 f17–23：接近STOP/人行横道，速度仍约0.104–1.493m/s，proceed均NO；未看到足以翻转标签的依据。

第二处STOP重复出现负代理间距，进一步支持需要核准导出控制体与有效停车位置的对应关系。导出固定1.5m半长框不是道路停止线；本轮未扩阈值、改提示词或补YES。详见 [代理定位报告](RE5_SUPPORT_PROXY_20261014.md)。

完整本机证据：`checkpoints/v55_release_rgb_20261014/`；[发布核验回执](v55_release_rgb_verification_20261014.json)。数据、联系表、权重、历史凭据及用户本机设置不随本次发布。

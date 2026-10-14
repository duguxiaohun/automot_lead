# v57：STOP停车位置已接入原生标定

本次发布与服务器最新步骤见[SERVER_V57_RELEASE_RGB_20261014.md](SERVER_V57_RELEASE_RGB_20261014.md)，下文“未push”描述保留为此前审计时点。

最新交接补强见[V57_HANDOFF_REVIEW_20261014.md](V57_HANDOFF_REVIEW_20261014.md)：两入口现指向v57_handoff清单，标定合同不变，失败诊断可打包；服务器命令不变。

2026-10-14。此前连续停在隔离实验是实现未收尾，不是缺少本地数据或必须等待服务器。本轮已经接入原生 `StopDutyTracker`，完成真实回放、自动编译与短两／四图重载。无需人工逐题填标签，也没有用专家起步替代停车证明。

## 生产变化

新增 `stop_position.py`：采用**匹配控制体对应的驶入车道端点平面**作为自动弱规则的位置基准。它不是导出框近缘，也不声称等于绘制停止线。仍要求车头在基准前 0–3m、绝对速度≤0.1m/s、连续三帧，保留原车道归属、义务持久化及当前横穿检查。

从本地已有12份地图自动提取1706个支持的STOP驶入车道，形成766269字节的 `stop_boundary_catalog_v1.json`。记录地图SHA、道路／车道、唯一signalReference、连接车道、重建控制中心、端点和数值精度预算；仅由地图生成，不从训练标签或停车样本拟合位置。Town15/151/-1的非支持几何明确排除；无唯一STOP或条件不成立的道路不生成绑定。服务器随源码取得这份紧凑资产，无需另外安装或复制204MB地图。

实际帧的镇、道路、车道、控制ID、固定导出形状和世界位置必须匹配。控制体匹配容差沿用v56的float32数值预算，未改变停车距离阈值。建立后保存原始绑定、源帧、地图和目录SHA；换路、关联标志清零不会重绑。初始不匹配允许在仍处原车道、关联仍有效的后续帧重新尝试，但不回填先前停车帧。

断帧、坏几何、控制体丢失／漂移、目录资产变更会撤销停车完成证据，保留义务。没有旧框近缘回退。重新取得有效证据仍需连续三帧；退休仍要求已完成且同一控制体完整通过自车后方。停车完成本身不跳过横穿交通检查。

`teacher_replay`核验绑定、建立帧、三帧位置／速度和来源；`teacher_data`验证建立帧源文件SHA。建立来源也纳入`instance_sources`和风险窗口，使源异常隔离及审计证据范围覆盖绑定历史。任务v57／teacher22，新协议v37和空registry32，独立预约不变，未授予人工准确率认证。501份源码清单包括新目录资产与生成器，两个Phase4入口已更新；尚未push。

## 本地实际结果

四条既有完整train路线共929帧：HardBreak Town12 1902、Parked Town13 1748、Accident Town12 26、Accident Town13 1153。**使用原生函数，没有monkeypatch。** 最终gzip回放→共享RGB4→短视图→重载约35.62秒，两模式各771道train题、0 val、0 test；ID／标签配对。

| 同四路线 R-E5/proceed | YES | NO | UNKNOWN |
|---|---:|---:|---:|
| v56原生 | 0 | 199 | 28 |
| v57原生回放 | 20 | 199 | 4 |
| v57自动编译后 | 20 | 199 | 不作为监督 |

20条YES全部通过原生编译过滤，来自4个物理组，均为readiness自动规则监督。对应三帧停车记录的车头间距2.503–2.744m，速度绝对值最大0.08965m/s，满足原阈值。原始883对两／四图题配对；对v56共同语义键29 UNKNOWN→YES、4 UNKNOWN→NO，另外新增233、移除67个RGB4语义键，包含放行后实例状态变化，不能只当33条静态改标。

2787份源文件和12份源地图SHA在末次核验；原v56库被新合同拒绝，旧SHA与产物未覆写。有效四路线产物：`checkpoints/stop_position_v57_20261014/native_bundle_final/`。中间小库保留仅供调试，不用于正式实验。

另用既有17条完整train路线做十事件原生回归：2205帧、约163.33秒，两模式各1457 train／0 val／0 test，十事件齐全；6615份源SHA重放前后和重载后核验。R-E5/proceed YES为10条／2物理组（与上述四路线集合重叠，不能相加作为独立样本）。两模式×world1/4四份首轮计划同序，各事件1024次，未实际GPU消费。完整产物位于`ten_events/`，采样回执为`ten_event_verification.json`。空留出和局部关键格缺口仍明确，不能据此宣布正式train/val/test均衡验收。

本轮实际查看28不同RGB：HardBreak f33–39、Parked f33–39、Accident Town12 f32–38、Accident Town13 f35–41；84源查看后核验。可见STOP、路面标记、侧路车辆及自车由静止转向前的过程；这些是既有开发窗口复看，不是独立盲审，也不据自车起步作安全结论。没有新提示词改动依据。

相关回归596项通过，1项短视图固定ID测试因base缺torch未完成（此前pvi亦缺tokenizers）；没有宣称完整套件全过。实际两模式小库加载已另行完成。新测试覆盖负旧代理间距仍可通过合法新边界、几何错配拒绝、已完成后证据撤销、目录漂移、来源和绑定篡改。早期测试断言把内部整数ID当字符串、代理形状测试未适配新fixture，均已修正并保留失败日志。

## 适用边界与后续

**“停车位置尚未接入”这一代码阻塞已关闭。** 这是采用匹配车道边界的自动弱监督规则，不等于认证历史采集地图版本或真实道路法规停止线。目录不支持或控制不匹配时保持UNKNOWN；不能把1706个地图候选都称为经过RGB认证的停车点。

下一阶段是正式池数据验收：发布同步v57后，用现有源数据生成一次新版正式库，检查R-E5正反例、其余关键格、物理组隔离、train／val／test十事件间等量，再两模式首轮及保存重载。YES:NO不强制等量；U-E5已知缺格仍只按existing_data_development报告，不扩展豁免。Phase3、旧readiness和STOP盘点均不重复。

服务器在**本轮代码发布同步后**，于现有`/mnt/workspace/cee3sgh/AutoMoT`目录运行：

```bash
conda activate automot
bash ./sync_automot.sh
DATA_ROOT=lead_data \
MODEL_DIR=checkpoints/Qwen3.5-4B \
REPLAY_WORKERS=2 \
OUTPUT_DIR="checkpoints/phase4_v57_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase4_existing.sh
```

此入口先检查501源，再做CPU建库／自动留出／短视图／预算与准入；不训练。旧完整构建曾用约20小时，新版没有全源计时，不承诺耗时更短。不要复用旧v54/v55/v56回放拼接新合同。按脚本打印路径交回`handoff/fullflow_report.zip`和`handoff/fullflow_package.json`；失败也交这两份诊断文件，不清理旧产物。当前未push，因此现在不要拿未同步的旧入口运行。

完整本地证据：`checkpoints/stop_position_v57_20261014/`；[核验回执](stop_position_v57_verification_20261014.json)。本轮未远程运行、未训练、未push，未修改用户`.vscode`。

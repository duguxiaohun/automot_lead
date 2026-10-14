# R-E5 覆盖准入修复与 STOP 代理来源核验

> **状态更新：服务器STOP盘点已交回核验，下面历史盘点命令不再执行。** Phase3不重做，Phase4目前集中停车位置归属校准。见[盘点验收](STOP_INVENTORY_RECEIVED_20261014.md)和[最新位置诊断](STOP_LINE_CALIBRATION_20261014.md)。

2026-10-14。**R-E5 放行正反例的准入漏检已修复；停车位置代理尚未完成校准，Phase4 暂不启动旧池训练或全库重建。** Phase3 三轮已关闭，不重复。只使用现有数据。

## 本轮代码

- `rule_baseline.critical_support()` 增加 `R-E5/proceed/readiness` 的 YES、NO 自动训练支持，分别报告题数和物理路线组数。人工参考、其它边、catchup、val/test 不能补格。
- `server_finish.raw_training_support()` 与过滤后准入共用 `CRITICAL_EDGES`，避免两处清单再次分叉。原始支持缺失时停止编译；源池有支持但最终过滤后缺失，仍阻塞训练准入。
- `existing_data_development` 仍只豁免已知 `U-E5/proceed/YES`，不豁免 R-E5，也不更改十事件训练、固定 val/test 等量要求。事件间 1:1 不代表 YES:NO 1:1。
- 新增只读 `stop_proxy.measure(frame, stop_id)`，分别输出导出框前缘、中心、后缘与自车前缘的距离，并绑定源文件；始终声明 `unproven_stop_line`、`label_authorization=false`。缺演员不会变成停车完成。

本轮没有修改老师、标签、阈值、提示词、训练超参。生产合同仍为 v55 / teacher v20，冻结协议和旧发布清单保持原字节。准入源码属于短视图绑定合同，因此旧短视图会明确拒绝加载；已合格的 v55 父库可以创建新的轻量视图，无需复制回放或 RGB。**这不授权旧 v54 父库冒用 v55 合同。**

## 实际数据核验

复用旧 17 路线 v54 的 1471 题，移除全部 8 条 R-E5 放行 YES 后，新关键报告由 8 题 / 2 物理组变为 0 / 0，并被开发模式硬阻塞。此前漏掉的是这个关键支持字段，不能据此声称旧完整准入必然通过。

复用已生成的 v55 父库，新视图仅 23,752 字节，两模式各 1381 题、ID/答案配对一致。R-E5 readiness 为 0 YES / 140 NO、NO 来自 3 物理组；两模式诊断均保留缺格。小池还有其它关键缺格、空留出和缺预算，不可用于正式训练，也不能外推服务器全源分布。本轮未重新回放 17 条路线。

重新读取两条实际路线 f24–39 共 32 帧，96 份 RGB/meta/bbox SHA 前后及目视后均一致。实际复看既有 f24–30 两段连续 RGB，共 14 不同帧、42 源文件；是开发复核，不是新盲审，也不是 32 帧全部目视。

## 停车代理的来源限制

本机 `lead/lead/expert/expert.py:2622` 将 STOP 框中心设为停止牌触发区中心，但尺寸**固定写成** `Vector3D(1.5, 1.5, 0.5)`；`expert_data.py:1960` 将其作为 stop_sign 的 extent / position / yaw 导出。这不是原始 trigger extent，也不是道路绘制停止线。两条路线 32 帧的导出尺寸都符合这个固定形状；历史采集服务器的源码 SHA 未核验，不能据此断言全部旧数据由本机版本产生。

| 路线 f30 | 速度 m/s | 框中心相对自车原点 m | 框中心距车头 m | 框前缘距车头 m | 框后缘距车头 m |
| --- | ---: | ---: | ---: | ---: | ---: |
| HardBreakRoute / Town12 1902_0 | 0.000975 | 1.983806 | −0.462385 | −1.964783 | 1.040013 |
| ParkedObstacleTwoWays / Town13 1748_1 | 0.002502 | 1.973113 | −0.473078 | −1.974461 | 1.028306 |

可见画面与速度支持车辆有停下，不能证明现有停车位置门已满足。约 −1.96m 是导出框前缘的距离，**不能直接解释为越过道路停止线**。改用框后缘虽会得到正距离，但没有证据说明该边界是有效停车位置；改用中心仍有约 −0.46m 的负距离。故不能扩大阈值、任选一个边界、使用专家 cleared 标志或后续起步来生成 YES。

进一步找到本机已有 `Town12.xodr`、`Town13.xodr`。按直线段坐标投影进行只读探查，在两个控制体中心约 3.09m / 3.05m 处找到 `StopLine2` 候选；但最近线所在 road 并不直接等于自车当前车道的 junction 连接 road，不能用“最近停止线”自动建立归属。已保存地图 SHA、原 signal 属性和 laneLink（`map_probe.json`）；未认证历史采集地图版本，未支持曲线段投影，未将这些候选用于改标。这为后续校准提供了现有资产线索，不需要新采集。

下一项本地校准需要从现有地图/采集定义中建立可复核的位置对应关系，并验证完整实例上的停车持续、同一控制体身份、横穿交通及后续状态影响。若现有资产无法提供对应关系，应继续 UNKNOWN 并报告能力缺口；不通过假补标签消除准入阻塞。自动数据流程无需逐题人工标注，但这一位置规则不能靠少量画面猜测完成校准。

## 验证与限制

- 117 项相关测试通过，覆盖 STOP 生命周期、准入、采样和原始支持入口，含本轮 16 条新增参数化用例。
- 另外 2 项未能完成：基座 Python 缺 torch；换用已有 pvi 环境后分别缺 peft、tokenizers。首次命令还有一次测试节点名称错误，日志保留。没有宣称完整回归全部通过。
- 新 `release_sources_v55_re5_admission_20261014.json` 共 489 项源码 SHA matched，无额外 Python 文件；两个 Phase4 建库 launcher 指向新清单，shell 语法通过。**尚未提交或 push，这不是立即建库指令。**
- 原始证据：`checkpoints/re5_support_proxy_20261014/`；轻量核验回执：`re5_support_proxy_verification_20261014.json`。未训练、未远程执行、未新增标签或全库回放。

## 服务器下一步

继续按 [STOP 修复手册](STOP_V55_FIX_20261014.md) 的只读盘点步骤处理既有正式回放，已生成报告则直接交回，不重复执行。发布同步后、报告尚不存在时：

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
conda activate automot
P4_PREPARED=checkpoints/phase4_existing_20261013_184628_526027234
python -m qwen3vl_local.audit_joint.stop_inventory \
  --index "$P4_PREPARED/inputs/production/index.json" \
  --output checkpoints/phase4_stop_inventory_v54_20261014.json
```

交回这个 JSON 即可。盘点只定位已有证据，不授权无 STOP 记录的旧路线直接复用。先结合停车代理校准及盘点确定更新范围，再生成当前合同正式池、核验 R-E5 等关键支持和十事件固定等量留出，最后两模式首轮/重载。现有训练产物保留，不删除、不自动续训。

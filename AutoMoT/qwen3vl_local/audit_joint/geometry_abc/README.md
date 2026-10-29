# 独立几何 a／b／c 实验

本入口从 `geometry_v1` 的原生轨迹位置父库派生三组区域标签，不调用六动作标定，不要求当前或未来六动作有效。旧 `geometry_study` 四组交集实验、六动作入口及其数据和权重保留。

## 表示规则

三组固定当前原点和当前 Driving 车道切线（朝自车前向定向），横向小于 −2 m 为 LEFT，大于 +2 m 为 RIGHT，其余 CENTER。径向距离是水平直线距离，不是沿轨迹弧长。

| 方案 | 径向分区 |
| --- | --- |
| a | 保持原规则：`q = r/t − v_current`，`delta = min(1, max(0.5, 0.2*v_current))`；q 小于 −delta 为 NEAR，大于 delta 为 FAR，其余 REFERENCE |
| b | 当前道路 R3：20／24 m；R1、R2、R4、R5：3／13 m |
| c | 全部已知道路：3／13 m |

边界来源（2026-10-29 补记，原先只记在 `checkpoints/` 的审计产物里）：横向 ±2 m 来自车道宽度统计，见 `../PHASE3_DIRECTION_PROGRESS_3X3_20261021.md`（全部合格帧 99.07% 的当前车道宽 2.75–4 m，中位 3.5 m）。b／c 的径向半径来自 `checkpoints/geometry_strategy_audit_20261022/`（`distribution.json`、`fixed_proposals.json`、`research.py`）：在当时的训练候选池（排除已收到的测试组与固定验证组，251,431 条记录）上，按每个事件、事件内每个物理路线组等权，取 2 s 端点直线距离的等量三分位，其它道路 2.27／12.71 m、R3 高速 19.68／23.67 m；再对照车速分布（其它道路车速第 67 百分位约 6.6 m/s，2 s 约 13 m；高速第 33–67 百分位 9.8–11.3 m/s，2 s 约 20–23 m）取整为 3／13 m 与 20／24 m，其它道路下界由 2.27 上调到 3 m。取整后 2 s 的近／中／远加权占比为其它道路 36.5／32.5／31.0%、高速 34.8／33.3／31.9%；短时距偏向 NEAR（高速 0.5 s 全为 NEAR），c 在高速上 2 s 有 92.5% 为 FAR。2026-10-29 已在 codon 上按同一成员与权重只读复算，数字一致；本次没有改变任何标签或阈值。

b／c 四个时距共用绝对边界，等号归 MIDDLE，FAR 无上界。b／c 的 MIDDLE 不表示匀速；本次不重拟合阈值，不使用验证／测试结果调整距离圈。未知道路类别不能默认为普通道路，因为 b 需要道路类别。

未来端点为 +0.5、+1、+1.5、+2 秒；过去端点为 −2、−1.5、−1、−0.5 秒。过去位置减当前位置，a 的历史也使用当前速度及正回看时长。历史缺失输入 UNKNOWN，未来缺失屏蔽相应 loss；不要求四个端点全部有效，也不把缺失填成 KEEP。

同一原始样本在三组可以得到不同标签。例如高速道路上距当前位置 15 m、时距 2 s、当前速度 7.5 m/s 的中间端点：a 为 CENTER_REFERENCE，b 为 CENTER_NEAR，c 为 CENTER_FAR。

## 建库、采样及训练

- 使用验证过的几何父库，继承物理路线划分和开发集隔离；不移动旧测试成员。
- 仅保留道路类别明确、至少一个未来几何端点有效的记录。三组使用相同成员和有效端点；标签及类别分布分别计算。
- 采样质量仅统计 a／b／c；十事件等额，未曝光帧优先，标签质量使用 natural／sqrt 各半的软偏好，不强凑九类等额。
- 每物理帧每轮最多一次、三轮最多三次，容量不足拒绝。三组共用计划和顺序，实际成功更新才推进曝光。
- 默认每事件每轮 1,280，四卡 B1，三轮每组 9,600 次更新、warmup 2,500；三组合计 28,800 次更新。旧约 2.12 s/步只能作粗估，三组纯训练约 17 小时，另加准备、验证与测试，非本次 GPU 实测。
- CPU 检查 train／val／test 十事件覆盖、固定验证成员的三组纯类支持、三轮容量。固定每事件按 ID 取最多 16 个验证成员；按已支持纯类 macro recall 选模，完成预算后只测试被选 checkpoint。
- 覆盖通过不代表统计支持充分。CPU 报告保留缺类、混合比例及单一物理组集中度；测试结果需连同支持量解释。

新表示合同为 `current_reference_geometry_abc_v1`。数据、准入、计划、权重及 expert 条件绑定新合同；旧四组交集库或权重不自动迁移。`geometry_study` 源码保持原样，新运行代码是独立版本，后续修改需分别验收。

## 运行

在当前机器已有 AutoMoT 应用根执行。统一公开入口 `geometry_study_abc.sh` 现在指向本方案，输出根为 `checkpoints/geometry_abc_v1`，不会覆盖 `checkpoints/geometry_study_v1`。

```bash
# 仅在本改动已发布到白名单仓库后同步
bash ./sync_automot.sh

# CPU：自动使用本机项目数据、CARLA 地图和兼容几何父库
bash qwen3vl_local/audit_joint/geometry_study_abc.sh cpu

# 只读状态，含 CPU 阶段及各组轮末报告
bash qwen3vl_local/audit_joint/geometry_study_abc.sh status

# CPU ready 后检查本机模型、四张 GPU、输出碰撞
bash qwen3vl_local/audit_joint/geometry_study_abc.sh check

# 串行完成 a、b、c 各自训练→验证选模→重载测试；首错停止
bash qwen3vl_local/audit_joint/geometry_study_abc.sh train all
```

也可 `train a`、`train b`、`train c` 单独执行。默认选择四张可用 GPU，也可用 `GPU_IDS=0,1,2,3` 指定。已有运行不覆盖、不隐式续训。CPU 审计可重复复用兼容产物，`.cpu.lock` 阻止并发准备；CPU 完成不自动启动 GPU。

报告位置：`ready_1280.json`、`cpu_status.json`、`reports/<dataset SHA>/1280/pool_report.json`；训练日志在 `operations/a_1280.log` 等文件。CPU 拒绝时保留拒绝原因及容量报告，容量通过不能覆盖评测覆盖失败。

## 验证边界

新增回归包含“记录结尾仍有未来位置，但原六动作当前／未来全部不可标定”的真实规则反例；a／b／c 必须保留该样本。另测三组不同标签、共享计划、原生小 Qwen 前向／反向／保存重载、两进程 CPU DDP、expert 表示绑定和原始源篡改拒绝。最终本机结果见同目录 `verification_20261023.json`；CPU 证据不能替代 4B GPU 或 CARLA 在线验收。

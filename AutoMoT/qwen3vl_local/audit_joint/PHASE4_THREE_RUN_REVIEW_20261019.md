# 三份 Phase4 中断包：保留已完成数据，只续接缺失阶段

本轮三份 ZIP 的字节数、整体 SHA、CRC、成员集合及逐成员 SHA 均通过；各包的 573 项源码记录逐项匹配 v85 发布清单。它们不是三次训练失败：全部是 CPU 建库流程被磁盘守护以 75 停止，均没有 `automatic_admission.json`，尚未训练。用户已确认 **v86 续接尚未启动**。

**当前不需要把三份任务各重跑一遍。** 使用用户所选机器上实际存在的资产，按其完成阶段续接；两机资产不互通，同一 cee 根路径也不代表两个容器一定能看到彼此目录。没有新证据要求修改原生标签、阈值或提示词。

## 1. 三份包分别完成到哪里

| 下载包 | 已记录的输出目录（checkpoints 下） | 停止阶段 | 最小续接范围 |
| --- | --- | --- | --- |
| 无后缀，cr02 | `phase4_teacher_v24_20261018_121758_779650998` | `rgb4_rule_dataset` | v86 严格复验并复用已完成回放，续建题库及后续阶段 |
| `(1)`，cee 路径 | `phase4_teacher_v24_20261018_121925_502799670` | `automatic_admission` | **复用 data4、references4、view，只做准入和刷新首轮预算** |
| `(2)`，cee 路径 | `phase4_teacher_v24_20261018_130500_102842314` | `student_view` | 复用 data4、references4，在新输出完成 view，再准入 |

三份均记录 8614 路线、1062401 帧的完整回放。后两份原生训练 JSONL 相同 SHA，均为 **300694 道自动规则题**；自动留出 val 28226、test 26355，分别来自 432、452 条路线。它们是留出原始题数，不是最终十事件等量清单。后两份 manifest SHA 因路径绑定不同，不应改写成相同 SHA。

`(1)` 的已存预算报告使用约 2.52 TiB 可用空间、约 26.20 GiB 首轮声明需求，显示 sufficient=true；但当时 guard 的保留量为 2770279071744 字节，接近全部剩余空间。`(2)` 保留量为 2769880612864 字节。旧公式 `max(20 GiB, 启动空闲量−12 GiB)` 造成的写入额度问题确实影响这两份任务，v86 已改为绝对 20 GiB。

**纠正归因：cr02 这一份实际保留量已经是 21474836480 字节（20 GiB），不能把它的停止也归因于只允许新增 12 GiB。** 它触发的是实际空闲量低于保留线；旧日志没有打印触发时字节数，不能认定磁盘 100% 满。三份旧预算都不能代表今天的实际空闲空间。

## 2. 标定支持与事件等额

本次原始 train/readiness 支持统计（过滤及最终采样前）：

| 关键能力 | 原始 YES / 物理组 | 结论 |
| --- | --- | --- |
| U-E2 return | 2 / 2 | 有支持但很少，最终是否保留仍需准入 |
| U-E3 proceed | 281 / 96 | 原始支持存在 |
| U-E5 proceed | 0 / 0 | 仍无可监督放行能力，开发模式明确记录限制 |
| U-E6 proceed | 728 / 99 | 原始支持存在 |
| U-E7 complete | 18 / 13 | 原始支持存在，不能直接等同最终题数 |
| R-E5 proceed | cr02 868 / 187；两份 cee 869 / 187 | STOP 修复后的原始正例存在 |

不能将三个包合并成独立路线支持，也不能把其他边的 YES 当成 U-E5 proceed YES。U-E5 不靠修改阈值、出画或截止帧补标签；现有开发政策允许这一项能力缺失，其余关键缺格仍阻塞。

本地真实小库的 RGB2/RGB4、B2、world=1/4 计划均核对为 **十事件各 1024 次，共 10240 次**。不同模式 stride=2/1、不同 world 的题序并不相同，本轮没有宣称严格配对题序或 GPU 实际消费通过。197 道局部题会被重复采样，等额不是独立支持充足的证据。

服务器本次正式 **train/val/test 的过滤后十事件等额还没有完成验收**；不是已发现不等额，而是缺最终准入结果。继续使用当前固定、无放回的等额 val/test 准入与规则一致率选优；不要求 YES:NO 等额，不需要人工逐题标注。

## 3. 本地逐帧 RGB 复核

原生 v24 重放 9 条完整 train 路线、627 帧，核验 1881 份 RGB/metas/bboxes SHA；两模式各 197 题、val/test 均空，配对 ID 和答案一致。重放及小库构建约 43.20 秒，不能外推正式全池耗时。

实际查看十个事件窗口的 **56 个不同 RGB 帧**，核验对应 168 份源文件。UE1/RE5 共用路线且窗口重叠，已去重；这是已知训练例的开发复核，非盲审，也不宣称均为历史上首次查看。

| 事件 | 路线缩写及逐帧窗口 | 画面与原生结果对照 |
| --- | --- | --- |
| U-E1 | BlockedIntersection Town12 346_0，58–63 | 前车、STOP/路口条件共存；58–61 NO，62–63 UNKNOWN，没有因局部清空直接放行 |
| U-E2 | VehicleOpensDoorTwoWays Town12 2091_0，60–65 | 橙色对向车从左侧经过，60–61 NO、62–63 depart YES；64 导航用尽终止，不作为完成正例。局部可见间隙判断不等于全绕行安全许可 |
| U-E3 | ParkingCutIn Town13 1009_1，39–44 | 近侧切入车和前方车辆仍在；39–42 NO，43 恢复运动后删失旧等待实例，不补完成 YES |
| U-E4 | VehicleTurningRoute Town06 Scenario4_17，51–56 | 雾中骑行者横穿，51–54 NO、55–56 UNKNOWN，不把画面边缘消失直接改 YES |
| U-E5 | InvadingTurn Town13 1148_0，57–62 | 夜间车辆侵入并接近画面边缘，57–60 NO、61–62 UNKNOWN；未获得侵入后可观察的可靠放行正例 |
| U-E6 | OppositeVehicleRunningRedLight Town07 Scenario8_19，40–45 | 消防车横穿，40–43 NO，44 RGB 不连续终止；可见绿灯不独立证明该实例全部优先权成立 |
| U-E7 | CrossJunctionDefectTrafficLight Town13 002180，35–40 | 蓝车横穿，两实例35–38均NO；39恢复运动删失，不补complete YES |
| R-E2 | noScenarios Town05 ll_2，46–51 | 雨雾与两侧车辆，46–49 NO、50–51 UNKNOWN；布尔间隙条件成立仍可能因质量门弃权 |
| R-E3 | HighwayExit Town13 000839，22–27 | 22进入走廊 complete YES，与23仍受限 release NO并不矛盾；24 release YES、26实例完成。不能把complete提示理解为前方无车 |
| R-E5 | 与U-E1同路线，60–65 | 60–63 NO、64–65 UNKNOWN；clear转真但priority仍未知，STOP义务没有被自动清零绕过 |

此轮没有发现足以修改标签、阈值或提示词的新确定缺陷。保留夜间、雾天、信号关联与有限图像可观察性限制；56帧不能证明106万帧全源标签均正确。自动建库无需逐题人工审核，仍应报告能力范围和规则一致性，而非独立人工准确率。

完整本地产物：`checkpoints/phase4_three_rgb_20261019/`；逐帧事实与终止原因见 `windows.json`，原图拼图只留本地。收包证据在 `checkpoints/phase4_three_received_20261019/`。

## 4. 这次代码修改与验证

新增 `phase4_admission_only.py`，只复用显式指定的本机完整题库/自动留出/视图。默认没有回放、原生编译或留出导出路径；只有显式 `--complete-view` 才在新输出生成视图。缺文件、视图父库不匹配、输出重叠/已存在、源码不匹配先拒绝。准入沿用原生读取、物理路线隔离、关键支持、B2 两模式 world1/4 及固定等量评测检查；当前磁盘刷新预算、guard绝对保留20GiB，生成原有30MB上限fullflow交接包。

- **234 项相关回归通过，7.43 秒**：包含新入口10项、恢复、打包、规则准入、原生teacher与STOP/十事件测试；早先82项属于重叠子集，不相加。
- 实际 CPU worker 对上述真实小库读取、meta小模型预算、原生准入和打包：正确因缺关键格、空val/test返回2，保留两模式诊断；原库32个文件SHA前后不变，未回放/重建/训练。
- 实际外层 CLI：578份发布源码匹配、20GiB空间报告、既有 piped-core 保护拒绝启动、失败包校验通过。本机没有绕过该保护，不冒称服务器外层启动已验证。
- 新旧已登记原生Phase4/学生/采样/提示词源码SHA不变；没有新标签版本，不需要因为本工具重建数据或改旧SHA。

本轮代码、清单、报告**尚未提交或push**；不是服务器已安装版本。

## 5. 发布后的最短执行步骤

用户任选机器，只使用该机实际存在的本次 v24 目录。**这不是要求再运行三组。** 已有完整view优先只做准入；若所选机器只有第1组，按[v86恢复页](BUILD_DISK_RECOVERY_20261018.md)复用回放完成剩余建库，不重新计算路线。

新入口发布同步后，在该机器AutoMoT根目录原Python环境执行。下面 `P4_PREPARED` 必须填本机已有 data4、references4、view 的目录，例如第2组的 `checkpoints/phase4_teacher_v24_20261018_121925_502799670`，不能照搬别台机器不存在的路径。

```bash
P4_PREPARED="checkpoints/填写当前机器已完成题库的目录"
P4_AUDIT="checkpoints/phase4_admission_$(date +%Y%m%d_%H%M%S_%N)"
ulimit -S -c 0 2>/dev/null || true
nohup python -m qwen3vl_local.audit_joint.phase4_admission_only \
  --prepared "$P4_PREPARED" --output "$P4_AUDIT" \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  > "$P4_AUDIT.launch.log" 2>&1 < /dev/null &
echo "PID=$! 输出目录：$P4_AUDIT"
```

只有已完成data4/references4、未完成view的第3组，在该命令末尾重定向之前增加 `--complete-view`；不会覆盖旧view，也不会重新编译原生题库。不得对第1组使用本入口来跳过缺失题库。

查看 `$P4_AUDIT.launch.log` 和 `$P4_AUDIT/build.log`；新终端须恢复实际输出路径，不重新执行date表达式。末尾应生成handoff，正常完成或阻塞都交：

- `handoff/fullflow_report.zip`
- `handoff/fullflow_package.json`

若输入前置失败尚未建输出，交launch.log。准入仍需读取正式库和回放证据，有CPU/I/O耗时，不能承诺几分钟完成；本入口不会训练。报告通过后才安排当前新基线首轮，不重跑Phase3、readiness或旧拟合。**被引用的旧输出必须一直保留，成功后也不能删除。**

# v50本机连续RGB续审与服务器混版检查

按用户要求，服务器交接与本机标定复核同时推进。本轮只修改只读服务器审计入口及文档，生产v50合同、标定规则、提示词、旧题库和权重不变。

## 本机实际执行

专用tmux低优先级单CPU，对5条既有训练路线完整回放755帧，原生生成1116条问题；耗时45.81秒，峰值162460 KiB（含编译、压缩诊断、Phase3动作证据及联系表生成）。两/四图各编译464条，语义配对一致。与既有同路线回放比较，1116题的事件、实例、边、阶段、目标及facts完全一致；其中四路线比较v49，U-E7路线比较前轮v50，不声称源SHA相同。另逐帧计算755份Phase3原始动作证据，不强行认定两阶段同命题。

实际逐格查看5张连续联系表、70个不同RGB帧，另看3张原图（含在70帧内）。210份RGB/meta/bbox源SHA和21份回放/编译/联系表回执成员核验通过。这是知道规则结果后选窗的开发审计；与历史观察有重叠，不是70个首次看见的帧、全路线目视或G2盲审。

| 事件/路线 | 实际已看连续帧 | 画面与当前规则对照 |
| --- | --- | --- |
| U-E2，ConstructionObstacleTwoWays/Town01/001275 | 86–99，原图91 | 黄拖牌从右侧退出，锥桶/施工区域在91仍可见，随后经过；91的pass YES仅判断从DEPART进入PASS所需的lateral_entry_complete，不是绕障完成。92因frozen_navigation_exhausted_not_completion截止，completed=false，后续没有return题。该路线返回监督缺口有明确生命周期原因，不能将驶过施工区域直接补为return YES。 |
| U-E3，ParkingCutIn/Town12/1681 | 108–121 | 左侧蓝车逐渐从侧方退出，前方车辆仍可见。108–111的proceed保持NO；112因observed_resumption_censors_stale_wait截止，completed=false。缺放行正例不能靠延长旧等待或把自车已经前进当成许可。画面描述不单独认定哪辆车就是teacher actor3697。 |
| U-E5，InvadingTurn/Town12/1450 | 50–63 | 黑车50–53接近并从左侧驶过，54后出视野，58后又有红色来车。规则54–55为decisive_actor_not_visible的UNKNOWN，56在目标连续到后方后截止，completed=false。未把出视野当成YES；后续来车也不能自动继承旧实例。 |
| U-E6，OppositeVehicleRunningRedLight/Town03/Scenario8_107 | 58–71，原图64 | 雨雾中自车驶离路口，对向车辆连续通过。58–64的release_ready/corridor_clear均true，但priority_satisfied=false，合取结果为NO；65以observed_junction_exit_censors_stale_wait截止，completed=false。不能仅凭自车运动或前方局部有空隙推翻优先权条件。 |
| U-E7，CrossJunctionDefectTrafficLight/Town03/002134 | 41–54，原图54 | 横穿/转弯车流仍可见，近段车道逐渐展开，远端浓雾遮蔽。52–53为event_resolved=true但stable_progress=false，complete仍NO；54稳定门满足才为YES。短图像对精确稳定性和远端冲突不足以独立认证，保留可观察性问题，不凭雾中车仍可见直接判错。 |

本轮没有确定到应立即改写YES/NO的新错标。更明确的问题是：所查U-E2返回、U-E3/5/6放行正例的缺失包含实例提前截止/可见性/优先权限制，不能只靠增加重复采样解决。下一轮正式池统计应分报这些原因和不同物理组；本机5路线不能代表全源每类均缺。

## 存储诊断

对本轮5份原生回放在内存做gzip级别6及逐字节解压还原：14,250,605字节变为499,687字节，约3.51%。这些文件高度重复，说明压缩值得评估；不将该比例外推全源，也没有修改production文件格式、压缩旧实验或认证完整建库峰值预算。当前同盘45.71 GiB可用条件仍按 [空间报告](SERVER_READINESS_REVIEW_20261011.md#7-收到同盘占用结果复制式建库的预算不通过) 处理。

## 服务器入口与验证

`server_readiness.source_check` 保持发布文件SHA检查，增加已登记audit_joint/Phase3/Phase4/qwen35目录内清单外.py/.pyc检查，包含嵌套conftest；排除probe_output与缓存，不递归目录链接，也不声称认证所有外部依赖。入口在pytest前执行检查，不通过则不启动pytest；继续执行环境预检、盘点和失败打包，commands登记source_preflight_exit_code及pytest_started。没有忽略失败测试或删除额外文件。

新增4项回归；完整相关回归 **2260项通过，103.57秒，退出0**。另在独立440文件源码副本中插入会报错的清单外测试，实际运行shell入口：该文件没有执行、原文件保留、pytest_started=false，最终退出2并生成50306字节/7成员核验通过的诊断ZIP。测试基座路径故意缺失，preflight退出1如实记录；该试验不是服务器或模型成功运行。

发布清单仍440文件，仅更新服务器工具/launcher/测试三份源码SHA，release为phase4_v50_20261011_readiness2；生产合同未变。执行步骤与所需两个反馈文件见 [服务器当前步骤](SERVER_V50_START_20261011.md)。不重复旧拟合、不启动当前复制式全量建库、不要求再交相同磁盘盘点。

完整数据/RGB/压缩诊断/日志保留在 `checkpoints/label_rgb_followup_v50_20261011/`，轻量证据见 [verification](local_rgb_followup_v50_verification_20261011.json)。只推白名单源码与轻量文档/核验摘要，不推checkpoints、RGB或权重；本机没有执行GPU训练或远程任务。

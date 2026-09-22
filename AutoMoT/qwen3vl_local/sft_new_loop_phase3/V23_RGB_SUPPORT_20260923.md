# Phase3 v23：RGB 输入有效性、主要动作采样与独立路线支持

本轮修改同步到 `action_prior`、`action_expert_ablation/qwen_simple`、`bev_only`。
默认 Phase3 索引为 `sft_new_loop_phase3_data_v23`，prompt 为 `v23_grounded_stage`。
需要新索引、新 full map、新 run；旧 run 继续使用原源码，不能修改 manifest 绕过来源校验。

## RGB 复核与报告修正

沿用上一轮独立核验的 13 个锚点、95 张不同 RGB、350 条纵向原始 meta 重放；
本轮补看 `/tmp/p3v21/panels` 中 HazardAtSideLane 左借道/右回归、PedestrianCrossing f17、
ParkedObstacleTwoWays f53、ConstructionObstacleTwoWays 初始化片段等既有面板。
这不是重新逐张审完报告中的全部 83 张图片，也不是全数据目视审计。

- 初始化画面有明确异常：ConstructionObstacleTwoWays f0 是晴天空路，f1 出现雨雾、障碍车和周围车辆。
  Accident、无灯路口的既有复核也支持此问题。不能将其解释成真实瞬间运动。
- StaticCutIn f113 的最低速度在 f117（+1 s），1.011 m/s；DECEL 是未满足持续近停，
  不是“近停落到窗口外”。f125 是已在行驶中的增速，不能归入零速起步歧义。
- ParkedObstacle f28 障碍车可见；HazardAtSideLane 白天也有清晰障碍物和骑行者。
  这类错例不能笼统归为看不清。Hazard 的 f62–69 提前预测右回归，实际跨线 f82，
  显示动作触发时刻与固定预测窗存在错位；本轮不按这一条测试路线调阈值。
- “同集 context＋速度查表拟合高，所以没看图”不成立；反过来，超过某个留一查表基线，
  或高速仍答 STOP，也不能证明一定使用 RGB。该基线不是所有非视觉模型的上限。
- 相邻帧像素差小不等于输入完全相同：模型还读速度、导航与场景；像素差大也不证明动作可预测。
  7 道 confirmed-pullaway 测试题不是这种标签的全训练支持量。
- 四图/两图的逐帧配对检验有路线内相关性，不能直接当独立样本显著性或跨训练种子结论。
  先前 final=7680 与 val=6000 的比较不对应同一权重。

## 稀少样本如何处理

历史全量候选池中，极小的原始组合是 RAMP_MERGE_EXIT 的 RESUME＋LEFT：8 帧、5 条物理路线。
它不是需要单独学习的新动作。Phase3 构建、binary/choice 训练、验证及独立 eval 均按主要动作分配额度：
该组合并入 LEFT，binary 的 RESUME:YES 与 LEFT:YES 仍原样监督。原始组合计数继续保留用于审计。

容量回流沿用 v22：预算小于池容量时不重复稀少格子；预算超过整个事件池时才按完整池循环。
不为稀少组合另建均衡桶。Action 三入口已经使用跨事件共享的七类 token，本轮绑定相同新采样合同。
`--action-balanced` 默认全局单帧最多 2 次，`--event-balanced` 仍为 8 次；显式配置可覆盖。
Phase3 没有套用 Action 的 2 次上限，必须区分两套预算。

不能按“少于100帧”把合法标签改为 KEEP：旧 Action train 中 UE6 的 KEEP/DECEL/RESUME
分别有32/57/86帧，却有8/26/35条物理路线；抹去它们会使 UE6 更接近只输出 STOP。
剔除初始化历史后，旧候选划分仍有 confirmed-pullaway train/val/test=1514/32/74条 context 候选，
不能因测试集只抽到7题就删除此规则。

KEEP 仍表示有完整证据的保持阶段；UNCOND 表示不提供有效动作条件，不能代替错误动作真值。
本轮对明确坏输入执行排除，不伪造 KEEP；full map 保留原因并撤去 eligibility，可审计为 UNCOND。
没有按模型答错、天气、肉眼不确定或标签低频批量删数据。

## 实现

1. `history_rgb.py` 共用 `post_initialization_history_v1`：anchor < 4 排除，最早完整历史为 f1–f4。
   Phase3 两/四图和 Action 单/四图、BEV-only 共用可用帧集合。`_history` 不再复制 f0。
   full map 记录 `input_exclusion_reason`，三入口读取时再次过滤。
2. `sampling.py` 按主要动作合并额度，并按去除 Rep/录制时间的物理路线轮转；
   构建时 train/val/test 都优先路线多样性。报告显式标注物理路线口径。
   实际出题支持仍须看每次采样报告，候选池有5条路线不等于每个动作子类都有5条。
3. `split_coverage.py` 在原32帧目标上新增默认5物理组目标，保护训练的路线支持底线，
   只移动未曝光 train 整组。路线不足记录 `insufficient_support`；不会用复制填补独立支持。
   Phase3 原帧容量失败机制保留，路线不足单独报告。
4. `action_prior/split_support.py` 在 Action 自己的三 split 上运行同一策略（32帧/5组），
   不照搬 Phase3 split、不看动作标签/模型分数。整组移动包含该路线的背景帧；
   原索引文件不改，运行时使用确定性有效划分。计划缓存并写入 `event_balance_source_audit.split_support`，
   不足显式报告，既有可用性检查保留。三入口共享数据读取和计划，源码/策略绑定恢复合同。
5. 提示词要求按当前响应/恢复阶段判断，事件名不直接推出 STOP；区分借道与回归。
   STOP 优先仍以满足持续近停/等待定义为前提；动态切入的 RESUME 同时覆盖起步及行进中增速。
   物理速度阈值、预测窗、确认起步判据、STOP > 首跨 > 速度顺序保留。
   Action 主线自动复用共享动作因果句；消融仍保留各自输入定义，不额外加入 Phase3 问答。
6. 训练结束先提交梯度累积尾部、保存 final，再对 final 单独做已启用的自由生成验证，输出
   `final_generation.json`、`final_generation_val_cases.jsonl`；不替换 best 的守卫与选优记录。
7. 四包测试/导出验证案例共新增170个已曝光物理组，累计1779组 train-only；
   名单绑定 mapping hash。不能继续把这些路线称作未接触的盲测集。

## 历史全量容量回放

来源 `/tmp/p3audit/capacity_v21/{phase3_final,action_data,action_full}`。
仅回放既有标签、新输入过滤和新划分，不把旧文件伪装为通过当前合同的生产产物。
复现脚本 `/tmp/p3_v23_work/replay.py`，机器结果保存在本地 `/tmp/p3_v23_work/replay.json`；自动回放 JSON 按白名单规则不入库，核心数量见下表。

| 项目 | Phase3 | Action（三入口共享） |
|---|---:|---:|
| 初始化历史排除 | 3696条 context 候选 | 34456帧索引行 |
| 补充 holdout 的未曝光物理组 | 54 | 18 |
| 新 train 数量 | 183762条 context 候选 | 817258帧 |
| 新 val 数量 | 2121条 context 候选 | 66718帧 |
| 新 test 数量 | 4117条 context 候选 | 75057帧 |
| val/test 各事件候选支持 | 均≥32帧、≥5物理组 | 均≥32帧、≥5物理组 |

两列分母不同，Phase3 候选可以同帧多 context；以上也不是最终均衡题数。
Action 行数是已有数据索引应用新划分/初始化过滤的回放，未重新扫描全部原始 route 时长。
5组只是最低支持目标，不是泛化保证；每个 event×action 的支持还可能不足。

## 验证和边界

Phase3 全套601项 CPU 回归通过；Action/两个消融相关328项通过（另2项合同测试因本机缺 runner 源码未纳入此通过数）。
覆盖真实数据准备发布/复用、共享训练循环、组合标签保留、物理路线轮转、初始化排除、整体分组迁移及 final 验证记录。
更广检查还受本机缺 `mot_lead_offline_runner.py`、`peft`、`matplotlib` 限制，未伪造依赖或绕过校验。
未执行完整新生产索引重建、真实 Qwen/BEV 训练或 GPU 效果实验，没有效果提升结论。
固定未来窗口与短历史输入之间仍有预测不确定性；本轮不宣称靠措辞消除此问题。

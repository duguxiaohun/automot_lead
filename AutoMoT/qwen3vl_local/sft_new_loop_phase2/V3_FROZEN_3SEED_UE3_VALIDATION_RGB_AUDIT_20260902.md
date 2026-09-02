# v3 frozen 3-seed UE3 validation RGB 审计（2026-09-02）

## 1. 结论

本轮不能把 `no seed produced production-ready best_generation` 解释成“融合后的模型整体退化”或
“UE3 prompt 仍然太保守”。逐帧看完 25 个 unique 假阴性 case 的 100 张原始 RGB 后，主要问题是
validation UE3 slice 把少数 route 的连续、弱可观测或时间窗过宽帧重复计分：

- 32 个 UE3 validation 样本只由 `DynamicObjectCrossing` 17 帧和 `StaticCutIn` 15 帧组成；
- 25 个 unique 假阴性只来自 5 条 route；
- 19/25 个 case 被三个 seed 同时判错，但其中只有 `StaticCutIn f84` 是三 seed 共享、且 RGB
  仍较明确支持 UE3 的直接模型错误；
- 其余三 seed 共享错误主要是事件尚未出现、事件已经结束、ROAD_CORRIDOR 与可见局部路口几何冲突、
  2RGB 端点不足，或“远距离横穿是否属于 immediate future corridor”本身不确定。

因此：

1. 三个 seed 仍全部不晋升，unseen-456 保持未触碰；不能绕过现有 guard；
2. production prompt v3 暂不修改。当前 prompt 已明确“只判 newest moment”“弱证据判 NO”
   “distant vehicle 不算 UE3”“明显局部路口下 ROAD_CORRIDOR 可 invalid”，这些规则正是多数错例中
   模型采用的合理边界；
3. 下一步优先修 validation 的 route/连续帧过度加权和可观测性审计，不应继续盲目改 prompt、重训；
4. 本审计归因只能作为诊断，不能回写当前 frozen-384 的正式分数，也不能用来偷偷降低 UE3 门槛。

## 2. 原始结果

实验：`v3_frozen_3seed_unseen456_20260831`，模型实际输入为四帧历史的首尾 2RGB；审计额外查看
四张原图 `t0 -> t1 -> t2 -> t3`。

| seed | step | overall exact | UE3 recall | UE6 recall | INVALID exact | applicable RE exact |
|---|---:|---:|---:|---:|---:|---:|
| 20260810 | 4000 | 0.8021 | 0.4062 FAIL | 1.0000 | 0.9062 | 0.5833 |
| 20260811 | 4000 | 0.7917 | 0.3125 FAIL | 1.0000 | 0.9375 | 0.7500 |
| 20260812 | 4000 | 0.7812 | 0.3125 FAIL | 0.8438 | 0.9062 | 0.7917 |

UE3 每 seed 32 条。假阴性分别为 19、22、22，共 63 个 seed-case 记录；去重后 25 个 RGB case：

- `DynamicObjectCrossing`: 15；
- `StaticCutIn`: 10；
- 三 seed 同错：19；
- 仅一个 seed 错：6。

## 3. 逐帧 RGB 归因口径

`MODEL`：四帧能看到目标持续横向进入，t0/t3 也足以支持，且问题域合理。

`LABEL_OR_SPAN`：最新帧尚未出现目标、动作已经结束、正标签时间窗明显过宽，或内部
ROAD_CORRIDOR 标签与可见局部路口几何冲突。

`2RGB_INFORMATION`：四帧能逐步建立运动方向，但模型看到的 t0/t3 端点不足以区分“进入视野”与
“横向进入 ego corridor”。

`AMBIGUOUS`：即使四帧也无法稳定确认 prompt 所要求的 immediate future corridor 占用，不适合据此
单向修改 prompt。

夜间图除原图外只做了临时亮度副本辅助辨认车辆轮廓；所有结论仍以包内原始 RGB 为准，亮度副本未入库。

## 4. 25 个 case 的 RGB 证据

### 4.1 DynamicObjectCrossing

| frame | failed seeds | t0 -> t3 可见证据 | owner |
|---:|---|---|---|
| 25 | 811 | 夜雨中左侧车辆连续向右横移并逼近 ego corridor；另外两个 seed 正确 | MODEL |
| 59 | 810/811/812 | 夜间多车道上右侧车辆始终主要位于相邻车道，未见明确跨越可见虚线 | LABEL_OR_SPAN |
| 60 | 810/811/812 | 与 f59 同一连续片段；尺度变大但 lane-relative 横向侵入仍不明确 | LABEL_OR_SPAN |
| 86 | 810/811/812 | t0-t3 未见可确认的横穿车辆 | LABEL_OR_SPAN |
| 87 | 810/811/812 | t0-t3 未见可确认的横穿车辆 | LABEL_OR_SPAN |
| 88 | 810/811/812 | t0-t3 未见可确认的横穿车辆 | LABEL_OR_SPAN |
| 89 | 810/811/812 | t0-t3 未见可确认的横穿车辆 | LABEL_OR_SPAN |
| 90 | 810/811/812 | 目标只在 t3 的最左边缘首次小范围出现，尚无可见横移轨迹 | LABEL_OR_SPAN |
| 91 | 810/811/812 | 中间帧开始出现左侧车辆，但 t0 无目标、t3 仍在远侧边缘 | 2RGB_INFORMATION |
| 92 | 810/811/812 | 四帧可见目标从左侧逐步出现；首尾端点仍难区分进入视野与入侵路径 | 2RGB_INFORMATION |
| 93 | 810/811/812 | 横向车辆已可见，但 newest 是明显的局部侧路/冲突口形态；三 seed 判 invalid 有视觉依据 | LABEL_OR_SPAN |
| 94 | 810/811/812 | 车辆继续从左侧进入局部冲突区；ROAD_CORRIDOR 与可见路口几何仍冲突 | LABEL_OR_SPAN |
| 108 | 810/811/812 | 雾中远处车辆与前车重叠，横穿目标极小；是否已影响 immediate corridor 不稳定 | AMBIGUOUS |
| 109 | 810/811/812 | 远处横向关系比 f108 稍清楚，但目标仍很小且距 ego 很远 | AMBIGUOUS |
| 110 | 810/811/812 | t3 能看到远处横向车辆，但“远车 NO”与正标签的 immediate 定义存在边界冲突 | AMBIGUOUS |

Town02 的 f86-f94 是同一 route 的连续 9 帧，Town12 的 f108-f110 是同一 route 的连续 3 帧，
Town05 的 f59-f60 也是同一 route 的连续 2 帧。把这些逐帧视作独立 Bernoulli 样本会显著夸大少数
route 的影响。

### 4.2 StaticCutIn

| frame | failed seeds | t0 -> t3 可见证据 | owner |
|---:|---|---|---|
| 77 | 810/811/812 | 近车从前方移到画面右侧，newest 更像驶离 ego corridor，而非仍在进入 | LABEL_OR_SPAN |
| 80 | 811 | 极暗环境下车辆关系变化仍可辨；另两个 seed 正确 | MODEL |
| 82 | 811 | 左前车辆位置连续变化并接近 ego 路径；另两个 seed 正确 | MODEL |
| 83 | 812 | 同一低照度切入过程；另两个 seed 正确 | MODEL |
| 84 | 810/811/812 | 左前近车在连续帧中明显改变横向关系，是本批唯一清晰的三 seed 共享 UE3 漏判 | MODEL |
| 85 | 812 | 低照度下近车横向交互仍在；另外两个 seed 正确 | MODEL |
| 86 | 812 | 目标仍在近侧冲突区；seed12 错分为 UE5，另外两个 seed 正确 | MODEL |
| 87 | 810/811/812 | newest 中主要切入动作已经过去，模型转为 UE5/NO，正标签尾部偏宽 | LABEL_OR_SPAN |
| 88 | 810/811/812 | newest 缺少继续横向进入证据，事件尾部已结束 | LABEL_OR_SPAN |
| 89 | 810/811/812 | newest 只剩远处/既有车辆，缺少仍在发生的 UE3 | LABEL_OR_SPAN |

## 5. 归因计数

按 unique case：

| owner | unique cases | seed-case records | 含义 |
|---|---:|---:|---|
| MODEL | 7 | 9 | 6 个单 seed 波动 + f84 三 seed共享漏判 |
| LABEL_OR_SPAN | 13 | 39 | 事件前后沿、无跨线证据、或问题域几何冲突 |
| 2RGB_INFORMATION | 2 | 6 | 四帧比首尾端点更有信息 |
| AMBIGUOUS | 3 | 9 | 远距离目标与 immediate corridor 定义冲突 |
| 合计 | 25 | 63 | 与原始假阴性记录完全对齐 |

这不是正式重算后的 recall。它只说明原始 `0.4062/0.3125/0.3125` 不能被当成纯模型能力值；
63 条 FN 中只有 9 条 seed-case 记录有较清晰的直接模型责任。

## 6. 对 prompt 的决定

production v3 保留，不改 hash。尤其不能做下面几种“为了过门槛”的改法：

- 删除 `distant vehicles / weak evidence -> NO`，会重新引入普通远车和邻车的 UE3 假阳性；
- 删除 newest-state 合同，会让已经驶离的 f77/f87-f89 继续被判 UE3；
- 要求“只要从边缘出现就是 UE3”，会把 ego 视差和普通进入视野误判成 lateral entry；
- 弱化 INVALID 的几何判断，会掩盖 Town02 f93-f94 的 ROAD_CORRIDOR/局部路口标签冲突。

如果后续独立、route-balanced RGB 审计仍稳定复现 f84 类型，才考虑只增加一条很窄的低照度规则：
车身轮廓虽然暗，但只要相对可见道路边界持续横移进入 ego corridor，仍可判 UE3。当前只有一个三 seed
共享 case，不足以立即改 production prompt。

## 7. 下一实验决策

1. 不重训、不跑 unseen；先用现有三个 step-4000 adapter 做新的 validation-only 复评。
2. 新复评必须按 `(scenario, route_id)` 分散取样，同一 class 先轮转不同 route，再取同 route 的第二帧，
   防止一条 9 帧连续 span 主导 UE3 guard。
3. 同时报 frame-level 与 route-macro 指标；原 frozen-384 结果继续原样保存，不能覆盖。
4. 为正类报告 `observable / span-boundary / domain-conflict / ambiguous` 审计字段，但人工字段只用于诊断，
   不直接参与 checkpoint 选优。
5. 只有 route-balanced validation 仍通过 UE3、UE6、INVALID、RE 四项 guard，才对未触碰的 unseen-456
   做一次验收；否则停止当前路线，转为清洗事件时间窗/RS 问题域监督，而不是继续堆 prompt。

# v3 route-diverse full UE3 RGB 审计（2026-09-02）

## 1. 审计对象与口径

证据包：`checkpoints/ue3_route_diverse_full_rgb_audit/`。它包含 route-diverse validation rescore 中
全部 32 个 UE3 正例，而不只是错例。三个 final adapter 评估了完全相同的 case 身份：

- seed 20260810：21 FN，frame recall `0.34375`；
- seed 20260811：25 FN，frame recall `0.21875`；
- seed 20260812：26 FN，frame recall `0.18750`；
- 三 seed 全对 4 例，三 seed 全错 19 例，有 seed 分歧 9 例。

审计对每例按 t0→t1→t2→t3 查看四张原始 stitched RGB。模型实际只看 t0/t3；
StaticCutIn 原图极暗，辅助查看了临时亮度版来辨认车身轮廓，结论仍以原图为准，
亮度版未写入仓库或审计包。不使用 scenario 名称作为判定证据。

## 2. 逐帧视觉分类

| visual class | cases | 含义 |
|---|---:|---|
| `VISIBLE_ACTIVE` | 12 | 四帧及首尾端点支持 newest 时刻仍正在横向进入 |
| `PRE_EVENT` | 5 | 目标尚未出现，或仅在 newest 边缘首次出现 |
| `POST_EVENT` | 6 | 横向进入已结束，当前不应靠历史续标 |
| `DOMAIN_CONFLICT` | 2 | 可见局部侧路/冲突口与 `ROAD_CORRIDOR` 问题域冲突 |
| `2RGB_UNOBSERVABLE` | 2 | 中间帧补充了出现过程，t0/t3 不足以确认横向入侵 |
| `AMBIGUOUS` | 5 | 邻车/视差或远距离 immediate corridor 边界不稳定 |

机器可读的 32 行决策表是 `ue3_route_diverse_rgb_decisions_v1.jsonl`。

### 2.1 DynamicObjectCrossing / Town05 Scenario3_45（f23-f25）

f23-f25 是三 seed 全对的对照组。夜雨中近距离车辆从左侧持续向 ego future corridor
横移，车身尺度大，相对车道边界的变化清晰，t0/t3 足以判定。三例均为
`VISIBLE_ACTIVE`。这说明模型并非完全不理解 UE3。

### 2.2 DynamicObjectCrossing / Town05 Scenario3_2（f59-f60）

夜间多车道中车辆尺度变化可见，但缺少明确跨越可见车道边界的证据，也难以
排除 ego 前进视差。两例均为 `AMBIGUOUS`，不能用它们单向放宽 prompt。

### 2.3 DynamicObjectCrossing / Town02（f86-f94）

- f86-f89：t0-t3 没有可确认的横向车辆，均为 `PRE_EVENT`；
- f90：目标仅在 t3 最左边缘初次小范围出现，仍为 `PRE_EVENT`；
- f91-f92：四帧能看到目标逐步出现，但首尾 2RGB 不足以区分“进入视野”与
  “横向入侵 ego corridor”，为 `2RGB_UNOBSERVABLE`；
- f93-f94：横向车辆已可见，但 newest 是明显局部侧路/冲突口，为
  `DOMAIN_CONFLICT`。三 seed 因几何冲突而不给 UE3 有视觉依据。

这一条 route 的 9 个滑窗被全部当作 UE3=YES，是当前 frame recall 被拉低的最大原因，
而不是 9 个独立的模型错误。

### 2.4 DynamicObjectCrossing / Town12（f108-f110）

雾中目标与前方车辆重叠且距离很远。即使四帧也无法稳定确认 prompt 要求的
immediate future corridor 占用，三例均为 `AMBIGUOUS`。

### 2.5 StaticCutIn / Town12（f75-f89）

这 15 例来自同一低照度 route 的连续滑窗：

- f75-f76：近车从侧前方进入 ego 近场，为 `VISIBLE_ACTIVE`；
- f77-f79：近车已向侧面离开，newest 不再支持正在进入，为 `POST_EVENT`；
- f80-f86：低照度下仍能辨认近车相对道路边界/ego 路径的横向变化，为
  `VISIBLE_ACTIVE`；
- f87-f89：主要交互结束，为 `POST_EVENT`。

其中 f82 三 seed 全对，f84 三 seed 全错。两者都处在同一清晰活动段，说明存在真实的
低照度稳定性问题；但证据仍只来自 1 条 route，不足以改 production prompt。

## 3. 诊断性重算（非正式指标）

只对 12 个 `VISIBLE_ACTIVE` 正例计算 UE3 命中：

| seed | visible TP/12 | frame recall | unique routes | route-macro recall |
|---|---:|---:|---:|---:|
| 20260810 | 6/12 | 0.5000 | 2 | 0.6667 |
| 20260811 | 5/12 | 0.4167 | 2 | 0.6111 |
| 20260812 | 6/12 | 0.5000 | 2 | 0.6667 |

这不是新 validation 分数，不能用它选 checkpoint 或打开 unseen。它只说明：

1. 原始 `0.3438/0.2188/0.1875` 同时混入了大量非模型责任帧；
2. 排除这些帧后，真正可观测的 UE3 仍未达 0.625 guard；
3. 12 例只来自 2 条 route，证据量仍不足以判定通用 prompt 错误。

## 4. 对代码、prompt 和实验的决策

- production prompt v3 保持不变，hash 不变。它对 PRE/POST/远车/问题域的保守判断与多数 RGB
  证据相符。
- 决策表不自动修改 dataset，避免把人工看过的 validation 泄漏进 train，也不覆盖原
  frozen 分数。
- 不继续训练当前 3-seed SFT，不打开 unseen-456。当前最大的工程收益不是继续堆
  prompt，而是停止将 scenario event span 的每个滑窗当作独立精确帧真值。
- 如果仍要做最后一次 SFT 实验，必须先从未用于本 validation 的 train routes 中抽取至少
  15 条 UE3 routes，用同一逐帧口径确认 active span；否则停止文字 QA 循环，转向
  LeadMoT/CARLA 小规模闭环 A/B。

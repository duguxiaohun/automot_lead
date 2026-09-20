# Phase3 v17：判断题也显式标注 KEEP

先前 v15/v16 将选择题的保持阶段改为 KEEP，但判断题仍用所问变化动作全 NO 表示保持。
这使两个题型对同一驾驶阶段的表达不一致。v17 在全部十类 UE/特殊 RE 的判断题中增加
`KEEP: YES/NO`，并共用选择题的场景动作因果描述。

## 语义与标注

| 情况 | 判断题 | 选择题 |
| --- | --- | --- |
| 有效场景，完整证据表明继续当前阶段 | KEEP:YES，其余动作 NO | KEEP |
| 减速配合首次左跨线 | DECELERATE:YES、LANE_CHANGE_LEFT:YES、KEEP:NO | LANE_CHANGE_LEFT |
| 当前持续停车等待 | STOP:YES、KEEP:NO | STOP |
| 场景前提被明确反驳 | INVALID_ACTION_CONTEXT:YES，全部动作含 KEEP 均 NO | 不进入 choice 样本 |
| 证据缺失或歧义 | 不补成 KEEP | 不补成 KEEP |

机动域（UE2/UE4/RE2/RE3）KEEP 表示继续当前车道和速度阶段，允许小幅调速；
纵向域（UE1/UE3/UE5/UE6/UE7/RE5）KEEP 只表示速度阶段保持，不断言没有变道。
事件仍可持续，例如绕障跨线已经完成、接下来继续行进时可以是 KEEP；后续又有增速或回归跨线时
应标相应动作，不能仅因“已经变过道”固定为 KEEP。

保留原始五动作证据；KEEP 按当前题域确定，不加入共享的原始 `ACTION_KEYS`。
binary 仍可同时回答一个纵向动作和一个横向动作 YES；KEEP 与这些动作互斥。
choice 沿用 STOP > 首次跨线 > 速度动作的投影。v8 速度/车道轨迹规则和采样划分不变。

## UE2 判断题例子

场景：绕障跨线已完成，静态障碍事件仍有效，接下来继续在当前车道行进，没有新的显著速度阶段变化。
KEEP 的动作释义为：

> Continue in the current lane with roughly the current speed, including after a completed bypass lane change while the obstacle event remains active. Small adjustments or passing within the lane do not constitute another lane change. This means continuing without a new ego lane crossing or meaningful speed-stage change.

回答示意（实际按题目给出的稳定乱序输出）：

```text
DECELERATE: NO
STOP: NO
RESUME: NO
LANE_CHANGE_LEFT: NO
LANE_CHANGE_RIGHT: NO
KEEP: YES
INVALID_ACTION_CONTEXT: NO
```

判断题纵向域共五行，机动域共七行，invalid 固定最后。
`scene_context` 仍只放道路、事件和给定历史事实。减速/停车的避碰、观察邻车、等待间隙和准备绕行目的
仍在动作释义中；所有场景不再显示未来数值时间窗或速度阈值，历史 RGB 时间仅说明输入。

## 训练、评估与兼容

- 默认仍 `4rgb + choice`；`ACTION_OUTPUT_MODE=binary` 新训判断题。
- prompt 为 `v17_binary_keep`，索引默认 `sft_new_loop_phase3_data_v17`，
  索引格式 `v5_binary_keep`，标签协议 `primary_choice_v3_choice_and_binary_keep`。
- 索引必须有布尔 KEEP，且与题域动作、invalid 和完整证据一致；训练、eval 和 preflight 拒绝旧/矛盾索引。
- 严格 parser 要求 KEEP 行，audit 还要求 EVIDENCE_KEEP；缺行或解析失败不能当保持。
- binary KEEP 的支持数、precision/recall、最佳模型守卫和错误样本审计均已接入；全 NO 预测不算 KEEP 命中。
- 新训练重建索引；旧 run 使用原源码恢复。action_prior 继续使用其原始五动作/NONE 协议，未改为接收 KEEP 文本。

从 `AutoMoT/` 启动判断题新训练：

```bash
ACTION_OUTPUT_MODE=binary bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

## 验证范围

394 项 CPU 回归通过；PyTorch 不可用，三个依赖模块和四个依赖测试未运行，未执行模型训练/推理。
新增检查覆盖十类场景全部合法纵横组合、KEEP 与 choice 对齐、invalid、缺证据、严格解析、
实际 train/eval KEEP 计数分支、最佳模型守卫和错误审计。

复用既有 26 条开发路线重建 1183 个候选、384 行局部索引（十类各 32 行，invalid 64 行）；
其中 78 行显式 KEEP:YES，原始动作、证据、顺序和 split 相对 v16 未变。
重放既有 26 片段/442 帧面板对应的 RGB 指纹、轨迹证据和标签检查通过。
本轮没有新增人工 RGB 审阅、全量生产索引重建或模型效果结论。
本地可再生结果位于 `probe_output/binary_keep_20260920/`，不入库。

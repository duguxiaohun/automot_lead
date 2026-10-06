# 审计风险落实为构建准入检查

第十/十一轮确认的异常消失、重影和实例边界风险原先只有登记，构建器不检查是否解决。
本次新增 `risk_review.py`，将两份已绑定审计账本中的12条已知风险路线接入构建、加载和预检。
这是数据准入规则，不是自动异常检测，不批准新的转移标签，不新增状态。

任务合同升级v11，单独train默认data_v11；须新数据目录、新run。提示词v6、运行快照v8、
人工标注v4、标定规则v5保持。Phase3/Action稳定默认未改。

## 实际行为

1. 已知风险路线必须采用逐帧 `reviewed_transition_band`，并提供 `risk_review`。
   旧式整段事实不能用于跳过风险复审。没有记录、漏掉当前登记风险或登记版本过期，构建拒绝。
2. 审核范围包含每道题实际输入的全部历史RGB，不只当前帧。每帧绑定RGB来源摘要、
   本帧 `observed_until`、处置状态、具体理由；可用帧还要求参与者身份、转移范围与实例边界三项复核。
3. 所有输入帧均为usable才允许保留原条件标签；任一帧uncertain或excluded，整题变为UNKNOWN，
   进入review_queue，保留对应处置记录，不能作为YES或NO监督。处置优先级excluded高于uncertain。
   这是训练样本隔离，不表示该交通条件的答案必定NO。
4. 原band截止仍有效：截止及之后不构建旧问题，不因风险审核通过而延长catchup。
   不将报告粗定位帧自动当截止，不固定扩窗。异常之前经复审可用的片段仍可构建，未封禁整条路线。
5. 加载数据再次核对逐帧风险记录和处置，即使更新JSONL哈希，也不能删除审核证据或把未解决风险改为监督。
   manifest及preflight输出 `risk_review_check` / `risk_review`，记录登记摘要和实际题目处置计数。
6. 这些内部证据不进入模型消息，模型仍只看因果RGB与当前/候选下一状态，回答YES/NO。

现有12条登记路线不包含默认453题，因此默认train/val/test/review_queue保持逐字节一致。
未来发现的新风险应先登记再构建；代码不会从未登记的RGB中自行发现异常。

## 如何补审核

从AutoMoT目录运行（输出必须为新文件）：

```bash
python -m qwen3vl_local.sft_new_loop_phase4.risk_review \
  --annotations /path/to/candidate_annotations.json \
  --data-root lead_data --rgb-mode 4 \
  --output /path/to/risk_review_template.json
```

模板按候选标注列出命中的风险、登记ID以及该模式所有实际历史帧，自动计算文件摘要。
模板是待填写材料：reviewer/evidence_id/reason为空，状态uncertain，检查项为false，不能直接放行。
完成逐帧审阅后，将相应 `risk_review` 对象写入候选标注，再走普通dataset构建命令。
2/4RGB历史集合不同，4RGB审核覆盖更广；不能拿仅覆盖2RGB的记录冒充4RGB已审。

单帧记录含：

```json
{
  "status": "uncertain",
  "reason": "记录当前帧中能确认与不能确认的内容",
  "observed_until": 89,
  "rgb_sha256": "实际图像SHA256",
  "participant_identity_reviewed": false,
  "transition_scope_reviewed": false,
  "instance_boundary_reviewed": false
}
```

usable并不生成YES：仍须原逐帧条件/完成证据决定答案。参与者不必全部离开画面才可推进，
但不能以消失或重影轮廓数量代替条件证据。这里的三个复核项表示审核者已核查对应问题，
不表示代码能证明身份、道路拓扑或事件边界正确。错误填写的“已审核”仍需人工审计发现。
不引入未来actor轨迹或完整路线到模型输入。

excluded题保留在待审库中以便追溯，不进入训练；若需要在异常后重新开始采样，必须新建
合适的事件/转移区间、独立审核所用历史帧，而不是复制旧事件的catchup。

## 验证与边界

- 311项Phase4测试通过，含20项新增准入回归；日志 `/tmp/phase4_risk_admission_tests.log`。
- 2RGB/4RGB实际pipeline MODE=check通过，各453道题RGB身份核验、各14整池采样计划通过。
- 两模式cap1下budget1/world1及budget4/world4各453轮均覆盖全部题目。
- 四份JSONL与第十一轮产物逐字节一致，仍453题/133待审，输入冲突为0。
- 合成风险反例验证历史帧影响当前题、未知/排除隔离、旧格式拒绝、截止优先、证据绑定、
  加载时再次拒绝和无提示词泄漏。合成事实仅在测试/probe中，不加入默认人工标注。
- 模板CLI实际输出成功，但没有自动填写或声称完成逐帧复审。

产物在 `probe_output/risk_admission_v11/`，含validation.json、两模式数据/日志及合成待审模板。
本轮无新增RGB目视审阅、曝光或人工标签，完整覆盖仍沿用第十一轮339条/38,369帧、
155/278格三ID、剩361条/60,299帧、18格来源不足。

ready=false：389个转移支持格及12个分段支持格仍缺，独立val/test为空，本机默认完整4B目录缺失。
未正式4B/GPU/DDP/CARLA验收；本次防止已知风险无审核进入训练，不能代替真实监督和独立评估。

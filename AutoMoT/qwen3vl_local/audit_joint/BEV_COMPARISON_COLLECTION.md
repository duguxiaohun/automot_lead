# 旧 BEV 与 ABC 补充审计收集器

在存有旧运行的服务器 AutoMoT 根目录、已激活环境中执行。首次先用该服务器已有的 `bash ./sync_automot.sh` 同步白名单文件。收集器直接运行，不调用训练或版本准入入口。

```bash
python -B qwen3vl_local/audit_joint/collect_bev_comparison.py \
  --run checkpoints/action_expert_ablation/bev_only/run_20260928_211328 \
  --output checkpoints/audit_exports
```

同一台服务器确实存在 ABC 三组完整运行时，可以在上面命令追加：

```bash
  --abc-run checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636
```

此目录只在当前机器存在时使用；不能拿另一台机器的路径代入。两台项目根分别为 `/home/cruser1/lda/AutoMoT` 和 `/mnt/workspace/cee3sgh/AutoMoT`，命令均在各自根目录执行。无需跨机复制模型。

输出为新建的 `checkpoints/audit_exports/bev_comparison_日期_时间_PID/`，其中 `bev_comparison_review.zip` 与 `receipt.json` 合计不超过 **30,000,000 bytes**。结束打印结果目录、包路径、大小、SHA 和缺失项。发回 ZIP 即可；清单已在 ZIP 内。建议下载到 `审计资料/02_原始交包/Action_旧语义BEV/run_20260928_211328/`，不覆盖原 training_audit.zip。

## 收录与边界

- 旧版 `eval_test/metrics.json` 原文：总体、已有事件/动作分组指标、test 身份、EMA/step/合同及数据散列。
- 完整已保存的训练 epoch/window JSON、validation、performance、评测进度和轻量 sampling JSON。不再仅限旧 training_audit.zip 的最近200窗口。额外保留旧 training_audit.zip。
- 已有 `eval_test/cases/rank*_case*.json` 的每条原始 sample、condition、metrics 和来源 SHA；坐标转为精确 float32（可表示时）或 float64 二进制，按每例 offset/dtype/shape 还原，不四舍五入。
- 超限时仅移除坐标二进制，仍保留全部逐例指标和身份，清单明确标记。若核心仍超限，拒绝发布，不随机抽样或制造完整包。
- best/latest.pt 与当前配置指向 BEV 文件只计算 SHA，不反序列化、不加载权重；当前 BEV SHA 与训练记录对照。未比较张量身份。
- 当前机器存在旧 test.jsonl 时导出身份字段并核对历史 SHA。该成员表不能冒充真实逐例预测；不匹配时明确标记。
- 当前训练源码只作为参考，逐文件比较旧合同 SHA，不把更新后的源码称为原训练源码。
- 不启动 GPU、推理、建库或训练，不修改旧结果。未保存的逐例预测无法通过打包补造；缺失会明确报告。
- 原 TensorBoard 大事件文件不收录；已有结构化指标完整收录。train.log 仅收最后256KB；大于16MB的 sampling 细节文件仅记录 SHA 和遗漏说明。

## 比较顺序

1. 先核对旧 test 的样本数量、自然/事件均衡口径、最佳模型 step 和 EMA。
2. 比较旧版与 a/b/c 的 route/waypoint ADE/FDE、分事件分动作表现；尽可能计算统一的十事件与物理组宏平均。
3. 有逐例证据时，核对共同样本并分析差值、胜率、P90/P95及退步场景；无共同成员不能把总分差当配对提升。
4. 合并已保存训练/验证曲线，比较归一化 FM loss 与米制轨迹误差的关系，以及呈现量、更新数、学习率、耗时和吞吐。
5. 保留历史输入、采样、划分、BEV文件身份差异的限制；旧 val 不充当 test，oracle 区域表示不充当 Phase3 预测接入。

本地通过标准库合成验证：513例坐标/身份/指标完整还原、float64回退、321窗口全收录、超限回退与拒绝、缺失案例、显式三组ABC、当前机器路径适配及输入字节不变。这是本地合成验证，不是远程真实运行验收。

标准库回归命令（无需 pytest、torch 或模型文件）：

```bash
python -B qwen3vl_local/audit_joint/tests/check_bev_comparison_collection.py
```

本收集器放在 audit_joint 包外工具位置，不加入 Action/ABC 训练模块的源码扫描范围；没有修改旧合同、数据、权重或 ready。

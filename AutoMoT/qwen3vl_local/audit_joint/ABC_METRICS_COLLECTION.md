# ABC 指标补充收集器

仅收集当前机器已有的 a/b/c BEV 指标。使用 Python 标准库，不加载模型、GPU，不训练，不需要旧语义 BEV 目录，不改写旧运行或合同。

在保存三组完整运行的服务器 AutoMoT 根目录、当前已激活环境下执行：

```bash
bash ./sync_automot.sh

ABC_AUDIT_RUN=checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636 \
python -B qwen3vl_local/audit_joint/collect_abc_metrics.py
```

两台服务器分别使用其真实项目根 `/home/cruser1/lda/AutoMoT` 或 `/mnt/workspace/cee3sgh/AutoMoT`。上述运行目录必须在当前机器存在；若不同，只修改 `ABC_AUDIT_RUN`。不会自动选取另一轮、跨机复制模型或重跑训练。没有停止的任务应继续使用启动时源码快照，不能热更新其运行环境。

收集内容：

- 各组根目录 JSON，包括配置、合同、完成证据及已有 metrics。
- 全部已有 training_audit、validation、performance、progress JSON/JSONL。
- test/eval_test 下非逐例 JSON/JSONL 及 eval.log。
- 已有 training_audit.zip、TensorBoard 事件文件。
- 不收权重、原始 RGB/LiDAR、逐例 cases 或训练源码。

输出 `checkpoints/audit_exports/abc_metrics_日期时间/abc_metrics_supplement.zip`，最后打印结果目录、包路径、大小、SHA 和缺失项。包内清单逐文件记录 SHA/大小；下载 ZIP 即可。

硬限 30,000,000 字节。超限时明确省略原始 TensorBoard，保留全部核心指标重打包；核心仍超限则拒绝交付，不截断或抽样指标。正在写入的文件拒绝收集。仅 ctime 元信息变化不作为本工具读取变化条件；绑定设备、inode、大小、mtime，记录实际读取字节 SHA。本工具不是训练准入回执，也不修改旧 SHA。

重点补收 `route_fm_mse` 和 `waypoint_fm_mse`。尚未保存的分项无法靠打包补造；缺失目录会列明，单项指标是否存在由后续审计确认。TensorBoard 的存在也不保证它记录了全部目标指标。

本地标准库合成验证共 8 个场景通过：模拟两台服务器目录，各自验证完整打包、超限后保留核心指标并省略 TB、核心超限拒绝、缺失运行拒绝。逐文件 SHA 正确，权重/逐例目录未收录，原输入字节不变。这是布局模拟，不是远程真实采集验收。

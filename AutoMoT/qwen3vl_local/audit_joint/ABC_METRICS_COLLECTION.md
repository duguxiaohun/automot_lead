# ABC 指标补充收集器：a/b/c 分别打包

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

## 默认输出三个独立包

从本版起，原命令默认分别打包，不再将三组放进一个包：

```text
checkpoints/audit_exports/abc_metrics_split_日期时间/
  delivery.json
  a/a_metrics_supplement.zip
  b/b_metrics_supplement.zip
  c/c_metrics_supplement.zip
```

**每个 ZIP 独立不超过 30,000,000 字节，三包合计可超过30MB。** 每组独立打印结果目录、包路径、大小、SHA、缺项；结束打印三组总回执。下载三个 ZIP 即可，回执已包含各包自身的完整来源身份。

一个组缺失或超限会记录错误并继续处理其余组，已经成功的包保留。仅部分完成时总回执为 `partial`、命令最终退出1，不会把两包冒充三包。需要只补某组时：

```bash
ABC_AUDIT_RUN=checkpoints/action_expert_ablation/geometry_abc_bev_only/run_20261026_122636 \
python -B qwen3vl_local/audit_joint/collect_abc_metrics.py --variant b
```

默认 `--variant all`；也支持 a/b/c。每次新建交付目录，不覆盖旧结果。

## v2：修复去掉 TensorBoard 后核心包仍超限

旧版按每个文件分别压缩，并把 `training_audit.zip` 当作不透明文件再次打包；重复指标及重复合同很难跨文件压缩。v2 不减少任何已选 epoch/window/event 记录，改为：

- 将原始来源字节合并到一个快照，以 ZIP/LZMA 进行跨文件压缩。
- 相同 SHA/大小且逐字节一致的内容只存一次，各来源保留独立路径映射。
- 展开 training_audit.zip 的 JSON/JSONL/Markdown/文本/日志成员后共同去重，保留包内独有指标。原 ZIP 容器记录 SHA/大小，不重复嵌入容器字节；非指标成员列明省略。
- 浮点数字、原始文本、完整时间步均不重写、不四舍五入、不抽样。
- 核心和可选 TB 使用同一次读取的快照，压缩重试不重新扫描运行目录。

包格式为 `abc_metrics_stream_v2`：`MANIFEST.json` 的 `sources` 提供每个来源在 `evidence.bin` 中的 offset、bytes 和 SHA256。读取该区间即可逐字节还原来源；名称含 `::` 表示嵌套审计包成员。完整 evidence.bin 也有 SHA。这是压缩存储格式变化，不是训练合同变化。

硬限现在对每个组分别计算：每包 30,000,000 字节。该组含 TB 超限则保留该组全部核心指标、明确省略该组原始 TB 再压缩。若单组完整核心去重压缩后仍超限，仅该组拒绝发布，保留其目录内的 `collection_inventory.json`，继续其它组，列出来源数量、原始/去重大小及最大的25个来源，不再只报一个无诊断信息的错误。不能承诺任意不可压缩数据都装进30MB，也不会靠丢弃指标假装完整。

正在写入的文件拒绝收集；仅 ctime 元信息变化不作为读取变化条件。绑定设备、inode、大小、mtime，记录实际字节 SHA。本工具不生成训练准入回执，不修改旧 SHA。

重点补收 `route_fm_mse` 和 `waypoint_fm_mse`。未保存的分项不能通过收集补造；缺失目录会列明，审计 ZIP 内的回退证据单独保留。TensorBoard 存在也不保证记录了全部目标指标。

本地标准库回归：

```bash
python -B qwen3vl_local/audit_joint/tests/check_abc_metrics_collection.py
```

验证合包超限但分组三包各自通过、组间内容完全隔离、首组失败不阻断后两组、单组补收、默认CLI在两机布局产出三包，以及完整321窗口/组、原始数值字节还原、嵌套包独有记录与去重、TB保留/省略、超限诊断、输入变化拒绝、重复ZIP成员拒绝、两台服务器布局模拟，以及真正超过30MB的旧打包方式回归。不代表远程实际采集已通过。

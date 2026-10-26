# Action ABC 主机 OOM 修复与恢复

收到的 `sigkill_review_1792978275793583592.zip` 对应 `geometry_abc_checkpoint_fixed/run_20261025_203923`。a_bev_only 探针已完成，a_qwen_simple 探针保存后 rank0 被 SIGKILL，尚未进入正式训练。

内核在 2026-10-25 21:29:23 明确记录 `global_oom`，杀掉 Python（匿名 RSS 约 19.57 GiB），并杀掉 journald、nvidia-persistenced；训练日志在同一时间窗口报告退出。容器内存限额 100 GiB、累计峰值 97.43 GiB、oom_kill=1。这是主机内存不足的直接证据，不能归为 CUDA OOM 或 LoRA 权重错误。内核 PID 与容器 PID 属于不同命名空间，现有材料没有 PID 映射，故不声称已逐 PID 证明被杀进程就是 rank0；累计峰值和 oom_kill 也不是本次独占计数。

## 修复范围

此前四个 rank 各解码并长期保留全部 train/val/test；探针也采用这个路径。正式 DataLoader 还会 spawn 多个进程，来源回执缓存无容量上限，probe/evaluate 用普通 torch.load 将不需要的优化器张量一并载入。保存后的 CPU staging/reload 进一步压缩剩余内存。

- 新增 `indexed_rows.py`。同一次流式读取计算完整 split SHA、行偏移与每行 SHA；完整验证 manifest、构建代码、隔离账本和路线回执。正式消费按原计划索引读取，每次缓存未命中均核验同一份字节再解码，最多缓存 8 条解码记录。
- 训练保持全部样本可访问，七轮顺序和每 rank 切片不变；不缩小数据池、不截断评测。spawn 子进程只携带偏移/散列和整数视图，不携带全池解码结果或父采样器 ID 字典。探针在分配模型前只保留原极值/补足成员；评测释放训练索引。
- 验证的联合事件分配只保留原分配算法所需身份字段，再回取同一原记录。极值选择流式保留必要 ID，选择规则不变。
- 探针使用正式配置的 DataLoader worker 数（当前默认每 rank 8），每个 worker 完成一次所选 train 样本的 CPU 解码，并保持存活直到模型保存、后续反向及重载结束。这些 CPU 重读不增加 optimizer 更新或正式曝光；各 rank 的 worker 数与存活阶段证据纳入准入，旧无此证据的探针拒绝。
- 路线来源回执改为最多 32 项 LRU；淘汰后重读须再次核验原 SHA。RGB/LiDAR/metadata 仍由已核验的原字节解码。
- 本轮原生 ZIP checkpoint 用私有 mmap 重载，probe/evaluate 复制 decoder/EMA 后立即释放映射字典，避免触碰未使用的优化器 storage。最终预算检查和交包验收同样不载入无关张量。不放宽权重/合同检查，不加载旧 Phase3 LoRA 或旧 Action decoder。
- 每 rank 进度加入 RSS/HWM、主机和 cgroup v1/v2 证据；父进程在每个任务前后单独保存内存证据，子进程被 SIGKILL 后仍能记录。成功包和失败包均收集这些证据。无法读取的字段明确标 unavailable，累计计数不自动等同于本次故障。

169 份数据构建来源逐字不变；标定、提示词、划分、配额、七轮/12719 更新/813792 呈现、优化器、学习率和共享训练循环均不变。此前 checkpoint 保存协调修复继续保留。精确扩展 processor 源码兼容表，不改旧回执；新的模型/数据/来源准入仍执行，旧探针不复用。

## 本地验证与边界

- 真实全池 843059 条记录（train731645、val52938、test58476）与旧 JSON 解码逐字段一致；七轮、四 rank、全部计划 ID 顺序一致；原验证成员 320 条及其选择 SHA 不变；a/b/c 条件诊断和已收到 processor 极值成员选择一致。
- 单进程完整索引后 RSS 617353216 字节（589 MiB），相对导入基线增加约 219 MiB；包含七轮计划时 RSS 749670400 字节（715 MiB）。旧实现仅 50000 条的对照 RSS 已达 1341464576 字节（1.25 GiB）。两者记录数不同，不称正式训练峰值同比。
- 真实四进程 Gloo 各加载完整索引/计划，合计 RSS 2902056960 字节（2.70 GiB）；每 rank 2 个 spawn 加载 worker，合计 RSS 4739649536 字节（4.41 GiB），完成 100 条真实 RGB/LAZ/metadata 消费，顺序及来源核验通过。RSS 相加包含共享页重复计数，并非机器独占用量。
- 214 项不同相关 CPU 测试通过：ABC/保存/内存观测 104 项，共享训练/进度 16 项，旧语义消融 94 项。含真实小 ABC 模型优化/保存/mmap 重载、索引版原生训练循环尾窗口/曝光、探针实际 spawn worker 存活与失败关闭、四进程保存后反向、输入篡改拒绝、真实 shell 恢复与自动交包。
- 本机没有加载完整 Qwen/BEV 权重，没有真实 CUDA/NCCL 或默认每 rank 8 个 worker 的全配置长训；不能据此宣称 GPU 长训验收通过、性能提升或所有平台 OOM 已消除。服务器库与本机略有成员差异，保留服务器原划分，不重建为本机版本。

本机证据在 `checkpoints/action_memory_fix_20261026/`：`verification.json`、`full_pool.json`、`four_process.json` 及测试 XML。原始诊断包、模型、数据、日志不发布到 Git。

## 服务器恢复

保留失败目录。该次失败只包含探针，可从它读取兼容 processor；新的源快照和探针会重新生成，不能把旧 run-dir 指定给新代码热更新。整套训练仍从 Qwen3.5 基座、允许的 BEV 感知预训练和新 decoder 开始。

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
bash ./sync_automot.sh
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261025"))[0])')
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --recover-from checkpoints/action_expert_ablation/geometry_abc_checkpoint_fixed/run_20261025_203923 \
  --maps "$ABC_ACTION_MAPS" \
  --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc_memory_fixed
```

准入和六组新探针通过后自动训练、测试、审计并打包，主包 30 MB 上限不变。失败仍保留日志并自动交 `failure_delivery/diagnostic.zip`，现在包含任务前后内存和各 rank 内存进度。不通过跳过失败探针或忽略 OOM 来继续后续训练。

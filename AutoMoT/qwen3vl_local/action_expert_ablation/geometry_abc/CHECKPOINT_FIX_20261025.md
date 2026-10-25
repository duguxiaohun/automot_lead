# 首次 checkpoint 保存导致其它 rank 通信超时

收到的 `action_nccl_review_20261025_201341.zip` 对应 `run_20261025_162304`，执行版本 `a8d7e02d...`。六个 probe_complete 回执均存在；此次失败发生在 **a_bev_only 正式训练**，不是准入或探针。17:22:20 已完成第 1/12719 次更新、每卡 16 次呈现，共 64 次呈现，global FM loss=1.602519。

rank0 随后停在 `checkpoint/write`，17:32:22 退出栈仍是 `torch.save → zip_file.write_record`。rank1/2/3 已进入下一 micro 的 `train/forward`，NCCL 第 19 次 BROADCAST 等待 600 秒超时。DataLoader worker 的 SIGTERM 出现在通信超时后，不能据此判定数据加载器首先出错。服务器代码身份与本机修复前完全一致。

## 已确认的代码缺口与限制

原共享 `save_training_checkpoint` 在采集各 rank RNG 后，仅让 rank0 保存；其它 rank 立即返回并进入下一轮 GPU 通信。没有保存完成确认。原探针保存后紧接 barrier 和重载，缺少保存后继续 DDP 前向/反向的检查。日志证明 rank0 长时间停在写入、其它 rank 跑到下一轮，但不足以区分底层共享盘慢写、I/O 卡住与 CUDA/通信相互等待；不能声称已证明服务器磁盘故障，也不靠延长 NCCL 超时掩盖问题。

## 修复

共享 Action 保存函数使用 `action_prior/checkpoint_io.py`：

- rank0 将模型、优化器、EMA、RNG 等 tensor 转为 CPU，保留 state_dict metadata、schema、合同、游标及数值。
- 其它 rank 使用 Gloo CPU 通信等待保存结果，不提前进入下一轮 CUDA collective；NCCL 训练本身不改。Gloo 控制组按进程组复用，等待仍有 600 秒上限，真实存储永久卡住仍会失败并报告 checkpoint 协调问题。
- 写入唯一临时文件，flush/fsync 后原子替换目标；正常返回异常会广播给所有 rank。失败不覆盖上一个完整 checkpoint，临时文件保留诊断。进程崩溃、系统级永久阻塞不承诺可恢复。
- 各 rank 明确记录 `checkpoint/cpu_snapshot`、`checkpoint/wait_for_writer`、`checkpoint/write`、保存字节及分段时间。
- ABC 六组探针在保存后立即再次执行训练集前向/反向，再重载 checkpoint；没有中间 barrier 掩盖缺口，不额外执行 optimizer 更新，不使用 val/test 反向。正式入口要求新的逐 rank 证据，旧探针不复用。
- 失败自动交包加入各 rank progress JSON。现有 processor 仅在精确源码兼容策略及数据/模型复核后复用；本轮只扩展保存、探针和诊断代码的已审查 SHA，不改图像 packing 或标定。

169 份数据构建来源保持原样，标签、prompt、split、采样、七轮/12719 更新不变。旧语义 Action/BEV/Qwen 使用同一保存函数，因此同样获得协调修复；不修改其采样或数值训练循环。Phase1/2/3/4 的独立训练实现、默认实验选择及已有结果不改。

## 本地验证

针对性回归包括真实四进程 Gloo、原生 ABC 小模型/优化器/EMA checkpoint 保存、模拟慢写盘、保存后实际 ABC 前向/反向、rank0 写盘失败全 rank 同步报错、旧完整 checkpoint 保护、metadata/RNG 保持、旧 probe 拒绝及失败包包含进度。将保存函数替换回修复前版本，两个新增四进程回归都失败；修复后通过。

相关测试 305 项通过（ABC/保存 83 + 共享训练/优化/恢复等 173 + 旧消融 49，互不重复）。扩大检查另外发现 11 项既有失败：9 项旧 decoder 配置兼容测试、2 项比较 shell 的默认每类样本数期望 50 而实际 30；修复前保存实现及未改动脚本下同样存在，未为通过测试改动旧模型/比较默认。CPU Gloo 不是 CUDA/NCCL 验收，本机未启动真实四卡或远程训练，服务器仍须运行新版六组探针。

证据：`checkpoints/action_checkpoint_fix_20261025/`。收到的服务器库记录 train=731643、val=52939、test=58476；不把它与本机 843059 条库冒称逐条相同，也不因本次超时重建或改变服务器划分。

## 服务器重新执行

这次已有正式训练目录和一次更新，不能将 `run_20261025_162304` 当作“仅准备失败”恢复，也不能给它改 SHA 或指定旧 execution version 继续。保留全部旧产物；从最初仅准备的 `run_20261024_232223` 读取可复用检查，新建独立实验，从原基座/BEV 感知权重及新 decoder 初始化。失败实验的 1 次更新不计入新实验预算，不读取旧 Action/decoder 或 Phase3 LoRA 权重。

```bash
cd /mnt/workspace/cee3sgh/AutoMoT
bash ./sync_automot.sh
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261025"))[0])')
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --recover-from checkpoints/action_expert_ablation/geometry_abc/run_20261024_232223 \
  --maps "$ABC_ACTION_MAPS" \
  --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc_checkpoint_fixed
```

成功时自动训练、测试、审计、交包；若仍卡在写入，新的各 rank 保存阶段与耗时会直接进入失败诊断包，需据此检查服务器存储，而不是反复重复训练或关闭门控。

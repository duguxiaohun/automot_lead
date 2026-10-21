# 方案一首轮周期验证失败：原因、修复与重启

用户交来的 `phase3_report.zip` 是方案一运行失败包。错误出在方案一与原生验证器的接口，先前测试遗漏了「固定验证成员存在零个有效未来窗口」的真实情况。不是要求用户再次标定，也不是历史 UNKNOWN 必须消除。

## 已收到的证据

- ZIP 6,547,616 字节，SHA256 `54591b2d71fe01d330aef889cf27f4dc90862a0da143532b13a6246688c6a50d`；CRC 和 19 个成员 SHA 全部通过。
- 报告主机 `szh-gpu-cr02`，`derive_common`、`prepare`、四卡 `probe` 均成功，`train` 阶段 5950.13 秒后失败。
- 四个 rank 的首轮 batch 日志各 2560 条，合计 10240 次呈现，33733 个有效监督窗口。它们记录成功 backward，不等于已保存可恢复 epoch。
- 失败点是 global step 2560 的 `evaluate_loss`：`fixed validation cannot skip inputs rejected by max_length/packing`。
- 实际生产 tokenizer 的 A–F 为现有 token 32–37；输入 1432–1479 tokens。旧四卡 probe 峰值 reserved 约 15.74–16.14 GiB，不是此次 OOM，也不是上下文超长。
- 本次服务器原生索引 SHA 为 `925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99`，沿用原固定评测名单。不要把此前本地不同池生成的新名单结论套到这次服务器运行。
- 没有收到 base 运行结果；本包不能判断 base 是否训练成功，也不构成重跑 base 的理由。

## 具体原因和修复

方案一允许缺失未来监督；一个未来半秒格的两个原生标签有任何一个不可用，该格就无 loss。某些固定验证锚点的四格全部无效，旧 `runtime.pack` 返回 None。原生训练器把 None 专门当成输入超长/打包拒绝，要求固定验证不能静默丢题，因此在第一轮末停止。旧测试检查了 pack 返回 None，却没把该情况送进完整周期验证；旧 probe 也只测验证前向形状。

修复仅位于独立方案一适配层：

1. 固定 160 个 loss 验证成员保持不变，预先记录哪些成员无监督。
2. 只将至少有一个有效窗口的成员传入原生 loss 计算。loss 仍按有效锚点求平均，每个锚点内部按有效窗口归一化；不把缺失标签算作零 loss 或正确答案。
3. 每次日志增加 requested_samples、unlabeled_samples、有效/请求窗口数及覆盖率。原生 skipped 继续专指输入拒绝；真正的 max_length/packing 问题仍失败，整体验证无监督也仍失败。
4. 开训前四卡 probe 现在实际执行完整固定名单的周期验证，检查成员与全局分母一致；不再只验证 shape。probe 参数不带入正式训练。

未修改原生 trainer、base 冻结发布清单、标定规则、历史填补、未来 targets、提示词、事件/action 配额或训练预算。仍为三轮 7680 更新。

## 本地验证

72 项相关回归通过（35.92 秒），包含：

- 修改前的小模型原生循环准确复现同一个异常；修复后真实 tokenizer/单图打包/四查询 loss 的三轮训练完成。
- epoch 1/2 接续与连续训练的参数、optimizer、scheduler、RNG、采样历史逐项一致；验证样本含零监督和部分监督，核对实际写出的分母与窗口数。
- 四个真实 Gloo/DDP 进程执行新增 probe helper 和原生验证汇总；各卡有效锚点为 0/1/2/1，全局 10 个请求、4 个有效、6 个无标签、14 个有效窗口，四卡结果一致。
- 全局零监督、真实 packing 拒绝仍在全部 rank 报错，不以放宽检查消除异常。
- 固定公共库/launcher、正式 val/test 对照、运行时刷新与标签代码变更拒绝等原回归。

另对本地完整派生库执行 runtime-only refresh：保留旧文件，新目录只更换 runtime.py/worker.py 的绑定，不重新标定。具体本地记录数、三轮计划/评测名单不变的验证和源 SHA 见同日轻量 JSON 回执。本地池与服务器池不同，不用本地零监督数量外推服务器数量。本轮没有执行修复后的 H20 完整训练。

## 为什么这次不能接着第二轮

当前原生顺序为「周期验证 → epoch 消费核验 → 完整 recovery 保存」。本次在周期验证就退出，且 `save_steps=0`、关闭 best 保存，尚未写出本次 epoch 1 的可恢复断点。不是根据 ZIP 不含权重推断——打包器本来就排除权重。不能从 batch 日志还原参数/optimizer/RNG，因此修复后方案一需要从头训练，不能伪称接续；原失败目录保留。

## 同步后只重启方案一

在有本次完整数据的那台机器的 `AutoMoT/`、automot 环境执行，先沿既有 `bash ./sync_automot.sh` 同步源码。不要搬另一台机器的数据/缓存，不重新建库，不重复 base。

第一次先显式刷新运行代码绑定，再启动新输出目录：

```bash
python -m qwen3vl_local.audit_joint.phase3_single_frame.refresh_runtime \
  --dataset checkpoints/phase3_common_v1/single_frame/dataset.json \
  --output checkpoints/phase3_common_v1/single_frame_validation_v2 \
&& GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame \
  --dataset checkpoints/phase3_common_v1/single_frame_validation_v2/dataset.json \
  --output phase3_single_frame_validation_v2
```

刷新会校验原始证据，并写出一份保留原记录的新绑定文件，因此仍有 CPU/磁盘读取耗时和一份派生 JSON 空间；不扫描重建原生题库、不重新计算动作。`single_frame_validation_v2` 已成功生成后只执行第二条，不能再次覆盖该目录。不要加 `--resume` 指向本次失败目录。

默认本机 `lead_data` 与 `checkpoints/Qwen3.5-4B`，与本次包一致。新 run 为 `phase3_single_frame_validation_v2/run_single_frame_nearest_v1/`；正常或中途失败均交其 `handoff/phase3_report.zip` 和 `phase3_package.json`，前置未形成包则交 launch 日志。总训练量未增加；原首轮约 99 分钟是本次实测，不能保证后续耗时或性能。

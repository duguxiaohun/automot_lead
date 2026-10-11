# 服务器下一步：先诊断和盘点，暂不训练

本机继续做 G1 关联、证据来源及 G2 试标准备。服务器现在需要执行的是 **CPU 检查 + 旧一轮模型 U-E7 训练题推理**；不重复 U-E1 的181题，不续七轮，也不让 v48 接着旧 adapter 训练。

以下命令均从对应版本的应用目录运行（目录内已有 `qwen3vl_local/` 和 `checkpoints/`）。新版通过既有 `sync_automot.sh` 同步到独立目录；旧一轮实验仍留在原目录/原环境。不要在应用目录遗留 `.git` 中 pull、不要把新版学生/老师文件覆盖旧实验。脚本的同步参数使用服务器已有方式。

## 1. 新源码目录：CPU 回归与资产盘点

```bash
ulimit -S -c 0
python -m pytest -q \
  qwen3vl_local/audit_joint/tests \
  qwen3vl_local/sft_new_loop_phase4/tests \
  qwen3vl_local/sft_new_loop_phase3

python -m qwen3vl_local.audit_joint storage-inventory --root checkpoints --core-dir .
python -m qwen3vl_local.qwen35.preflight --model-dir checkpoints/Qwen3.5-4B
```

模型不在默认位置时只替换真实 `--model-dir`。以上不训练、不下载、不删除文件；preflight 仅检查模型/vendor/环境，不代表四卡实际内核或 E0 性能已通过。

G0 还缺 Phase3 正式 `frame_index.jsonl`、明确 prompt variant、原 adapter（如已丢失保留 missing），Action 原三 split、原参数 JSON 和完整事件 map。先查既有目录，不为填缺口启动重建：

```bash
find checkpoints -type f \( \
  -name frame_index.jsonl -o -name sft_new_loop_phase3_adapter_config.json -o \
  -name effective_pool_audit.jsonl -o -name pipeline_ready.json -o \
  -name manifest.json -o -name '*argv*.json' \
\) -print
```

将测试结果、preflight 与盘点结果交回；随后按实际存在的资产填写 [G0 手册第3–4步](SERVER_RUN_20261008.md)。不再提交一模一样的缺失报告，也不把旧v42题库传给v48加载器后改hash“兼容”。新版G0需要匹配版本；旧基线补齐在旧源码环境做。

## 2. 旧一轮实验目录：U-E7 已采训练题推理

此步骤用于区分“已有监督未拟合”和“训练/验证状态不一致”，不产生梯度更新。需要保留原 `train_fit` 工具、原两组 run、原学生视图及其引用的原 data4；若旧目录缺工具，只能补独立审计工具，不能替换学生或生产文件。

下列 run/view 是已收交接包登记的路径。若服务器移动过资产，先核对实际位置和来源；不要用 `latest` 猜旧 run。每组分别查询 `complete`、`proceed`，包含原生全部阶段，题量由真实已采集合决定。

```bash
set -euo pipefail
ulimit -S -c 0
for mode in 2 4; do
  if [ "$mode" = 2 ]; then
    UE7_RUN=checkpoints/phase4_full_epoch1_rgb2_20261009/run_20261009_235753_477334207_rgb2
  else
    UE7_RUN=checkpoints/phase4_full_epoch1_rgb4_20261009/run_20261010_020535_587421198_rgb4
  fi
  for edge in complete proceed; do
    UE7_OUT="checkpoints/phase4_ue7_fit_20261011_rgb${mode}_${edge}"
    bash qwen3vl_local/audit_joint/run_guarded.sh \
      --space-path checkpoints --min-free-gib 20 -- \
      python -m qwen3vl_local.audit_joint.train_fit fit \
      --run "$UE7_RUN" --epoch 0 \
      --dataset checkpoints/phase4_short_observed_20261009/view \
      --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
      --event U-E7 --edge "$edge" --phase all --output "$UE7_OUT"
    python -m qwen3vl_local.audit_joint.train_fit verify-fit "$UE7_OUT" \
      --run "$UE7_RUN" --dataset checkpoints/phase4_short_observed_20261009/view
  done
done
```

默认自动选一张空闲 GPU，四项串行。显式 pin 示例：将循环内的 launcher 改为 `GPU_IDS=0 bash qwen3vl_local/audit_joint/run_guarded.sh ...`，其余参数保持上面的完整命令。不是四卡训练，不需要分给两种输入各两张卡。

如需先做CPU选择预检，同一 `fit` 命令加 `--prepare-only`，同时将输出改成另一个带 `_prepare` 的新目录；正式推理使用未存在的输出目录。不能对prepare输出直接重跑或将它计作拟合完成。任何来源不符、OOM或无已采支持导致非零退出时先交失败报告；不改门禁或启动训练补题。

完整 `verify-fit` 须 `verified`、退出0；只有内部一致性或缺原视图的退出2不能升级通过。

## 3. 旧目录打包四份结果

四项成功后使用已有 capture，保留全部逐题 raw/指标/回执：

```bash
python -m qwen3vl_local.audit_joint pack \
  --capture checkpoints/joint_missing_assets_20261009_layoutfix/capture \
  --result-dir ue7_rgb2_complete=checkpoints/phase4_ue7_fit_20261011_rgb2_complete \
  --result-dir ue7_rgb2_proceed=checkpoints/phase4_ue7_fit_20261011_rgb2_proceed \
  --result-dir ue7_rgb4_complete=checkpoints/phase4_ue7_fit_20261011_rgb4_complete \
  --result-dir ue7_rgb4_proceed=checkpoints/phase4_ue7_fit_20261011_rgb4_proceed \
  --output checkpoints/phase4_ue7_fit_handoff_20261011.audit.zip
python -m qwen3vl_local.audit_joint verify-package \
  checkpoints/phase4_ue7_fit_handoff_20261011.audit.zip
```

capture 路径如已变更，替换为原始可核验目录，不新造一个“通过”capture。某项失败时，`pack` 仅传实际已生成的结果目录并保留失败报告，不要求其余缺失目录存在。ZIP 硬上限30MB；手工交回本机 checkpoints 即可，不推数据/权重到 Git。

## 4. 现在不需要训练什么

- 不续旧一轮至七轮，不重跑181道U-E1拟合。
- 不用v48源码加载/续训旧adapter，不改旧manifest SHA。
- 不立即全量重建v48：先确认G0资产、磁盘和新实验范围，旧回放保持。
- E0四rank实际内核/profiler、E1数值等价及新基线训练另一步实施；现有preflight不能替代。Phase3权重丢失时的新训练是新基线，不是旧实验恢复。

U-E7结果与G0资产清单交回后，才能给出绑定真实路径和确定数据版本的下一次训练命令；此时先不运行 `run.sh` 的默认多轮流水线。

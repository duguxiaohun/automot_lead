# 服务器CPU验收通过：下一步为存储预算与当前正式池

> 2026-10-12 v52发布后当前服务器步骤：见[新版限量试建交接](SERVER_V52_NEXT_20261012.md)，同步后仅运行 `bash qwen3vl_local/audit_joint/server_v52_pilot.sh`，交回 `pilot_report.zip` 与 `pilot_package.json`。不重复readiness/旧拟合，不直接全量建库或续七轮；本地107帧重算/28帧RGB续审见 `audit_joint/RELEASE_RGB_V52_20261012.md`。

**后续状态：** v52已修复累计索引/发布保护，并接入固定自动首轮准入、训练重载及test入口。正式池与预算仍待交付，readiness保持关闭；详见[修复与执行步骤](BATCH_BASELINE_V52_20261012.md)。

**2026-10-12更新：** v51共享引用/压缩已完成本机验证；后续按[存储与RGB审计](STORAGE_RGB_V51_20261012.md)补正式池和完整峰值预算。不要重跑已关闭的readiness或旧fit，也不要把旧v42资产填入新合同直接续训。

**2026-10-12 08:31新包已验收：** 440源码文件与111测试文件清单一致，服务器2262通过/3项历史产物依赖跳过，source/pytest/preflight均退出0。沿用用户覆盖同步方式，6个额外文件保留；同步与CPU入口问题已关闭，**不要重复readiness、相同磁盘盘点或旧拟合**。详见 [当前开训审计](CPU_PASS_TRAINING_AUDIT_20261012.md)。

### A. 当前必须先完成的收尾

1. 新产物存储减少production重复复制并完成完整峰值预算。当前45.700 GiB可用、保留20 GiB后25.700；旧规模复制式新建约39.132 GiB，未计规则留出和权重/优化器。本机原生5路线464题编译/重载确认20.81MB库中14.27MB为额外回放副本。该存储改造尚未实施，不能直接启动现有复制式全量流水线。
2. 按当前合同生成Phase3正式索引及Phase4完整候选回放、单份四图库/短视图与规则留出，做原生准入及逐事件/条件支持检查。旧v42资产保留，不能改hash兼容；局部开发池不是正式全源。
3. CPU采样与数据准入通过后，新合同两图/四图各一轮＋保存重载，再进入后续正常训练。事件之间1∶1，不要求YES:NO等额；不等待全部独立人工盲审/性能实验/CARLA。

**下方B–D仅为原有资产齐备后的模板，当前存储实现与预算未满足，暂不执行。** 下一次真正需要服务器做的工作是新版正式数据准备及一轮新基线，不再是重复CPU环境检查。

### B. 新合同资产齐备后：自动训练库和规则留出

在应用目录执行。下列四个路径必须替换为实际资产，`P50_NEW`须尚不存在。`P50_WEAK_REGISTRY`应是当前 `full_pipeline.default_registry()` 生成的弱监督registry；仓库空 `teacher_registry_v25.json` 是严格批准登记，不能拿它冒充弱训练许可。

```bash
P50_POOL=/实际路径/v50/candidates.json
P50_PROD=/实际路径/v50/full_production/index.json
P50_WEAK_REGISTRY=/实际路径/v50/weak_registry.json
P50_NEW=checkpoints/phase4_auto_v50_new
test ! -e "$P50_NEW" || exit 2
mkdir -p "$P50_NEW"

# 不传人工annotations；只编译一份RGB4，仍需一份production副本的空间。
bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$P50_NEW" --min-free-gib 20 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --candidate-pool "$P50_POOL" --production-index "$P50_PROD" \
  --teacher-registry "$P50_WEAK_REGISTRY" --max-teacher-questions-per-route 0 \
  --data-root lead_data --rgb-mode 4 --output-dir "$P50_NEW/data4"

bash qwen3vl_local/audit_joint/run_guarded.sh --space-path "$P50_NEW" --min-free-gib 20 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.teacher_evaluation \
  --candidate-pool "$P50_POOL" --production-index "$P50_PROD" \
  --data-root lead_data --rgb-mode 4 --output-dir "$P50_NEW/rule_refs4"

python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
  --dataset4 "$P50_NEW/data4" --output "$P50_NEW/student"
```

守护是余量监控，不是硬磁盘配额；先预算再运行。不要调用默认同时编译data2/data4的整套流水线来替代这里的单份编译。原生val/test可以为空，由显式自动留出包补足；两者仍须严格物理组隔离，独立老师审核预约不会参与选优。

### C. CPU准入和采样检查

```bash
P50_VIEW="$P50_NEW/student/view"
P50_REF4="$P50_NEW/rule_refs4"
for P50_MODE in 2 4; do
  WORLD_SIZE=4 RANK=0 python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.train \
    --dataset "$P50_VIEW" --rgb-mode "$P50_MODE" \
    --selection-mode rule_consistency --teacher-references "$P50_REF4" \
    --sampling-only --epochs 1 --output-dir "$P50_NEW/cpu_rgb$P50_MODE" \
    > "$P50_NEW/cpu_rgb$P50_MODE.log" 2>&1 || break
done
```

该命令只检查/打印计划，不初始化GPU、启动四rank或写训练权重。两份日志均须成功；任一缺事件/隔离/合同错误就先补数据，不跳过门禁。另提交正式池的逐事件×边×阶段×答案题数和物理组数，重点核对U-E3/5/6 proceed YES、U-E7 complete YES、U-E2 return；十事件等量不等于这些条件都有支持。等额val/test取最少事件的不同题数，禁止重复补足。

### D. 覆盖和CPU准入通过后：两模式各完整一轮

```bash
for P50_MODE in 2 4; do
  DATASET="$P50_VIEW" DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B \
  OUTPUT_DIR="$P50_NEW/runs_rgb$P50_MODE" RGB_MODE="$P50_MODE" \
    bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh \
      --selection-mode rule_consistency --teacher-references "$P50_REF4" --epochs 1 || break
done
```

此时才使用四卡；每模式默认完整10240次呈现，脚本随后新进程保存重载核验。保留原题库、视图和参考包路径，独立评测/重载会核验其逻辑路径、解析目标和SHA。旧adapter不续用。交回run、固定评测计划、逐题输出、采样和重载回执，核对正常再安排后续轮数；规则一致率不能写成独立正确率。G0/G1/G2未因此自动通过。

---

## 历史记录（以下旧指令不覆盖上方v50步骤）

**2026-10-11最新收尾顺序：** 用户确认只要求事件间1∶1；参照Action共用均衡实现后的实测与最短开训步骤见[开训收敛报告](TRAINING_SCOPE_20261011.md)。现需当前Phase3正式索引/variant、当前Phase4全池manifest及training_scope统计，避免把47路线局部缺额推断为全源缺额。E1–E6等优化不作为新基线前置；已完成的旧拟合不重复。下面历史命令保留，旧版本数据不交给新合同加载。

**本地开发更新至v49：** 已修复未执行机会回滚晚于实例截止检查的问题，并完成局部题库差分；见[v49报告](LABEL_AUDIT_V49_20261011.md)。旧服务器结果继续保留原合同。下述CPU检查可用于同步后的新独立目录，训练/旧checkpoint复验仍须匹配各自原版本，不能让v49加载旧题库或直接续旧adapter；本轮未push。

**接收更新：四组U-E7结果已交付并完成本机范围内复核，下面第2–3步仅保留作历史复现，不要再次执行。** 见[接收报告](UE7_FIT_RGB_REVIEW_20261011.md)。现在服务器只补尚未交付的第1步CPU回归/preflight/资产盘点；已有结果直接交回。继续暂停训练，待本地实例、完成定义及新版正式池准入差分明确后再登记新基线。

本机继续做 G1 关联、证据来源及 G2 试标准备。原服务器清单是 **CPU 检查 + 旧一轮模型 U-E7 训练题推理**（推理现已完成）；不重复 U-E1 的181题，不续七轮，也不让 v48 接着旧 adapter 训练。

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

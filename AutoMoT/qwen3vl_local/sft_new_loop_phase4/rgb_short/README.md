# 短两图基线：先准备一次，再四卡短跑

2026-10-09 用户确认：各自 RGB4 保持原样，RGB2 均使用 `[-2,0]`（4 Hz 下0.5秒）。不是新增三图，不统一修改两个任务的标签。Phase3预测采集行为，Phase4判断当前条件；本轮不修改这两种语义。

| 输入 | Phase3 | Phase4 |
|---|---|---|
| 保持的四图 | `[-3,-2,-1,0]` | `[-6,-4,-2,0]` |
| 新短两图 | 原四图位置 `[1,3]`；`2rgb_short` | 原四图位置 `[2,3]`；本目录入口的 `--rgb-mode 2` |
| 历史两图 | `2rgb_endpoints` 仍为 `[-3,0]` | 原生旧入口仍为 `[-4,0]` |

旧名称不悄悄改义；原生 Phase4 的生产、老师、人工审核合同和旧 checkpoint 保持。新训练/评测通过独立 `rgb_short` 入口，checkpoint 绑定学生输入合同、视图身份与源码，拒绝混用旧 adapter。为保持已冻结生产源码SHA，学生编码器/训练循环从旧实现独立派生，没有运行时替换旧模块或篡改旧 manifest。

## 1. Phase4 只准备一份四图库和小视图

所有命令从服务器 `AutoMoT/` 执行，使用原 automot Python 环境。更新代码后，**本节替代之前续建旧 data2/data4 的命令**。如果旧四图库已经完成，走复用分支；没有则从已有回放编译一次四图库。两个分支择一，输出目录须未使用。不要与旧建库任务并行写同一原目录。

尚无完整 data4（本次已知服务器状态）：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path checkpoints --min-free-gib 20 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
  --phase4-base checkpoints/joint_remote_20261008_211924/phase4_data \
  --data-root lead_data \
  --output checkpoints/phase4_short_rgb_20261009
```

已有完整四图库时，改用 `--dataset4`，不再编译或复制production：

```bash
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path checkpoints --min-free-gib 20 -- \
  python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
  --dataset4 checkpoints/joint_remote_20261008_211924/phase4_data/data4 \
  --output checkpoints/phase4_short_rgb_20261009
```

`--phase4-base` 先核验旧请求、候选池、registry和完整逐路线回执，仅调用原生四图编译器，绝不启动老师回放。会新存一份production（按已收到报告约18.33 GB），加上题目和元数据；短两图不再复制第二份production。原13 GB data2残片保留。20 GiB守护预留不是总空间保证；失败后不要不断换目录重试，若四图库manifest已完成可用第二分支只生成小视图。

主要产物为新 `data4/manifest.json`（编译分支）及 `view/student_view.json`。视图引用原四图库的逻辑/解析路径和文件SHA；父库必须保留，搬迁或软链接改指会被拒绝。旧 `pipeline_ready.json` 不补写，也不把这个实验视图交给旧 `full_pipeline --skip-build` 或原生 `train.py`。

新视图同时为2/4图筛选相同题目：剔除新短输入下同split的相反答案冲突，剔除已登记开发曝光的val/test路线，逐项保存排除原因。不搬动旧split；test标签不参与train筛选。Phase3/Action全池尚不齐，`joint_independence_certified=false` 保持。

两图标签是从完整四图题目转来的**实验参考目标**，不是重新审核过的两图真值。原四图证明不作为两图审核批准；评测输出 `transferred_reviewed_reference` 和 `teacher_consistency`，选优用开发参考宏平均一致率，不写独立人工准确率。四图控制组的每题图像、提示词、目标保留，题集与两图采用相同排除。

## 2. 先CPU采样，再四卡短跑

准备完成后，先运行实际加载与四rank采样计划（没有CUDA或模型加载）：

```bash
WORLD_SIZE=4 python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.train \
  --dataset checkpoints/phase4_short_rgb_20261009/view \
  --rgb-mode 2 --output-dir checkpoints/phase4_short_sampling_no_weights \
  --sampling-only --epochs 1 --epoch-samples 40
```

把 `--rgb-mode` 改为4可检查同一题序。输出目录参数在sampling-only下不会创建。若排除后缺少基础train/val/test支持，正式训练会在模型加载前阻塞并列原因；不拿test补val。

四卡短跑沿用原模型环境，自动选择四张空闲卡：

```bash
DATASET=checkpoints/phase4_short_rgb_20261009/view \
OUTPUT_DIR=checkpoints/phase4_short_smoke RGB_MODE=2 \
bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh \
  --epochs 1 --epoch-samples 40
```

显式选卡的等价示例（二者择一）：

```bash
GPU_IDS=0,1,2,3 DATASET=checkpoints/phase4_short_rgb_20261009/view \
OUTPUT_DIR=checkpoints/phase4_short_smoke RGB_MODE=2 \
bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run.sh \
  --epochs 1 --epoch-samples 40
```

40次全局呈现，每卡10次前向，默认accum8对应2次更新；随后完整val生成、保存adapter及恢复状态。它仅证明运行链路，不能作为效果结果。四图控制组用 `RGB_MODE=4` 和不同 `OUTPUT_DIR`；两种模式串行执行，每次各用四卡。短跑后先检查loss、显存、实际耗时和保存/加载，正式训练再去掉40次限制（默认每事件1024次、每轮10240次）。本轮未代替服务器执行短跑。

每个run写 `student_view.json`、`run.json`、`data_admission.json`、采样/验证/逐题结果；联合打包器已支持该视图清单，沿用 `audit_joint pack --result-dir NAME=实际run目录` 的30 MB交接规则。完整离线评测入口为 `python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.evaluate --help`；adapter输入/题库与请求不符时在模型加载前拒绝。当前未接入CARLA或修改运行时执行许可。

## 3. Phase3 重建索引与新输入

Phase3继续使用原四帧索引，短两图只改变学生选择的两张图及相应提示词。新索引生成不需复制RGB，也不要求旧adapter恢复。缺原始Phase1/2 collection或RGB时，需先恢复这些上游输入，不伪造题目。

只建索引和CPU预检，不训练、不测试旧adapter：

```bash
PIPELINE_CHECK_ONLY=1 PACK_RESULTS=0 HISTORY_RGB_MODE=2rgb_short \
DATA_DIR=checkpoints/phase3_short_index_20261009 \
PIPELINE_ROOT=checkpoints/phase3_short_prepare_20261009 \
bash qwen3vl_local/audit_joint/run_guarded.sh \
  --space-path checkpoints --min-free-gib 20 -- \
  bash qwen3vl_local/sft_new_loop_phase3/run_full_pipeline.sh
```

这会按原候选提示词及开发隔离配置建库；实际默认/显式参数写入pipeline记录。索引补齐后先重新检查三任务可用池隔离，再使用原train.sh，以 `HISTORY_RGB_MODE=2rgb_short` 进入短两图训练，`4rgb`保留原四图。旧adapter丢失时新训练登记新基线。`RUN_REGRESSION=1` 对短两图会明确拒绝借用历史端点结果；新输入没有同输入历史参考。

本轮不将E0/E1优化、完整G1/G2盲审或Action恢复作为单任务短跑的前置条件；这些缺口与独立验收边界仍保留。

本机验证：相关完整回归1361项通过，收尾短输入专项35项通过（包含真实小模型CPU反传及模拟四卡启动，未运行GPU）。原生人工题443/273/186经共同排除后为443/239/123，805道保留题的实际RGB身份核验通过；该统计仅针对本机人工题库，不代表服务器全量弱题数量。小视图45620字节，详见[验证记录](verification_20261009.json)。

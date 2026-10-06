日常使用只需 `bash qwen3vl_local/sft_new_loop_phase4/run.sh`，自动连跑四图和两图，详见 [一行入口](QUICKSTART.md)。下文保留高级单实验用法。

# Phase4 一条命令运行全量两图／四图实验

从 `AutoMoT` 目录执行。使用已安装 torch/transformers/peft 等训练依赖的 Python；基座默认 `checkpoints/Qwen3.5-4B`，原始数据默认 `lead_data`。

```bash
bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh && \
SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
```

第一条默认四图，第二条两图。每次依次执行：全量配对数据准备／核验 → 主机和采样预检 → 训练（每轮完整验证集选优）→ 最优 adapter 完整测试集评估 → 本地 `audit.zip` 打包。测试集不重复采样凑 1:1。

Phase4 仅预测 YES/NO，默认 `ACTION_OUTPUT_MODE=binary`；也可显式传入这个变量。`choice` 不适用，不需要四种实验组合。

## 数据与训练范围

首次构建自动扫描全部合格路线，保持冻结物理路线划分，逐帧回放老师，合并已有审核题，生成 `DATA_DIR/data2` 和 `DATA_DIR/data4`。默认 `DATA_DIR=checkpoints/phase4_v38_full`。每路线老师题上限为 **0（不截断）**；配对丢题会报错。

每轮覆盖全部合格训练题，严格按十事件的**呈现次数 1:1**，通过重复小事件题池实现。最小轮预算由最大事件题池和实际 DDP world size 决定。不给定 `--epoch-samples` 时自动计算；过小预算拒绝。旧 cap4/cap8 不限制此模式。默认 7 轮、梯度累积 8，沿用全量采样与 checkpoint 恢复合同。

全量指全部合格监督，不将未知／未标注帧补成 NO。老师仍只覆盖 UE1/UE4，其余事件沿用人工题，属于全量弱监督实验。大规模重复不增加稀缺事件的信息量；六事件人工留出评估缺口不因本次入口改造而消失。

## 常用环境变量

| 变量 | 默认值／用途 |
|---|---|
| `HISTORY_RGB_MODE` | `4rgb`；两图用 `2rgb_endpoints` |
| `TRAIN_MODE` | `ddp`，最多选择 4 张空闲可见 GPU；也支持 `single` |
| `GPU_IDS` | 显式指定 GPU，例如 `0,1,2,3` |
| `PYTHON` | `python`；可指定虚拟环境解释器 |
| `MODEL_DIR` | `checkpoints/Qwen3.5-4B`，必须为已安装的完整基座 |
| `DATA_ROOT` | `lead_data` |
| `DATA_DIR` | `checkpoints/phase4_v38_full`，两图／四图共用的建库目录 |
| `BUILD_WORKERS` | `16`，按路线并行，范围 1–64 |
| `EPOCHS` / `ACCUMULATION` | `7` / `8`；额外训练 CLI 参数继续传递 |
| `PIPELINE_ROOT` | `checkpoints/sft_new_loop_phase4_pipeline/<时间戳>_rgb<2或4>` |
| `OUTPUT_DIR` / `RUN_ROOT` | 默认 `PIPELINE_ROOT/train`；两者同时给出必须一致 |
| `SKIP_BUILD=1` | 只复用已完成、源码／配置／内容匹配的双模式题库 |
| `SKIP_TRAIN=1` | 从显式 `RUN_ROOT` 或 `OUTPUT_DIR` 的 `best.json` 继续测试／打包 |
| `SKIP_EVAL=1` | 跳过测试，仍打包已有训练记录 |

跨 `&&` 共用的参数请先 `export`，不要仅给第一条命令设置：

```bash
export PYTHON=/path/to/training-env/bin/python
export MODEL_DIR=/path/to/Qwen3.5-4B
export GPU_IDS=0,1,2,3
export BUILD_WORKERS=24
bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh && \
SKIP_BUILD=1 HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
```

不要给两次实验指定相同 `OUTPUT_DIR`。默认会自动分开。若复跑测试，已有 `test` 或 `audit.zip` 产物仍遵守不覆盖约束。

## 只准备数据并检查，不训练

```bash
TRAIN_MODE=check bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh && \
SKIP_BUILD=1 TRAIN_MODE=check HISTORY_RGB_MODE=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
```

该模式不要求基座或 GPU；建库后验证实际采样。`TRAIN_MODE=preflight` 只做数据预检，`TRAIN_MODE=host-preflight` 额外检查模型／主机。三种检查均不会读取 `best.json`、测试或打包。

## 构建与重跑

构建持有目录锁防止两个进程同时写。同源码、同配置的再次运行可复用完成产物；中断后重跑 `SKIP_BUILD=0`，保留并核验已完成路线，未完成输出另存诊断。两套题库都通过严格加载和无丢题配对后才写 `pipeline_ready.json`。

源码或标注变更时需要新的 `DATA_DIR` 重新构建。v37 原产物保留，不能修改旧 manifest 哈希来迁就 v38 源码。外部生产产物必须同时提供 `CANDIDATE_POOL`、`PRODUCTION_INDEX`、`TEACHER_REGISTRY`，且合同匹配、逐帧记账完整；这属于高级用法，默认命令不需要这些参数。

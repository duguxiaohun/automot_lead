# Phase1 / Phase2 两图、四图独立判断题运行

从服务器 `AutoMoT/` 目录运行，先激活训练环境并用 `bash ./sync_automot.sh` 同步已发布源码。默认模型是 `checkpoints/Qwen3.5-4B`。下面四组都是原生 YES/NO 判断题，从基座分别训练；Phase1 与 Phase2 无 adapter 传递或成功依赖。两图取原四帧的 `[0,3]`，四图取 `[0,1,2,3]`。

## 数据准备

源码同步不携带本机数据和 ready。目标机器已有本阶段数据时复用并检查，缺失时独立 CPU 建库；不要在正式 GPU 命令里临时开启建库。每阶段准备一次，覆盖两个图模式：

```bash
HISTORY_RGB_MODES="2rgb_endpoints 4rgb" DATA_AUDIT_DIR=checkpoints/sft_new_loop_phase1_data_cpu/judgment_release_20261025 bash qwen3vl_local/sft_new_loop_phase1/prepare_data.sh
HISTORY_RGB_MODES="2rgb_endpoints 4rgb" DATA_AUDIT_DIR=checkpoints/sft_new_loop_phase2_data_cpu/judgment_release_20261025 bash qwen3vl_local/sft_new_loop_phase2/prepare_data.sh
```

这是两条独立命令；各自 CPU 准备成功后即可运行本阶段。阶段状态、完整日志、10 秒心跳与 7200 秒超时保留；失败先检查报告。既有同名审计目录不覆盖，重新检查时换 `DATA_AUDIT_DIR`。

## 四条独立训练、测试、打包命令

Phase1 两图：

```bash
RUN_BUILD=0 PIPELINE_TIMESTAMP=judgment_release_20261025 HISTORY_RGB_MODES=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

Phase1 四图：

```bash
RUN_BUILD=0 PIPELINE_TIMESTAMP=judgment_release_20261025 HISTORY_RGB_MODES=4rgb bash qwen3vl_local/sft_new_loop_phase1/run_full_pipeline.sh
```

Phase2 两图：

```bash
RUN_BUILD=0 PIPELINE_TIMESTAMP=judgment_release_20261025 HISTORY_RGB_MODES=2rgb_endpoints bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

Phase2 四图：

```bash
RUN_BUILD=0 PIPELINE_TIMESTAMP=judgment_release_20261025 HISTORY_RGB_MODES=4rgb bash qwen3vl_local/sft_new_loop_phase2/run_full_pipeline.sh
```

默认每组自动选四卡，batch=1、梯度累积=1、三轮：Phase1 配额 768，共 9216 次更新；Phase2 配额 1024，共 5376 次更新。两种图模式各是一组独立训练，步数不是两组之和。保持此前采样算法与比例；改变卡数、累积或配额需重新验算，不能套用这些步数。

同一组四卡一次启动一条命令；同时运行须显式分配互不重叠的 `GPU_IDS`。命令固定本次运行名，已有输出会拒绝覆盖；重复实验应换 `PIPELINE_TIMESTAMP`，不要删除旧证据或隐式续训。

默认流水线包含训练、base/LoRA production 与 audit 评测及审计打包；成功后日志打印 `audit bundle` 路径。交回 `.tar.gz` 审计包即可核对完整 cases、数据/源码/模型身份、更新数、曝光和指标；权重另行保存在训练目录，不放入审计压缩包。历史最优为 Phase1 v5 四图、Phase2 v3 首尾两图；当前训练合同不能直接继承历史成绩，是否可严格配对由交包证据判断。

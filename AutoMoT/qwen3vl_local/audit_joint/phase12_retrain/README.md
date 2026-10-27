# Phase1 / Phase2：独立新训练、评测和交包

用户于 2026-10-27 明确：修改标定或提示词后，必须用该版本重新训练，再评测新权重。
旧 adapter 换提示词只能作为辅助诊断，不代表新训练后的效果。

## 本版本实际改变什么

- Phase1：5 个逐帧确认的 **train** 帧修正（1 个 STATIC_OBSTACLE、4 个 R1→R5），
  原生八问中 RS2/RS4 定义分别加入 v2 的同向通道与可识别灯头说明。训练、验证、测试
  用同一个新提示词身份；两个说明合并用于本次新训练，尚无新权重精度结果。
- 不把上轮 test 标定候选写入本次数据：val/test 字节保持，原评分口径保持。
  不按场景或路线扩散5条修正，不强行让它们入选训练；标签改变可能影响原生分箱成员，
  采样算法、比例和重复上限保持。5条修正不等于全训练池已重新逐帧审计。
- Phase2：继续现有 `direct_event_visual_v5_highway_ue3` 标定和提示词，从 Qwen3.5 基座
  独立新训练。没有将 Phase1 的 RS 修正移植为 EVENT 规则，也不依赖 Phase1 权重或成功。
- 两图固定原 history 的首尾 `[0,3]`，四图 `[0,1,2,3]`，均为 YES/NO。
- 每模式四卡、累积1、三轮：Phase1 配额768 / 9216更新；Phase2 配额1024 / 5376更新。
  每阶段两个模式先后训练；两台服务器上的两个阶段完全独立。不能把两模式合计更新误作单模型步数。

## 一行式入口

在各服务器自己的 `AutoMoT/` 根目录激活训练环境，先同步已发布的白名单源码。
默认使用本机 `checkpoints/Qwen3.5-4B`、`lead_data` 和本阶段标准数据目录；
不假定另一台服务器的路径或 run 存在。

服务器一（Phase1）：

```bash
bash qwen3vl_local/audit_joint/phase12_retrain/run.sh all 1
```

服务器二（Phase2）：

```bash
bash qwen3vl_local/audit_joint/phase12_retrain/run.sh all 2
```

自动选4张空闲卡；明确选卡时前置 `GPU_IDS=0,1,2,3`。入口拒绝单卡套用四卡预算。
继承已激活 Python 环境，可显式前置 `PYTHON=/path/to/python`；不安装环境或下载模型。
本次命令固定训练/评测/预算参数，会清除旧 shell 中 RUN_TRAIN=0、PLAN_ONLY、旧 ADAPTER_DIR
等原生脚本参数，防止冒充重训。资源路径使用 CLI `--model-dir`、`--data-root`、
`--parent-data`、`--collection-dir`，不要用旧脚本环境变量重定向新版本。

`all` 每次创建独立 run，先模型资源检查，再 CPU 数据准备/真实RGB检查/三轮采样投影，
全部通过后才开始正式 GPU 训练。已有父数据复用；缺失时原生 CPU 建库，残缺目录拒绝覆盖。
每个 CPU 子阶段默认7200秒超时、10秒心跳；有错立即输出并非零退出，不隐式继续训练。
仅“暂时没有报错”不等于保证质量或保证不卡住，请看心跳和阶段状态。

本机可先仅 CPU 验证：`bash .../run.sh cpu 1 --run-dir <新的本机目录>`。
同一台服务器、同一份准入可 `bash .../run.sh train 1 --run-dir <该CPU目录>` 接续；
逐文件核验原数据/准入/源码，缺目录或已尝试过GPU训练均拒绝，不复用另一台机器的回执。
不能从本机 CPU ready 推断远程已准入。不同机器缺少部分审核帧会记录 absent_reviews；
至少实际匹配一条修正才能称 Phase1 修订数据，全部缺失则拒绝，内容/SHA不符也拒绝。

## 隔离与公平比较

运行创建独立内容寻址源码版本，只在新副本中改变 Phase1 提示词渲染和训练输出根。
原源码、旧版本、旧数据、旧 adapter、旧 latest 不改；新训练输出到
`checkpoints/sft_new_loop_phase{1,2}_retrain_runs/`。源码目录链接仍拒绝；
`Automot/checkpoints` 资产链接及本机RGB/权重目录按现有规则共享。

每次训练从同机 Qwen3.5 基座初始化独立 LoRA；两种模式不传递 adapter。
训练后沿原生 val 选模、重载新 adapter，做 test/base/LoRA/审计提示词评测并打包。
训练消费步数、来源与 adapter SHA 仍由原生完成校验检查。
`[DONE]` 仅指新训练、评测、打包完成，精度是否更好还要对审计包比较。

下载日志中全部 `[BUNDLE]` 文件：每模式的原生审计包（可能另有 balanced 包）
和小型 `training_revision_audit.tar.gz`。后者含5条训练修正、本机实际应用情况、
holdout散列、CPU准入和新源码版本绑定。权重保留服务器，不塞进审计包。

先核验新旧测试成员/问题/标签/RGB SHA，再报告整题、逐类 accuracy/precision/recall/F1、
格式、时延/显存和原正确损失。基座相同的“数据+提示词联合改动”不等于单因素实验。
已曝光的测试集只能作开发回归，最终泛化需独立未开发测试；本入口不偷偷重划测试集。

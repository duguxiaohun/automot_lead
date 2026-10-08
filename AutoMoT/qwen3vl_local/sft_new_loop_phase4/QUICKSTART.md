# 一行运行 Phase4

在 `AutoMoT` 目录执行：

```bash
bash qwen3vl_local/sft_new_loop_phase4/run.sh
```

无需填写数据集位置、输出目录、图数、采样预算或 `SKIP_BUILD`。入口会自动：

1. 查找兼容的 Phase3/Qwen3.5 训练环境。
2. 使用本机完整基座；若缺失，优先复用本地 Hugging Face 缓存，否则下载官方 Qwen3.5-4B 的固定版本。
3. 自动选择与当前源码和输入匹配的全量配对题库；没有就生成，中断后可复用完成的路线。同一命令重跑无需手改数据路径。
4. 依次运行四图、两图实验；每个实验包含训练、完整人工测试集评估和审计包。第二个实验自动复用题库，结果分目录保存。

默认七轮；按 Phase3 方式每事件每轮抽取 1024 次，十事件共 10,240 次。全量指题池来自全部合格路线，不要求每轮遍历全部题。事件内按容量分配、物理路线轮转；稀缺事件可能重复，日志报告实际重复量。原始驾驶数据沿用项目的 `lead_data`；不会把无标签帧生成假 NO。现有训练仍属于弱监督实验。

模型准备使用 [官方仓库 Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，固定 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。首次缺失时需要网络下载，已有完整模型无需网络；仅模型准备阶段联网，训练和评估仍离线，不使用远程代码或在线推理。下载不会覆盖已有不完整或不同版本的模型目录。

最初 v38 入口验收：15 项专项测试通过，实际只读配置检查自动选中了 `phase4_v38_full`。当时 v38 冻结源码合同及已有题库未变。后续 v39 修复了软链接路径读取，当前源码需新建匹配题库，入口会自动处理。测试覆盖无参数入口、环境探测、自动复用／新目录选择、缺模型准备、两实验分目录、第一阶段失败即停止。模型下载和训练编排使用模拟进程验证，本轮未下载权重或启动 GPU 训练。

当前默认采样见 [v40 说明](PHASE3_STYLE_SAMPLING_V40.md)。高级单实验入口和原环境变量仍保留；日常运行不需要设置参数。

`lead_data` 下的场景、路线或文件可以软链接到其他磁盘。旧版出现 `candidate route escapes data root` 时，更新到 v39 后重跑同一命令即可；无需移动原始数据或指定新输出目录。见 [软链接修复说明](SYMLINK_DATA_V39.md)。


## 自动结果压缩包

Phase4 默认开启打包，独立于 Phase3 的 `RUN_AUDITS` / `RUN_REGRESSION` / `PACK_RESULTS` 环境变量。`run.sh` 顺序完成四图、两图后分别生成：

- `checkpoints/sft_new_loop_phase4_pipeline/all_<运行ID>/rgb4/train/audit.zip`
- `checkpoints/sft_new_loop_phase4_pipeline/all_<运行ID>/rgb2/train/audit.zip`

单组 `run_full_pipeline.sh` 的包位于该组 `$OUTPUT_DIR/audit.zip`，默认是 `<pipeline目录>/train/audit.zip`。成功时日志打印 `[Phase4 audit] archive=<完整路径> bytes=<大小>`。测试失败时不会继续打包；检查模式不会打包。显式 `SKIP_EVAL=1` 仅打包已有训练记录，不能视为完成测试。

包中包含训练 JSON 记录、逐题结果及 `test/metrics.json`、`test/cases.jsonl`（若存在），不含权重或原始 RGB；不按行数截断结果。默认压缩包上限为 30,000,000 字节，超过会明确失败，不会静默删题。

四卡、梯度累积 8 时，每卡每轮 2560 次计算、全模型 320 次参数更新。旧的每轮全题重复策略已退出默认流水线。老师目前自动覆盖 UE1/UE4，其余八事件沿用人工题；这与全路线扫描是两个不同概念。

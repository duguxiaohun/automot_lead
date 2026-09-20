# Action 默认训练与中途审计

适用于 `action_prior`、`action_expert_ablation/qwen_simple` 和 `bev_only`。
三条入口共用实现；正常训练直接沿用原命令，无需再追加优化细节、验证或审计开关。

## 直接运行

以下命令在 `AutoMoT/` 下执行，每条普通命令与紧随的 GPU_IDS 命令是替代写法，只选一种。

```bash
bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
```

先验、采样、模型和数据路径仍按实际实验配置；本轮只精简新增的优化细节开关。

## 训练未结束时，把这个文件带走

**当前 run 目录下的 `training_audit.zip`**，不必等全部训练结束或最终 test/probe。
默认目录示例（若自定义 OUTPUT_DIR，以实际 run 路径为准）：

- `checkpoints/action_prior/latest/training_audit.zip`
- `checkpoints/action_expert_ablation/qwen_simple/latest/training_audit.zip`
- `checkpoints/action_expert_ablation/bev_only/latest/training_audit.zip`

自动更新时间：首个 optimizer update 保存后、既有定期 checkpoint 保存后、每个 epoch
训练结束保存后、每次完整验证结束后，以及 SIGTERM/SIGINT 安全保存后。
每个 epoch 的完整验证前先更新一次，验证结束后再更新一次。验证尚未完成的包明确标记
`validation_pending=true`，不会将历史 best 分数当作这一轮的成绩。

包内包含：

- `metrics.json`：已提交 step、epoch、是否完整、待办验证、当轮全 rank 训练统计、验证指标、
  best 分数、性能统计和各 epoch 的简要曲线。
- `config.json`、`training_plan.json`、存在时的先验/条件合同：实际训练条件、优化参数路由、
  LR/参考验证周期、总预算、模型/源码来源身份。
- `audit/`：各 epoch 详细训练/验证统计，以及最近200个日志窗口的 loss、LR、梯度范数、
  样本呈现数和代表参数更新幅度。
- `validation/`：最近200份已完成的验证记录；`AUDIT_MANIFEST.json` 记录实际收录、SHA256和大小裁剪。

ZIP 硬上限30,000,000字节，核心摘要/配置必须完整，详细历史超预算时明确列出遗漏。
不包含权重、缓存、RGB/视频、原始 TensorBoard、完整运行日志或本机聊天。
本包用于中途审计，不是训练恢复备份；恢复训练仍用 `latest.pt`。

ZIP 先写同目录临时文件，再原子替换。SIGKILL/掉电无法执行最后一次保存；仍可使用上一次
已发布的完整 ZIP，以包内 step 为准，不能声称它包含被强杀时尚未提交的数据。
打包失败会在终端警告并保留上一份ZIP，已保存checkpoint不受影响。包只在本机生成，不自动上传。

## 固定默认行为

- 隐藏矩阵使用 Muon；输入/最终输出、embedding/query、norm/bias 使用 AdamW。
- 两种优化器共用 embedding/query/norm/bias 不衰减的规则。
- Muon momentum=0.95、Newton–Schulz=5次，采用 `0.2*sqrt(max(rows,cols))` RMS尺度；
  基础 LR 默认2e-4，辅助 AdamW betas=(0.9,0.95)。参数、优化器状态和EMA为FP32，CUDA NS为BF16。
- 5% warmup 后，剩余更新预算按1:2:4:8划分 cosine 周期；每段降到0再回原峰值。
  短预算自动减少周期；EMA保持0.999，不随重启清空优化器状态。
- 每个 epoch 和共同参考周期末，用 EMA 对完整 val 验证并参与 best；重合只验证一次。
  即便显式选择单余弦基线，也按同样参考周期验证，保证同预算下四组合的选优机会相同。
  小验证只作诊断，最终不足一轮也完整验证；最后成绩单独写 `validation/final.json`。
- 首步、每100步和 LR 周期首尾，监控各非空优化组代表参数的真实更新 RMS/相对范数及算法耗时。
  不代表全部层的最大更新；CUDA计时会在监控步同步。

上述细节不再提供 CLI/环境开关；实际值记录在 `action_optimization_v4` 合同，恢复必须一致。
移除了之前新增的 decay 策略、周期验证、固定验证间隔、监控频率、周期数量/倍率和 Muon 内部调参开关。
旧命令若带这些选项，应删除后新开 run；旧 run 必须用原源码恢复，不隐式转换旧优化器状态。

只保留需要时的两个对照选择 `--optimizer`、`--lr-scheduler`（环境变量 OPTIMIZER/LR_SCHEDULER），
默认无需传。四组合为 AdamW/Muon＋单余弦/重启余弦；原来的 LR、epoch、batch 等基础训练参数保留。

## 性能与验证边界

`train/samples_per_second` 扣除验证、checkpoint与审计打包时间，仍包含数据等待、冻结条件模型、
decoder和其他循环开销；`train/overall_samples_per_second` 包含这些耗时。
`performance/session_*.json` 是本次进程会话的最终汇总，末尾验证/保存也计入；恢复后另开会话，
不把停机时间算作训练。中途ZIP的performance是发布前的当前会话快照，不包括正在生成该ZIP的时间。

已有 eval.sh 的 `--raw` 可对同一 checkpoint 做 raw/EMA 配对；使用相同val并分开输出目录。
CUDA小矩阵检查共用入口仍可独立运行，不加载Qwen/BEV，不读取数据：

```bash
python qwen3vl_local/action_prior/check_optimization_cuda.py --gpus 1
GPU_IDS=0 python qwen3vl_local/action_prior/check_optimization_cuda.py --gpus 1
python qwen3vl_local/action_prior/check_optimization_cuda.py --gpus 2
GPU_IDS=0,1 python qwen3vl_local/action_prior/check_optimization_cuda.py --gpus 2
```

真实条件模型训练、双卡NCCL和真实收敛/吞吐仍需远端验收；本机小矩阵和模拟训练回归不能替代。

本轮验证：两包完整CPU检查615项通过、7项CUDA跳过；22项失败仍为缺失只读runner（17项）
或peft（5项）。中途ZIP、验证中断、三入口计数恢复、打包失败保留旧包、历史过滤和开关精简
均有回归覆盖。RTX 4070 Ti SUPER上的6项单卡CUDA检查通过；双卡检查未执行。

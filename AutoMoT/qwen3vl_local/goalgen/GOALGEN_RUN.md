# GoalGen Runbook Index

GoalGen 运行手册按版本拆分：

| 版本 | 运行手册 |
|---|---|
| v1 | [`GOALGEN_V1.md`](GOALGEN_V1.md) |
| v2 | [`GOALGEN_V2.md`](GOALGEN_V2.md) |

最短命令索引：

```bash
# v1 数据
python qwen3vl_local/goalgen/build_dataset.py --mode v1 \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 0

# v1 训练
DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp

# v2 数据
python qwen3vl_local/goalgen/build_dataset.py --mode v2 \
  --keyframes lead_data/keyframes_all_scenarios.json \
  --data-root lead_data \
  --samples-per-scenario 0

# v2 训练
VERSION=v2 DDP_GPU_COUNT=4 bash qwen3vl_local/goalgen/train.sh ddp
```

详细 eval / probe / counterfactual 命令请看对应版本文档。尤其注意：

- v1 counterfactual 默认可覆盖完整状态机。
- v2 eval/probe 都会校验样本只来自 middle 子目标之间转换，误传 v1 val 会报错。
- v2 counterfactual 必须只围绕 middle 子目标之间转换，不能设计 init/final；
  `probe.py` 会拒绝 v2 下的 `--counterfactual-scope all`，并且 v2 的
  `--counterfactual-config default` 只生成 middle-only 候选。

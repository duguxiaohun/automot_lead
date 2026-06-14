# GoalGen Plan Index

GoalGen 的 v1 / v2 现在拆成独立版本文档，避免把两套数据分布和实验口径混在同一页里。

| 版本 | 文档 | 适用场景 |
|---|---|---|
| v1 | [`GOALGEN_V1.md`](GOALGEN_V1.md) | 全量 4 类 transition，含 `initial` / `final` 两端，从零训练 |
| v2 | [`GOALGEN_V2.md`](GOALGEN_V2.md) | 只保留 middle 子目标之间 2 类 transition，从 v1 warm start |

共享代码入口仍是：

```text
qwen3vl_local/goalgen/build_dataset.py
qwen3vl_local/goalgen/train.py
qwen3vl_local/goalgen/train.sh
qwen3vl_local/goalgen/eval.py
qwen3vl_local/goalgen/probe.py
```

关键边界：

- v1：`--mode v1` / `VERSION=v1`，数据在 `checkpoints/goalgen_v1_data`，产物在
  `checkpoints/goalgen_v1_dit`，counterfactual 默认可覆盖全状态机。
- v2：`--mode v2` / `VERSION=v2`，数据在 `checkpoints/goalgen_v2_data`，产物在
  `checkpoints/goalgen_v2_dit`，counterfactual 默认只允许
  `middle[0]→middle[1]` / `middle[1]→middle[2]`，不能设计 init/final；
  `eval.py` / `probe.py` 都会校验 v2 样本只来自 middle 子目标转换；
  `probe.py` 会拒绝 v2 下的 `--counterfactual-scope all`，且 v2 的内置
  `--counterfactual-config default` 只生成 middle-only 候选。

修改 GoalGen 版本边界、训练配方、eval/probe 命令或 counterfactual 口径时，优先改对应的
`GOALGEN_V1.md` / `GOALGEN_V2.md`；本索引只保留入口和边界摘要。

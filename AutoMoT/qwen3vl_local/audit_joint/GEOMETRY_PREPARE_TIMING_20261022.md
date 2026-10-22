# geometry 准备耗时观测

2026-10-22。审计没有发现新的阻断问题，本轮补充全池性能测量所需的运行记录，没有提前削弱校验或宣称加速。

新增独立 `geometry_prepare_profile.py`，由 `geometry_v1/remote.sh cpu` 默认调用。它在当前进程临时包裹原有函数进行计时，正常或异常退出都会恢复原函数；实际仍调用原有 `data_prepare.main`，参数解析、校验、生成、三轮计划不变。

报告追加到 `checkpoints/geometry_v1/prepare_timings.jsonl`。每次调用包含独立 run_id、状态、原错误、总耗时，以及：

- `geometry.prepare` / `geometry.plan`：数据准备及计划总时间。
- `candidates.enumerate`：实际消费候选路线生成器的时间。
- `dependencies.hash_files` / `outputs.hash_files`：输入绑定和缓存输出校验时间。
- `discovery.enumerate` / `discovery.validate_candidates`：旧库查找和资格验证时间。
- `native.build` / `native.validate`：原生建库与校验时间。
- `<family>.cache/dependencies/build/validate`：缓存阶段调用次数、累计时间和失败次数。
- `selections` / `cache_actions`：实际数据选择、build/reuse 结果；发现路径直接复用也会记录。
- `adapter_hashing`：缓存适配器成功散列的调用数和累计字节数，重复读取重复计数。不保存全池路径集合，避免为观测再持有一份巨大的文件名单。

各阶段耗时包含子调用，彼此有重叠，不能加总；文件计数不覆盖原生校验器内部所有散列，也不代表唯一文件数、物理磁盘读取量或端到端吞吐率。总时间不含计时入口之外的模型/地图前置检查与报告写入时间。失败报告保存的是已走到阶段的数据，不是完整运行耗时。

计时文件独立于数据、来源证明和 complete 回执。重复测量只追加记录，不污染缓存；非本工具格式文件拒绝追加。计时文件写入失败时保留原生失败异常，不用报告异常掩盖原错误。没有更改 `data_prepare.py`、`versioned_data.py`、原生 geometry Python 文件或 v90 发布清单，因此这次加入观测本身不会改变生成缓存身份、要求重标定。

验证包含实际合成 geometry 原生建库和第二次复用，逐字比较数据目录；正常无计时入口同样命中原库。另测试异常后函数恢复、CLI 参数转交、成功/失败记录追加、异种文件保护及远程入口接线。完整回归结果与 SHA 见 `geometry_prepare_timing_verification_20261022.json`。合成数据和地图替身不等于真实全池性能测试。

下一步在任选服务器同步本轮白名单后，照常运行 CPU 入口，回传 `prepare_timings.jsonl` 与日志。先看首次生成和再次复用的阶段占比，再决定优化哪一段。远程同步仍只用 `bash ./sync_automot.sh`，本轮尚未 push。

提示词收益仍需实际配对推理；本轮没有新预测分数、GPU、服务器或新增 RGB 语义审计，没有调整标定、mask、配额或训练预算。

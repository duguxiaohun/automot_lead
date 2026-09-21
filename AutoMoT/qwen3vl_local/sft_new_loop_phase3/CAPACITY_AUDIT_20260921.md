# v21 全量容量与训练前检查

本轮检查数据容量、划分、采样与下游消费，不新增 RGB 人工判读，不改变 v9 动作阈值、标签或提示词。产物写入 `/tmp/p3audit/capacity_v21/`，未覆盖现有 checkpoints、生产 split 或训练 run。

## 实际发现与修复

全量扫描 42 个 collection scenario 后，旧固定 seed + train-only 开发隔离导致 Phase3 val 缺 LEAD_BRAKE、ONCOMING_INVASION、JUNCTION_RULE_CONFLICT、SIGNAL_FAILURE；test 还缺 DYNAMIC_CUTIN。并非源数据没有这些类别：对应未曝光物理组分别仍有 27、46、48、44、53 个。只检查此前 242 条开发路线无法发现此问题。

`split_coverage.py` 现只从未曝光 train 物理组补充 holdout：同一物理组的所有 context、Rep 和重复采集一起移动，保护 train 各 context 非空，不移动现有 val/test 或开发路线。选择只依赖 context 容量、物理身份和固定 seed，不读取动作值、RGB 判读或模型分数。默认 `--min-holdout-context-frames 32` 对齐默认生成验证单桶预算，修改后的比例以实际 manifest 为准。`split_coverage.json` 保存调整前后容量、移动清单和未解决缺口；无可用来源时失败并保留候选，不能靠重复补足源容量。仅补非空会留下 test 某类 2 帧，该中间方案未作为最终默认。

最终移动 11 个未曝光物理组，开发集合仍为 1609 组 train-only。规则及新增 split 源码绑定 mapping hash，需要重建派生索引。seed 保持 20260920；已改变划分的新 run 不能与旧 split 混用。

Action 自动准备现将 Phase3 全候选池与 Action 自有 split 解耦：准备器使用 val/test ratio=0 生成候选容器，随后按 Action 三 split 构建 full map、执行其开发路线隔离。候选容器的 train 标记不把 Action val/test 移入 train；Action 不从 Phase3 均衡问答行取标签。此容器不能当作完整 Phase3 SFT 索引。自动缓存准备身份升级为 v2，显式旧文件仍严格校验。

Action 三条路径共享新增 token 支持检查：七类全部显示，区分呈现次数、不同帧、物理路线、UNCOND 原因、缺失动作及 eval 有而 train 没有的动作。开启 token 却没有任何有效训练动作时，训练计划提前失败；实际每轮采样后再次检查，包含 uniform 的 DDP 尾部截断。支持不够不补 KEEP，不强行伪造稀有动作。每轮 `sampling/epoch_*.json` 的 token 支持是整轮计划，截断训练的实际观测仍以训练审计为准。

显式 Action 预算不可行时，现在错误中给出 requested、repeat cap、各桶容量及独立容量上界；跨桶共享帧仍由联合分配器判断。自动预算=0 使用可行预算。均衡模式的 DDP 尾部统计修正为 0，不把采样增减量称为 DDP 丢弃。

## Phase3 全量结果

| 项目 | train | val | test |
| --- | ---: | ---: | ---: |
| 每 context 索引行数 | 965 | 34 | 32 |
| INVALID 索引行数 | 1930 | 68 | 64 |
| 索引合计 | 11580 | 408 | 384 |

全量索引 12372 行通过正式 `check_index`，物理路线不跨 split，开发路线无 holdout 泄漏。

用正式 `_balanced_work` / `_validation_work` 回放 choice/binary 各七轮，seed=`20260920 + epoch*1000003`，训练每桶 1024、验证每桶 16/32，检查 world size 1/4 的分片。

- choice：每轮 10240 次呈现、9650 个不同题，单题最多 2 次；val/test 的 16/32 单桶采样均无重复题。
- binary：每轮 12288 次呈现、11444–11454 个不同题，单题最多 4–5 次；16 单桶验证无重复，32 单桶验证含自动 INVALID 的少量重复（val 384 次/383 题，test 384 次/382 题）。人工同 RS 负例不靠重复凑数。
- 所有 context 配额满足，INVALID 覆盖与预算自动增容回归通过。上述呈现次数不是独立源帧或独立物理路线数，binary/choice 分母不可混用。

**仍存在的实际支持限制：** LEAD_BRAKE、ONCOMING_INVASION、JUNCTION_RULE_CONFLICT、SIGNAL_FAILURE 在最终采样的 val/test 各仅 1 个物理组。32 帧容量不证明跨路线泛化或统计稳定性；这些结果必须带物理路线支持数解释。缺少人工 same-RS 验证支持仍按既有规则标 `insufficient_support`，不恢复已撤销的人工负例开训门槛。

## Action 全量核验

原始输入发现 9582 条路线，排除 839 条异常时长路线；68912 个锚点缺少完整窗口。基础 Action 索引为 train 801543、val 92142、test 99804 帧。此数字是映射/开发隔离前的输入容量，不能替代有效标签数量。

实际 full map 共 1051417 帧。执行正式 `read_rows` 的映射与开发隔离后，train/val/test=843913/70058/79518；其中旧 val 的22084帧、旧 test 的20286帧移入 train。三个 split 七类 token 均有支持，无 eval 有而 train 无的动作。事件评估十桶及普通背景均非空；val/test 的事件计数包含 special_filtered，不能当作可用动作标签数。

| 动作 token | train | val | test |
| --- | ---: | ---: | ---: |
| UNCOND | 675236 | 58929 | 66226 |
| DECELERATE | 24665 | 1620 | 2020 |
| STOP | 84936 | 5463 | 6878 |
| RESUME | 15231 | 925 | 1429 |
| LEFT | 11072 | 812 | 813 |
| RIGHT | 10436 | 809 | 687 |
| KEEP | 22337 | 1500 | 1465 |

uniform 与 event_balanced 各回放7轮、world=1/4，共28个整轮计划：

- uniform 每轮843913帧（1 rank）/843912帧（4 rank，丢1个尾帧），均无重复，六种有效动作每轮都有支持。
- event_balanced 每轮95136次呈现，十个事件各7928、背景15856；65563个不同帧，全局单帧最多8次，满足既有repeat cap。限制桶为 UE3 的991个可用训练帧，991×8=7928；不能把7928当作7928个不同标签。
- uniform 中 UNCOND 较多是显式记录的普通背景503569、过滤51790、未确认119877之和；没有由缺 eligible 候选静默补成 KEEP 或 UNCOND。

可复算结果见 [CAPACITY_REPLAY_20260921.json](CAPACITY_REPLAY_20260921.json)；完整机器报告及脚本位于 `/tmp/p3audit/capacity_v21/`。591项Phase3与136项Action/消融测试分别通过；两个测试套件合并进同一pytest进程曾发生CPU/CUDA测试环境交互，最终按独立进程核验。以上均不加载真实模型、不启动NCCL。

## 可复算入口与边界

Phase3：在完成新版构建后，使用原 `train.py --sampling-only`，分别传 `--action-output-mode choice` / `binary` 与实际 `--focus-balance-count`。本轮七轮脚本位于 `/tmp/p3audit/capacity_v21/replay_phase3.py`。

三条 Action 入口共享纯数据检查器，不加载 Qwen/BEV、不启动 NCCL、不写训练目录。先让 pipeline 准备当前 full map，再将 `EVENT_BALANCE_INDEX` 设为训练配置中记录的实际文件路径：

```bash
python qwen3vl_local/action_prior/audit_data_capacity.py \
  --data-root lead_data --data-dir checkpoints/action_prior_data \
  --event-balance-index "$EVENT_BALANCE_INDEX" \
  --high-level-action-token --event-balanced --world-sizes 1 4 --num-epochs 7 \
  --report checkpoints/action_capacity.json
```

uniform 条件使用 `--no-event-balanced`；其余输入、seed、重复上限、预算应与真实训练一致。该命令只读现有索引；不是模型文件检查、RGB 可见性审计、GPU/DDP 实跑或训练效果证明。更新源码后需重建新 mapping 的派生产物，旧 run 必须使用原源码恢复。

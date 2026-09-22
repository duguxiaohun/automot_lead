# Action Expert 消融实验

## 2026-09-23 与主线同步的RGB输入和支持量规则

qwen_simple/bev_only与action_prior共用anchor≥4过滤、物理路线开发隔离、独立Action holdout支持补齐，
以及主要动作容量回流；目标每事件32帧、5物理组，迁移/缺额写入训练计划。
单图/四图/BEV-only使用同一有效帧范围。合法稀少动作保留标签，组合不另设等量桶。
action-balanced默认单帧上限2，event-balanced仍8；采样不自动打开token，也不向消融注入Phase3问答。
新条件必须新full map、新run；旧run使用原源码。
详见 [v23 RGB与容量审计](../sft_new_loop_phase3/V23_RGB_SUPPORT_20260923.md)。

## 2026-09-22 action-balanced：事件内动作均衡

新增 `--action-balanced` / `--sampling-mode action_balanced` / `ACTION_BALANCED=1`，主线和两个消融共用。
默认使用 `event_balanced`，省略采样参数等价于 `--event-balanced`；加入 `--action-balanced` 即切换为动作均衡。
新训练只保留这两种采样，不再提供 uniform/随机采样。多个正向 CLI 模式按最后出现的值选择。
环境变量先解析、CLI 后覆盖；`ACTION_BALANCED=1` 选择动作均衡，`ACTION_BALANCED=0` 选择事件均衡；
`EVENT_BALANCED=1` 仍可用，同时设置两个环境开关时 ACTION_BALANCED 优先。
旧 `--no-event-balanced`、`--no-action-balanced`、`EVENT_BALANCED=0`、`--sampling-mode uniform` 均报错，
需要切回事件均衡时使用 `--event-balanced`。默认 event 也自动准备/复用 full map；不会自动开启动作 token 或文字动作先验。
续训恢复保存的模式，不注入新默认；旧 uniform（含未记录模式的历史 run）必须使用原源码，不能静默改采样续训。
验证/测试仍遍历原有效 split，不执行训练均衡采样。
本轮525项相关CPU回归通过；7项因本机缺只读 runner 源码未执行，未绕过生产校验；未验证真实GPU训练。

新模式分两层：

- UE1–UE7、RE2/RE3/RE5 各抽同样次数；按用户选择，确认普通背景继续占两份，比例 `1:…:1:2`。
- 每个特殊事件内，动作尽量均衡，但稀少格子达到实际容量后将缺额回流给同事件其它动作。
  五变化动作和 KEEP 保持原标签；例如 KEEP 只有4帧、STOP有1000帧，预算100时抽4/96，而非重复成50/50。
- 动作使用共享 Phase3 candidate/full map 投影的**逐帧主要 token 标签**。并发事件帧仍使用同一个主要动作，
  只进入支持该动作的事件域：RE2+RE5 的右变道帧只计入 RE2×右变道，保留两事件事实及全局 token。
  普通背景仅 confirmed_regular；filtered/unconfirmed/缺映射帧不参与均衡池。
- 无样本支持的动作格子报告为缺失，不制造 KEEP、不补 UNCOND、不为不可能动作凑数；缺整个事件/背景池则开训前失败。

特殊事件之间仍严格等量，事件内改为容量约束下的近似动作均衡，**不保证各动作次数相等**。
整个事件池不足预算时才循环完整池，再对余量做容量内均衡；不按任意帧数阈值删除或重标稀少动作。
也不保证每个 micro-batch、单个 rank 或 max_train_steps 截断前缀均衡；DDP 从共同完整计划分片，不截断整轮配额。

`--action-balanced` 只改变采样，不会自动开启 token、文字动作先验或 Qwen 场景提示。
即使 token 关闭，也会读取动作标签用于采样与审计，但不会送入 decoder；可做相同采样下有/无 token 对照。
开启 `--high-level-action-token` 时继续默认使用已加入的弱分离正则。

复用 `--event-balanced-epoch-samples`（默认0自动）、`--event-balance-max-frame-repeats`（action-balanced新训练默认2，event-balanced仍为8）、
`--event-balance-route-diverse`。预算0根据事件池及跨事件共享帧的联合容量求最大可行预算，小动作格子不再限制整轮；
显式预算须是 `lcm(12, world_size)` 的倍数，并满足全局重复上限。
最大重复次数按同一帧在**整个 epoch 所有事件/动作桶的总出现次数**限制；先最少偏离容量内动作目标，再最大化不同帧覆盖及路线轮转。
共享帧冲突允许同事件回流，目标/实际配额和超目标次数均记录。公平比较应使用共同可行的明确呈现预算/更新数、seed、split 和 full map，
不能只对齐轮数。验证/测试不重采样，默认仍按自然分布 ADE 选 best；显式 event_balanced_ade 仍是原事件加权指标。

**完整训练池回放：** 已核验的 v21 标签池在新采样器下，1/4 rank 各7轮通过；新默认上限2时
每轮 **23784** 次（各特殊事件1982、背景3964），21251个不同帧，最大重复2次。
显式上限8的旧对照仍为95136次/轮；不同自动预算不可只按相同轮数比较。
移除4条 `RE5×右变道` 的采样归属，没有删除这4帧或改成 KEEP/UNCOND；原1440结论已被本修订替代。
这是历史候选/训练池的数值回放，未全量重建 v22 生产产物或训练模型。
新源码须重建 Phase3 candidate/full map 并新开 run；Action 自动准备按新 hash 建新缓存，显式旧映射会拒绝。
详见 [全量容量回放](../action_prior/ACTION_BALANCED_20260922.md)。

默认2是较保守的实验起点，不是已验证的最佳超参数。显式 `--event-balance-max-frame-repeats 1` 可禁用整轮重复，
显式8可复现原容量方案；环境变量 `EVENT_BALANCE_MAX_FRAME_REPEATS` 同样可用，CLI优先。
恢复训练使用保存值，旧配置缺字段按历史8解释；严格源码合同仍要求旧run使用原源码。
上限降低也减少每轮优化器更新数，warmup/验证时刻随计划变化；不会自动延长轮数补回旧预算。

稀少组合不自动改为KEEP或UNCOND：KEEP代表真实保持阶段，UNCOND代表不提供动作条件；数量少不证明标错。
七个动作向量跨事件共享，UE3×RESUME的41帧并非RESUME全部训练支持（全训练池15231帧）。
低于100帧或10条物理路线只标记复查线索，**不参与过滤、重标或配额计算**，也不依据val/test调整训练标签。
如后续对不可靠标注做UNCOND消融，应另存原动作及mask理由，不改变事件/轨迹或归入普通背景；当前没有启用此屏蔽。

`training_plan.json` 的 `sampling.action_balance` 保存支持/缺失动作、可行配额倍数、动作格子的独立帧和物理路线支持；
`sampling/epoch_*.json` 保存实际事件/动作次数、全局重复、不同帧与路线覆盖。
计划及逐轮报告的 `support.cells/events` 还记录可用帧、物理路线、呈现次数/可用帧比值和复查标记；该比值不是单帧实际最大重复数。
训练日志新增 `group/action_balance/<事件>/<动作>/...`，token 关闭也可审计该课程。
两层规则、标签内容身份和采样预算绑定恢复合同；只改采样方式也应新开 run，旧 run 仍用原源码。
闭环只将采样作为训练溯源；开启 oracle token 的模型仍因缺在线动作 provider 而拒绝闭环。

```bash
# AutoMoT/ 下，两个消融均可自动准备 full map/candidate
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced --high-level-action-token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --action-balanced --high-level-action-token --rgb-frame-count 1
# 环境等价；CLI 模式选项优先
ACTION_BALANCED=1 HIGH_LEVEL_ACTION_TOKEN=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
```

## 2026-09-22 动作 token 弱分离正则

开启 `--high-level-action-token` 的新训练默认增加弱分离：`--action-token-separation-weight 0.01`、
`--action-token-separation-margin 0.5`。仍为七个可学习的 1024 维向量，KEEP 不细分，UNCOND 也参与。
对完整 embedding 表的 21 对不同类别计算 `mean(relu(cos(e_i,e_j)-margin)^2)`，
总训练 loss 为 `FM loss + weight * separation loss`。只惩罚过高相似度，不要求动作全部正交；
归一化只用于正则分支，BEV concat 与注意力输入不变。该设计参考防坍塌思想，**不是 LeJEPA/SIGReg 的复现**。

正则使用 FP32（包括 BF16 autocast 内），所有类别等权，每个 micro-step 与 FM loss 一起除以实际累积窗口长度，
DDP 不额外乘 world size。标签/Phase3 规则、采样和 Qwen prompt 不变。
验证仍计算原始 FM MSE 和轨迹指标，best 仍按原 ADE 策略选取，不把正则加入验证分数。
权重设 `0` 可做无正则对照；token 关闭时正则不执行。CLI 优先于环境变量
`ACTION_TOKEN_SEPARATION_WEIGHT` / `ACTION_TOKEN_SEPARATION_MARGIN`。

TensorBoard 和 `training_audit/windows` 记录 `fm_loss`、`action_token_separation_loss`（未加权）、
`action_token_separation_weighted`（加权），`loss` 是总训练目标。
`action_token/cosine/<类别>__<类别>` 共 21 对、`cosine_max/mean` 和每类 `norm` 在首步、日志窗口、轮末与最终步记录；
`action_token_initial.json` 保存本次进程起点（恢复时为恢复步）的范数/相似度。
审计 ZIP 包含近期窗口。权重为 0 时两个 separation loss 日志为 0，相似度仍监控。

这是软约束，不保证 decoder 一定利用 token，也不保证轨迹指标提升；相似度不能代替正确/替换 token 的固定噪声轨迹对照。
默认系数是实验起点，尚未做真实模型/GPU效果验证。算法版本、权重和 margin 写入条件合同与训练计划，
更改需新 run；旧 run 使用原源码，不能跨源码直接续训。缺这些字段的历史配置按权重 0 解释，仍保留严格源码检查。

```bash
# 在 AutoMoT/ 下执行；两个消融入口与主线共用正则
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --high-level-action-token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --high-level-action-token --rgb-frame-count 1
# 独立新 run 的无正则对照
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --high-level-action-token --action-token-separation-weight 0
# 调整系数；也可通过同名 CLI 覆盖。固定四卡可前置 GPU_IDS=0,1,2,3
ACTION_TOKEN_SEPARATION_WEIGHT=0.02 ACTION_TOKEN_SEPARATION_MARGIN=0.5 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --high-level-action-token
```

## 2026-09-21 后续：共享容量检查

两条消融同步使用主线新的候选准备与 token 支持检查：候选容器不依赖 SFT 的 val/test 桶；训练全为 UNCOND 时拒绝，每轮实际采样后再检查并报告七类支持。简短 Qwen / 无 Qwen 条件不变，不强制补齐不存在的动作。真实全量核验与纯数据审计命令见 [容量审计](../sft_new_loop_phase3/CAPACITY_AUDIT_20260921.md)。

## 2026-09-21 Phase3 v21 同步范围

两条消融复用主线的 Phase3 candidate/full map、主要动作投影、动作 token 和开发路线隔离，无单独的旧速度标定副本。启用动作 token 时，确认起步修订来自 v21；开关仍默认关闭。事件均衡仅改变采样，不能据此声称模型已获得动作标签。

`qwen_simple` 保留 LeadMoT 简短导航提示词，`bev_only` 没有 Qwen 提示词；主线自然先验的 UE1/信号故障文案更新不注入这两个消融。这样保留各组原有条件定义。比较时使用同一新 full map、有效 split、seed 和预算；event/action 两种模式均使用 full map 同步开发路线隔离。

新来源合同需要新产物、新 run，已有生产映射/缓存/checkpoint 尚未全量更新。实际 7900 行 v21 候选的共享消费者回放和 14 项消融入口测试通过；完整范围、164 项测试及复用方式见 [主线 v21 同步说明](../action_prior/run.md)。

## 2026-09-21 动作 token / 单当前图 demo

两条消融与主线共用 `--high-level-action-token`（默认关闭）和 `--rgb-frame-count 1|4`（默认4）；也支持 `HIGH_LEVEL_ACTION_TOKEN=1/0`、`RGB_FRAME_COUNT=1/4`，CLI 优先。

动作 ID 经可学习 `Embedding(7,1024)`，在 120 个 BEV token 后追加一个 token；七类为五个变化动作、KEEP、UNCOND，KEEP 不细分。`bev_only` 在本地序列做 self-attention；`qwen_simple` 同时读取冻结 Qwen KV。两者均不因此向 Qwen prompt 添加动作/RS/EVENT 文字。Phase3 标注来源、投影、准备、审计、训练和恢复走共享实现，不复制训练循环。

```bash
# BEV 基线与动作 token 实验分别新开 run
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --no-high-level-action-token
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --high-level-action-token
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --high-level-action-token
# Qwen 四图＋动作 token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token
# Qwen 仅当前一图＋动作 token
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --high-level-action-token --rgb-frame-count 1
# 环境变量等价写法；恢复时未显式设置则沿用原配置
HIGH_LEVEL_ACTION_TOKEN=1 RGB_FRAME_COUNT=1 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
# 单图、不带动作 token 的对照
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --no-high-level-action-token --rgb-frame-count 1
```

`train.sh` 同样支持这些选项；token 模式首次自动准备 Phase3 candidate/full map。当前图始终为 anchor 的完整三视角拼接 RGB；不改变 BEV 单帧 RGB＋LiDAR，所以 `bev_only` 的图数选项不改变模型有效条件。请共用 `DATA_DIR=checkpoints/action_prior_data`、full map、seed、卡数、采样及预算比较，且检查训练计划中的有效 split/七类覆盖。开/关 token 的对照应使用同一 `--event-balance-index`；该路径参数不切换 event/action 模式或开启文字先验。

普通 RE/过滤或未确认帧为 UNCOND；证据完整的保持为 KEEP。缺 eligible 标签、坏哈希、冲突不降级为 UNCOND。逐例审计保留来源原因，验证提供 `group/action_token/*`。动作来自未来轨迹标注，属于离线 oracle 条件增益实验，不能当 Phase3 预测或闭环成绩。开关和图数绑定 checkpoint；eval 自动恢复，不能临时切换；新条件新训、旧 run 用原源码。

主线单图 LoRA 输入变化、来源搬迁和完整合同说明见 [action_prior/run.md](../action_prior/run.md)。


优化细节已统一为默认值，无需追加新开关。每个 epoch 训练结束及完整验证后，自动更新当前 run 的 **`training_audit.zip`**；训练被中途终止时可直接带走此文件审计。包内有进度、各轮训练/验证指标、loss/LR/更新幅度和实际配置，详见 [默认训练与中途审计](../action_prior/OPTIMIZATION.md)。

新训练与主线同步默认 `muon_adamw + cosine_restarts`，共7轮；warmup只用首轮更新数的5%，周期依次占1/2/4轮，warmup包含在首周期内。所有优化器、周期、warmup 参数直接复用主线，无消融副本；四组合对照、参数与恢复边界见 [共享优化说明](../action_prior/OPTIMIZATION.md)。

从远端 `AutoMoT/` 目录运行。本目录只做两个不使用 RS/EVENT 标定或 LoRA 先验的 action expert 对照，
轨迹 decoder、Flow Matching、BEV、数据索引和训练循环直接复用
`qwen3vl_local/action_prior/`，不再分别维护主线和消融的训练循环。

## 两个实验

| 子目录 | 条件输入 | 目的 |
| --- | --- | --- |
| `qwen_simple/` | 4 张 stitched RGB + LeadMoT 原简短导航 prompt 的 base Qwen KV + frozen BEV | 对照“没有推理/先验摘要”的 Qwen 视觉语言 encoder |
| `bev_only/` | frozen LEAD BEV(RGB+LiDAR) + speed/target/next target/final goal + route/waypoint query；Qwen prefix KV 长度为 0 | 对照“完全没有 Qwen 图文 KV”的 action expert |

两者都不读取 Phase1/Phase2 adapter 或 `prior_labels.jsonl`。开启 `--event-balanced` 后只读取共享 full map
用于采样与真实事件分桶评测，不把这些标签送入 Qwen/BEV/decoder，也不生成先验复核指标。
`bev_only` 仍沿用 action_prior 的 LEAD BEV encoder 输入与导航状态；这个 BEV backbone
仍融合当前 stitched RGB 与 LiDAR BEV，所以它不是纯 LiDAR/完全无视觉实验。区别是 Qwen
engine 不初始化，decoder 每层拿到的是 zero-length prefix KV。

## 共享代码与实验边界

主线 `action_prior/train.py` 和两个消融入口共同调用
[`action_prior/training_core.py`](../action_prior/training_core.py)：

- 模型与 FM 配置构造、FP32 Muon/AdamW 状态、共享学习率调度、EMA。
- 每轮样本打乱、DDP 分片、梯度累积及不足完整窗口的更新。
- FM loss、固定噪声验证、Euler 采样指标、频繁验证及 epoch/final 验证选优。
- TensorBoard 核心日志、checkpoint 保存、恢复配置校验、待完成验证和预算停止。

`common.py` 只保留消融的输入构造、配置/条件合同、简化审计与入口组装；
主线通过审计接口继续记录先验指标；消融只在启用均衡时追加共享 full-map 事件桶，不产生先验 case 审计。
后续修改上述训练行为应修改 `training_core.py`，三条入口会共同使用新实现。
先验生成、Qwen simple prompt 和无 Qwen 输入仍分别由各自 runtime 管理；修改这些条件分支
不会自动改变其他实验的定义。

`action_prior` 与 `qwen_simple` 的对比同时移除了先验、分析摘要并更换 prompt，
因此衡量的是整条先验推理链的收益，不能单独归因为“生成推理文字”的收益。
`qwen_simple` 与 `bev_only` 衡量 Qwen 图文分支的收益；BEV 的 RGB 融合分支仍然存在。

## 数据索引

默认复用主线索引：

```bash
python qwen3vl_local/action_expert_ablation/build_dataset.py \
  --data-root lead_data \
  --output-dir checkpoints/action_prior_data
```

该 wrapper 直接调用 `action_prior/build_dataset.py`，所以异常时长 route 剔除、4Hz anchor、
物理 route 分 split、route10/waypoint8 监督都与主线一致。若目录已存在不会覆盖。
两个消融的 full pipeline 默认共享该目录；主线指定同目录时也使用同一份共享构建函数。
首次启动时会用 `.build.lock` 文件配合 `flock` 做进程级构建锁，拿到锁后再次检查
`manifest.json` 和 `train/val/test` 三个 split，避免并发实验
同时写同名临时文件。锁文件残留不会阻塞后续启动，进程退出会自动释放文件锁。

## 快速运行

```bash
# Qwen + simple prompt
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh

# BEV-only
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh

# 只训练
bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh
bash qwen3vl_local/action_expert_ablation/bev_only/train.sh

# smoke：默认 4 个 optimizer update
bash qwen3vl_local/action_expert_ablation/qwen_simple/smoke.sh
bash qwen3vl_local/action_expert_ablation/bev_only/smoke.sh

# 独立 eval
bash qwen3vl_local/action_expert_ablation/qwen_simple/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/qwen_simple/latest/best.pt
bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt
```

常用环境变量与 action_prior 对齐：`GPU_IDS`、`DDP_GPU_COUNT`、`DATA_ROOT`、`DATA_DIR`、
`LEAD_BEV_CKPT`、`NUM_EPOCHS`、`LR`、`GRAD_ACCUM`、`VAL_STEPS`、
`SAVE_STEPS`、`LOGGING_STEPS`、`NUM_WORKERS`、`TRAIN_SAMPLED_METRICS`。
`MODEL_DIR` 只对 `qwen_simple` 有效；`bev_only` 不初始化 Qwen。
`OUTPUT_DIR`/`RESUME` 可用环境变量或 CLI `--output-dir`/`--resume` 传入，launcher 会统一
创建/复用真实 run 目录，训练日志和 checkpoint 写在同一个目录。
`run_full_pipeline.sh` 会在训练前固定本次 `RUN_TAG` 和实际 run 路径，最终 eval 直接使用
该路径下的 `best.pt`，不会再通过可能被并发实验改写的 `latest` symlink 找权重。
CLI `--data-root` / `--data-dir` 会同时用于索引构建、训练和最终 eval。
若 `--resume` 指向 `latest/latest.pt` 这类软链接，pipeline 会先解析为真实 checkpoint 路径，
训练与最终 eval 都使用解析后的同一个 run。显式传入的 `MODEL_DIR` / `--model-dir` 和
`LEAD_BEV_CKPT` / `--lead-bev-ckpt` 也会传给最终 eval，再由权重哈希验证同一性。
仅传 `--resume` 时，训练入口会先读取 checkpoint 所在 run 的 `config.json` 恢复原始
LR、epoch、梯度累积、数据索引等非默认参数；launcher 在选 GPU 前读取
`training_plan.json` 恢复原始 `world_size` 默认值。命令行显式传入的参数仍优先生效，
用于数据/Qwen/BEV 路径搬迁等场景。`train.sh --resume` 不会再自动注入脚本默认 LR、
epoch、梯度累积或默认索引；只有用户实际传入的 CLI 参数或环境变量会作为覆盖传给 Python。

## 与主线完全共用 event-balanced

三个 full pipeline / train.sh 共用 `action_prior/event_balance_common.sh` 解析开关和环境变量，
自动准备调用 `action_prior/prepare_event_balance.py`。不传开关默认 `event_balanced`；
`--event-balanced` 与 `EVENT_BALANCED=1`、`--sampling-mode event_balanced` 等价，
`--action-balanced` 切换动作均衡。CLI 优先于环境变量，多个正向 CLI 开关按最后一次取值。
两个消融无需传 `--dataset-priors`，也不会构建 Phase1 标定先验索引或加载 Phase1/2 模型。

```bash
# 三组固定同一个 action split 索引；full map 缺省时自动构建/按内容缓存复用。
DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
GPU_IDS=0,1,2,3 DATA_DIR=checkpoints/action_prior_data bash qwen3vl_local/action_prior/run_full_pipeline.sh --dataset-priors --event-balanced
bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced

# 环境变量写法；CLI 可切换另一种模式。
EVENT_BALANCED=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
GPU_IDS=0,1,2,3 EVENT_BALANCED=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh
EVENT_BALANCED=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced
GPU_IDS=0,1,2,3 EVENT_BALANCED=1 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --action-balanced
```

默认 UE1–UE7、RE2、RE3、RE5 十个 special 桶各权重 1，`REGULAR_BACKGROUND` 权重 2。
只有确认常规帧进入背景；未确认和被过滤的特殊帧不会伪装成常规。全局配额、跨桶共享帧去重、
最小费用流最大化唯一帧数、每帧重复硬上限和 DDP 分片都直接使用主线实现。
权重表使用正整数；自动 epoch 预算也按该表逐桶计算容量，不再写死背景除以 2。
`event_balance_route_diverse` 是组内物理路线轮转偏好，不是路线硬配额。

| 调整项 | 唯一实现/用法 |
| --- | --- |
| 桶、比例、重复和分配算法 | `action_prior/event_balance.py`：`SPECIAL_BUCKETS`、`EVENT_BALANCE_WEIGHTS`、`build_event_balanced_epoch` |
| 标注映射、候选/缓存准备 | `action_prior/build_event_balance_index.py`、`prepare_event_balance.py` |
| 训练默认值、数据读取、split 隔离、预算 | `action_prior/config.py`，两个消融直接引用 |
| 梯度更新、验证、选优 | `action_prior/training_core.py` |
| 事件桶与 ADE/FDE 聚合 | `action_prior/metrics.py`，三组直接引用 |
| epoch 呈现数 | `EVENT_BALANCED_EPOCH_SAMPLES` / `--event-balanced-epoch-samples`；默认 0，自动选择可行预算 |
| 每帧总重复上限 | `EVENT_BALANCE_MAX_FRAME_REPEATS` / `--event-balance-max-frame-repeats`；event-balanced默认8，action-balanced默认2 |
| 组内路线轮转 | `EVENT_BALANCE_ROUTE_DIVERSE=0/1` / `--no-event-balance-route-diverse` / `--event-balance-route-diverse` |
| best 选优 | `BEST_SELECTION_METRIC` / `--best-selection-metric`；默认 `natural_ade`，可选 `event_balanced_ade` |

采样配置与 full map 内容身份写入 checkpoint 合同。比例/逻辑更改后用同一版本重新训练三组；
不能给既有 checkpoint 临时换课程或用新源码强行续训。共享的是采样与训练行为：主线 planning
自然先验、分析 prompt 仍只维护在 `action_prior/prompts.py`，两个消融没有副本，也不消费它们。
`qwen_simple` 的简短导航 prompt 继续由 LeadMoT 提供，`bev_only` 无 Qwen prompt。
`--event-balanced-scene-priors`（包括对应环境变量）在消融入口明确拒绝，避免破坏无先验定义。

```bash
# 已有索引后只训练；将路径替换为 pipeline 打印的 [event balance index]。
EVENT_BALANCE_INDEX=checkpoints/shared_event_map/full_event_mapping.jsonl \
  bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh --event-balanced \
  --event-balance-max-frame-repeats 4 --best-selection-metric event_balanced_ade
GPU_IDS=0,1,2,3 EVENT_BALANCE_INDEX=checkpoints/shared_event_map/full_event_mapping.jsonl \
  bash qwen3vl_local/action_expert_ablation/qwen_simple/train.sh --event-balanced \
  --event-balance-max-frame-repeats 4 --best-selection-metric event_balanced_ade

# 同版本续训自动恢复课程/预算；标签搬迁只覆盖路径，同时传给最终 test。
bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh \
  --resume checkpoints/action_expert_ablation/bev_only/latest/latest.pt \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
GPU_IDS=0,1,2,3 bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh \
  --resume checkpoints/action_expert_ablation/bev_only/latest/latest.pt \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl

# 独立评测沿用 checkpoint 课程；搬迁时需连同 manifest.json 一起移动。
bash qwen3vl_local/action_expert_ablation/bev_only/eval.sh \
  --checkpoint checkpoints/action_expert_ablation/bev_only/latest/best.pt \
  --event-balance-index checkpoints/moved_event_map/full_event_mapping.jsonl
```

`train.sh` 需已有 Action split 索引，Python 会在模型预检前自动准备 full map；从零开始建议用 full pipeline。显式路径缺失会报错，不会替换为默认文件。
续训不自动准备新的 full map；源内容与合同不符会拒绝。三组还应核对相同 seed、world size、累积数、
epoch 呈现预算、重复上限、best 指标和 `training_plan.json` 中的 sampling source/quotas。
共享 full map 按同一规则将 Phase3 开发物理路线限制在 train；val/test 保持自然分布遍历，不进行均衡重采样。
开发名单直接复用 Phase3 当前构建器（2026-09-14 共 709 组），不在 action 维护日期列表。
验证预检与实际指标均按全部真实语义桶统计，包含 `special_filtered`；训练仍只采 eligibility 通过的帧。
验证覆盖全部桶时额外提供均衡加权 ADE/FDE；选择 `event_balanced_ade`
但 val 缺桶会预检失败，不能悄悄退回总体 ADE。

均衡模式追加 `sampling/epoch_*.json` 的配额、唯一帧、重复次数审计，以及 TensorBoard/metrics 中的
`group/event_balance/*`、桶覆盖和 `event_balanced_*`；默认训练不做 Euler 采样时只记 FM MSE
与覆盖，ADE/FDE 在验证中生成。两种训练模式都记录共享事件指标。

## SIGTERM / SIGINT 安全停止

训练进程捕获第一次 `SIGTERM` / `SIGINT` 时只记录请求。所有 DDP rank 在当前梯度累积窗口
完成后的 optimizer 安全点同步请求，再共同原子保存 `latest.pt`；如果信号在 validation 中到达，
则丢弃未完成的 validation 指标并保存验证前 cursor。rank0 同时写 `termination.json`，其中包含
signal、optimizer step、cursor 与 checkpoint 路径。保存完成后进程以 `128 + signal`
（`SIGTERM=143`、`SIGINT=130`）退出，因此 `run_full_pipeline.sh` 不会误跑后续 eval。

launcher 在多卡时优先把外部信号转发给 torchrun workers，让 torchrun 等待安全保存；单卡时直接
通知训练进程。退出时会显式关闭已登记的 DataLoader iterator；短生命周期 validation loader
禁用 persistent workers，减少强制退出后的 semaphore 清理告警。`SIGKILL`、节点掉电、GPU/进程
永久卡死无法由 Python 捕获，仍只能退回最近一次周期 checkpoint。

恢复使用终止标记记录的 checkpoint：

```bash
bash qwen3vl_local/action_expert_ablation/bev_only/train.sh \
  --resume checkpoints/action_expert_ablation/bev_only/latest/latest.pt
```

resume 开始时会把旧 `termination.json` 归档到 `termination_history/`。本机制修改了执行源码指纹；
改动前生成的 checkpoint 仍须使用对应旧代码恢复，不能绕过严格合同。

## TensorBoard 对比

```bash
bash qwen3vl_local/tb_serve.sh checkpoints/action_expert_ablation/qwen_simple/latest/tb
bash qwen3vl_local/tb_serve.sh checkpoints/action_expert_ablation/bev_only/latest/tb
```

重点看：

- `train/loss`
- `train/route_fm_mse`
- `train/waypoint_fm_mse`
- `train/lr`
- `train/grad_norm`
- `train/samples_seen`：到当前 step 的累计训练 case 呈现数
- `train/step_samples`：当前一次 optimizer update 实际使用的 case 数
- `train/samples_per_second`
- `val/loss`
- `val/route_ade_m`
- `val/waypoint_ade_m`
- `val_epoch/*`

每个完整 optimizer step 是 `world_size * grad_accum_steps` 个 case，默认四卡、`GRAD_ACCUM=16`，
即 64 个 case，与 `action_prior` 一致；epoch 最后一个累积窗口可能不足完整 step，但在同索引、同卡数、
同累积数和同 seed 下尾部也一致。因此 TB 横坐标的 step 代表已经看过的训练样本量可直接对齐。
窗口日志中的 `loss(window_global,N 样本均值)` 也是全 rank 样本均值；`train/samples` 是这个
日志窗口的样本数，不能当作累计样本数。`train/samples_seen` 按训练游标计算，续训后连续，
包括不同 epoch 对同一帧的重复呈现，不含 val/test。

公平比较前核对三组的 `config.json`、`training_plan.json` 和 checkpoint 的 split hash：

| 对比项 | 必须对齐 |
| --- | --- |
| 数据预算 | 同一 train/val/test 索引及内容、GPU 数、梯度累积、seed、epoch/step 上限 |
| 优化设置 | optimizer、参数路由、LR、scheduler、实际周期、warmup、Muon 超参数、weight decay、EMA、decoder dropout、dtype |
| FM 定义 | route/waypoint loss 权重、坐标缩放、trajectory layers/heads |
| 评测口径 | 同一 val 子集/全量集、验证 seed、Euler 步数、EMA 或 raw 权重 |
| 曲线显示 | 同一 tag、日志间隔、TensorBoard smoothing，并核对 `samples_seen` |

降低 GPU 数、同时提高累积数即使保持 64 case/step，也可能改变 epoch 尾部与随机数消费，
不能直接声明严格同预算配对。默认训练只算 FM MSE；如需训练 ADE/FDE，三组都统一设置
`TRAIN_SAMPLED_METRICS=1`，同时披露增加的采样开销。

FM MSE 下降快只说明向量场训练误差下降快；最终效果看从纯噪声 Euler 采样的
`val_epoch/route_ade_m`、`val_epoch/waypoint_ade_m`、对应 FDE 与独立 test。
`best.pt` 按加权采样 route/waypoint ADE 选取。完整 epoch 使用全量 val；
用于 best 选择的验证始终遍历全量 val（包括 epoch 中途截断预算），周期验证才受 `val_max_samples` 限制。
smoke 的训练预算很短，不能与正式训练成绩混用。
正式 test 不参与训练选优。

## Checkpoint

checkpoint schema 为 `action_expert_ablation_checkpoint_v1`，会保存：

- `ablation_variant`
- `decoder_config`
- `flow_config`
- EMA/optimizer/scheduler/RNG
- 条件合同 `condition_contract`
- shared dataset hash
- `cursor.validation_pending`，用于在 epoch/final validation 前中断后先补验证和 best 选择

resume 会在打开新的 TensorBoard writer 前归档旧 `tb/` 中 step 大于 checkpoint step 的
event，并保留 checkpoint step 本身的记录，避免续训曲线出现未来 step 或重复 step。
执行指纹绑定消融入口、共享 action_prior/LeadMoT/BEV 执行依赖和关键运行库版本；
未使用的 Phase1/2 prompt 不参与 `bev_only` 或 `qwen_simple` 身份。
`qwen_simple` 与 `bev_only` checkpoint 不能互相 resume/eval；合同会拒绝跨条件加载。

## 版本与验证

本次共享循环重构保持模型结构、FM 定义和 checkpoint 容器 schema；执行源码指纹会改变。
**旧 run 的续训/评测需要其原代码版本**，不能因 schema 相同就绕过合同检查。
新的配对实验应从同一份新代码、相同 seed 分别训练三组 decoder。

源码回归测试（不加载真实 Qwen/BEV、不读取真实 LEAD 数据、不启动 CARLA）：

```bash
python -m pytest -q qwen3vl_local/action_expert_ablation/tests qwen3vl_local/action_prior/tests
```

新增 CPU 小模型测试覆盖三入口的数据/更新/指标一致性、两个消融的中途/epoch/截断预算恢复，
以及训练和 validation 收到真实 `SIGTERM` 后的安全 cursor 保存与精确恢复；已完成预算再次续训不执行额外样本。
测试还包含真实 TensorBoard event 裁剪检查，缺少 `tensorboard` 时会明确 skip；
本机未执行这一真实文件路径。CPU 测试不能替代真实 GPU/DDP 或 Qwen/BEV 前向检查。
远端首次运行可先使用上面的 `smoke.sh`，确认 loss 有限、best/latest 保存和独立 eval 可运行，
再按正式预算训练。smoke 是独立短预算 run，不能直接把它改成正式长预算续训。

2026-09-14 event-balanced 复审：修复开发名单漏排 397 组、验证预检误用候选桶、自动预算写死背景权重、
主线与消融首次并发构建未共用锁的问题。主线与消融测试共 **282 passed / 1 skipped**。
CPU 小模型验证了三组均衡采样顺序、配额审计、预算、事件桶指标和参数更新一致，以及课程恢复；
shell 桩检查自动准备、CLI/环境优先级、显式路径与最终 eval 搬迁转发；内容合同检查路径搬迁可用、
课程变更拒绝。跳过项是缺少 TensorBoard 依赖的真实事件文件裁剪；未运行真实 GPU 训练。

首次上机可用独立短预算完整 pipeline 验证均衡采样、权重加载、保存与 test。以下每组只训练 4 次更新，
仍会在选 best 时遍历全量 val，并完成全量 test，因此总耗时并非只有 4 步。确认后用上面的正式命令另开 run。

```bash
OUTPUT_DIR=checkpoints/action_expert_ablation/qwen_simple_smoke \
  bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --max-train-steps 4
GPU_IDS=0,1,2,3 OUTPUT_DIR=checkpoints/action_expert_ablation/qwen_simple_smoke \
  bash qwen3vl_local/action_expert_ablation/qwen_simple/run_full_pipeline.sh --event-balanced --max-train-steps 4
OUTPUT_DIR=checkpoints/action_expert_ablation/bev_only_smoke \
  bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --max-train-steps 4
GPU_IDS=0,1,2,3 OUTPUT_DIR=checkpoints/action_expert_ablation/bev_only_smoke \
  bash qwen3vl_local/action_expert_ablation/bev_only/run_full_pipeline.sh --event-balanced --max-train-steps 4
```

2026-09-14 主线续训接口已与两个消融对齐：三个 `run_full_pipeline.sh` 均支持
`--resume 路径` / `--resume=路径` / `RESUME=路径`，并提前解析真实 checkpoint 路径。
主线显式数据/Qwen/BEV 路径覆盖也会传给最终 test/probe；原课程、LR、epoch 和卡数默认值
由原 run 恢复，不把新训练默认参数带入续训。主线操作见[续训示例](../action_prior/run.md#续训)。


### 2026-09-22 训练审计与统一对比

本次bev_only无/有action token训练结果见
[TRAINING_AUDIT_20260922.md](bev_only/TRAINING_AUDIT_20260922.md)。
两个消融目录新增 `compare_checkpoints.sh`，统一调用主线同名脚本；
编辑 `action_prior/compare_checkpoints.sh` 的CKPT_DIRS即可比较多个run；
CASES_PER_CATEGORY/EVENT_CASES/ACTION_CASES配置每类及每split案例数，0跳过。
输出RGB轨迹投影＋道路/车辆框俯视图PNG/PDF；已有第三人称图需对应标定，可加入同屏。
细节见 [CHECKPOINT_COMPARISON.md](../action_prior/CHECKPOINT_COMPARISON.md)。

2026-09-22 对比入口补齐：结果默认写到 `AutoMoT/test/run_<时间>/`，与checkpoints同级。
默认最多自动选4张最空闲GPU，卡不足时减少并发；模型少于卡时按case拆片（4卡2模型各2份），模型多于卡则排队动态补位；
`--gpus`/`GPU_COUNT`设置自动上限，`GPU_IDS`显式pin优先。独立日志与scheduler记录，失败/中断回收本次子进程。
并发相关检查使用真实CPU子进程，未验证实际多GPU模型推理。

# 本机版本化数据准备与 Phase3 明确语义提示词

2026-10-22。本轮是数据生命周期与新提示词实验接线，不是新模型精度验收。服务器任选，所有路径基于当前 AutoMoT 应用根；不使用远程 Git、不迁移另一台机器的资产。同步仍只用 `bash ./sync_automot.sh`。本轮尚未 push。

## 已实现的行为

新增统一数据准备入口 `python -m qwen3vl_local.audit_joint.data_prepare`，支持 `phase1 / phase2 / base / single_frame / geometry / phase4 / recipe`。只准备数据与必要的原生验证/计划，不启动训练。旧独立训练入口继续保留；不是自动扫描 checkpoints 中每一种未知文件格式并擅自转换它们。

共用缓存位于 `checkpoints/generation_cache/<family>/<semantic_version>/<完整依赖SHA>/data`。不使用时间戳选最新，也不把服务器名称作为资产路径开关。人类可读版本用于说明语义，内容身份用于决定能否复用：来源文件、标定/派生实现、配置、划分及输入成员都属于对应生成阶段的依赖。

| 内容 | 当前行为 |
| --- | --- |
| 相同版本、相同依赖、完整结果 | 校验输出SHA与原生合同后直接复用，不调用生成器 |
| 同版本但输入改变 | 产生不同内容SHA的新目录，保留旧版本 |
| 缺失数据 | 在当前机器调用原生生成器 |
| 残缺/失败目录、输出损坏 | 明确拒绝复用与覆盖，保留失败证据 |
| 多份内容不同但都符合要求的旧库 | 报告歧义，要求指定一份；不按mtime猜最新 |
| 只有提示词/允许的训练运行代码变化 | Phase3标签不重算；训练/评测合同另行绑定 |
| 单帧派生标签的一条路线输入变化 | 只生成该路线分片，复用其它分片，然后重新组装完整库 |
| 全局标定、划分或共享依赖变化 | 使相关阶段失效；没有明确独立依赖的部分不能假装只改一个event |

缓存有跨进程文件锁、依赖前后核对、完整标记、失败记录与输出散列。文件夹存在不是成功标记。不会通过修改旧manifest中的SHA、修改旧checkpoint源码合同或强填未来UNKNOWN来让检查通过。

## 各入口的实际接线范围

### Phase3 base / single_frame

`phase3_launch.sh` 默认接入，无需再手填旧索引的时间戳路径：

1. 优先现有公共目录；默认目录不存在时，在本机 checkpoints 内有限深度查找 `common_receipt.json`。只接受当前原生发布、当前原始数据根、索引/清单/固定评测成员均通过验证的库。
2. 公共库使用 `phase3_common_inputs_v2`，校验标注 JSON、全部候选路线的 metadata 内容及成员、RGB 文件成员/挂载位置和隔离证据。任一生成依赖改变，在 `phase3_common_input_<SHA>` 新目录生成原生公共索引，保留旧成员和旧回执。没有标注导出时，仅有完整 v2 依赖证明且原始依赖仍匹配的库可复用；旧库缺少该证明时不能原地补写当前 SHA，需要原标注重新生成。原生索引只读取一般 RGB 的存在性；隔离证据图片另做内容校验，不把这个检查称为全池像素审计。
3. 单帧库先寻找当前数据根、索引、配置、标签来源一致的已有库；完整原生来源验证通过则直接用。仅运行代码变化时，生成带父库/变更名单的运行合同派生版本，records保持不变。
4. 没有可复用的完整单帧库时，按路线生成 `phase3_route/native_halfsecond_v1` 分片。每片绑定该路线全部metadata、当前图片、相关隔离证据、锚点完整内容及标签代码。整条路线的前后文都在依赖中，不只看锚点文件。
5. 合并前仍做全局物理路线 train/val/test 隔离验证；合并库用原 `dataset.load` 重验。公共目录保存 `single_frame_ready.json`，后续启动使用它选择的真实路径。

路线增量针对单帧派生库。原四图公共索引仍由原生完整生成器负责；如果其全局输入/划分改变，本轮没有为旧原生索引伪造局部迁移。首次遇到没有任何派生缓存的新机器，仍需生成全部必要路线；后续才能复用分片。

实际入口核对还复现了当前geometry版LeadMoT配置与原Phase3发布合同冲突。新增`phase3_native_runtime.py`：仅从已同步白名单中的原版兼容源码构建可复用快照，保留当前应用的geometry源码；启动时显式传入本机数据、模型、collection、输出路径。真实子进程通过原261源校验，第二次直接复用，无Git依赖、不改旧SHA。单帧缓存派生继续在原run_stage的时间/20GiB保留空间监控下运行，不因改成缓存而取消原守护。

### Phase1 / Phase2

新 `data_prepare --method phase1|phase2` 接入版本缓存，保留原生全局均衡、划分及标签生成器。同请求第二次不重建。依赖包括标注文件、审计表/人工记录/覆盖证明、RGB存在成员、生成实现及配置。原生成器只依据图片是否存在选择成员，未将像素内容作为标签生成输入；后续训练前的RGB读取检查继续执行。

旧 Phase1/2 manifest 并非都保存原始生产源码指纹。对这种库，不能仅凭目录名宣布与当前生产源码完全一致。可用 `--existing 本机frame_index.jsonl` 走原有方法/版本/三split/布尔标签校验直接复用；回执明确 `original_producer_source_identity=unknown`。不会给旧库补写一个虚假的当前生产SHA。原 full_pipeline.sh 的固定目录复用逻辑保留；要采用新内容版本缓存，先运行新入口，再将回执中的index交给原训练/评测入口。

这两个原生生成器包含全局采样与holdout选择，本轮没有拆成任意路线补丁。当前以生成文件为代码依赖单位，相关文件变化可能保守地使整个该阶段失效；不承诺只改文件中的一行就一定只重算一个event。

### geometry_v1

`geometry_v1/remote.sh cpu` 已接入：现有地图链接与assets只校验，已有匹配数据优先复用，缺失时生成内容版本目录；三轮四卡计划单独缓存。`data_ready.json` 保存实际dataset和plan路径，train读取该路径；旧固定 `data/dataset.json` 仍兼容。存在 run/test 时继续拒绝覆盖或自动续训。

原生geometry合同、候选/过滤后覆盖搜索、开发组隔离和九类标定均不变。几何生成目前仍是原生全池阶段；共享地图、几何规则或划分改变，不能套用六动作单帧的路线分片替代其全局覆盖流程。

### Phase4

新 `data_prepare --method phase4` 先执行独立的 v90 发布源码校验（旧 v89 清单保持原字节，接续、补 view、建库和首轮入口统一指向 v90），在本机查找已完成的CPU准入状态，按原生teacher/view/reference与预算检查复用；新回执支持此前“数据与view在两个不同目录”的 `existing_assets` 引用。不同机器的已收结果包不会仅凭hostname或文件夹名被当成本机可用资产。

没有完成结果时，可发现唯一兼容的 data4 + references4 中间产物并调用原有接续入口；显式 `--existing` 也可选择未完成目录。没有可复用资产才进入原有CPU准备流程。旧teacher全局合同仍有效，不改版本号伪装旧回放。资源准入仍可能因当前源码/资产/空间不同而拒绝；不能把找到目录当作已通过服务器验收。

其它训练/测试“数据生成”阶段可以使用 `--method recipe --recipe 本机JSON`。recipe需声明 `operation=generate_data`、完整sources/inputs、family/version，以及原生build/validate命令数组；`{output}`为当前版本目录，`{python}`为当前Python。它不是训练结果缓存，必须由实现者列全依赖；不能把未知checkpoint输出登记为当前标签版本。

## 提示词与标定

此前报告记录逐帧审计215张不同RGB、重放80个未来起点，当时未发现足以直接改原生规则的确定错标。原预测和手工笔记现已删除，图册是重建版；这些数值属于历史记录，不能视为本轮重新认证。训练有大量动作转换监督，不能直接归因为没见过转换。右变道换图净收益集中于一个锚点的四个相关窗口，也不能当作四个独立场景的正确性证据。

本轮不修改速度/变道阈值、UNKNOWN填补界限、未来mask、event/action配额或训练预算。仍为原生0.25s起点、每两个起点合成一个0.5s查询，四查询覆盖未来2s的起点；每个起点的原生前瞻判定可能引用更远轨迹，**不是未来2s执行轨迹的逐段真值**。历史仍使用用户允许的完整离线原生标签及nearest_short。

`native_clarified_v1` 明确预测记录中的行为而非推荐安全动作；event固定动作域，不代表障碍/等待状态持续；每个未来起点重新判断，已完成跨线不能继续算新的即将跨线，历史STOP/RESUME不强制后续延续。它追加在原系统提示词前，不改图像、速度、导航、历史、动作集合或时间格。

新增 `phase3_prompt_runtime.py`，为新训练的prepare/probe/train/val/test全部接入同一变体，保存独立prompt名称、实际提示词哈希和适配源码SHA。旧prompt断点不能套用新prompt继续训练；数据标签可以复用，训练状态不能因此自动兼容。原始默认prompt保留，未验证效果前不把实验假设标成已改善的base。

仍使用Qwen现有A–F词表，不新增embedding；单图四并行读出、最多四个有效loss，原mask不变。新入口不重启方案二。

## 发布后的命令

以下命令在当前AutoMoT应用根执行，当前Python来自已激活环境。**本轮尚未push，先发布再在服务器同步执行。** 数据不存在时生成可能耗时；训练预算仍四卡7680次更新，不因缓存缩短或增加预算。

先只准备/复用数据，不启动GPU训练：

```bash
bash ./sync_automot.sh && \
python -m qwen3vl_local.audit_joint.data_prepare --method single_frame \
  --receipt checkpoints/single_frame_data_ready.json
```

此后用原 `phase3_launch.sh --method base|single_frame` 时，同样自动准备/复用；无需重复手填index和dataset路径。若确实要新训练明确语义版本，使用新实验名：

```bash
GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_launch.sh \
  --method single_frame --prompt-version native_clarified_v1 \
  --output phase3_single_frame_clarified_v1
```

建议先跑上一份审计报告中的268次 `--prompt-review` 对照，再决定是否花费完整训练预算。已训练权重的诊断使用旧模型合同+显式推理干预；新训练使用独立提示词合同，两者目的不同。

Phase1/2可将上面的method换成phase1/phase2，查看回执的index；Phase4换成phase4。geometry继续使用已有 `bash qwen3vl_local/audit_joint/geometry_v1/remote.sh cpu`。所有方法都只使用当前机器本机路径，不指定必须cr02或cee。

## 验证边界

已验证：缓存重复调用不执行生成器、依赖变化新建版本、旧版本保留、并发锁、失败/损坏拒绝、无最新目录猜测、原生单帧分片与完整建库records/raw_proof逐字段一致、修改一路只重算一路、运行源码变更时records不变、注释输入变化后公共索引新版本、真实小Qwen新prompt四查询前向/反向/保存合同、torchrun/bootstrap参数边界、原匹配图诊断与Phase4接续回归。

Phase1/2缓存测试调用原生参数解析与校验器，昂贵生成器使用小型替身；Phase4新发现器测试使用准入替身，既有原生接续回归另外执行；不冒称全池/H20运行通过。未在服务器执行、未重训、未测新精度、未新增RGB语义盲审。完整测试数、耗时及源码SHA见同目录 `versioned_data_prompt_verification_20261022.json`。

## 本次复核修复

首次 geometry 建库的生成器复用、公共索引原始依赖漏检、Phase4 发布合同冲突及转换成对统计已修复。当前验证以 [修复说明](DATA_PREPARE_REVIEW_FIXES_20261022.md) 和 `data_prepare_review_fixes_verification_20261022.json` 为准；前一份回执保留为历史记录。

## 候选恢复与混合版本发现补充

geometry 自动复用现在要求完整候选输入证明，覆盖被筛除路线及缺失/损坏 RGB 的恢复；生成缓存协议升级为 geometry_v2，九类标定语义不变。无此证明的旧库不自动认定完整，旧文件保留，必要时新建一次可验证版本；显式选择仅代表固定历史产物。公共索引自动发现只跳过明确缺原始证明且无标注的旧候选，继续寻找完整 v2；显式请求与损坏文件仍严格拒绝。详见 [边界修复](DATA_REUSE_BOUNDARIES_20261022.md) 与 `data_reuse_boundaries_verification_20261022.json`。

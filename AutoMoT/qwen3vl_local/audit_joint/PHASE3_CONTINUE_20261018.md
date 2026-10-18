# Phase3 原断点接续与固定数据引用

本页取代 PHASE3_RECOVERY_HANDOFF_20261018.md 的首次诊断步骤。用户已停止清理；本入口没有删除操作，不再收集同一份只读诊断，不重新生成原题库。

## 21:19 接续回包：规则依赖遗漏已修复

收到cr02的83344字节ZIP，CRC、ZIP SHA及全部成员SHA核验通过。输出为 `phase3_continue_20261018_211943_895233955/result`。原生CPU断点读取2.62秒通过，确认next_epoch=2、global_step=2560。随后四个worker均在最开始的原生 `check_index` 报缺少 `mapping_rgb_decisions_v18_20260920.jsonl`；尚未加载训练模型、建立NCCL或执行新训练步，不是原NCCL超时再次复现。

原因是接续器复制源码时遗漏JSONL规则资产。修正版复制Phase2/Phase3包顶层的JSONL证据文件，并将其SHA纳入runtime快照与worker检查；不递归复制 `probe_output` 等生成数据。CPU断点读取后，先在同一隔离runtime中调用原生 `check_index` 校验choice/binary两模式，成功后才启动torchrun。原生mapping合同、数据SHA和训练源码均不改，不补造缺失规则文件或重新生成题库。

旧失败输出保留，使用新发布工具和新输出目录，仍接续原epoch_0002。两图两组不重跑，四图binary仍在choice成功后执行。反馈文件名和下面的流程保持一致。53项相关测试通过，含真实隔离runtime的完整原生索引准入、JSONL缺失/改动拒绝、CPU失败不启动torchrun，以及原断点恢复测试；未进行本次服务器GPU训练验证。

## 已收到的证据与本次执行范围

cr02 报告对应应用根 `/home/cruser1/lda/AutoMoT`：

|原 pipeline|已核对状态|本次动作|
|---|---|---|
|20261017_223254_126705688_2rgb_short_choice|三轮训练、test、审计打包完成|保留，核对已有回执，不重跑|
|20261018_034614_519706091_2rgb_short_binary|三轮训练、test、审计打包完成|保留，核对已有回执，不重跑|
|20261018_095559_213634718_4rgb_choice|第三轮中途 NCCL 超时，第二轮完整状态存在|核验 epoch_0002，从第二轮末续原三轮计划，再 test、审计、打包|
|4rgb_binary|尚未开始|前一组成功后，复用相同题库完成三轮训练、test、审计、打包|

明确恢复点为第三个目录的 `train/training_states/epoch_0002`；服务器原生读取器仍须核验张量和 optimizer/RNG。原包只验证了断点成员与元信息，没有重读权重正文。恢复起点 global_step=2560，不能从没有保存的3720步接续。原总轮数3、四卡、B2、累积2、原学习率进度及 sequence 采样保持；四图 binary 使用同一配置，仅切换输出模式，原生入口重新计算其采样及学习率总步数。

单元素 ALLREDUCE 超时的根因尚未确证。rank2 最后一条 batch 比其它 rank 少一条，不能据此认定坏数据、OOM或通信硬件故障。新增各rank日志、NCCL trace/dump/desync环境配置及30秒父进程心跳；没有延长原3600秒超时，没有自动重试，也不声称已经在服务器GPU上修复根因。

## 独立源码恢复

不要先整体同步、`git pull` 或回退服务器源码。只从本次发布提交取下面五个轻量文件，在当前目录创建的工作子目录运行：

- `phase3_continue.py`
- `phase3_legacy_sequence.py`
- `phase3_legacy_epoch_fixture.py`
- `dataset_catalog.py`
- `phase3_continue_release.json`

原 checkpoint 绑定的 sequence SHA 为 `8477a5333b0da3e04ff84366045bd73824a5327e1c824e470c275c1cd254886e`，精确字节来自历史提交 `69c1a18a4`。后续 Phase4 更新使当前共享文件 SHA 变化，即使 P3 默认采样行为相同，也必须保留原绑定，不能改回执 SHA。

入口复制本机 `qwen3vl_local` 源码到新输出的 `runtime`，只在副本内恢复精确旧 sequence 文件。原pipeline还绑定了测试文件，后续 `test_epoch_consumption.py` 的夹具修正也按历史SHA在副本内恢复。两份原字节都来自 `69c1a18a4`；不修改训练算法或忽略原SHA。保留服务器额外的P3源码文件；完整核对 checkpoint 源码集合/SHA 和原 pipeline 的 Python/shell/JSON 源码集合/SHA。任一其它差异会停止。数据和基座通过原本机路径引用，原始配置里的相对路径字符串不改，基座原inode/stat绑定不改。这个运行副本和原题库必须保留，供后续恢复与结果复查使用。

启动前核验完整断点文件 SHA、index/manifest SHA、基座身份，并由原生 `training_state.read` 在CPU读一次状态；随后原生训练入口仍执行全部恢复门禁。输出必须是新目录，且不能放在原运行、原题库、原始数据或基座目录内。已有目录拒绝覆盖。

调用形式（实际发布提交和下载/后台启动完整命令在本次回复中）：

```bash
AUTOMOT_SERVER=cr02 GPU_IDS=0,1,2,3 python ./phase3_continue.py \
  --root /home/cruser1/lda/AutoMoT \
  --completed-choice checkpoints/sft_new_loop_phase3_pipeline/20261017_223254_126705688_2rgb_short_choice \
  --completed-binary checkpoints/sft_new_loop_phase3_pipeline/20261018_034614_519706091_2rgb_short_binary \
  --checkpoint checkpoints/sft_new_loop_phase3_pipeline/20261018_095559_213634718_4rgb_choice/train/training_states/epoch_0002 \
  --output "$PWD/result"
```

机器名只是用户声明的来源标签；入口同时记录实际hostname和根目录，不根据标签猜路径、不迁移资产。此处目录来自已经收到的cr02报告；在另一台机器不能直接套用这组绝对路径或通过修改SHA复用断点。

原生epoch保存仍保留20GiB动态空间守护，启动最低21GiB只是下限，不保证两组全部保存所需空间。没有删除旧checkpoint，也不关闭空间保护。

## 固定数据入口、版本及水印

本次四组使用同一个 `phase3_four_data_20261017_223254_120427458/frame_index.jsonl`，SHA：
`925053e34c47f35dfe4987f14a1be3a8abdc206d75bea31964dc4ca9b8fcee99`。

新增本机目录 `checkpoints/dataset_catalog/phase3/`：

- `current`：指向明确登记的数据目录的固定软链接。
- `current.json`：当前版本和不可变记录的指针。
- `versions/<内容SHA>.json`：index/manifest SHA、原生产源码SHA、native规则版本、依赖目录、登记时Git/dirty状态、真实hostname、用户机器标签与路径。重复登记相同版本不增加数据副本。

旧数据没有生成时 Git/机器证据的字段标为未知；登记机器和登记Git不能冒充历史生成来源。水印放在绑定原数据SHA的旁车记录内，避免修改原manifest使checkpoint无法恢复。Phase4登记同时绑定train/val/test、候选池、review queue、registry和外部原生回放引用；它只是引用登记，不等于新库准入。

未来显式复用已登记P3数据可以把 `INDEX` 指向固定入口解析后的路径：

```bash
INDEX=$(readlink -f checkpoints/dataset_catalog/phase3/current/frame_index.jsonl)
```

再将该 `INDEX` 传给原全流程入口。恢复旧checkpoint时仍使用它原来绑定的路径，不擅自改成此别名。登记器提供 `resolve` 检查已登记版本、所有文件与源码依赖；源码变化或引用漂移会拒绝复用。时间戳运行目录仍保留，用于隔离训练结果和故障证据；共用题库不会因登记操作复制。固定入口不代表可覆盖仍被旧训练/回放引用的数据。

## 增量生成的实际边界

`dataset_catalog.py plan --previous OLD.json --requested NEW.json` 新增按分片依赖判断的复用规划：独立分片局部源码变化只使该分片失效；共享规则、输入、split或schema变化使全部分片失效；缺少依赖不能复用。规划不执行旧标签重签名或新库准入。

**当前原生Phase4仍绑定全局teacher身份和跨帧实例状态，尚未接入跨teacher版本的event级部分重生成。** 不能把“该event规则文字没改”等同于因果依赖不变。原有v24已完整回放的恢复继续使用v86 `RESUME_FROM`，复用完整原生路线；不因本次发布重放全池，不迁移旧v23标签为v24。今后若实现event级生产，需要先拆分并验证原生状态/共享依赖，再接入规划结果，不能用此规划器绕开合同。

## 结果、失败与反馈

当前工作目录的新子目录保存下载、`setup.log`、后台`launch.log`、PID及`result/`，不使用 `/tmp`。

`result/workflow.json` 记录正在执行的stage及每个已完成stage；`choice_train.log`、`binary_train.log` 和各组 `rank_logs` 留完整进程输出；两组都有独立pipeline和打包回执。原生 test 从训练产物重载adapter，四卡评估，保留base对照、binary审计prompt及原审计打包流程。原流程未启用历史成功题回归，因此完成状态仍是 `completed_without_historical_regression`，不能声称无退化。

成功或中途失败都反馈：

1. `result/handoff/phase3_continue_report.zip`
2. `result/handoff/phase3_continue_package.json`
3. `launch.log`

下载/极早期路径检查失败还没产生handoff时，反馈 `setup.log` 和存在的 `launch.log`。失败后不重复运行原四组命令；按新stage回执确定剩余工作，已完成训练不因后续评估故障重训。完整审计tar.gz在两组各自 `eval/`，需要效果复查时再交；handoff只含轻量状态、指标和日志，不含权重/RGB。

本地验证包括真实小模型的连续训练与断点恢复optimizer/RNG/曝光逐项相等、原生CPU checkpoint读取、损坏状态拒绝，以及实际4个torchrun CPU worker的参数传递。新GPU训练与本次服务器NCCL故障尚未验证。真实Qwen batch的两个既有测试因本机缺少transformers依赖未完成，未记为通过。精确测试数和源码SHA见 `phase3_continue_verification_20261018.json`。

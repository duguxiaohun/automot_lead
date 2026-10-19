# Phase3 独立采样恢复版与两条密集监督路线

> 本页保留为 v1 历史记录。当前代码、默认入口、辅助标签、对照组、覆盖门禁和显存预算已升级，执行以 [v2 审查修复说明](PHASE3_DENSE_REVIEW_FIXES_20261019.md) 为准；不要照抄本页旧密集路线配置。

本页取代继续跑 sequence 7680 实验的建议。当前首先验证 `base`，不再接着训练已收的 sequence 首轮。现有数据、权重、旧入口和 Phase4 共享模块保留，供追溯及原版恢复；不清理资产、不修改旧断点 SHA、不使用 SSH。本次新入口是 `audit_joint/phase3_base_trial.sh`，不要用旧 `run_full_pipeline.sh` 启动这次恢复实验。

## 1. 恢复范围与验收标准

恢复历史提交 `71cc49636eb367c1e1c06743953f48e61fd17f4c` 的独立事件/动作选题和单题 loss。历史已收四图 choice 的 val 为 242/320（75.625%），test 为 274/380（72.105%），这是固定题库的动作准确率，不是闭环驾驶成功率。本次没有重跑旧模型。新 sequence 首轮 val 185/320，尤其 VULNERABLE_CROSSING 仅 1/32；它尚未完成三轮，但用户已决定停止这条恢复路线。

| 项目 | 恢复版 base |
|---|---|
| 选题 | 十事件每轮各1024；事件内部按历史 support-aware 动作硬配额，保留历史路线分散选题 |
| 顺序 | 独立打乱；无少见题优先、锚点曝光优先、连续片段优先 |
| 跨轮增强 | seed + epoch × 1000003，恢复每轮题目/选项顺序变化 |
| 更新 | 四卡，每卡B1，累积1；每轮2560，三轮7680；lr=1e-5，warmup=2000 |
| 输入输出 | 原四图、原提示词、原动作词组 choice；不新增真实当前动作输入 |
| 数据标定 | 复用同SHA题库；teacher、标定、阈值、主动作投影、RGB窗口不变 |
| 保留修复 | 固定160 loss / 320 val / 380 test成员，原单题KV解码，完整epoch断点、实际消费审计、NCCL诊断、磁盘守护 |
| 另外保留 | 显式随机种子、梯度检查点；并非逐字回滚全部历史源码 |

动作配额是历史 `support_aware_quota` 的容量约束配额，不是把每个事件中不存在的动作强补到等量。有效全局 batch 是4；batch本身不由“是否sequence”数学决定，本实验主动恢复历史配置。更小batch不会自动保证准确率恢复。

独立适配仅在新worker进程内安装；共享文件磁盘字节保持不变。旧 sequence 选题函数在本入口被替换，恢复状态只保存策略与下一轮，不保存或使用曝光次数来影响下一轮选样。旧sequence断点禁止接入。

先看首轮固定val及分动作、分事件结果，再接续余下5120更新。不能凭首轮改善宣布恢复；完整三轮后只评测新模型test，并核查 STOP/RESUME/DECELERATE/KEEP 和横向动作的召回、误报。验收通过后再把该 **final adapter** 作为两条路线的共同起点，保留原base报告，不能覆盖。

## 2. 路线一：同一观察，多时点动作

`--experiment future`：当前四图预测 t=0、+1、+2秒的三个动作。当前原生规则的立即/纵向/横向窗口分别为6/8/12帧，帧间隔0.25秒，即1.5/2/3秒。BPE token没有固定动作持续时间，因此这里定义的是三个预测锚点，不能说每个token代表执行一秒。未来锚点各自仍使用原生动作标定，窗口可以重叠。

当前道路、情境、已存在的历史描述、速度和导航前提沿用；只使用当前输入信息，不插入未来速度/道路/导航/RGB。三个答案来自同路线、同split、同情境/道路的现有索引精确帧。缺标签的监督位置mask掉，不插值、不补KEEP；报告逐事件/时点覆盖。未来支持全缺时阻止训练，不能把一个loss宣称三个loss。稀疏索引不能凭空提供未来标定，本入口不自动重建大题库。

A–F固定对应 DECELERATE、STOP、RESUME、LANE_CHANGE_LEFT、LANE_CHANGE_RIGHT、KEEP；保留原动作定义及STOP优先规则，提示词显示本情境允许的选项。验证六个字母均为现有单token，不扩词表，不随机初始化新embedding。base仍输出原词组；新路线的编码变化单独记录。

这是**并行选择读出**：三个固定query位置的原LM head各预测一个字母，真实答案仅传给CE loss，不出现在input_ids；后面的读出看不到前面的真实答案。推理也用相同位置并行读出，不调用普通多token `generate()`。这比多词组自回归多了受监督的动作决策，但并不保证文本token总数更多。loss按有支持的query取平均，不能靠把loss乘三冒充信息增加。

## 3. 路线二：八图、五个监督位置

Qwen3.5-4B官方为262144原生上下文，32层采用3层Gated DeltaNet加1层普通注意力的重复结构。最大上下文不是H20微调显存预算，也没有可直接套用的固定“最佳图片数”。依据：[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)。

两个明确区分的可执行模式：

* `--experiment stream`：真正一次输入八图，在第4、5、6、7、8张之后分别设置query，五个loss共享同一次前向的视觉和语言前缀。按因果顺序插入各锚点当时可用的道路/情境/速度/导航前提，省去动作历史描述。标准因果注意力和因果递归保证较早query看不到未来图像和前提；答案不回填。**可见历史是4/5/6/7/8帧，不是严格移动四帧窗口。** 每次更新重新计算，训练不复用带旧权重或detach的跨步KV。
* `--experiment windows`：八图组成 `[0:4]…[4:8]` 五个严格四帧窗口，在同一张卡作为五个语言上下文并行计算五个loss。冻结视觉塔对相同输入patch去重，一般编码8图而非20图；完全相同图片可进一步去重。不同窗口的语言状态独立，**不声称共享语言KV**。

仅给普通注意力设置滑窗mask无法清掉DeltaNet中较老图像的递归状态。因此不声称同时实现“严格四帧遗忘＋任意共享语言KV”。两种模式的实现和证据分开；stream最接近用户要求的一次整段输入，windows提供严格窗口对照。

统一的是提示词模板及A–F映射，不把“当前真实动作”作为输入；保留当前可用的道路/情境与导航前提，避免把原情境域动作投影变成无条件猜测。五窗口必须来自同路线、情境、道路、split且四图历史逐帧重叠，不按最近帧强拼。连续样本仅作为**监督单元**，其选取仍由独立事件/动作硬配额决定，无少见题或长片段优先。

每图最多442368像素，保留原1152×384拼接图面积，对应约432视觉token；八图约3456视觉token，另计文本。实测总长必须≤8192，超出拒绝，不截掉图像/标签。只对query位置计算全词表logits，避免为所有图像token分配巨大248320维logits。正式训练前单卡执行两次真实前向/反向/AdamW更新，记录GPU型号和峰值显存，要求至少8GiB剩余且peak reserved≤总显存85%。探针不保存模型，不改变初始adapter；不是所有输入/其它并行任务的不会OOM保证。

三种新目标均保持四卡、B1独立锚点、累积1、三轮7680更新。相同更新数下，future最多三倍、stream/windows五倍query监督，因此不是相同监督量的纯速度比较。五个高度重叠窗口也不是五个独立场景。GPU探针、覆盖报告和实际训练记录用于确认收益成本。

新路线固定val/test成员采用共同四图、单query的A–F读出，方便观察当前动作是否改善；全词表argmax错误/格式外token均算错，六选一约束后的结果另报。它和base的动作词组生成不是完全相同的解码协议，不能只凭总分把改进归因于loss密集度。另外在同一批固定锚点上报告各未来时点/窗口的支持数、准确数和缺失项，不替换缺失片段。无标签推理API为 `qwen3vl_local.audit_joint.phase3_base_dense.objectives.predict`，训练和推理使用同一位置读出协议。原生eval遇到dense adapter的独立prompt合同会拒绝，不能把它装成原base进行推理。

## 4. 当前先在cr02运行恢复版base

在原automot环境运行。日志在当前AutoMoT目录，数据只引用一次，不复制或再生成。若四张空闲卡不是0,1,2,3，修改GPU_IDS；不得同时占用其它任务的卡。

```bash
cd /home/cruser1/lda/AutoMoT
bash ./sync_automot.sh && (
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  P3_RUN="phase3_base_$(date +%Y%m%d_%H%M%S_%N)"
  mkdir -- "$P3_RUN"
  nohup env AUTOMOT_SERVER=cr02 GPU_IDS=0,1,2,3 PYTHON="$(command -v python)" \
    bash qwen3vl_local/audit_joint/phase3_base_trial.sh \
    --experiment base --stage quick \
    --index checkpoints/phase3_four_data_20261017_223254_120427458/frame_index.jsonl \
    --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
    --output "$P3_RUN/result" > "$P3_RUN/launch.log" 2>&1 < /dev/null &
  printf 'PID=%s\n目录=%s\n' "$!" "$PWD/$P3_RUN"
)
```

完成标志是 `result/workflow.json` 的 `status=quick_epoch_completed_quality_pending`，`active_stage=null`，各stage退出码0，`training.optimizer_updates=2560`，并产生handoff。看到训练结束打印或 loss 降低本身不算整个流程完成。中断也会打失败包。

反馈两文件：`result/handoff/phase3_base_report.zip` 与 `phase3_base_package.json`；若前置失败没有handoff，则交 `launch.log`。用实际输出目录查看 `tail -f .../result/train.log`。本轮不要求重跑旧模型或继续sequence实验。

首轮审阅后续训，只把参数改为：

```bash
--experiment base --stage complete \
--resume /绝对路径/本次phase3_base首轮/result/train/training_states/epoch_0001 \
--output /绝对路径/全新接续目录/result
```

其它输入保持一致。首次也支持 `--stage complete` 一次跑全部三轮；quick和complete使用同一个三轮LR计划。新base的epoch断点不能与sequence断点混用。

## 5. base通过后分别启动新路线

同一入口、同一index/data/model、同样新建日志目录；独立选择其中一种，不串行叠加三个训练结果：

```bash
--experiment future --init-adapter /已验收base接续目录/result/train/final --stage quick
# 或
--experiment stream --init-adapter /已验收base接续目录/result/train/final --stage quick
# 严格滑窗对照
--experiment windows --init-adapter /已验收base接续目录/result/train/final --stage quick
```

只有独立采样base完成7680更新的adapter可作为初值；用户仍需先确认效果，程序不会把“跑完”冒充质量通过。密集路线的覆盖、GPU探针、训练、固定val顺序执行，首错停止；仍交相同命名的handoff两文件（由不同实验目录区分）。它们的complete必须引用**各自**quick断点，并保持同一个init-adapter及experiment。test只在完整预算后运行，不用于反复选版本。

## 本地验证与尚未完成的验收

最终101项相关测试通过（12.37秒），236项源文件绑定核验通过。测试覆盖历史函数逐项采样/单题loss及梯度、epoch恢复与不中断参数完全一致、实际Qwen处理器的三种输入与像素限制、真实小Qwen的因果RGB隔离、共享视觉前向/梯度一致、缺标签mask和跨split/断帧拒绝，以及真实处理器进入原生训练并生成optimizer断点。

本机没有H20或完整服务器题库/4B权重；尚未执行新base的服务器训练，也未测得两个新路线的准确率、全池连续标签覆盖或H20峰值。不能宣称恢复超过70%、加速多少或保证不爆显存。该实现提供可执行、可拒绝不满足条件输入的实验路径；“完美”和最终效果需要新报告验证。

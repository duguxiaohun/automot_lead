# Phase4 v37 全量训练：每轮全部合格题，事件呈现严格1:1

本版按用户确认的策略实现：每轮覆盖全部合格训练题，小事件通过重复补足到与最大事件相同的呈现次数。不是事件加权损失，也不是此前每轮400次的小预算抽样。事件内部保留全量题池的边/答案比例，不声明边内YES/NO也为1:1。未标注/UNKNOWN/异常否决帧不当作NO；“全量”是全部合格训练路线产生的可用监督，不是把每张原始图强制赋标签。

## 实现与规模

- `--sampling-policy full_event_equal`：每事件配额至少为最大事件题池大小，并按DDP整除要求向上对齐。每题每轮至少一次，事件次数严格相同，没有尾部丢弃或占位样本。轮间确定性打乱，额外重复优先给累计见得少的题；checkpoint绑定题池、seed、预算、world和历史。
- 该显式策略不应用旧的单题cap4/同帧cap8。实际重复量进入报告；其他旧采样策略仍保留原上限。过小的显式epoch预算报错，绝不悄悄截断全量。
- 建库传`--max-teacher-questions-per-route 0`，取消100题/路线截断，保留可见性、异常、冲突和人工否决。manifest记录选择策略，全量入口拒绝仍有路线截断的库或会丢题的配对视图。
- `train_full.sh 2/4`固定全量模式并绑定另一图数数据；主机预检贯通真实进程数和梯度累积。训练每轮完整遍历人工val并按事件宏平均选优；test由独立evaluate入口完整遍历，不做训练式等额重复。

全部8614路线、1062401帧已按最终源码重新回放；train/val/test路线仍7268/658/688。2/4图训练题均为88986，比v36各新增13497道，且v36所有训练题语义ID保留。原443人工训练题逐行保持，val273/test186/review539逐字节保持。全部88986题严格配对，无配对丢弃。

每轮单进程793040次，四进程793040次；每个事件各79304次（四进程）。每轮唯一题覆盖88986/88986=100%，不是累计七轮才覆盖。七轮四进程共5551280次呈现；accumulation=8时每轮约24783个优化步。2/4图、world1/4均实算七轮，两个图数在相同world下每轮题序一致。

| 事件 | 不同训练题 | 每轮呈现（world4） | 平均每题重复 |
|---|---:|---:|---:|
| R-E2 | 32 | 79304 | 2478.25 |
| R-E3 | 23 | 79304 | 3448.00 |
| R-E5 | 32 | 79304 | 2478.25 |
| U-E1 | 79304 | 79304 | 1.00 |
| U-E2 | 44 | 79304 | 1802.36 |
| U-E3 | 21 | 79304 | 3776.38 |
| U-E4 | 9426 | 79304 | 8.41 |
| U-E5 | 35 | 79304 | 2265.83 |
| U-E6 | 41 | 79304 | 1934.24 |
| U-E7 | 28 | 79304 | 2832.29 |

这些重复是用户选择的严格呈现1:1所致，并未增加稀缺事件的独立路线或正例证据。关键边和六事件人工评估缺口保留；这是全量弱监督训练，不代表完整十事件驾驶能力已认证。

## 运行

本机默认`AutoMoT/checkpoints/Qwen3.5-4B`仍缺失。先把MODEL_DIR改为真实完整基座目录，PYTHON使用已安装项目依赖的环境；不自动下载模型。以下为现有本机环境：

```bash
cd /home/codon/automot_lead
export PYTHON=/tmp/automot-qwen35-env/bin/python
export MODEL_DIR=/实际完整模型目录/Qwen3.5-4B
export DATA_ROOT=/home/codon/automot_lead/AutoMoT/lead_data
P4=/home/codon/automot_lead/AutoMoT/qwen3vl_local/sft_new_loop_phase4

# 两图全量训练（默认自动选择最多4张空闲GPU）
OUTPUT_DIR="$PWD/AutoMoT/checkpoints/phase4_full_2rgb_$(date +%Y%m%d_%H%M%S)" \
  bash "$P4/train_full.sh" 2 ddp --epochs 7 --accumulation 8

# 四图全量训练，使用相同训练配置
OUTPUT_DIR="$PWD/AutoMoT/checkpoints/phase4_full_4rgb_$(date +%Y%m%d_%H%M%S)" \
  bash "$P4/train_full.sh" 4 ddp --epochs 7 --accumulation 8
```

也可显式`export GPU_IDS=0,1,2,3`选择设备；单卡将`ddp`改为`single`。共享GPU时依次运行两组实验。

不启动GPU的采样检查：`bash "$P4/train_full.sh" 2 check --epochs 1`和对应4图命令。完整训练主机检查：`bash "$P4/train_full.sh" 2 host-preflight`。不要再指定`--epoch-samples 400`或`10000`，默认预算自动满足全量与1:1。

epoch边界恢复时使用新的OUTPUT_DIR，并追加`--resume /旧run/epoch_000`，保持数据、图数、seed、GPU进程数及其他配置相同；`--epochs`仍为总目标轮数。旧版本训练不能跨合同直接恢复。

本版全量数据已构建，无需再次重建。若迁移服务器，带上`phase4_v37_full`中的两套完整数据与production、匹配源码及原RGB/metas/bboxes；设置实际DATA_ROOT/MODEL_DIR，使用专用全量入口。不得通过改旧manifest的SHA冒充匹配源码。

## 验收

941 passed in 86.54s (0:01:26)。实际双模式pipeline check、全部训练/人工评估/待审RGB核验、world1/4七轮全量覆盖和配对题序、全量专用shell入口均通过。每轮10000的预算因不够覆盖全池而退出2；默认host-preflight只因缺MODEL_DIR退出2。

新产物`AutoMoT/checkpoints/phase4_v37_full`；合同v37，老师规则v8未改、预约v17同路线，严格注册表v12仍空。没有新人工标签/独立RGB审核，没有下载模型或运行GPU/CARLA。旧v36产物保持。验收绑定SHA仅对记录时点有效。

主要证据：`verification.json`、`paired_full_sampling_verification.json`、两模式`verification2/4.json`、`paired_preflight2/4.log`、`final_source_acceptance.json`。前瞻覆盖报告采用穷尽循环的不变式推导；另有真实七轮采样计划逐项验证，不能把预检计划当作已完成训练。

# Phase4 v33：逐分支覆盖、严格配对评估与启动检查

本轮接受审计中的监督覆盖、六事件评估缺失和采样规模结论。代码补齐可检查的覆盖报告、评估入口、风险接入和启动配置；没有把缺题变成标签，也没有增加新的规则老师。完整十事件闭环仍未验证。

## 逐分支覆盖

`branch_support.py` 明列每个事件/分支/是否返回原车道的全部转移边，并分别计数 train/val/test、readiness/catchup、YES/NO、物理路线及静止样本。UE2 的不返回和本车道通过、UE4 的跟随和绕行都单列，零样本分支不会消失。`preflight` 输出 `branch_support` 与按关键边优先的 `task_plan`，这是补标清单，不是自动标签。catchup 不能填补 readiness 缺额。非绕行模板的返回标志统一为默认 True，仅用于规范分组键，不表示跟随/横穿事件必须返回原车道。

七轮覆盖新增 `pool_fraction_seen`。预算不可行、时间线为空时结果为 null；不能写成已经覆盖或除以零。epoch 在此表示受容量限制的采样预算，不表示完整遍历题池。事件等额、边内答案平衡、单题 cap4 和共享帧 cap8 保持，因此大池不能解除稀缺事件造成的预算瓶颈。

## 严格配对评估

`evaluate` 的 `cases.jsonl` 新增 `paired_identity`，绑定物理路线、split、实例状态、转移边、阶段、参考答案、来源类别、因果历史、RGB 文件/像素 SHA、提示词和模型输入 SHA。`paired_eval` 只接受两边非空且完全相同的题目集合与身份；重复、漏题、换图、换答案、换参考类别均拒绝，不能取交集后声称公平比较。

UNKNOWN 和 MALFORMED 计为错误。人工参考和老师参考分开报告，后者仅为老师一致率。该工具不证明两次训练预算相同，不替代独立盲审，也不是 CARLA 闭环评估。旧预测文件缺少输入身份，须用新 evaluator 重新生成，不能补写哈希冒充同输入预测。

```bash
cd AutoMoT
PYTHONPATH=. /tmp/automot-qwen35-env/bin/python -m qwen3vl_local.sft_new_loop_phase4.paired_eval \
  --left /path/to/run_a/cases.jsonl \
  --right /path/to/run_b/cases.jsonl \
  --output /path/to/paired_report.json
```

## 启动和构建

`train.sh` 默认指向 `checkpoints/phase4_v33_full/data4`；`RGB_MODE=2` 切换 data2。启动首先检查数据合同、目录和本地模型文件，缺少模型明确报告 `MODEL_DIR`，不下载权重。`host-preflight` 执行主机预检但不训练；`check` 仍只验证数据和采样。路径通过环境变量传入，拒绝用训练 CLI 参数另行覆盖，保证预检与训练看同一组路径。

```bash
# 仓库根目录：有限采样检查，不加载 GPU 模型。
PYTHON=/tmp/automot-qwen35-env/bin/python \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh check

# 需先有完整的本地基座权重；本机默认权重仍缺。
PYTHON=/tmp/automot-qwen35-env/bin/python MODEL_DIR=/path/to/local/Qwen3.5-4B \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh host-preflight
```

`run_full_pipeline.sh` 不传生产索引时会明确说明只构建已有审核题/条件流；传 `PRODUCTION_INDEX` 时必须同时传匹配源码的 `TEACHER_REGISTRY`，两者不配对则在扫描数据前拒绝。全量弱库构建示例：

```bash
P4_BASE="$PWD/AutoMoT/checkpoints/phase4_v33_full"
PYTHON=/tmp/automot-qwen35-env/bin/python MODE=check RGB_MODE=4 \
  DATA_ROOT="$PWD/AutoMoT/lead_data" \
  CANDIDATE_POOL="$P4_BASE/candidates.json" \
  PRODUCTION_INDEX="$P4_BASE/full_production/index.json" \
  TEACHER_REGISTRY="$P4_BASE/weak_registry.json" \
  DATASET=/path/to/new/data4 OUTPUT_DIR=/path/to/new/check4 \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/run_full_pipeline.sh
```

## RGB 与风险登记

接入用户报告的 12 个训练曝光物理组和 7 处原图确认的不连续，精确绑定相邻帧及原始 RGB SHA。撤回的缩略图猜测、未确认的遮挡不升级为确认风险。本轮另看了 3 条既有训练路线、8 帧组成的 3 张缩放联系表，复核其中三处消失及绕行位置；无新增完整序列、人工标签或独立 RGB 审核。用户报告的 12 条全序列筛查、1,926 帧、34 张原图复核属于继承审计，不计入本轮新增工作。

累计 199/278 格达到三个 ID 的筛查数量，原清单仍余 228 条；筛查量不能等同于逐题认证。风险与来源详见 `thirty_third_audit_exposure_20261006.json`；原只读报告及证据索引已保存在 v33 产物的 `inherited_readonly_audit`，不依赖临时目录长期存在。

## 未完成的实际接线

Phase4 动作先验导出与桥接接口不等于 Action/CARLA 接管。现有 Action runner 仍拒绝缺少在线提供者的高层先验路径；尚缺真实在线实例、导航、机动安全许可和执行回执。未以离线真值或专家控制量替代这些输入。本轮未改 Phase3/Action 默认行为，未执行 GPU 训练或 CARLA 验收。

完整核验计数见 [thirty_third_audit_verification_20261006.json](thirty_third_audit_verification_20261006.json)。产物使用新目录 `AutoMoT/checkpoints/phase4_v33_full`，保留 v32 原产物与来源绑定。

## 本轮实际验证结果

850 项专项测试通过；全部 8,614 条合格路线/1,062,401 帧已重放，物理 split 未改变。

| 模式 | 训练题 | 弱题 | 每轮呈现 | 七轮不同题 world1/world4 | 池覆盖比例 | RGB 核验题数 |
|---|---:|---:|---:|---:|---:|---:|
| 2 图 | 75373 | 74930 | 400 | 705/705 | 0.9353% | 76371 |
| 4 图 | 75373 | 74930 | 400 | 705/705 | 0.9353% | 76371 |

原 443 道人工训练题逐行保留；val273/test186/review539 逐字节保留。18 个旧 UE1 静止正例保持，旧过期实例错误题的回归检查通过；运动 UE1 readiness YES 与弱 restrict 仍为0。新增风险是否减少题量由构建排除账本记录，登记风险本身不产生监督。

留出老师参考每模式 val8345/test7952；新盲审961卡/56路线全部未填，严格批准0，无独立准确率。本机默认模型检查明确失败，采样检查通过不代表主机已能训练。

配对 CLI 使用真实已审核题的输入结构和人为 UNKNOWN/MALFORMED 预测完成机械检查；未运行模型推理，不能把该检查报告作为模型性能。弱库 trainable=true，formal_data_ready=false；本轮不声明完整十事件闭环可训或已接通 Action/CARLA。

## 关键分支的实际 readiness 支持

| 分支/边 | YES | NO |
|---|---:|---:|
| U-E2/default/return / depart | 0 | 2 |
| U-E2/default/return / return | 0 | 0 |
| U-E4/cyclist_bypass/return / depart | 0 | 0 |
| U-E4/cyclist_bypass/return / return | 0 | 0 |
| U-E4/cyclist_follow/return / complete | 0 | 73 |
| U-E7/default/return / proceed | 0 | 4 |
| R-E2/default/return / enter | 0 | 0 |

UE2 不返回、本车道通过分支仍为零题；val/test 缺少 U-E2, U-E3, U-E4, R-E2, R-E3, R-E5。这些问题需要可靠监督及独立审核，本轮没有新增标签。

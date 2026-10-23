# Phase3 五类当前运行说明

当前正式版本为 base bounded_v3、single v3、abc v2，三包均已完成本机正式全池 CPU 准入。统一入口为 `bash qwen3vl_local/audit_joint/phase3_compare.sh`。历史回执保持原样，最新验收范围见 [发布复核](PHASE3_RELEASE_REVIEW_20261023.md)。

| 方法 | 当前实现 | 监督与输入 |
|---|---|---|
| base | phase3_base_bounded | 四图、原生六动作单题、独立采样 |
| single | phase3_single_v3 | 单图、六动作历史与四个未来窗口、独立六动作有效性 |
| a/b/c | geometry_abc_v2 | 单图、当前参考几何历史与未来；三组共享成员、mask及计划，各自生成标签 |

每组十事件各1,280次呈现／轮，共12,800次；四卡、每卡B1、累积1，每轮3,200更新，三轮9,600。各组允许独立采样。seed=20260904、lr=1e-5、warmup=2500及原生余弦、最终第三轮测试政策统一。single／abc 全有效验证集诊断，宏最佳轮只记录，不用于选择测试模型。

single／abc 每个物理帧跨事件每轮最多一次、全程最多三次。base 按用户授权允许有限重复：物理帧每轮最多两次，同事件累计按容量设限、共享帧全程最多六次。当前计划每轮额外物理呈现485次（3.79%），三轮累计最大五次。超过对应容量上限仍拒绝。图数、历史含义、查询数与loss保留方法差异；不能仅按各自分类准确率给轨迹能力排序。

## 独立标签与 loss

single 复用已校验原生父库的当前 RGB／速度／导航／车道参考与物理划分，从原始轨迹重新计算六动作。父库保留几何未来无效的记录；single 不读取其几何标签作有效性门，不与 abc 未来交集。仍要求原生当前动作有效和至少一个原生未来窗口有效。A/B/C/F 继续按各起点的局部速度与证据判定；D/E 先原生确认变道，再投影到当前车道方向；历史允许后续离线证据和既有短缺口填补。共享父库当前观测要求与既定划分仍存在，不能称为全部原始候选无条件可用。

abc 不调用六动作标定，保持当前原点／速度／车道方向。a 规则不变；b 当前道路 R3 用20／24米，其他用3／13米；c统一3／13米。横向±2米，所有历史／未来时距共用固定米数，边界归中区、远区开放。

abc 从原词表取 A–I 九个 logits，先取列再做九类内部 log-softmax／交叉熵；每个锚点对有效查询平均，DDP 再对等量锚点平均。无效查询不贡献梯度，其他词表 logits 不影响 loss。预测 `probabilities` 是和为1的九维区域概率；全词表原始 argmax、格式诊断及区域总词表概率质量单列。没有新增 token，也不对字母编号做 MSE。新合同与源码身份绑定数据、准入、训练、预测及 expert；旧库和旧权重不会自动改写为新 loss。

## 类别支持与边界

正式 CPU 报告包含全体固定验证成员、类别端点数与物理组数；评估额外报告稀疏类、支持较充分类别子集宏平均、保持原有类别集合的 leave-one-group-out 宏平均范围，以及移除某组导致类别消失的次数。这些是敏感性诊断，不是置信区间。20端点／5物理组仅是明确标记的诊断筛选线，不是准入或可靠性认证。

训练集边界审查自动报告近界3／4／5米候选（外界13米）的分时距类别量和物理组支持；不读取 val/test 拟合阈值，不自动变更标签。当前全池统计仍为：c左近／右近在3米时96／28，4米时947／256，5米时1822／908。4米候选尚未完成新的训练集边界RGB／轨迹审查，因此保留3／13米基线。

## 执行

在服务器已有 AutoMoT 根目录、含依赖的 Python 环境中先同步和查看状态：

```bash
bash ./sync_automot.sh
bash qwen3vl_local/audit_joint/phase3_compare.sh status all
```

本机 ready 不代表另一台服务器已有兼容 ready。服务器首次准备运行 `cpu all`；已有当前版本且来源未变化时验证复用缓存。本机无需重复全池建库。CPU 不要求模型或GPU；支持 `--maps` 指定完整地图目录。准备失败时查看原因，不降低准入标准。

全新任务且五组输出均不存在时：

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu all
bash qwen3vl_local/audit_joint/phase3_compare.sh check all
bash qwen3vl_local/audit_joint/phase3_compare.sh probe base
bash qwen3vl_local/audit_joint/phase3_compare.sh train all
```

`check` 校验正式准入、输出碰撞、完整模型与四卡；`probe base` 在独立目录执行前台真实模型探针，不完成正式训练。探针已有输出时拒绝覆盖。`train all` 按base→single→a→b→c在同一组四卡顺序运行，共48,000更新；每组完成训练、测试及验收后才继续，首错停止。

已有 base 内层训练完成、仅外层收尾失败时：

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh verify base
```

该命令不重训、不运行GPU，仅验收已有产物。完成base后不要重新 `train all`；剩余四组均未开始时依次 `train single`、`train abc`，否则按实际缺少的方法启动。single／a／b／c也支持 `verify`，核验保存的结果，不重新推理。

输出为 `checkpoints/phase3_compare_base/run_base_bounded_v3`、`checkpoints/phase3_single_v3/runs/six_1280`、`checkpoints/geometry_abc_v2/runs/{a,b,c}_1280`。保留旧输出，拒绝覆盖和隐式续训。旧 `geometry_study six`、旧coverage等入口保留历史版本含义，不代表当前合同。远程无需Git历史，仍只用 `bash ./sync_automot.sh` 同步。

## 当前证据与限制

正式全池：base bounded_v3、single v3、abc v2均ready。single保留204,910锚点（train202,516／val573／test1,821）；abc保留259,493锚点（train256,469／val740／test2,284）。single在六动作有效性过滤后仅移动绑定的未曝光train完整物理组补齐事件，原val/test不移动；abc不受六动作筛选限制。详细证据见 [CPU准入](PHASE3_CPU_ADMISSION_20261023.md) 与 [base有限重复](PHASE3_BASE_BOUNDED_RUN_20261023.md)。

single跟车制动test为58锚点／219查询／1物理组，信号灯故障仅1查询；动态切入12查询、STOP／RESUME目标质量3／9、1物理组。abc跟车制动test仅2锚点／1物理组。验证缺类、稀疏测试支持、既有开发曝光和边界视觉审查仍限制结果解释。分类loss已实现，统计均衡不等于边界语义已验收。

最新CPU回归与发布范围见 [发布复核](PHASE3_RELEASE_REVIEW_20261023.md)。CPU及合成分布式测试不能替代真实4B四卡探针、训练或在线效果。本轮未执行这些验收，也未修改既有ready、标签或权重。

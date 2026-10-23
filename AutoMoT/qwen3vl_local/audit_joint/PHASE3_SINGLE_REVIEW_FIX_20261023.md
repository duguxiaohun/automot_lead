# Phase3 single 测试覆盖与地图入口修复

本次用户指出的两个阻断都可以复现。旧 single v2 在当前父库的 test 跟车制动四个候选上全部无效，重复建库无法产生该事件的监督；本机旧地图发现只查项目 CARLA 目录，无法找到实际位于兄弟 `lead` 目录的地图。

## 当前入口

`phase3_compare.sh` 的 `single` 现指向独立 `phase3_single_v3`，输出为 `checkpoints/phase3_single_v3`。base和abc标签／训练运行时保持原样；v2库、源码与旧划分保留。新旧single的权重与准入不混用。

```bash
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu single
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu abc
bash qwen3vl_local/audit_joint/phase3_compare.sh status all
bash qwen3vl_local/audit_joint/phase3_compare.sh check all
bash qwen3vl_local/audit_joint/phase3_compare.sh train all
```

仍需在已有AutoMoT根目录、正确的Python依赖环境运行。`cpu all` 会先运行base已有的CPU流程；base仍要求其本地模型等前置资产，不能把它当作完全不需模型的single/abc检查。完成正式CPU准入后才check/train。本轮未push；发布后远程仍使用 `bash ./sync_automot.sh`，不要求Git历史。

## single v3 划分

先逐条按原生六动作重算并过滤，再按真实有效成员做单独的物理组覆盖修复：

- 原val/test物理组固定，绝不挪回训练。现有曝光清单与原生父库development组的并集禁止进入新holdout；若已有holdout违反这个条件，明确拒绝。
- 只在val/test缺事件时，从原train中未列入上述清单的完整物理组补充。选择只使用有效性、事件存在、物理组身份哈希及训练容量，不使用动作取值、RGB内容或模型分数。
- 移动的是整个有效物理组，包含其全部事件和锚点。记录`parent_split`，保留原始证据ID；原父库及旧产物不重写。
- 每次候选移动先保护每个训练事件的1,280个不同帧，再执行原生跨事件共享帧容量规划。共享帧冲突时回溯其它组；状态预算用尽报unknown，完整无解报infeasible。失败不发布部分移动，正式ready继续拒绝。
- 划分证明、原／新assignment、来源清单、移动组、训练预算及搜索状态均内容绑定；重载核对父库原split和物理组。准入和训练checkpoint间接绑定同一证明。

默认仍是三个split各十事件的最低覆盖要求，不把最少一个有效样本说成稳定评测。新增各split每事件的有效查询数、物理组数、类别质量及稀疏支持提示。少于2组、20查询或2个观察类别只是明确标注的诊断线，不是偷偷修改准入门槛。

“未列入已知曝光清单”并不证明未知远程训练从未使用过该组。新划分仍明确 `independent_holdout_claim=false`。如果有额外已知曝光记录，可以在 `cpu single` 或 `cpu all` 时多次传 `--exposure-manifest`；格式沿用 `{"additional_development":{"groups":["Scenario/physical_route_group"]}}`。这些文件的内容和路径绑定准备缓存；变化不能沿用旧ready。

## 地图解析

统一CPU入口先选择一个完整地图源，再把平坦视图传给single/abc，避免改动冻结的abc v2运行时。优先级为：

1. 当前应用的 `CARLA_0915/CarlaUE4/Content/Carla/Maps`；兼容顶层OpenDrive与各Town/OpenDrive。
2. 当前应用内 `lead/3rd_party/fail2drive/toolbox/carla_xodr`。
3. 应用父目录内 `lead/3rd_party/fail2drive/toolbox/carla_xodr`。

本机实际采用第3项，已确认十二Town齐全。远程若项目CARLA地图完整仍采用第1项；本机缺项目CARLA目录不代表远程缺地图。

可用统一CPU入口的 `--maps` 指定完整平坦XODR目录或CARLA Maps根目录，`--geometry`指定已有可验证父库。显式地图路径不成功时不偷偷回退；不混拼多个不完整地图根。重复Town内容冲突、旧视图内容不一致或解析过程内容变化均拒绝，来源及SHA写入map_resolution.json；建库仍按父库地图SHA校验。旧abc v2模块的历史独立CPU命令保持原样；本次默认路径修复适用于统一入口，新single v3直接CPU入口也使用该解析器。

## 真实证据与边界

本机复算三个事件的全部当前父库候选，共425条逻辑路线：

| 事件 | train有效锚点 | val有效锚点 | 旧test有效锚点／查询 |
|---|---:|---:|---:|
| LEAD_BRAKE | 3164 | 34 | 0／0 |
| SIGNAL_FAILURE | 1430 | 9 | 1／1 |
| DYNAMIC_CUTIN | 1346 | 3 | 3／12 |

跟车制动旧test帧346、347、349、350仍全部无效，没有编造标签。当前绑定曝光清单之外有14个具有有效跟车制动监督的train物理组。按新规则身份哈希顺序，首个候选是 `HardBreakRoute/Town12_4357_0`：58个有效锚点、219个查询，父库内只有LEAD_BRAKE事件，移出后该事件train剩3,106个不同帧。因此找到了具体可用的监督来源；正式全池后处理还会对所有事件、全部组和共享帧容量作最终决定。

这一复算不是完整single全池准入，不能据此宣布五类正式流程全通过。信号灯故障旧test一个查询、动态切入旧test一个物理组的问题仍保留为警示。新增真实规则合成三split十事件的测试确认：旧test六动作全无效时，新版本补充合法train组，父库字节不变，新库可重载与缓存复用；禁止全部供体时仍拒绝。

完整本机证据见 `checkpoints/phase3_single_review_fix_20261023`，精确回执见 `phase3_single_review_fix_verification_20261023.json`。未GPU、未在线、未新RGB边界视觉审查，abc边界及九类内部softmax loss未变。当前未启动正式全池CPU建库，不产生新版正式ready。

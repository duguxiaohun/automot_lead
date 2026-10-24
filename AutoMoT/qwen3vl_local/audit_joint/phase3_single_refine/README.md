# single RGB refinement：独立快照与成对提示词实验

本入口针对收到的 `phase3_single_v3/six_1280` 最终第三轮，保持 A–F、离线历史、当前车道左右参考、四个未来查询及全词表 loss。它不读取 base 的新提示词作为自己的规则，也不改旧生产入口、ready、标签、采样或权重。

当前同时提供独立候选训练流水线及同权重推理实验，**尚未证明候选性能提升**。`original` 精确复现收到版本的文本；`clarified` 消除方向参考冲突，说明历史取后一个原生起点、连续低速/起步例外、每个未来起点的判定窗口。模型仍只接收当前 RGB，轨迹和未来图像只用于离线审计。

`frozen/` 保存收到版本的规则、历史填补、窗口压缩、左右参考转换、原提示词、RGB不确定清单和源隔离规则。`provenance.json` 保留原来源 SHA 和独立副本 SHA。独立副本改了导入路径，规则身份自然不同；隔离校验器也绑定独立规则，绝不伪装成旧源码 SHA。隔离账本是本机文件快照，不冒称它属于收到的162份Python源码。


## 新增完整训练流水线（2026-10-24）

本机正式CPU检查已经通过：204910有效锚点，三轮各12800呈现/3200更新，共9600；实际解码35231个将被计划训练/验证/测试/保护样本使用的输入。标签、划分、事件配额、未见优先及重复上限保持。动态切入供给1346，距每轮1280仅多66；远程若原始数据不同，必须在该机CPU阶段实查，不能降低阈值或复制本机ready冒充通过。

代码同步到保有项目资产的机器后，在AutoMoT应用根目录用已激活环境执行：

```bash
bash qwen3vl_local/audit_joint/phase3_single_refine/run.sh all
```

默认复用项目内Python环境、lead_data、Qwen3.5-4B和原single v3已验收数据；原native ready不存在时才调用原single CPU准备。独立输出是 `checkpoints/phase3_single_refine_train_v1`，不会覆盖旧six_1280、base或原权重。此命令会在全部检查通过后实际开训；需要只准备时用 `cpu`。

顺序为 CPU准入 → 模型/磁盘/四卡检查 → 独立四卡probe → 三轮训练 → 逐轮权重验证/训练保护对照 → 最终第三轮test → 完成验收 → 自动ZIP。GPU自动按空闲显存选择四张，也可沿用GPU_IDS；同一批卡统一传给torchrun。显存下限只是早期检查，实际是否能运行仍以probe为准。

probe会用真实processor检查三轮计划成员、全部val/test及保护样本，随后执行真实前向、反向、allreduce、保存和重载，输出到独立probe目录。正式训练重新初始化模型和seed，probe更新不计入9600步或训练曝光。

本候选将验证移到三轮优化完成以后，分别加载三个epoch权重，保留逐轮指标；固定第三轮test政策不变。这样评测异常不会使已完成训练丢失。训练平均loss为跨rank平均；每100步保存进度、当前rank0 loss、轮内ETA与完整短曲线，解释写入progress.jsonl。

```bash
bash qwen3vl_local/audit_joint/phase3_single_refine/run.sh status
```

状态包含阶段、失败原因、训练进度和审计包位置。首次CPU原始来源SHA校验可能较久；随后RGB解码每2000个打印进度。GPU日志在独立根目录logs下，并同步打印到终端。

验证或测试失败后可只补评测，已成功的逐轮验证由回执校验后复用：

```bash
bash qwen3vl_local/audit_joint/phase3_single_refine/run.sh evaluate
```

仅补验收和打包，无需GPU或模型加载：

```bash
bash qwen3vl_local/audit_joint/phase3_single_refine/run.sh package
```

再次运行all/train不会覆盖已完成训练；同目录使用互斥锁。未完成的正式训练或probe仍保留并拒绝隐式续训，需要先审查失败原因；不把部分权重误报为完成，也不偷偷从零重训。

自动交付路径：`checkpoints/phase3_single_refine_train_v1/delivery/single_refine_audit_<内容指纹>.zip`。结束时打印 `AUDIT_PACKAGE`、`AUDIT_DIRECTORY`、`AUDIT_BYTES`；`delivery.json`也保存完整绝对路径和SHA。硬上限为 **30,000,000字节**，包含完整test、三轮val/保护样本逐例结果、loss曲线、采样/曝光、模型回执、准入和源码。排除权重、RGB和完整日志，不删除核心case来压缩。先做大小演练，最终ZIP再次做大小/CRC/成员SHA检查，内容寻址文件名避免覆盖不同交付。

本机大小演练给全部4008次评测成员各预留4096字节不可压缩预测空间，再留6MB公共材料，总估计25.84MB。实际发布仍以ZIP大小检查为准；估计不是任意输入的数学上界。

已有53项审计测试加新增流水线测试，共74项相关CPU回归通过；包含真实小Qwen训练/保存/验证/test及真实四Gloo优化/异常协调，GPU进程编排测试替身明确。尚未执行4B四卡probe或正式训练，不能承诺未知远程硬件/文件系统永不失败。旧同权重提示词实验入口如下，仍可单独运行。

## 文件职责

- `audit.py`：全本机有效池结构复算、收到的全部test原始meta重算、分层选RGB及图册。收到test是开发证据，不再作为未曝光评测。
- `pool_replay.py`：从原始meta和地图重建**全部既有有效成员**的历史/未来标签，输出独立压缩库；不重新枚举父库已排除的候选，不发布训练ready。
- `objective.py` / `prompt.py`：原提示词与澄清候选；现有A–F token不扩词表，真实processor检查上下文，不截断；每次前向关闭KV缓存并清理rope状态。
- `experiment.py`：完整val、同adapter、六种输入模式成对推理；另设按事件×纯动作、每层最多四个不同物理组的train回归对照，分开报告。
- `regression.py`：每一个旧对变错、修正旧错、事件×动作、各时距、各类precision/recall/F1、物理组宏平均、历史冲突及转换检出/误报。净增长不抵消退化；同时报告原始全词表格式与未限制输出，避免事件解码限制掩盖退化。

## 在持有本次完整训练数据和权重的 AutoMoT 应用根目录运行

使用当前已激活Python环境；默认复用项目内 `lead_data`、`checkpoints/Qwen3.5-4B`、`checkpoints/phase3_single_v3/ready_1280.json` 和 `checkpoints/phase3_single_v3/runs/six_1280/epoch_3/state.pt`，无需重新填写这些路径。文件必须实际存在且匹配回执；本地收到的审计ZIP未含state.pt和完整4B权重。

```bash
python -B -m qwen3vl_local.audit_joint.phase3_single_refine.experiment prepare
```

完整来源校验后固定val全部成员、独立train对照、RGB、代码、模型/权重期望SHA和输入文本。输出默认 `checkpoints/phase3_single_refine/plan`；拒绝覆盖。不得用本机573条val替代远程模型绑定的另一套val。

确认选定GPU空闲后运行（默认物理GPU 0；可通过GPU_IDS选一张卡）：

```bash
GPU_IDS=0 python -B -m qwen3vl_local.audit_joint.phase3_single_refine.experiment replay
```

输出默认 `checkpoints/phase3_single_refine/replay`。每个成员依次运行 original、clarified、两者去历史、两者黑图六种模式；每次重新前向，不自动训练或覆盖既有产物。

`regression.json` 保留完整val指标；`guard_regression.json` 单独报告已曝光train对照。任一旧正确丢失或分层下降，返回 `tradeoff_review_required`、CLI退出2。完成回执只表示推理完成；不表示效果通过。去历史/黑图属于分布干预，不能直接当作因果证明。

要重新做本机独立全量标签审计，选择一个尚不存在的输出目录：

```bash
python -B -m qwen3vl_local.audit_joint.phase3_single_refine.pool_replay \
  --output checkpoints/phase3_single_refine/local_label_rebuild
```

默认复用已建库的 `checkpoints/phase3_single_v3_maps`。源库或原始meta变化、独立快照不匹配、成员非连续、标签/有效性不一致均不会报告重现成功；本命令无正式训练准入含义。

## 验证与决策边界

初版53项相关CPU测试通过；新增流水线后的74项总回归见上文。初版包含小型原生Qwen单图四查询前向/反向与token检查、提示词交替重放、上下文拒绝、隔离规则身份、成对统计与train/val分离。编排测试使用合成数据/替身的范围在测试中标注；不能称4B GPU验收。

验证集没有纯左变道；train对照可发现明显回归，但不能证明左变道泛化。暗光、稀疏事件、重复开发曝光、预测起点之后尚不可观测的交通变化仍是限制。同权重成对实验可单独检查文本影响；独立训练流水线则用于验证重新训练后的效果。两者均需审查旧正确损失与分层退化，不能仅凭平均指标宣布改善。此入口不改变旧训练默认值；独立训练ready只在本机实际CPU检查通过后生成，不代表4B GPU已验证。

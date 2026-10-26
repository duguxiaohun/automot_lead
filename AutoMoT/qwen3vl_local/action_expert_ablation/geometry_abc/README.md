# ABC 区域条件轨迹消融

2026-10-26：已跑完七轮却在性能汇总报进程组未初始化时，使用[完成恢复入口](../../ABC_COMPLETION_RECOVERY_20261026.md)，先核验已训结果，避免重训。该入口只支持文档指定的旧版本，默认继续原六组。

只训练 BEV 的 a/b/c：使用 `run.sh all --branch bev_only`；三组探针、测试及交包同步限定。完整恢复命令见 [BRANCH_SELECTION_20261026.md](BRANCH_SELECTION_20261026.md)。默认仍为六组。

2026-10-26：主机 OOM 修复、全池内存验证与恢复命令见 [MEMORY_FIX_20261026.md](MEMORY_FIX_20261026.md)。保存后 SIGKILL 的旧探针需用新版本重新执行。

独立于旧语义 Action 实验。固定使用 `audit_joint/geometry_abc_v2` 的标定和提示词，运行 `a/b/c × bev_only/qwen_simple` 六组；不引用路线一的新提示词，不修改旧训练入口、库或权重。

四个未来区域为数据集真值，时距为 0.5/1/1.5/2 秒，测试的是条件表示的上限。两个分支都使用当前速度、Phase3 定义的导航三点、当前车道参考方向，以及过去 -2/-1.5/-1/-0.5 秒的位置区域。历史缺失为独立 UNKNOWN，不补成近区。Qwen 只接当前一张 RGB，BEV 使用当前 RGB/LiDAR；导航实际数值不再采用旧 Action 的 route-lookahead。未来区域只进 decoder，不作为文本答案进入 Qwen。尚未接 Phase3 预测器或 CARLA 闭环。

## 本地准备与运行

在 `AutoMoT` 根目录执行，`PYTHON` 可指定完整环境解释器。准备不需要模型、GPU 或 CARLA server，但需要可导入的 CARLA 地图库、Pillow、NumPy、laspy/lazrs 及原项目依赖。

```bash
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh prepare \
  --data-dir checkpoints/abc_trajectory_v1 \
  --maps ../lead/3rd_party/fail2drive/toolbox/carla_xodr

bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh status \
  --data-dir checkpoints/abc_trajectory_v1

bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh check \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth

bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --data-dir checkpoints/abc_trajectory_v1 \
  --output checkpoints/action_expert_ablation/geometry_abc
```

`all` 自动建立 `run_<tag>` 和 `latest`，依次完成来源/划分/七轮计划校验、实际 processor 全池长度检查、打包演练、六组四卡短探针、六组训练、完整 test、条件干预审计与打包。所有短探针通过后才开始第一组长训练。默认自动选四卡，也可设置 `GPU_IDS`。默认权重路径如上，可显式覆盖；不会联网下载。`check` 只做 CPU 准入并打印目录，缺模型时退出 2，不能把数据就绪称为运行就绪。

已完成 `check` 的目录可用 `all --run-dir <打印的 run 目录>`；此操作仍重新核验来源。`probe --run-dir ...` 只执行六个短探针。失败的正式训练目录保留证据，不能用同一目录隐式续训或覆盖半成品。训练已经完成时，可用 `evaluate --run-dir ...` 补测试/审计，或 `pack --run-dir ...` 只补审计打包；均需传正确的 `--data-dir`，前者还需原模型路径。已完成组的完整性仍会被检查。

## 数据和预算

按当前 Action 代码的物理组哈希划分（seed 2026，val/test 各 0.1），应用代码已声明的开发曝光隔离。用户未保存历史 Action 结果，不绑定不存在的旧 run，不声称和旧实验逐成员相同，也不认证未知外部曝光。全有效性过滤后才检查每事件 holdout 至少 32 帧/5 物理组；必要时只移动完整、未被开发曝光保护的 train 组，保留父划分和移动报告。

从原始事件映射构建；不用旧六动作有效性筛选 ABC。背景须有确认的常规事件证据。每个保留样本实际解码 RGB/LiDAR，要求 2 秒连续真值及四个区域都有效；不对缺未来补零或末点复制。逐路线保存拒绝原因和原始内容 SHA，缓存复用必须同时通过源内容、文件成员和合同校验。改变准备源码或数据后使用新的输出目录。

每组 7 轮，每轮 116,256 次呈现：十事件各 9,688，背景 19,376；四卡 B1、累积 16，最后 32 个全局样本按实际分母更新，每轮 1,817 步，共 12,719 步。整数联合分配限制跨事件共享物理帧每轮最多 11 次，优先覆盖未见帧；区域偏好为 natural/sqrt 各半、权重上限 4，同时考虑 a/b/c 与时距。六组共用成员和七轮顺序。训练保存实际逐样本消费并与计划逐轮比对。

沿用旧 FM、优化器、EMA 和验证选模训练循环，验证保持事件等额，最终完整 test 包含背景。旧语义条件结果只能作历史参考；四个未来区域比旧单语义 token 提供了不同的信息量，不能据此单独断言表示本身优劣。

## 自动交付

主包为 `<run>/delivery/abc_trajectory_audit.zip`，严格不超过 **30,000,000 字节**。最后打印结果目录、包路径、大小、SHA256，并写 `.receipt.json`。打包前重新核验七轮消费、checkpoint、完整 test 成员、真值及指标。

包内包含全部成员/标签的精确整数 NPZ、六组全部 test 的四项 ADE/FDE、事件/物理组统计、配置/合同/完成回执、embedding 诊断和同噪声区域替换/时距置换诊断。浮点压缩仅在精确往返相等时用 float32，否则保留 float64；不删除难例或抽样冒充全量。全部轨迹、权重和完整预测文件清单保留在结果目录，包内绑定其 SHA；按时距误差复算需这些完整预测。无未来条件是分布外诊断，不是训练过的无条件基线。

另报 split×event×表示×时距×区域的完整支持量和物理组数量，包括零支持项。验证诊断按未来/历史、四个槽位、区域覆盖选样，每个已有支持的组合优先四个不同物理组，并补事件代表；同一成员只推理一次。原区域整体替换、时距置换之外，新增八个逐槽位替换，固定样本和 FM 噪声，报告每格响应比例、分位数和物理组支持。任一已有支持格内出现不超过 1e-6 米的孤立槽位响应，标记 `condition_use_review_required`，交付完整包后退出 2，不能由另一个样本有响应抵消。无支持格为 N/A，少于四个物理组明确列为限制，不单凭支持少拒绝。零响应仅要求复核，不证明坍塌；非零响应不证明精度。历史干预只替换 decoder 的历史槽位，Qwen 观测文本保持固定；UNKNOWN→区域0单列为缺失性干预。全四标签相同的样本不能用于判断时距置换是否生效。

CPU 准入还重放所有缺失历史，记录原生缺失原因；模型输入保留 UNKNOWN/mask。训练沿用原计时器并补各 rank 峰值显存/整体吞吐，测试记录同步后的前向时延及排除各 rank 首次前向后的统计。低当前速度/未来运动、低当前速度/小位移、末段低速度等分层按预先固定的真值条件报告，不能用于约束预测。

输出轨迹也会按当前参考重标区域。原生 v2 标签来自完整 xyz 的世界水平投影，轨迹 decoder 只预测 ego xy，因此重标诊断明确采用预测 z=0；同时单列 GT 在相同零高度投影下与原 oracle 的一致率，不向预测借用未来真值高度、不修改原标签。包内保留全部重标区域整数，可重算区域一致率；逐时距误差、运动分层和逐例时延的独立复算需结果目录中的完整案例。

`prepare` 用真实成员规模与六组不可轻易压缩的模拟误差演练 26MB 上限，留 4MB 给最终回执。演练包不是模型结果。核心结果超限就拒绝发布，不能默默丢数据。失败时另输出小型诊断包。

本机 CPU、小模型和 Gloo 验证不替代真实 Qwen/BEV 四卡探针，也不证明精度提升。实际验收证据见 `checkpoints/abc_implementation_20261024`。

## 2026-10-24 独立审计后的修复

运行时加载器现在读取并散列当前 metadata、RGB 和 LiDAR 的实际字节，与准入绑定的 manifest/路线 receipt 对照后，从**同一份已验证字节**解码。metadata 包含连续未来和 route 目标；这些内容也必须一致。不会先 hash 路径、再重新打开已被替换的文件。每次消费都检查内容，缓存仅保存已验证的 receipt，不用 mtime 代替内容散列。数据根路径可迁移，但迁移后的实际内容仍必须完全匹配；sample 不能单独绕到另一个根目录。

每轮实际消费增加逐样本来源验证计数，完整 test 案例带三类实际输入 SHA，最终打包重核来源证明与计划消费一致。准入额外绑定完整 manifest SHA；修改 manifest/receipt 不能给变化的输入重新标记旧身份。

正式训练用 optimizer 的成功 step 后钩子、在 `zero_grad` 前采集已 DDP 汇总并裁剪的十行 token 梯度，保存时使用已采集值。每个 rank 保存更新数、累计梯度范数及未来/历史各槽位的区域曝光；共享 embedding 梯度不声称是两种角色的因果贡献分解。最终四 rank、12,719 步的诊断及文件 SHA 必须完整，否则不能交付成功包。

标定、成员、划分、七轮计划和数据构建源码未改，现有 `abc_trajectory_v1_local` 数据可继续经校验复用。运行源码改变，旧准入/probe 不兼容：在新输出目录重新执行 `check`，再在模型齐备时执行 `all`。修复验收见 `checkpoints/abc_review_fixes_20261024`。完整 processor 的本地耗时和真实四卡大模型能力仍需模型文件齐备后实测。

## 第二轮审计：条件覆盖及最长输入

processor 全池结果现在分别保存 a/b/c × train/val/test 的最大文本长度、最大视觉 token 数、成员 ID 和 image_grid_thw。每组四卡反向探针选本表示的训练集两类极值（去重），再以稳定 ID 补足四条训练样本；val/test 极值只由未包 DDP 的模型在 eval + no_grad 下前向，不参与 optimizer 更新。Qwen 实际 prefill 处记录并核对 processor 形状，反向前即拒绝与报告不一致的极值。六组均保存实际 RGB、LiDAR 点数及 decoder BEV/KV 形状。BEV 使用相同选样和真实形状记录，不将文本长度称为 LiDAR 点数极值。

正式开训及最终打包重新核验选样、split、processor SHA、四 rank 消费回执及 holdout 更新数为零。所有回执收入主包；容量演练加入六组新诊断完整选样报告，保留 4MB 余量和全部 test。旧 processor/probe 因源码及字段改变不能复用，需新输出目录重新 `check`/`all`，不能修改旧 SHA。当前代码和全池验证证据见 `checkpoints/abc_coverage_probe_fixes_20261024`；本机缺完整模型，不能把 CPU 检查当成真实四卡最长输入验收。

## 2026-10-25 初始化约束

保留 BEV 感知预训练权重；Qwen 分支使用原始 Qwen3.5。区域条件嵌入和 Action decoder 从头训练，不加载旧 Phase3 LoRA、旧 Action/decoder。worker 显式拒绝旧 adapter/init/resume，train/probe 不接收 checkpoint；evaluate 只接受本实验合同与完成证据通过的本轮权重。自身探针的保存重载核验保留，探针权重不进入正式训练。worker 源码更新后重新运行 check/probe，旧准入不直接复用；数据标定/划分/七轮计划未改。

## 准入期间更新代码的恢复

run.sh 现在自动固定本次源码版本。尚未正式训练的旧失败运行用 `all --recover-from <旧run>` 创建新运行，重新校验来源/计划/资源，按精确白名单复用或重做 processor；旧 probe 不复用。正式训练目录存在时不自动重训。具体命令和边界见 [运行版本说明](../../RUNTIME_VERSIONING_20261025.md)。

首次保存后继续训练的同步修复与重跑说明：[CHECKPOINT_FIX_20261025.md](CHECKPOINT_FIX_20261025.md)。新探针包含保存后继续前向/反向，旧 probe 不复用。

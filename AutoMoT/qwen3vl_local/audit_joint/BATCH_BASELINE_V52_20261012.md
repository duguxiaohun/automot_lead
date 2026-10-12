# v52：累计回放索引修复、自动规则首轮入口与RGB续审

服务器readiness继续关闭，不重复旧fit、环境检查或磁盘盘点。本轮修复v51复审的确定缺陷，并固定自动基线入口；没有修改生产标签、阈值或提示词。当前合同v52、同预约计划v32、空严格registry v27，老师算法仍v17。旧v51产物和失败复现保留，不改hash兼容旧adapter。

## 1. 累计索引与发布保护

`teacher_replay.produce`现在在进程间排他锁内核验既有索引及全部回执，只计算尚未登记的请求路线，合并新旧回执，按冻结候选池顺序原子写入累计index。原文件损坏、合同/编码不一致、并发构建均拒绝；失败不替换旧index。不得删除`.production.lock`绕过活动构建。

共享引用安装在相同写锁内捕获源文件并写入`published.json`，封存准确的index SHA。已经发布的集合（包括显式发布的部分集合）不得追加路线；请求已有子集/重排/全集只核验，既有字节不变。扩展已发布集合应新建输出目录。发布标记是协作协议，不是操作系统写保护；手工修改文件仍由SHA核验拒绝。

**因此应先完成全部分批回放，再编译共享数据集。** 提前把部分回放安装为共享库会将该部分集合封存，后续追加会失败；这不是要求放宽准入。

本机使用真实原生代码、新目录5训练路线：先2条返回partial，再3条累计5条并complete。编译单份RGB4共享库后，再请求2/3/5条三次，每次index仍5条、源目录各文件SHA全部不变，短视图与共享库均重新加载成功。另有单元反例覆盖倒序批次、plain/gzip、旧回执损坏、部分集合封存后禁止扩展及构建/发布互斥。

## 2. 自动首轮的唯一推荐入口

采用 **单份原生RGB4规则库 → 短视图 → RGB4自动规则留出 → `rgb_short/run_automatic.sh`**。原生双库总入口保留原用途，本轮没有删除其人工参考准入，不把它用于此自动规则实验。自动留出不需要补全人工标注，衡量规则一致率，不是独立标签正确率。

新只读检查器`audit_joint.rule_baseline`在加载实际短视图、过滤规则train、排除老师独立审核预约和短输入冲突之后，调用与训练相同的`evaluation_policy.prepare`：

- 核验完整回放/未按路线截断、所有合格池的物理组隔离、十事件训练和val/test固定无放回等量；报告不会把原生人工val/test的统计当作规则留出统计。
- 原生计划首轮每事件1024次，总10240次，world1/4题序一致，两模式题目身份及题序一致。
- 分报题数与物理组；首轮门要求U-E2 return、U-E3/5/6 proceed、U-E7 complete的readiness YES与NO均有规则train支持（至少一个独特题/组，不等于独立质量认证）。只报告YES:NO分布，不要求1:1。
- 完整首轮预算必须是2模式×各保留1个checkpoint，保留至少20GiB。上限/证据缺项保持阻塞；启动时读取当前目标盘余量并取已声明余量的较小值。该读数是运行保护，不要求用户重复交磁盘盘点包；仍不保证操作者声明的容量上限正确。
- 检查开始/结束绑定数据及外部回放路径/SHA和参考包。报告以题数、SHA及条件统计为主，不重复嵌入全池train ID数组。

`run_automatic.sh`先运行上述检查，失败退出2且不调用GPU；通过后固定规则模式、seed=20260929、epochs=1、epoch_samples=10240，调用既有四卡训练与新进程重载链路，最后对本次`epoch_000`运行固定等事件test评测。使用本次RUN_TAG路径，不读取可能被其它会话更新的latest。首轮不接受额外训练参数，避免40题短跑、reviewed模式或7轮覆盖固定预算。

预算门接在这个训练入口前；它不自动授权/启动全源建库，也不替代建库阶段的容量预算和磁盘守护。原生工具和其它历史入口仍可独立调用，不能声称全仓库所有命令都强制经过新门。

## 3. 本机结果与RGB边界

本轮专用tmux、低优先级单CPU、core=0、6GiB地址空间上限，5路线755帧/1116原始问题，原始语义及Phase3同帧动作/证据与v50完全一致（排除源码派生ID后逐条比较）。68.87秒、峰值262300KiB，退出0。原生只建RGB4：464 train、空val/test；短视图RGB2 `[-2,0]` / RGB4 `[-6,-4,-2,0]`各464条，身份配对、四图跨度不变。自动参考导出/加载链路可运行，但该样本的参考为空；新准入按预期报告`empty val/test`及`missing first-epoch budget`，不是正式池通过。

完整回归 **2302 passed，121.89秒**；固定test出口补接后追加4项入口测试通过，其中1项新测试，使用命令桩核验调用顺序/参数与阻塞前不触发GPU，不冒称服务器训练实测。最后的短视图在`view_final`按最终学生源码重新建立并原生加载，旧view留存。机器结果见`batch_baseline_v52_verification_20261012.json`。

实际查看5联系表、70个不同RGB帧，无新增原图放大；不是全路线目视或G2。与v51复审70帧窗口不同，其中部分与更早历史复核重叠，不宣称新增70个历史未见帧。选窗210份源SHA；U-E5补充10帧参与者追踪30份源SHA，与选窗去重后共219份。

| 路线窗口 | 观察与同帧原生证据 | 结论 |
| --- | --- | --- |
| U-E2 / Town01 001275，f100–113 | 对向蓝车经过，施工障碍已在画面后方；原生此前导航f92已截止，此窗没有旧U-E2 return题 | 不将“驶过障碍”倒填旧实例return YES；正式池仍需另外的返回实例 |
| ParkingCutIn / Town12 1681_0，f122–135 | 前方车辆与路边队列可见，自车随后接近并等待；原生此处生成U-E1/3697/122跟车，proceed先NO/UNKNOWN，128–135为NO | 后继跟车不能当作旧U-E3放行/完成正例；按实例和事件分开 |
| U-E5 / Town12 1450_0，f64–77 | 红色车辆先后经过；来源追踪确认49的Audi为3752、64的Mercedes为3753、70出现的Dodge为3754，不是同一辆车消失后重现 | 不能按相近颜色/场景名合并参与者；旧3752在55后不可见、56截止不回填完成；此窗无旧U-E5题 |
| U-E6 / Town03 8_107，f68–81 | 驶出路口后多辆对向车经过；原生此前65已截止旧路口实例，此窗无旧U-E6题 | 场景还在动不代表旧事件仍应出题，更不自动补proceed YES |
| U-E7 / Town03 002134，f41–54 | 横穿车向右离开，自车随后通过路口；雾雨中仍有其它车辆；同实例complete在43/48/51–53为NO，54为YES | 保留稳定进展门及可观察性限制；没有足够新证据调整阈值或批准短输入可判断性 |

这轮没有发现需要修改标签/提示词的新确定缺陷。其余事件沿用历史证据，不宣称十事件全源逐帧审核完成。

## 4. 正式池完成后的执行步骤

以下是当前源码的接口顺序，**不是让服务器现在绕过尚缺的全源预算直接全量建库**。新合同的正式候选池、回放和容量上限确认后再执行；所有路径都应是v52新产物，不复用v42/v51 manifest。服务器readiness不用再跑。

1. 冻结正式候选池。在同一新回放目录顺序执行`teacher_replay --compression gzip --route-list 批次.json`，每批路线必须来自同一份候选池；最终累计complete后才能安装共享库。不并行启动多个批次写同一目录。
2. 只编译RGB4规则库，导出自动参考并创建短视图，示例在AutoMoT根目录（P4_NEW/DATA_ROOT为已确认的新产物与原数据路径）：

```bash
export P4_NEW=/实际路径/phase4_v52_rules
export DATA_ROOT=/实际路径/原始数据
python - <<'PY'
import os
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase4.full_pipeline import default_registry
from qwen3vl_local.sft_new_loop_phase4.identity import write_json
write_json(Path(os.environ['P4_NEW'])/'weak_registry.json',default_registry())
PY
python -m qwen3vl_local.sft_new_loop_phase4.dataset \
  --candidate-pool "$P4_NEW/candidates.json" \
  --production-index "$P4_NEW/full_production/index.json" \
  --teacher-registry "$P4_NEW/weak_registry.json" \
  --max-teacher-questions-per-route 0 --rgb-mode 4 \
  --data-root "$DATA_ROOT" --output-dir "$P4_NEW/data4"
python -m qwen3vl_local.sft_new_loop_phase4.rgb_short.build \
  --dataset4 "$P4_NEW/data4" --output "$P4_NEW/short"
python -m qwen3vl_local.sft_new_loop_phase4.teacher_evaluation \
  --candidate-pool "$P4_NEW/candidates.json" \
  --production-index "$P4_NEW/full_production/index.json" \
  --data-root "$DATA_ROOT" --rgb-mode 4 --output-dir "$P4_NEW/references4"
```

3. 使用完成的首轮预算先运行只读准入，交回JSON；无须先训练。缺条件支持时补真实来源，不能改答案或用重复采样冒充覆盖。

```bash
python -m qwen3vl_local.audit_joint.rule_baseline \
  --dataset "$P4_NEW/short/view" --teacher-references "$P4_NEW/references4" \
  --budget /实际路径/first_epoch_budget_request.json \
  --space-path /实际路径/新实验目录 --output /新的路径/automatic_admission.json
```

预算在数据建好后的当前余量基础上计算新增占用；已经落盘的库不重复算新增，相关分项可显式填0并注明实测依据。优化器、两份adapter、日志/报告/临时峰值仍需上限，不能用空字段当0。多挂载点分别预算；本入口按输出盘检查，不认证其它盘容量。

4. 准入通过后每模式各一轮（固定入口已经自动重载、固定test评测；不能用旧adapter续训）：

```bash
export DATASET="$P4_NEW/short/view"
export TEACHER_REFERENCES="$P4_NEW/references4"
export FIRST_EPOCH_BUDGET=/实际路径/first_epoch_budget_request.json
export MODEL_DIR=/实际路径/Qwen3.5-4B
RGB_MODE=2 OUTPUT_DIR=/新的路径/v52_rgb2 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
RGB_MODE=4 OUTPUT_DIR=/新的路径/v52_rgb4 \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh
```

反馈应包含：准入/预算JSON、manifest与引用描述、自动留出references.json、student_view.json；训练后再交两模式run.json、采样计划与执行记录、epoch_000指标/cases/checkpoint回执、reload_check报告/cases、test_fixed指标/cases。沿用≤30MB交接白名单；权重、optimizer、原始RGB和完整回放不打包。新增准入JSON可单独附上，不声称历史打包器已经自动发现所有新文件。

当前正式Phase3索引、Phase4全池/自动留出及真实首轮预算仍待服务器交付；这不要求重做readiness或旧拟合，也不要求先完成人工全量审核/G2。G0未完整通过、G1继续、G2未运行。本机未远程执行、GPU训练或push。

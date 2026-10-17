# v71：评估连续段、重复上限与正式全池准备

> 后续v72已修正段内action评分及开发曝光缓存绑定；已知v71 view也可升级，须使用新输出目录。每卡B2不变，详见 [ACTION_EXPOSURE_FIXES_20261017.md](ACTION_EXPOSURE_FIXES_20261017.md)。

每卡B2选择不变：Phase3累积2、四卡16样本/update；Phase4累积4、四卡32样本/update。没有新增batch比较、GPU试验或模型效果验收要求。

## 修正内容

共享评估选择器升为fixed_equal_contiguous_evaluation_v2。按实际stride在所有路线中优先选择最长连续锚点段；同帧问题保留为候选，取走一题后其余候选仍可选，不再把同帧多题人为视作断点。两图stride2、四图stride1决定成员选择及顺序，不仅用于最终排序。十事件等量、无放回、标签盲选、固定seed和输入顺序无关保持。8连续锚点优先于孤立点+7；原8帧片段限制已去除。

各模式按自己的时间间隔选评估段，因此两图和四图的val/test成员可能不同；不能再把结果当作逐题配对对照。训练候选成员仍严格一致。Phase4正式准入已只对训练成员作跨模式相等核验，并继续保存各模式评估成员SHA/事件配额。Action共享选择器也获得连续性修正。

连续训练策略的max_question_repeat变为每题每epoch硬上限，进入history绑定；改变或取消上限拒绝混用旧history。每事件容量不足在模型加载前拒绝，不静默放宽或填充。默认None明确表示不限，维持小事件覆盖完后按等量配额重复；回执分别记录max_question_repeat_limit（配置）与max_question_repeat（实际）。旧策略未传参数时仍按原默认4调用，不改变冻结原生采样代码。

正式rgb_short数据入口新增runtime_data：

1. CPU准备时使用现有validated_reservations和可选P4_PRODUCTION_CACHE，完整校验原生库及学生成员。
2. 每个RGB模式保存一份学生数据JSON与完成回执；只有全部校验和写入结束后才发布回执，写入保持20GiB磁盘余量。
3. 准入、训练、重载和test复用同一准备目录。run.sh在torchrun前先调用CPU准备；多rank缺少准备结果直接拒绝，不允许各rank退回独立全量验证。
4. 每rank仍需读取自己的学生JSON并进行自身评估准备，但不再解析原生teacher_provenance、遍历生产路线证明。单模式正常准备只调用一次native.load_dataset；不宣称所有数据I/O或参考集处理都消失。

准备绑定包含完整当前源码合同、读取适配器SHA，以及父库/view/sidecar/索引/生产路线的逻辑和物理路径、device/inode/size/mtime/ctime。首次完整SHA/证明检查由正式loader执行；复用时核对这些输入文件状态和准备JSON SHA，读前后复核。它是可信本机的派生缓存，不是外部签名证明；源数据/RGB必须按现有约定保持不可变，RGB仍由正式encoder核对。输入或源码变化需新准备目录，不改旧回执绕过检查。参考包及跨split检查仍由原有evaluation_policy执行。

## 旧学生视图与缓存

本轮修改学生运行合同。新增phase4_view_upgrade，仅接受v61/v70发布清单中已知运行/采样源码差异；其余生产、输入、标签、观测合同不允许漂移。实际完整加载父库并核对两模式成员/排除项/RGB身份sidecar后，在新输出目录写新view，不覆盖旧view，不允许用新合同恢复旧checkpoint。

phase4_first_epoch收到已知旧view时，自动先执行该升级，再执行准入和训练；不再要求用户单独排查或重建原生库。升级本身仍有CPU工作，默认900秒上限，失败停后续。phase4_load.py和phase4_production_cache.py的源码字节未变，现有v69路线缓存仍可复用，无需重新校验8614条证明。原生producer/teacher源码及合同未改。

## 正式B2入口

同步v71后，用户决定启动正式训练时，在服务器AutoMoT目录执行以下命令。它不是短benchmark，会运行两模式既有首轮、重载和固定test流程。已完成的原生校验与batch选型无需重跑。

```bash
export PREPARED=checkpoints/phase4_v57_20261014_232157_762998141
export DATASET=checkpoints/p4_batch_measure_20261016_093447_958074352/student/view
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
export DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B
export GPU_IDS=0,1,2,3
export BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32 GRADIENT_CHECKPOINTING=1
export OUTPUT_DIR="checkpoints/p4_batch2_$(date +%Y%m%d_%H%M%S_%N)"
bash qwen3vl_local/audit_joint/train_phase4_batch.sh
```

自动升级输出位于本次OUTPUT_DIR/view_upgrade/view；准备数据位于OUTPUT_DIR/validated_data；两个模式各自run_first_epoch_rgb2/run_first_epoch_rgb4。有效当前view无需升级。默认仍检查独立references4和磁盘预算，不能把train-only原生库native_validated等同正式准入。通用run.sh也已接CPU准备；直接torchrun需先用runtime_data模块准备并设置P4_VALIDATED_DATA。

## 验证范围

已覆盖连续段反例、同锚点多题、真实步长、标签盲选、重复容量拒绝/限制/跨epoch绑定、缓存篡改/变更/磁盘不足、四个真实CPU worker不再调用原生loader、shell先准备后torchrun、已知旧view升级/未知源拒绝及正式工作流接线。

本地17路线1457题、v70历史合同fixture显式升级成功，两模式成员与原数据一致；每模式准备后四次worker读取完全一致，共两次native.load_dataset而非八次。既有v69路线缓存直接复用。没有新增目标4B四卡实测，不外推服务器全池耗时；评估生成仍主要逐题，共享RGB也仍按独立样本编码。Phase3跨进程优化器恢复仍不在本次范围内。

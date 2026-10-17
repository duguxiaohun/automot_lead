# Phase4 RGB2首轮已完成：v80结果核验与接续

> 后续状态已更新：旧125707目录被用户删除；新162511运行的RGB2训练及重载均通过，test读取快照失败。请使用[最新接续说明](PHASE4_TEST_SNAPSHOT_20261017.md)，不要再执行本文旧目录命令。

## 已收证据

Downloads的phase4_epoch_report(2).zip与带(1)副本字节相同，只计一次。3156117字节、27个成员，ZIP/成员SHA及CRC通过；ZIP SHA为e1121d716037e5fcaea47cdde9608e784aaadc2feb152539fdf97fcf1ccee849。558份源码逐项匹配v79；运行/adapter来源合同与本机当前学生合同一致，run和adapter投影记录一致。权重未随ZIP提交，文件字节仍需服务器重载检查，不能用元数据SHA声称已本地核验权重。

实际目录：checkpoints/phase4_b2_epoch1_20261017_125707_138502059。

- 视图升级成功，525.23秒，其中双模式转换146.19秒；v79原900秒超时路径已通过本次实跑。
- RGB2四H20-3e、每卡B2/累积4，完整训练一轮。四rank各2560次呈现、1280微批、320次更新。逐微批ID、顺序、rank、slot和更新数均与10240次采样计划匹配，十事件各1024。事件身份承接已记录采样计划，本地未重新加载服务器完整池。
- 验证370题，十事件各37，固定成员/顺序与run计划一致；原始输出重解析、图像/题目配对identity摘要、混淆矩阵均核对。四卡消费94/92/92/92、合并无重复，分片成员摘要匹配。
- RGB2训练4375.21秒（约73分钟）、内置验证33.41秒、保存3.59秒。整个rgb2阶段8110.30秒，其余时间包含准备、加载等；缺细分证据，不能归因全部GPU训练，也不承诺所有CPU耗时已经消除。
- 之后torchrun在解析reload_check命令时退出2：`ambiguous option: --run could match --run-path, --run_path`。尚未重载权重，尚未执行固定test，RGB4未启动。不是NCCL warning、OOM或训练失败。

## 质量与事件覆盖

| 事件 | 验证题数 | 规则一致率 | 关键限制 |
|---|---:|---:|---|
| U-E1 |37|100%|全部proceed NO；只含1物理组，未检验放行YES|
| U-E2 |37|100%|全部depart NO；只含1物理组，未检验return|
| U-E3 |37|100%|全部proceed NO；只含1物理组|
| U-E4 |37|21.62%|29道proceed YES全答NO，2物理组|
| U-E5 |37|100%|10道YES来自restrict/hold；不能代替缺失的proceed YES|
| U-E6 |37|51.35%|22道YES答对4道，5物理组|
| U-E7 |37|81.08%|3道YES答对2道，未覆盖complete|
| R-E2 |37|100%|唯一YES是hold，不代表进入/放行全通过|
| R-E3 |37|100%|唯一YES是hold，1物理组|
| R-E5 |37|100%|37道proceed NO，未验证STOP放行YES|

总体316/370=85.41%，YES召回18/66=27.27%，NO召回298/304=98.03%，全部答NO基线82.16%。这是自动规则一致率，不是独立安全准确率。连续段优先选择在部分事件只取一条路线/单一答案，事件等量成立不代表条件和路线覆盖充分。不要求YES:NO强制1:1，也不在看过结果后改动本轮固定评测清单。

采样回执显示U-E4 proceed YES实际呈现262次，U-E6 proceed YES351次，因此不能把低YES召回全部解释成没采到正例；U-E1 proceed YES本轮0次、U-E7 complete YES2次，是其它真实监督覆盖限制。软action补偿不保证每轮每子条件都有曝光，事件1:1并不自动解决它。暂不据此更改冻结采样/提示词或重训本轮；先完成同版RGB4对照及两模式固定test，再针对模型错误、输入可观察性和采样支持分开处理，不直接续7轮。

正式pool关键支持：U-E5 proceed YES仍0；U-E2 return YES2条；U-E7 complete YES13条；R-E5 proceed YES760条/182物理组。自动建库无需逐题人工审核，但当前结果不能证明所有事件能力可靠或联合G0通过。Phase3不重跑。本轮未新增RGB目视、生产标签或提示词修改。

## 两项修复及接续方式

1. 原生torchrun解析器本地复现相同--run歧义。新增外围Python启动适配，在torchrun模块参数前插入`--`；常规Python参数和解释器/退出码保持不变。首轮、多轮和本次接续均接入。未修改checkpoint绑定的evaluate.sh、batch_runtime、学生/teacher源码、view、runtime快照或v69缓存适配器，现有RGB2权重与数据无需迁移。不要直接调用未适配的旧evaluate.sh/direct CUDA CLI；用本文接续入口。
2. 旧打包器将未压缩JSON也限制为30MB。本次RGB2证据约9.89MB、准入报告10.19MB；按同样结构构造两模式即约40.16MB，旧逻辑会省略部分证据。现在Phase4未压缩处理预算128MB、最终交付ZIP仍硬限30000000字节；采样/逐步/逐题记录不删减，超出ZIP上限拒绝成功回执。其它pilot调用保留原默认。新增32MB可压缩证据保留及压缩ZIP超限拒绝测试。

81项相关回归7.87秒通过，含真实torch解析反例、两个真实CPU torchrun worker、解释器/空格参数/退出码、接续不重训RGB2、首错停止/异常恢复、坏来源/缺权重/缺消费/缺验证拒绝、保存历史workflow、交接诊断和ZIP界限。未执行服务器GPU重载，不能称所有端到端步骤已验收。

## 发布同步v80之后执行一次

本轮已获用户明确白名单发布授权；收到发布提交号后，按原sync_automot.sh的clone+rsync路径同步，不用服务器遗留.git判断版本。不再运行新建首轮的train_phase4_batch.sh命令，否则会重复RGB2。既有任务应已退出，保持原输出目录和权重。

```bash
export P4_RUN=checkpoints/phase4_b2_epoch1_20261017_125707_138502059
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
python -m qwen3vl_local.audit_joint.phase4_continue   --root "$P4_RUN"   --data-root lead_data   --model-dir checkpoints/Qwen3.5-4B
```

默认空闲四卡；入口验证原合同、实际采样消费、验证成员和权重存在性后，顺序RGB2重载→RGB2固定test→RGB4首轮/重载/test。复用已升级view和validated_data；不重建库、老师回放、RGB2训练、Phase3或readiness。旧workflow先保留为workflow_before_continuation.json，旧handoff不覆盖。中途失败即停，不自动重试；已有接续输出时拒绝覆盖，交回诊断后再按实际完成阶段处理。

查看阶段和日志：

```bash
cat "$P4_RUN/continuation.json"
tail -n 40 "$P4_RUN/continuation_rgb2_reload.log"
tail -n 40 "$P4_RUN/continuation_rgb2_test.log"
tail -n 40 "$P4_RUN/rgb4.log"
```

成功或失败交回新的handoff_continued/phase4_epoch_report.zip和handoff_continued/phase4_epoch_package.json。首次启动前源检查/原资产缺失会直接拒绝，此时交终端报错即可，不能删除门禁文件或重训绕过。

## 发布前并行本地RGB复核

仅使用既有train路线DynamicObjectCrossing/Town02_Rep0_Town02_Scenario3_3_route0_01_09_03_56_07和Town07_Rep0_Town07_Scenario3_1_route0_01_08_19_58_04，未查看固定val/test RGB。原生从路线起点重放至68/56帧，共126帧；378份RGB/meta/bbox源SHA通过。实际逐帧目视Town02的61–68和Town07的49–56，共16张不同RGB，核对16道四图proceed题。

Town02行人由右向左横穿，61–64为NO，65–67为UNKNOWN，68为YES；Town07雾天行人继续向左，49–51为UNKNOWN、52–56为YES，后段左侧篱笆附近另有行人可见。当前重算的布尔条件、答案及题ID与已有回放相同；仅12处源异常位移浮点末位差异，最大1.42e-14，诊断比较容差1e-10，未改生产SHA/数据。视觉趋势与离开局部通行走廊相容，但RGB本身不能证明所有隐藏优先权与完整安全条件，雾天可观察性限制保留。

本轮未建立新的确定改标、阈值或提示词缺陷，不扩大审核为服务器接续的前置。原图与重算证据仅留本机checkpoints/v80_release_rgb_20261017，发布轻量摘要；不是全源/盲审认证。服务器仍只执行上面的接续命令。

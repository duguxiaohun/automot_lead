# v89 本地 RGB 复核与分离资产首轮入口

结论：本轮没有发现需要先改变生产标签、阈值或提示词的新证据。已收 v88 正式池可以用于限定范围的自动规则监督开发训练；U-E5/proceed/YES 缺失仍不能作为已学会的能力。CPU 准入已经关闭，不重复 readiness、回放、原生建库或独立准入任务。尚未执行新版本 GPU 首轮，不能宣布全流程已验收。

## 入口改动

原首轮入口已经支持外部 `--view`，并非完全不支持分离布局；不足是预算固定读取 `PREPARED/first_epoch_budget.json`。现在增加显式 `--budget`，B2 shell 对应 `FIRST_EPOCH_BUDGET`，可直接绑定旧目录的 data4/references4 与续接目录的新 view/预算。未传参数仍保持原路径行为。

显式文件缺失、损坏、非对象或空间不够时不回退旧预算；workflow 记录预算逻辑路径、SHA 和服务器路径上下文，handoff 保留失败诊断。预算不能放在将创建的输出目录中。现有真实空间重查、视图父库绑定、原生准入、四卡 B2/累积4、32呈现/更新、首错停止、20GiB守护、30MB交包继续生效。显式预算暂只用于新首轮；多轮/恢复入口拒绝默默忽略它。

v89 清单覆盖585个既有发布文件，只更新外围入口/测试/清单指针；历史清单保持原样。原生老师 v24、学生合同、采样 v5、提示词均未改变，不需要新库或视图升级。当前修改仍在本机，未push；以下命令须在发布并经原有 sync_automot.sh 同步后使用。

相关回归303项通过（13.08秒），585份发布源码匹配，shell语法与差分空白检查通过。测试覆盖显式新预算、旧预算缺失、损坏预算不回退、两模式传参、失败交包及误入多轮分支；不是服务器GPU验收。

## 本地真实数据与逐帧复核

9条完整 train 路线重新因果重放627帧，1881份RGB/meta/bbox源SHA通过；编译、两模式重载各197题，无val/test。578条原始问题及实例结束记录与v88逐字段相同，生命周期结构检查通过。另对绕行/变道/完成三条路线做只读几何诊断，未回写标签。

实际查看10个连续窗口、60张不同RGB，180份目视源文件核验。与上一轮v88的60帧无重叠，与此前phase4_three的56帧重叠2帧；未宣称相对全部历史都新增。这些是已知开发路线、可见标签复核，不是独立盲审。

| 事件 | 路线简称／帧 | RGB与规则核对 |
| --- | --- | --- |
| U-E1 | Town12_346_0／36–41 | 前车及对向车可见；39帧NO，40帧历史释放边界未确立而UNKNOWN，没有误改YES。 |
| U-E2 | Town12_2091_0／45–50 | 左侧来车经过；47→48帧NO→YES。actor3964在实例目标走廊当前交集True→False，与车辆经过一致。只支持限定可见区域判断，不能据此承诺完整绕行安全。 |
| U-E3 | Town13_1009_1／34–39 | 黑车从右侧停车位驶入，35帧建立实例并NO，未提前放行。 |
| U-E4 | Town06_Scenario4_17／45–50 | 雾中骑行者仍穿过行驶区域，六帧NO；能看见目标不等于雾天所有目标均可观察。 |
| U-E5 | Town13_1148_0／51–56 | 夜间窄路对向来车接近；53帧建立实例后NO；本窗口不包含侵入后可见离场正例。 |
| U-E6 | Town07_Scenario8_19／28–33 | 消防车从右向左横穿，六帧NO，未因画面信号显示绿色而忽略来车。 |
| U-E7 | Town13_002180／26–31 | 橙车横穿后还有其他路口车辆；保持NO。不能仅看橙车离开便认定路口可通行。 |
| R-E2 | Town05_ll_2／29–34 | 目标方向LEFT；左前车间距增加，30→31帧NO→YES。actor176当前交集True→False；后续画面右侧黑车不是左侧目标走廊占用。雾中YES仍限可见条件。 |
| R-E3 | Town13_000839／17–22 | 22帧几何入道已完成，complete YES但entry_gap_clear False，hold/renewed_restriction YES；“已完成入道”和“可继续放行”分开，未把完成当加速许可。 |
| R-E5 | Town12_346_0／46–51 | 前车与STOP仍在；48帧停车义务未满足、距有效停车参照约16.69m，NO，没有因前车运动直接放行。 |

几何诊断只是当前原生规则的解释核对，不是独立真值。提示词 `route_prompts.py` 已明确可见条件不涵盖未见后侧交通，`prompts.py` 已明确几何完成不撤销纵向限制，本轮没有新依据加长提示词或放宽门槛。

## 事件等量与能力边界

本地仅验证两模式、world1/4、B2每事件1024呈现；197题小库重复多，不能作为正式池或独立评估。正式池采用上轮已核验的300694训练题，每模式每轮10240呈现；固定val每事件37题、test每事件18题。两模式评测成员不同，不可把总分当严格逐题配对比较。实际GPU消费仍须首轮日志核验。事件间1:1不要求YES:NO等额。

正式支持限制：U-E5/proceed/YES 0；U-E2 return YES 2题/2组，U-E7 complete YES 13题/13组。未知证据不补YES，实例退出不伪装成功放行。自动规则训练不要求人工逐题审核全部题库，但抽查与准入不能证明每条自动标签都正确，也不替代联合Phase3/Action G0。

## 发布同步后的首轮命令

用户任选cr02或cee；所有路径必须是当前机器自己的资产，不互相套用。下面两个已填写目录来自本次用户确认的cee包：只有当前机器实际有这对已准入资产时才原样用；换机器时替换为该机自己的已准入原库和续接目录，不自动寻找“最新”目录。

在当前机器 `AutoMoT/` 内、激活训练环境后执行一次：

```bash
(
set -euo pipefail
export PREPARED=checkpoints/phase4_teacher_v24_20261018_130500_102842314
P4_ADMITTED=checkpoints/phase4_continue_20261019_113458_023744615
export DATASET="$P4_ADMITTED/view"
export FIRST_EPOCH_BUDGET="$P4_ADMITTED/first_epoch_budget.json"
export DATA_ROOT=lead_data
export MODEL_DIR=checkpoints/Qwen3.5-4B
for f in "$PREPARED/data4/manifest.json" "$PREPARED/references4/references.json" \
         "$DATASET/student_view.json" "$FIRST_EPOCH_BUDGET" "$MODEL_DIR/config.json"; do
  test -f "$f" || { echo "缺少本机资产：$f"; exit 2; }
done
export OUTPUT_DIR="checkpoints/phase4_v89_epoch1_$(date +%Y%m%d_%H%M%S_%N)"
export P4_PRODUCTION_CACHE="$OUTPUT_DIR/production_cache"
export PYTHON="$(command -v python)"
export EPOCHS=1 BATCH_SIZE=2 ACCUMULATION=4 UPDATE_SAMPLES=32 GRADIENT_CHECKPOINTING=1
unset RESUME_RGB2 RESUME_RGB4 GPU_IDS
nohup bash qwen3vl_local/audit_joint/train_phase4_batch.sh \
  > "$OUTPUT_DIR.launch.log" 2>&1 < /dev/null &
printf 'PID=%s\n输出目录=%s\n' "$!" "$OUTPUT_DIR"
)
```

默认自动选择空闲四卡；只做当前新基线RGB2/RGB4各一轮、重载及固定test，不重跑此前已成功的旧模型。冷缓存准备和启动时输入复核仍需要CPU时间，但不是老师回放，不承诺固定训练耗时。程序会依次写rgb2.log、rgb4.log，最终交包；不要因为某个中间stage完成而Ctrl-C。

记下上面输出目录后查看：

```bash
P4_RUN=checkpoints/替换成上面打印的输出目录名
cat "$P4_RUN/workflow.json"
tail -n 60 "$P4_RUN.launch.log"
tail -n 60 "$P4_RUN/rgb2.log"
tail -n 60 "$P4_RUN/rgb4.log"
```

运行成功或失败后交回 `P4_RUN/handoff/phase4_epoch_report.zip`、`phase4_epoch_package.json`。如果启动前失败没有handoff，仅交 `P4_RUN.launch.log`。最终以workflow退出码和handoff_result为准，日志不增长或某次protocol end不是整体成功标志。原库、续接目录、训练输出与权重均保留。

完整本地证据：`checkpoints/v89_launch_rgb_20261019/`；轻量核验回执 `v89_launch_rgb_verification_20261019.json`。本轮未训练、未远程执行、未push。

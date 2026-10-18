# 旧正式池与新版采样验收、机器无关入口（2026-10-18）

本次诊断成功，不是标定再次失败。`training_authorized: false` 是这个只读工具的固定属性，不是新的训练失败码。末尾单独的 `)` 是粘贴时缺少配对左括号的 shell 语法错误，发生在 Python 完成之后，不影响已经生成的包，不需要重跑。

用户明确：由用户自由选择服务器，后续不能指定必须在哪台执行。机器名称只作报告元数据；所有命令检查当前机器的实际资产。两台服务器数据、权重、缓存和运行目录不互通；不自动使用另一台的路径，不猜最新目录，不自动复制数据。

## 包的核验

- ZIP 33,796字节，SHA256 `404693a8710eb250a44c51d3f862e785c24b98c46a917a91a97056cc5219ec03`；CRC、唯一成员SHA、另交独立JSON逐字节一致。
- 571份发布源码与v84清单逐项一致；errors为空，173.637秒。此次执行记录来自cr02，这只是收到的事实，不是下一次机器选择要求。
- 训练manifest SHA `2c055bf78443227f179d06313f4d08218a6f37699b8bf7aaf4938f88b4d64c24`、train SHA `03031513d895982f6c0876e88d90d2fb1ab1db537aafc61bc58bd72fe0f05e0d` 与此前完整准入包一致；298770题全部为weak_rule_teacher。
- 八条合同差异全部解释清楚：teacher_rules/teacher_events两处逻辑修改，dataset/identity两处资产版本指针，以及旧plan37/registry32替换为新plan39/registry34。四个旧源码SHA与存档v83相符。不是八个新代码缺陷，也不是服务器同步失败。

## 事件等量和关键监督

两模式历史训练行上的新v5 CPU计划都为十事件各1024次（10240合计），各10036独立题，单题最多2次。不是GPU实际消费；正式学生过滤及训练消费仍由原生入口验收。

| 原生关键正例 | 全池题数/物理组 | RGB2计划曝光 | RGB4计划曝光 |
|---|---:|---:|---:|
| U-E2 return | 2/2 | 1 | 1 |
| U-E3 proceed | 277/95 | 91 | 97 |
| U-E5 proceed | 0/0 | 0 | 0 |
| U-E6 proceed | 708/97 | 354 | 373 |
| U-E7 complete | 13/13 | 3 | 2 |
| R-E5 proceed | 759/182 | 213 | 191 |

稀少题整轮漏采问题已在正式历史池计划层面关闭；每轮至少一次不保证学会，也不增加独立场景。只有U-E5/proceed/YES仍无支持，不用其它边YES代替。全池原始题量无需事件等量，采样后的训练和实际选定val/test才要求等量；YES:NO不强制等量。

此前同一正式池的完整准入记录是val每事件37、test每事件18，无放回、固定、按规则一致率选优。此次新包只重核了train，未重读服务器holdout，不能声称它重新核验了当前val/test，也不能把旧版holdout验收移给未来v24新版池。两模式选题成员不同，不是严格配对效果比较。

## 剩余工作已收敛

当前训练/保存/重载/test链路已有cee旧版两模式的成功证据；不要求重跑那些结果。不再要求重复readiness、盘点或本次label_release_check。

**新教师v24需要对应的新正式池，之后再验收两模式首轮。** 原题库仍绑定v22，不允许把旧SHA改成新SHA继续使用。老师条件会影响实例时序，不能只修改几道已知题的答案就宣称全源更新完成。当前没有经过验证的“旧老师产物免回放升级”路径；不能承诺本次原生更新无需回放或几分钟完成。

已有数据允许作为限定范围的自动规则开发实验，不依赖人工逐题标注。U-E5放行能力必须排除在可监督能力声明之外；若要求十事件每个关键条件都有可靠正负例，目前不满足。U-E2返回和U-E7完成的独立支持仍少。无新证据要求修改提示词、阈值或生产标签；本轮未新增RGB目视，不冒称十事件全源视觉认证。

## 本轮代码改动

- 新增local_paths只读前置检查：使用显式本机DATA_ROOT/MODEL_DIR/OUTPUT_DIR；缺数据根或模型配置、已存在/悬空/嵌套输入的输出路径提前拒绝。检查路径存在不等于数据、模型权重或磁盘预算准入；原生准入及磁盘守护保留。
- server_phase4_existing.sh和train_phase4_batch.sh从脚本定位当前AutoMoT根，避免另一机器或另一工作目录导致相对路径误解。所有相对配置路径统一相对这个根；绝对配置使用用户指定的本机路径。构建入口先检查路径，再创建新输出；直接prepare也检查。
- label_release_check不再强制选择cr02/cee/local；AUTOMOT_SERVER可选，始终记录真实hostname与逻辑/解析路径。
- 教师v24/plan39/registry34、学生合同、采样器、评测、提示词均未改变。新v85发布清单573份，仅外围入口变化，不引入新一轮标签版本或新的view迁移理由。
- 162项本机CPU回归通过（5.46秒），包括两种服务器目录布局、错误机器标签不改变资产选择、另一cwd启动、缺失/别名/悬空路径提前拒绝、训练首轮/续训/打包及事件采样。首次测试抓住一处新提示文本的shell引号问题，已修正，bash语法检查通过。
- 实际取消AUTOMOT_SERVER后运行本地CLI，3187历史题两模式模拟及ZIP核验完成，573源码匹配。不是远程GPU或完整新版池验收。

## 后续执行：用户任选有本机原始数据的服务器

本轮按用户授权发布v85；下列是同步后的正式池更新入口，**不是要求重跑诊断**。先确保没有任务正在使用将同步的源码，再按既有sync_automot.sh同步；不能使用遗留服务器git判断版本。

从任选服务器的AutoMoT根和原Python环境执行，若本机路径不同只调整DATA_ROOT/MODEL_DIR/OUTPUT_DIR；不填写另一台的运行目录。此命令会执行CPU回放、单份原生RGB4库、短视图、自动规则留出及准入，不启动GPU训练。新的输出完整保留旧题库和历史权重。

```bash
DATA_ROOT=lead_data \
MODEL_DIR=checkpoints/Qwen3.5-4B \
OUTPUT_DIR="checkpoints/phase4_teacher_v24_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase4_existing.sh
```

输入不存在会在扫描/回放前停止；旧的空间保护仍会检查当前磁盘并中止余量不足的任务，不保证未知机器的剩余空间一定够。不自动删除或转移旧资产。构建是重任务，不能承诺固定时间；这轮已有诊断无需再次运行。

完成或失败只交该次新输出的 `handoff/fullflow_report.zip` 和 `handoff/fullflow_package.json`。以新池实际准入为依据再给训练命令，不沿用另一台机器的成功目录或旧checkpoint。收到的本次包及核验材料在本机checkpoints/label_release_received_20261018，轻量回执为label_release_received_verification_20261018.json。

## v85 发布当轮补充验证与后台命令

162项回归再次通过（10.12秒，与上一轮重叠不相加），573发布源码匹配。重新原生回放2条完整train路线231帧、693源SHA，两模式各48训练题、无val/test；原始questions和instance_ends与前v24逐字段相同。实际查看24张不同RGB（相对此前140+40+16目视集合新增10张），72目视源SHA通过。UE4 Town02 65–67 UNKNOWN、68 YES时行人已向自车走廊左侧离开；69 RGB不连续终止不是完成。UE6 Town06 61–68为实例出现前，69–72横穿警车仍构成NO；light_hazard有独立否决时不能只看车辆位置放行。没有新确定阈值或提示词缺陷；信号关联可观察性限制仍保留，不作盲审或全源认证。完整证据只留本机checkpoints/v85_release_rgb_20261018，轻量回执v85_release_rgb_verification_20261018.json随白名单发布。

为避免终端关闭或误按Ctrl+C中断长任务，推荐在任选服务器的AutoMoT根目录、已激活Python环境中，先按原流程同步（确认没有使用源码的任务正在运行）：

```bash
bash ./sync_automot.sh
```

同步成功后再执行下面这一段。DATA_ROOT和MODEL_DIR默认是本机仓库中的相对路径；本机布局不同就替换为本机真实路径。无需AUTOMOT_SERVER变量，不复用其它服务器的缓存或checkpoint。

```bash
P4_OUT="checkpoints/phase4_teacher_v24_$(date +%Y%m%d_%H%M%S_%N)"
nohup env PYTHON="$(command -v python)" \
  DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B \
  OUTPUT_DIR="$P4_OUT" REPLAY_WORKERS=2 \
  bash qwen3vl_local/audit_joint/server_phase4_existing.sh \
  > "$P4_OUT.launch.log" 2>&1 < /dev/null &
echo "PID=$!  输出=$P4_OUT  启动日志=$P4_OUT.launch.log"
```

保留打印的输出目录；后续新终端把P4_OUT设置为那条本机路径，不能重新执行带date的赋值来查看旧任务。查看进度：

```bash
tail -n 50 "$P4_OUT.launch.log"
tail -F "$P4_OUT/build.log"
# 在查看日志的终端按Ctrl+C，只退出tail，不终止后台建库。
```

当前阶段在`$P4_OUT/build_result.json`，最终回执在`$P4_OUT/finish_receipt.json`。不要看到teacher路线进度完成就终止任务，后续还有编译、自动留出、视图及准入。完成或运行中失败交回`$P4_OUT/handoff/fullflow_report.zip`及`fullflow_package.json`（30MB上限）；若前置输入检查失败、尚未创建handoff，交启动日志即可，不循环重试，不删除旧资产。该步骤是CPU标签更新和准入，不启动GPU；完成后再安排首轮。

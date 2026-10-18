# v86：磁盘守护停止后的完整回放续接

本次日志显示8614/8614路线计算走完、耗时15854.8秒，随后磁盘守护退出。不能据此认定最终索引已落盘，也不能认定磁盘已满。原入口错误地将保留量设为 `max(20 GiB, free_at_start - 12 GiB)`，使任务所在文件系统新增约12 GiB就可能触发。其它任务写入同样占用此额度。

已修正为绝对保留20 GiB；可用 `MIN_FREE_GIB` 提高，不能通过此入口降到20以下。`build_storage.json`记录实际可用量、保留量、可继续写入量，完整构建需求仍未知，不宣称空间一定足够。守护触发日志包含实际free/reserve字节数。两个CPU建库入口均修正，训练磁盘预算不变。

## 续接方式

两台机器均可使用；必须使用执行机器本地实际存在的中断目录、原始数据和模型配置。不会从另一台机器找数据。同步前确认旧任务已停止；不要对仍运行中的源码目录同步。

在当前服务器AutoMoT根目录、原Python环境执行：

```bash
bash ./sync_automot.sh
```

将下面 `P4_OLD` 改为**这次8614条完成后因磁盘停止的目录**，不是此前v22/v57历史题库，也不是其它机器的路径。`DATA_ROOT`/`MODEL_DIR`使用当前机器实际配置。

```bash
P4_OLD="checkpoints/这里填写本次被磁盘守护停止的目录"
P4_OUT="checkpoints/phase4_v24_recovered_$(date +%Y%m%d_%H%M%S_%N)"

if [[ -f "$P4_OLD/inputs/candidates.json" ]]; then
nohup env PYTHON="$(command -v python)" \
  RESUME_FROM="$P4_OLD" OUTPUT_DIR="$P4_OUT" \
  DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B MIN_FREE_GIB=20 \
  bash qwen3vl_local/audit_joint/server_phase4_existing.sh \
  > "$P4_OUT.launch.log" 2>&1 < /dev/null &
echo "PID=$! 输出目录：$P4_OUT"
else
  echo "旧目录不正确：没有inputs/candidates.json；未启动。"
fi
```

旧目录不存在不会启动；先确认路径，不删除旧目录或改成普通prepare重跑。是否成功开工以launch.log和build.log为准。

```bash
echo "输出目录：$P4_OUT"
tail -n 60 "$P4_OUT.launch.log"
tail -F "$P4_OUT/build.log"
```

新终端需要把P4_OUT设回刚打印的路径，不重新执行date表达式。Ctrl-C退出tail不会停止nohup任务。

续接严格拒绝重新计算路线：
- 索引存在：原生accounting核验完整集合、教师合同、逐帧题目及文件SHA。
- 索引不存在：所有预期gzip路线文件齐全时，原生inspect_route核验每条，恢复完整索引，再做原生accounting。只安装缺失索引；保留原index.json.pending及所有路线文件，不覆盖既有索引。
- 缺路线、截断、旧合同、部分/损坏索引或已发布集合丢失索引：阻塞并交包，不自动回退全量回放。
- 输出使用新目录；旧题库/回执保留。后续新库共享引用旧回放，**成功后也不能删除或移动P4_OLD/inputs/production**。

这里不执行GPU训练。完整回放内容仍要读取核验，后续编译、自动留出、短视图和准入仍有CPU/磁盘耗时，不承诺几分钟完成。日志的`recomputed_routes: 0`表示复验不是重算。最终应检查`finish_receipt.json`；teacher到末行不等于建库完成。

## 反馈

无论通过或阻塞，交新P4_OUT下：
- `handoff/fullflow_report.zip`
- `handoff/fullflow_package.json`

实际ZIP仍不超过30MB，包含建库阶段、错误、存储计划和日志尾。路径检查在创建输出前失败则交 `P4_OUT.launch.log`。若仍因低空间停止，另附 `df -h "$P4_OUT"`，不自行删除旧库/权重或关闭守护。

## 本机验证及边界

- 116项相关回归通过（11.08秒），其中19项新测试覆盖绝对保留、缺索引重建、缺文件、gzip截断、旧合同、已发布索引丢失、部分索引、原生入口阻塞打包；恢复中禁止调用produce/route_records。
- 两条真实路线231帧副本，故意删除索引；恢复索引与原版逐字节一致，所有路线SHA不变，重算0条；原始数据未改。
- 实际CPU CLI使用真实回放和小模型配置完成恢复后，在原生关键覆盖门正常阻塞（该局部样本不是完整十事件池）；不将此声明为正式池准入。
- 实际shell前置、源码576项、空间计划及失败打包经过执行。本机是piped core collector，原保护拒绝启动子任务；未修改系统设置或绕过守护。服务器此前已启动过guard，不要求重复readiness。
- 教师v24、plan39、registry34、原生标定/学生/提示词/采样/评测源文件未改，训练/val/test十事件等额检查不变。U-E5/proceed/YES限制保留。
- 未新增RGB目视、GPU或远程执行；本轮专门修复空间预算和续接，不把存储修复称为标签全源认证。

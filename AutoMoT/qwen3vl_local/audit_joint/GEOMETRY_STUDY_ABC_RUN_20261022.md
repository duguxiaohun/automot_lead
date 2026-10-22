# geometry a / b / c 发布与执行

新增包外 `geometry_study_abc.py/.sh`，不改冻结的 159 个 study 源、不改在途接续绑定的 165 源。v2 审计的供体与退出竞争修复已复核。原版本、数据、权重和失败证据保留。

| 方案 | 径向分区 | 横向与时距 |
| --- | --- | --- |
| a | 原速度自适应参考带，阈值不变 | 当前车道参考横向 ±2 m，未来 0.5/1/1.5/2 s |
| b | 当前道路 R3：20/24 m；其余已知道路：3/13 m | 四个时距共用绝对边界 |
| c | 全道路 3/13 m | 四个时距共用绝对边界 |

均为九类，过去位置也统一当前自车原点、当前速度和当前车道参考。b/c 的中间带是 MIDDLE，不是“匀速”。三个方案仍用原四表示共同池和共同采样顺序；只训练 abc 不等于重新建立仅 abc 的样本池，也不绕过 six 共同有效性与准入条件。

每组四卡、每轮十事件各 1280、每帧每轮最多一次、三轮共 9600 次更新；三组共 28800 更新。旧 geometry 单步约 2.12 s，据此粗估每组纯训练 5.7 h，三组约 17 h，另加 CPU 准备、模型加载、保存与评测。不是本次 GPU 实测或完成时间保证；单轮 10 h 实测预算门保持。不得为了开训降低覆盖要求或自动减到 1024。

## 1. 同步与 CPU 测试

在所选服务器已有 AutoMoT 应用根运行，不手填资产路径，不重复拼接 AutoMoT：

```bash
bash ./sync_automot.sh
nohup bash qwen3vl_local/audit_joint/geometry_study_abc.sh cpu > geometry_study_abc_cpu.log 2>&1 &
```

只启动一次。CPU 入口按本机内容证明复用兼容父库，否则原生准备；自动检查项目 CARLA 目录含十二 Town，包括嵌套 Town11/12/13/15。然后构建/复用共同池、正式准入、三轮容量和方向审计，不自动训练。

```bash
tail -n 40 geometry_study_abc_cpu.log
bash qwen3vl_local/audit_joint/geometry_study_abc.sh status
```

完整 CPU 结果在 `checkpoints/geometry_study_cpu_audit_v2_1280/`：`status.json`、`pool_report.json`、`directions.json`、本地图册 `directions.html`。图册引用原始本机 RGB，不能仅下载 HTML 就声称可离线显示图片；生成图册不代表已人工目视。

退出 0 才表示正式条件通过且 ready；退出 2 为完整诊断后拒绝，先回传上述前三份 JSON 和日志，不直接训练。其它退出为执行失败。旧报告目录存在时拒绝覆盖，不反复运行 cpu。首次原生准备期间可能尚未生成审计 status，使用 cpu 日志观察阶段。

本机 `checkpoints/geometry_study_pool_20261022/pipeline_status.json` 对应已经启动的任务，继续等其 `full_report_v2`，不要套上述命令重复建库。本机在途任务不代表另一台服务器已经准备好。

## 2. 开训检查与三个方案

CPU 完成通过后：

```bash
bash qwen3vl_local/audit_joint/geometry_study_abc.sh check
nohup bash qwen3vl_local/audit_joint/geometry_study_abc.sh train all > geometry_study_abc_train.log 2>&1 &
```

`check` 重新核验 ready、共同计划、当前源码及本地模型分片，确认四张不同 GPU、a/b/c 的所有目标目录和日志都不存在。它不加载 4B 模型，不是 GPU forward/backward 或显存验收。实际模型、prompt、DDP 检查仍在每组原生训练入口中，失败即停止；没有通过缩短预算的假 smoke 后跳过原生检查。

`train all` 按 a→b→c 串行，每组正式三轮后按验证选模，再重载选中 checkpoint 做 test；任一组失败不启动下一组。外层还核对完成步数、表示、共同数据 SHA 和 test 指标落盘。它自动确认四卡，在原始数据和模型加载前明确拒绝卡数不足，不等到后续预算检查才报错。

也可只运行单组（不要和 all 同时运行）：

```bash
bash qwen3vl_local/audit_joint/geometry_study_abc.sh train a
bash qwen3vl_local/audit_joint/geometry_study_abc.sh train b
bash qwen3vl_local/audit_joint/geometry_study_abc.sh train c
```

默认自动选择四张空闲显存最多的卡；若需明确选卡，只用 `GPU_IDS=0,1,2,3 bash ...`。外层锁防止这套入口重复并发启动；不会阻止系统中其它训练使用 GPU。已有 run/log 不覆盖、不隐式续训。成功 a 后 b 失败时，不再运行 all 重训 a；先检查 b 的日志，未开始的 c 可单独运行。新外层固定 `a_1280/b_1280/c_1280`，忽略外部 `STUDY_RUN_NAME`，防止三组混用目录；显式语义重试使用原生 remote.sh 的独立名称，不删除旧失败目录。

## 3. 查看进度与交回结果

```bash
bash qwen3vl_local/audit_joint/geometry_study_abc.sh status
tail -n 30 checkpoints/geometry_study_v1/operations/a_1280.log
tail -n 30 checkpoints/geometry_study_v1/operations/b_1280.log
tail -n 30 checkpoints/geometry_study_v1/operations/c_1280.log
```

每 100 更新原生日志输出当前轮进度/loss；status 汇总已完成轮次、每轮 loss/用时、选中 checkpoint、test 是否落盘及日志末行。轮内 step 以日志为准，不能把启动检查完成当作训练完成。被 SIGKILL 的外层状态可能仍为 running，须结合进程和日志判断，不把它当心跳。

每组结果位于 `checkpoints/geometry_study_v1/runs/{a,b,c}_1280/`。交回 `startup.json`、`epoch_1/2/3.json`、`validation_1/2/3.json`、`training_complete.json`、`test/metrics.json` 和对应 operations 日志；无需传 `state.pt` 或基座权重。失败时先交对应日志和已有 JSON，禁止改 SHA 绕过检查。

比较时关注事件×时距×类别支持、宏召回、历史冲突与拷贝、转换检出、错误集中路线。三组 token 的含义和难度不同，分类准确率不能直接证明轨迹效果更好；最终轨迹优劣仍需相同成员和 expert 设置下的 ADE/FDE。该入口训练的是离散 token 模型，不自动训练 action expert 或运行 CARLA。

## 验证边界

原 81 项 CPU 回归通过；新增运行入口测试覆盖四卡不足/重复/无效编号、后续组输出冲突在先、准入失败不启动、三组串行、失败停止、缺评测回执不报成功和只读状态。真实训练子任务在新入口测试中用替身，原有小 Qwen backward/保存重载和两 Gloo rank 回归保留；不声称 4B/服务器 GPU 验收。轻量发布记录见 `geometry_study_abc_release_verification_20261022.json`。

发布副本验收：在无 `.git` 的精确暂存应用副本内运行 192 项相关测试，全部通过（49.22 s）；包含共享缓存与候选恢复边界。工作区先前 159 项测试与此重叠，不相加。

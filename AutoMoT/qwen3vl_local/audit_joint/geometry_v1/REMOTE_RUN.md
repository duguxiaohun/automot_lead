# geometry_v1 远程执行

任选 cr02 或 cee，在已有 AutoMoT 应用目录执行。两机均已由用户确认项目内
`lead_data`、`checkpoints/Qwen3.5-4B`、`keyframe_filter/collection_output` 存在。
CARLA 使用 `./CARLA_0915/CarlaUE4/Content/Carla/Maps`：Town01–07/Town10HD
在 `OpenDrive`，Town11/12/13/15 在各自的 `TownXX/OpenDrive`。不搜索其它挂载盘。

`remote.sh` 从脚本位置定位应用根，优先使用显式 PYTHON，其次项目 `.venv/bin/python`，
最后使用当前环境的 python。cee 即使在未激活环境的 root 终端也会选用项目已有 venv；
cr02 沿用已激活的 automot 环境。无需手填四个资产路径，也不要再 `cd AutoMoT`。

## 同步并后台 CPU 建库

先沿用已有项目同步脚本；确认新入口确实同步后再启动。该同步脚本是远程既有工具，
不属于本次发布；若同步后找不到入口，回传同步输出，不继续建库。

```bash
bash ./sync_automot.sh && test -f ./qwen3vl_local/audit_joint/geometry_v1/remote.sh
```

同步成功后执行：

```bash
( set -o noclobber; nohup bash ./qwen3vl_local/audit_joint/geometry_v1/remote.sh cpu > ./geometry_v1_cpu.log 2>&1 < /dev/null & )
```

上述后台命令的日志已存在时会拒绝覆盖。脚本核验模型和实际依赖导入，使用固定入口目录
`checkpoints/geometry_v1`；已有地图链接和 assets 先校验，数据与计划按完整依赖复用，缺失或输入变化才生成独立内容版本，旧库保留。
它在入口目录下建立 `maps` 软链接视图，将项目内十二个标准 Town 地图提供给原建库器，
不复制地图、不更改 CARLA。若同名候选内容不同则拒绝。路径及 SHA 记入 `assets.json`；
建库器仍对实际使用的地图做原有源文件绑定。

```bash
tail -n 50 ./geometry_v1_cpu.log
```

本次 CPU 入口退出码为 0 后写入 `checkpoints/geometry_v1/cpu.done`，表示数据准备和三轮计划均成功；不能用此前留下的标记替代本次退出状态。
`data_ready.json` 给出本次实际 dataset 和 plan 路径。回传 `geometry_v1_cpu.log`、入口目录下的
`assets.json`、`data_ready.json`、`prepare_timings.jsonl`，以及 dataset 同目录的 `receipt.json` 和
`split_coverage.json`、回执指向的 plan 文件，检查后再训练。失败时保留已有日志和回执。
不上传 dataset.json、原始图像或权重。CPU 全池建库耗时尚未实测，不受训练启动预算约束。

CPU 入口已默认接入独立计时器。`prepare_timings.jsonl` 每次追加一条成功/失败记录，不覆盖既有测量；
含准备总时间、候选枚举、依赖散列、发现与校验、原生建库、计划生成/复用的耗时，及散列调用数和累计字节。
各阶段是包含子调用的耗时，存在重叠，不能相加；散列计数只覆盖缓存适配器成功完成的 bind 调用，重复读取重复计数，
不含原生校验器内部全部 I/O。计时不改变数据生成代码指纹或缓存身份，不跳过前后校验；当前没有全池加速结论。
需要再次测量复用时，发布同步后可前台执行同一条 `bash ./qwen3vl_local/audit_joint/geometry_v1/remote.sh cpu`，
已有匹配数据照常复用，报告继续追加。训练目录与旧权重不会因此重写。

## CPU 结果核验后再训练

默认四卡、每事件每轮1024呈现、三轮7680次更新；warmup2000。
训练启动预算1小时，100步后实测单轮预算10小时；不是总耗时保证，没有 CLI resume。
默认沿用 run.sh 按空闲显存选择四卡；可用 GPU_IDS 显式指定四个编号。
训练成功后单卡重载第三轮 checkpoint 并执行 test；失败或预算停止不继续 test。

```bash
( set -o noclobber; nohup bash ./qwen3vl_local/audit_joint/geometry_v1/remote.sh train > ./geometry_v1_train_test.log 2>&1 < /dev/null & )
```

查看 `geometry_v1_train_test.log`；仅输出下 `train_test.done` 存在表示两步成功。
回传 `run/startup.json`、`run/epoch_*.json`、`run/validation_*.json`、`test/metrics.json`
及日志。这里只训练 Phase3 九类预测器，不训练连续 expert、不启动在线 CARLA。

需要合并额外的开发曝光 manifest 时，应在正式建库前使用 run.sh build 的
`--development-manifest` 接口；本便捷入口使用已发布的默认开发名单。

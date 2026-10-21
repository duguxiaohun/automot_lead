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

日志已存在会拒绝覆盖。脚本核验模型和实际依赖导入，随后独占创建固定输出
`checkpoints/geometry_v1`；已有输出会拒绝，不删除、不重建、不自动续接。
它在输出下建立 `maps` 软链接目录，将项目内十二个标准 Town 地图提供给原建库器，
不复制地图、不更改 CARLA。若同名候选内容不同则拒绝。路径及 SHA 记入 `assets.json`；
建库器仍对实际使用的地图做原有源文件绑定。

```bash
tail -n 50 ./geometry_v1_cpu.log
```

仅 `checkpoints/geometry_v1/cpu.done` 存在表示完整建库和三轮计划都成功。
回传 `geometry_v1_cpu.log` 及输出下的 `assets.json`、`data/receipt.json`、
`data/split_coverage.json`、`plan.json`，检查后再训练。失败时保留已有日志和回执。
不上传 dataset.json、原始图像或权重。CPU 全池建库耗时尚未实测，不受训练启动预算约束。

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

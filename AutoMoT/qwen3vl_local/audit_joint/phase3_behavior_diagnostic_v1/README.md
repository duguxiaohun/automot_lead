# Phase3 单帧模型错图配对诊断

目的：在已完成的 `single_frame_validation_v2` 模型上检查预测对 RGB 的依赖。
保持原始提示词、历史动作和未来监督，仅将图像替换为同 event、速度档、亮度档和尺寸，且来自不同物理路线的图像。
这不是训练，也不重建题库。使用原实验所在机器的模型、索引、原始 RGB、完整 epoch 3 训练状态及验证回执；两台机器的资产不能混用。

三个 JSON 是固定干预协议清单（相对图像路径、身份、SHA、配对及排除原因），不含图像、权重或动作真值。
种子为 11/29/47，各 264 个配对；主比较使用三组共同的 260 个接收样本。
缓存跨种子重复项后，共 268 次原图和 636 次错图前向，零参数更新。不能将子集得分直接与原 320 题总分比较。

## 旧模型源码隔离与四卡修复

后续 geometry_v1 修改了共享 leadmot/config.py，因此旧六动作模型须使用原版源码。当前启动器只读取 sync_automot 同步的白名单：`runtime_sources.json` 列明 484 项运行依赖及原始 SHA；除两项兼容副本外，其余从现有项目文件读取。`compat_sources/qwen3vl_local/leadmot/config.py` 和 `decoder.py` 是原始 v2 源码，独立发布，不覆盖当前 geometry 实现。

启动器先核验所有清单文件，再复制到新输出目录 runtime；保留 261 项原生发布校验及原生 job、dataset、checkpoint 检查。**不调用远程 Git，不要求 `.git`、历史提交或与本机相同的仓库布局**。清单中的 reference_commit 仅记录本机来源，不用于远程查询。缺失或 SHA 不符的文件在加载模型前报告具体路径，通过同步白名单解决；不改 SHA 绕过检查。

先完成 CPU 预检查，再启动默认四个独立推理进程，每卡载入一份模型；按接收样本划分，各卡 67 个不同原图，错图配对也归同一卡。汇总检查完整性、重复、原实验绑定和目标窗口；仍是 904 次总前向，不能保证耗时恰好缩短四倍。某一卡失败会停止其它进程，日志及失败回执保留。

## 运行

以下命令在原来运行训练的 `AutoMoT/` 目录、原训练 Python 环境中执行。
远程同步统一使用已有的 `bash ./sync_automot.sh`；保留本机数据、权重及原实验目录。

```bash
bash ./sync_automot.sh
bash qwen3vl_local/audit_joint/phase3_matched_image_eval.sh \
  --job phase3_single_frame_validation_v2/run_single_frame_nearest_v1/experiment/single_frame_job.json \
  --output phase3_single_frame_visual_diagnostic_mount_v1
```

默认自动选择四张显存占用最少的 GPU；显式指定四卡并后台运行：

```bash
nohup env GPU_IDS=0,1,2,3 bash qwen3vl_local/audit_joint/phase3_matched_image_eval.sh \
  --job phase3_single_frame_validation_v2/run_single_frame_nearest_v1/experiment/single_frame_job.json \
  --output phase3_single_frame_visual_diagnostic_mount_v1 \
  > phase3_single_frame_visual_diagnostic_mount_v1.log 2>&1 &
```

两种运行方式任选一种。显式 `GPU_IDS=0` 仍可只用一张卡，卡数来自指定列表；默认不足四卡会报错，不静默退回单卡。目录已存在时只展示原回执和日志末尾并停止，不自动覆盖或重复推理。不能凭“目录存在”判断上次成功或失败；先用下面的只读命令查看。只有确认上次失败原因已解决后，才明确指定新输出名重试。

```bash
bash qwen3vl_local/audit_joint/phase3_matched_image_eval.sh \
  --job phase3_single_frame_validation_v2/run_single_frame_nearest_v1/experiment/single_frame_job.json \
  --output phase3_single_frame_visual_diagnostic_mount_v1 \
  --inspect-existing
```

此命令不选择 GPU、不加载模型、不修改任何文件；将终端输出回传即可定位此前失败原因。回执 status 仅为已保存状态，不表示当前进程仍在运行。

需要只验证资产时，追加 `--preflight-only` 并使用不同输出名；该模式不会选择 GPU 或推理。
正式推理自身已先执行全部资产检查，无需先单独跑预检查。
必须保留原始源码合同；不能通过改 SHA 让其它版本的权重、标签或模型通过检查。

完成后收集输出目录中的 `receipt.json`、`metrics.json`、三个种子的 `*_cases.json` 和 `*_metrics.json`。
这些 JSON 不含模型权重或 RGB。打包时排除 runtime 源码快照，保留日志和分片结果：

```bash
tar --exclude=phase3_single_frame_visual_diagnostic_mount_v1/runtime \
  -czf phase3_single_frame_visual_diagnostic_mount_v1.tar.gz \
  phase3_single_frame_visual_diagnostic_mount_v1
```
若失败，交日志及已生成的回执，不删除原训练目录，不重训。

## 已有证据与限制

2026-10-21 本地逐帧审计覆盖 27 个锚点、25 个物理路线组、431 张不同 RGB（23 个 val 锚点及 4 个 train 开发锚点）。
320 个固定 val 的原生当前标签、1280 个历史标签、1280 个未来有效性和 964 个有效目标分布逐项重放一致；8738 个原始来源文件 SHA 不变。
未找到足以直接修改原标定或提示词的确定错标。当前发现：22 个当前变道锚点中 4 个未进入有效未来变道监督；严格纯标签转换子集 44 个锚点中 16 个预测偏晚、13 个未预测转换；1.5 倍偏好实际仅改变混合目标，纯变道目标的归一化抵消权重。
这些是开发诊断证据，不是全池盲审或修改标签的授权依据。

本机已验证无 Git 平铺目录和无关 Git 目录下的白名单复制/原生校验、真实四 CPU 子进程的分片与单进程结果精确一致、漏项/重复拒绝和失败清理，以及原生小 Qwen 的真实配对前向。四 CPU 子进程使用确定性预测替身，不冒称四卡 4B/H20 推理已通过。
保留原生标签、提示词和训练代码。原始审计图像及完整证据仅留本机 checkpoints，不随 Git 发布。

本轮旧 `phase3_single_frame_visual_diagnostic_v2` 已由回执确认为源码准备失败，未训练、未推理；保留该目录。当前 mount_v1 同时包含无 Git 部署与 RGB 挂载修复，无需删除旧目录、重训或重建数据。

## 2026-10-22 RGB 挂载路径校验修复

原生来源校验允许场景、路线或图片通过软链接挂载到 data_root 的真实路径以外；旧诊断额外使用 resolved_path.is_relative_to(root)，误拒绝这种合法布局。现按场景/完整路线/rgb/锚点核对逻辑路径，要求计划与原记录解析到同一文件，并核对已准入 raw_proof 的 resolved_path、stat、SHA 和计划图片 SHA、尺寸及亮度。检查后再次确认文件与挂载未变化；没有放宽原生标签和数据来源绑定。

不同文件的错误现在包含样本 ID、计划逻辑/真实路径和原记录逻辑/真实路径；每组成功时打印 outside_root_mounts 计数。外层进程失败会直接展示该 worker 的日志末尾。

已用场景/路线/图片外部软链接样例复现旧条件误拒绝；本机实际三份计划各 266/265/266 个接收或供图文件通过检查，本机这些图片都位于解析根目录内。不能从旧服务器通用错误确定其失败分支；四卡 H20 仍待本次实际验收。保留失败的 sync_v1 目录，新输出使用 mount_v1。

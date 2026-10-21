# Phase3 单帧模型错图配对诊断

目的：在已完成的 `single_frame_validation_v2` 模型上检查预测对 RGB 的依赖。
保持原始提示词、历史动作和未来监督，仅将图像替换为同 event、速度档、亮度档和尺寸，且来自不同物理路线的图像。
这不是训练，也不重建题库。使用原实验所在机器的模型、索引、原始 RGB、完整 epoch 3 训练状态及验证回执；两台机器的资产不能混用。

三个 JSON 是固定干预协议清单（相对图像路径、身份、SHA、配对及排除原因），不含图像、权重或动作真值。
种子为 11/29/47，各 264 个配对；主比较使用三组共同的 260 个接收样本。
缓存跨种子重复项后，共 268 次原图和 636 次错图前向，零参数更新。不能将子集得分直接与原 320 题总分比较。

## 运行

以下命令在原来运行训练的 `AutoMoT/` 目录、原训练 Python 环境中执行。
同步仅使用 fast-forward；如有本地改动冲突，保留文件并先处理冲突，不清理数据。

```bash
git pull --ff-only origin main
bash qwen3vl_local/audit_joint/phase3_matched_image_eval.sh \
  --job phase3_single_frame_validation_v2/run_single_frame_nearest_v1/experiment/single_frame_job.json \
  --output phase3_single_frame_visual_diagnostic_v1
```

默认自动选择一张显存占用最少的 GPU；显式指定单卡并后台运行：

```bash
nohup env GPU_IDS=0 bash qwen3vl_local/audit_joint/phase3_matched_image_eval.sh \
  --job phase3_single_frame_validation_v2/run_single_frame_nearest_v1/experiment/single_frame_job.json \
  --output phase3_single_frame_visual_diagnostic_v1 \
  > phase3_single_frame_visual_diagnostic_v1.log 2>&1 &
```

两种运行方式任选一种。目录已存在时拒绝覆盖，重试须明确指定新的输出名。
需要只验证资产时，追加 `--preflight-only` 并使用不同输出名；该模式不会选择 GPU 或推理。
正式推理自身已先执行全部资产检查，无需先单独跑预检查。
必须保留原始源码合同；不能通过改 SHA 让其它版本的权重、标签或模型通过检查。

完成后收集输出目录中的 `receipt.json`、`metrics.json`、三个种子的 `*_cases.json` 和 `*_metrics.json`。
这些 JSON 不含模型权重或 RGB；可将整个诊断输出目录打包交回。
若失败，交日志及已生成的回执，不删除原训练目录，不重训。

## 已有证据与限制

2026-10-21 本地逐帧审计覆盖 27 个锚点、25 个物理路线组、431 张不同 RGB（23 个 val 锚点及 4 个 train 开发锚点）。
320 个固定 val 的原生当前标签、1280 个历史标签、1280 个未来有效性和 964 个有效目标分布逐项重放一致；8738 个原始来源文件 SHA 不变。
未找到足以直接修改原标定或提示词的确定错标。当前发现：22 个当前变道锚点中 4 个未进入有效未来变道监督；严格纯标签转换子集 44 个锚点中 16 个预测偏晚、13 个未预测转换；1.5 倍偏好实际仅改变混合目标，纯变道目标的归一化抵消权重。
这些是开发诊断证据，不是全池盲审或修改标签的授权依据。

本机已验证纯诊断逻辑及原生小 Qwen 的真实配对前向；完整 4B/H20 推理尚待远程执行。
保留原生标签、提示词和训练代码。原始审计图像及完整证据仅留本机 checkpoints，不随 Git 发布。

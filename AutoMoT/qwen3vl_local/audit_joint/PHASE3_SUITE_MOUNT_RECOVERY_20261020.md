# Phase3 base/A/B：外部数据挂载修复及免重建接续

## 本次服务器状态

用户提供的cr02日志表明：共享数据生成exit=0，1098.90秒，新增22421条标签、26786个请求目标缺失；base CPU准入exit=0（6.07秒），A在CPU preflight的原始meta来源核验退出（19.24秒），B未启动。没有执行GPU探针、训练或测试。错误为 `ValueError: Raw route escapes data root`，不是NCCL、显存或已证实的空间不足。

原核验将 `lead_data/scenario/route` 解析成物理路径后，要求仍在根目录内；这会拒绝通过场景/路线软链接映射到外部盘的数据布局。日志没有给出具体链接目标；本地已用真实小型meta/RGB夹具在外部挂载下复现该检查条件，不能冒称已经检查服务器所有链接。

## 修改范围

- 使用逻辑 `scenario/route` 身份组织证据，单独记录解析后的路线、meta目录和每个源文件的物理目标。允许外部场景、路线、meta/RGB子目录以及文件链接。
- 仍检查精确帧meta复算、答案/速度/导航/RGB历史、split、原生规则、quarantine和文件SHA。缓存核验拒绝链接目标变化、源文件stat变化和meta成员变化；重新计算时也检查目标未在过程中切换。不同逻辑路线指向同一路线根目录会拒绝。
- 不复制RGB或meta，不改已有索引、标签规则、提示词、采样、三轮7680预算及模型初始化。
- `--prepared-data` 显式兼容唯一已发布生产者 `1103ece79` 的共享数据。兼容清单固定其原始发布SHA，检查253项未变源码依赖和六个数据生成函数的源码摘要；规则、映射、prompt、生成逻辑变化或未知发布均拒绝。不改旧receipt的release SHA。当前发布仍独立核验全部源码。

修复的是路径准入；26786个缺失目标是否影响A/B逐事件/时点覆盖，要由后续预检给出结果。本次失败发生在覆盖检查之前，不能保证修完路径后其他门槛一定通过，也不通过降门槛绕过。

## 服务器接续

同步白名单后，直接引用这次已生成的 `shared_data`，新输出目录保存修复后的核验及三组训练结果。base仅通过了旧CPU检查，当前新版会重新核验三组；不存在需要恢复的训练checkpoint。

在本次cr02 AutoMoT根目录、automot环境执行（若换机器，路径必须是该机已有资产，不迁移或猜测）：

```bash
bash ./sync_automot.sh && (
  set -euo pipefail
  ulimit -S -c 0 2>/dev/null || true
  mkdir phase3_base_ab_mountfix_v1
  nohup env GPU_IDS=0,1,2,3 SKIP_TRAIN=0 PIPELINE_CHECK_ONLY=0 \
    PYTHON="$(command -v python)" \
    bash qwen3vl_local/audit_joint/phase3_suite.sh \
    --index phase3_base_ab_v1/result/shared_data/base/frame_index.jsonl \
    --prepared-data phase3_base_ab_v1/result/shared_data \
    --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
    --output phase3_base_ab_mountfix_v1/result \
    > phase3_base_ab_mountfix_v1/launch.log 2>&1 < /dev/null &
  printf 'PID=%s\n日志=%s/phase3_base_ab_mountfix_v1/launch.log\n' "$!" "$PWD"
)
```

此命令不调用数据生成器。顺序为三组CPU准入→A/B四卡探针→base三轮/val/test→A三轮/val/test→B三轮/val/test→比较报告。三组均从同一Qwen初始权重开始，A/B不加载base结果。

`tail -f phase3_base_ab_mountfix_v1/launch.log` 查看总进度；详细错误位于新 `result/<组名>/<阶段>.log`。结束标志为 `[suite] exit=0` 和 `suite_workflow.json` 的 `three_methods_completed_quality_pending`，不等于质量达标。

成功或失败都交新 `phase3_base_ab_mountfix_v1/result/handoff/` 的八个文件；尚无handoff时交新launch.log。保留旧 `phase3_base_ab_v1/result/shared_data`，新训练会继续引用它。不要运行原来不带 `--prepared-data` 的命令重建。

## 本地验证

最终相关回归205项全部通过（185.68秒），含17项新增挂载及旧数据兼容检查；258项发布源码SHA匹配。实际创建外部软链接进行meta/RGB复算，并验证改指、文件变化、成员变化及生成依赖变化会拒绝。未在服务器执行修复或GPU训练。[轻量回执](phase3_mount_recovery_verification_20261020.json)。

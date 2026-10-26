# 两服务器源码快照与 BEV ABC 启动

cr02 的 `/home/cruser1/lda/AutoMoT/Automot/checkpoints` 是目录软链接。原 runtime_version 扫描 Automot 源码树时一律拒绝子目录软链接，因而在创建执行快照时误拒绝，尚未进入准入、探针或训练。另一台 `/mnt/workspace/cee3sgh/AutoMoT` 没有触发相同布局，不代表原规则兼容两机。

修复仅将明确路径 `Automot/checkpoints` 按资产目录共享；兼容真实目录、绝对软链接、相对软链接和悬空软链接。资产内部 JSON、权重和产物不进入源码版本，不递归扫描。悬空链接不制造资源，后续原生资源准入仍会拒绝缺失权重。其它源码目录（如 mot、team_code）软链接继续拒绝，资产链接目标被改动或替换仍拒绝；旧含真实 checkpoint 源文件的 v1 快照继续按原清单逐字验证，不迁移旧回执。

本轮 82 项相关 CPU 测试通过，覆盖上述布局、旧快照、篡改、真实子进程读共享权重/写共享结果、分支选择及恢复。以本机 2524 个真实源/依赖文件在隔离目录模拟 cr02 布局，执行 prepare 和 verify 成功；没有给本机原 Automot 添加链接，没有远程执行或加载 GPU 模型。

两机命令均从各自 AutoMoT 根、已激活环境启动。不要把另一台机器的旧 run 当作本机必有目录。下述命令仅在当前机器存在指定旧目录时请求原生恢复；原生恢复仍拒绝已有正式训练产物或不兼容证据，缺失时新建运行并正常准备。数据/资源检查不省略，不使用旧 Phase3 LoRA 或 Action 权重。

```bash
bash ./sync_automot.sh && \
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261025"))[0])') && \
(
  ABC_RECOVERY_ARGS=()
  ABC_RECOVERY="checkpoints/action_expert_ablation/geometry_abc_checkpoint_fixed/run_20261025_203923"
  if [[ -d "$ABC_RECOVERY" ]]; then
    ABC_RECOVERY_ARGS=(--recover-from "$ABC_RECOVERY")
  else
    echo "当前服务器没有旧运行目录，将正常重新准入。"
  fi
  bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
    --branch bev_only \
    "${ABC_RECOVERY_ARGS[@]}" \
    --maps "$ABC_ACTION_MAPS" \
    --data-root lead_data \
    --data-dir checkpoints/abc_trajectory_v1 \
    --model-dir checkpoints/Qwen3.5-4B \
    --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
    --output checkpoints/action_expert_ablation/geometry_abc_bev_only
)
```

只执行三组 BEV a/b/c 的探针、训练、测试与自动交包。每组七轮预算不变，输出目录由原入口创建独立运行。仍需 Qwen 基座路径完成共享准入/必要 processor 检查，不启动 Qwen 分支训练。如果另一套四卡任务正在运行，先结束或等待该任务，不并行争用 GPU。

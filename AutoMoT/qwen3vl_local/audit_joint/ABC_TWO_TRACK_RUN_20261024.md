# ABC 两路线发布与运行（2026-10-24）

两路线独立：Phase3 是只补强 a 提示词的新实验（b/c 与标定不变）；Action 是固定旧 abc v2 的 a/b/c × BEV/Qwen 六组 oracle 区域条件轨迹实验。旧 `phase3_compare.sh train abc` 仍复现旧版本，不是本次 refine 入口。旧任务已经完成时不要重新运行它覆盖旧 run。

以下命令默认已在服务器 AutoMoT 根目录，并已激活原训练 Python 环境。先使用服务器已有的 `bash ./sync_automot.sh` 同步最新 main。同步脚本本身不在本次发布内。本次只发布两个新包的源码、测试和说明；原始数据、地图、权重和原 abc adapters 不经 Git 上传。

## 1. Phase3 ABC 提示词小改

沿用上次 `phase3_compare.sh cpu/check/train abc` 形成的 canonical 数据与 run：

```bash
ABC_BASE_DATASET=$(python -c 'import json; from pathlib import Path; p=Path("checkpoints/geometry_abc_v2/ready_1280.json"); print(json.loads(p.read_text())["dataset"])')

bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh all \
  --baseline-dataset "$ABC_BASE_DATASET" \
  --baseline-runs-root checkpoints/geometry_abc_v2/runs \
  --model-dir checkpoints/Qwen3.5-4B \
  --output checkpoints/abc_refine_v1
```

若旧 run 使用自定义目录，应将 `--baseline-runs-root` 改为真正包含 `a_1280/b_1280/c_1280` 的目录。三组都需要完整 `epoch_3/state.pt`、receipt、training_complete 和 `test/cases.json`。轻量 `abc_review.tar.gz` 不能代替原 adapter。

`all` 自动完成新准备、旧test成员核验、同adapter成对val保护、完整processor、三个四卡探针、三组训练、test、回归与打包。每组3轮/9,600更新。数据与训练分别写到新run内部；已有原始数据、旧模型和旧ready保留。

仅先做CPU数据/计划检查（可选；正式all仍使用自己的新run/data）：

```bash
bash qwen3vl_local/audit_joint/phase3_abc_refine/run.sh prepare \
  --baseline-dataset "$ABC_BASE_DATASET" \
  --output checkpoints/abc_refine_v1_cpu/data
```

不要将此 prepare 命令误认为同adapter/processor/GPU验收通过。

## 2. Action ABC 六组

先用既有地图解析器生成统一的平坦地图视图。它只从项目CARLA及已知lead目录寻找完整十二Town，不混拼不完整地图源：

```bash
ABC_ACTION_MAPS=$(python -c 'from pathlib import Path; from qwen3vl_local.audit_joint.phase3_maps import resolve; print(resolve(Path.cwd(), Path("checkpoints/abc_action_maps_20261024"))[0])')

bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh all \
  --maps "$ABC_ACTION_MAPS" \
  --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc
```

地图在其它位置时，`resolve` 第三个参数可传该完整地图根的路径；不要直接将带Town子目录的根当成数据构建器要求的平坦xodr目录。

`all` 在数据不存在时建库，随后全量来源/历史/划分/七轮计划检查、processor、容量演练、六组四卡探针，全部探针成功后才开始长训；之后完整test、条件使用审计和自动打包。每组7轮/12,719更新，每轮116,256呈现（十事件各9,688，背景19,376），六组共同计划。训练和主评测用数据集真实未来区域，不接Phase3预测器或CARLA闭环。

仅先建库和CPU检查（可选）：

```bash
bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh prepare \
  --maps "$ABC_ACTION_MAPS" --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1

bash qwen3vl_local/action_expert_ablation/geometry_abc/run.sh check \
  --maps "$ABC_ACTION_MAPS" --data-root lead_data \
  --data-dir checkpoints/abc_trajectory_v1 \
  --model-dir checkpoints/Qwen3.5-4B \
  --lead-bev-ckpt checkpoints/tfv6_resnet34/model_0030_0_backbone_only.pth \
  --output checkpoints/action_expert_ablation/geometry_abc_checks
```

check打印实际run目录；如要接续该目录，在all命令中添加 `--run-dir <check打印的run目录>`。all仍重新核验来源。旧源码生成的准入/probe不可复用，应重新检查；正式训练半成品不支持隐式续训。

## 3. 环境、交付与验收边界

两个入口默认自动选四卡。同一组四卡上两条路线顺序运行。可通过 `PYTHON=/实际环境/bin/python` 指定解释器；需要指定卡号时，在入口前加 `GPU_IDS=0,1,2,3`。不要启动其它版本训练来绕过门控。

Action需要服务器原项目提供 `leaderboard/team_code/mot_lead_offline_runner.py`、`Automot/` 和相关环境依赖。这两个目录在用户发布白名单外，本次不上传、不改写。CPU准入会报告缺失依赖；代码已绑定本机验收用runner的内容SHA，发布验证记录列出该SHA供核对。原始LEAD数据、collection_output事件来源、完整地图和模型也须预先存在，不自动联网下载。

正常交付路径：

- Phase3：`checkpoints/abc_refine_v1/run_<tag>/delivery/abc_refine_audit.zip`
- Action：`checkpoints/action_expert_ablation/geometry_abc/run_<tag>/delivery/abc_trajectory_audit.zip`

两条路线都自动打印结果目录、包路径、字节数和SHA，主包硬限30,000,000字节。质量需复核时仍交付结果并退出2；前置条件或运行失败保留失败证据，不应当作训练成功。已完成训练需补包，可按各包README的pack入口执行，不重训。

本地独立审计250项不同CPU测试通过；全量Phase3记录及两路线计划保持原值；诊断支持格漏项为0。实际大模型processor、同adapter重放、四卡长输入探针及正式训练尚未执行，合成体积演练不能证明精度或GPU能力。后续以服务器真实回执为准。

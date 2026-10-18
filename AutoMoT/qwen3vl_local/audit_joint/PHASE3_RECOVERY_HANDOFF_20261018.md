# Phase3 中断恢复：先收集证据并保护现有资产

> 历史首次诊断说明，已收到并复核报告，不再执行下方首次收集命令。当前用户停止清理；使用 [PHASE3_CONTINUE_20261018.md](PHASE3_CONTINUE_20261018.md) 的独立源码接续流程。新工具、日志、报告全部放当前目录的新子目录，不使用下方历史示例的 `/tmp`。

本次用户指定 cr02，应用根为 `/home/cruser1/lda/AutoMoT`。工具本身没有机器到路径的映射；换机器必须使用该机的显式路径。用户列出的三个 pipeline 全部保留：

- `20261017_223254_126705688_2rgb_short_choice`
- `20261018_034614_519706091_2rgb_short_binary`
- `20261018_095559_213634718_4rgb_choice`

目前仅收到第3轮 step3720/3840 与 rank0/3 的单元素 ALLREDUCE 超时片段。不能判定根因、前两组全部完成、或断点文件完整。现有训练器在完整 epoch 结束保存 optimizer、scheduler、各 rank RNG 和曝光 history。若第二轮 checkpoint 有效，可从第二轮末恢复原3轮计划；不支持从未保存的3720步直接恢复。不能把仅 adapter 的目录当成完整断点。

## 获取独立工具

本轮不运行整仓库 `sync_automot.sh`，不切换 cr02 正在用于恢复的源码。仅在 `/tmp` 的独立 Git 仓库获取一个标准库脚本。下面 `P3_TOOL_REF` 使用本次回复给出的发布提交，也可使用 `main`，内容 SHA 不符会停止。不会 checkout 服务器训练工作区。

```bash
(
set -euo pipefail
ulimit -S -c 0 2>/dev/null || true
P3_DIAG=$(mktemp -d /tmp/phase3_recovery.XXXXXX)
echo "诊断目录：$P3_DIAG"
exec > >(tee "$P3_DIAG/launch.log") 2>&1
P3_TOOL_REF=main
git init -q "$P3_DIAG/tool_repo"
git -C "$P3_DIAG/tool_repo" fetch --depth=1 https://github.com/duguxiaohun/automot_lead.git "$P3_TOOL_REF"
git -C "$P3_DIAG/tool_repo" show FETCH_HEAD:AutoMoT/qwen3vl_local/audit_joint/phase3_recovery_inspect.py > "$P3_DIAG/phase3_recovery_inspect.py"
printf '%s  %s\n' f84058afd854baf67f1d79acba64c4ce866a503c04efbd8eaaa89cf070bcb517 "$P3_DIAG/phase3_recovery_inspect.py" | sha256sum -c -
P3_STATUS=0
AUTOMOT_SERVER=cr02 python3 -S -B "$P3_DIAG/phase3_recovery_inspect.py" \
  --root /home/cruser1/lda/AutoMoT \
  --run checkpoints/sft_new_loop_phase3_pipeline/20261017_223254_126705688_2rgb_short_choice \
  --run checkpoints/sft_new_loop_phase3_pipeline/20261018_034614_519706091_2rgb_short_binary \
  --run checkpoints/sft_new_loop_phase3_pipeline/20261018_095559_213634718_4rgb_choice \
  --disk-usage --output "$P3_DIAG/handoff" || P3_STATUS=$?
echo "退出码=$P3_STATUS；反馈目录=$P3_DIAG"
exit "$P3_STATUS"
)
```

如果 HTTPS Git 认证/连接失败，保留 `launch.log` 并反馈；也可用用户原有授权方式单独复制这个脚本，核对上述SHA后运行同一条 Python 命令。不要为取工具整体覆盖训练源码、修改全局 Git 配置或更换账户凭据。

## 反馈文件与实际检查边界

正常完成或部分目录检查失败，交回：

1. `handoff/phase3_recovery_report.zip`
2. `handoff/phase3_recovery_package.json`
3. `launch.log`

下载/启动失败且尚无handoff时只交 `launch.log`。退出2也可能已经生成失败诊断包，请保留并交回，不自动重试。

每个显式运行的子检查默认120秒，超时只结束该检查自己的子进程组；磁盘元信息扫描上限60秒，GPU进程查询上限10秒。都不执行训练、生成或张量反序列化。报告包含：原始训练Git/源码SHA/输入路径、checkpoint回执及源码差异、成员缺失/多余/越界、各rank各epoch末尾两行batch记录、每份训练/pipeline日志最后96KB、评测指标和打包manifest、当前主机/进程与磁盘占用。

默认不读取多GB checkpoint正文；`--verify-checkpoint-files` 是显式深查选项。即使SHA全匹配，也只证明文件绑定，实际恢复仍由原训练器核验配置、数据、基座stat、张量、optimizer与RNG。部分报告失败、缺失或超时不被当作不存在依赖的证明。报告ZIP只含检查JSON，不含权重/RGB/原始数据；成员SHA、ZIP SHA、大小及检查器SHA另存回执。

所有请求运行、发现的输入/基座/恢复点列入保护清单。Phase3/4目录和pipeline子目录目前一律 `unreviewed_keep`；没有生成可执行删除名单，也没有删除入口。进程快照不是无活动写入者的保证，磁盘扫描也不是可回收空间认证。

## 收包后恢复与清理顺序

先确认两图两组的train/test/打包状态和四图故障位置，校验明确的恢复点及其原源码。共享sequence源码已在后续Phase4更新中变化，不能改SHA绕过绑定。若要回查原版，只在独立目录恢复并逐文件对比，不回退当前工作区，也不移动原基座或数据。通过后给出只接续未完成组及后续四图binary的命令；不让全四组重跑。

清理必须另行列出精确目录、引用关系和预计空间，排除这三组、共享数据、基座、恢复状态、正在用的Phase4资产及需要保留的证据。不会按日期或 `phase3*`/`phase4*` 通配符删除。当前诊断没有全局依赖图，不能据此自动清除任何旧版本。

## 后续数据复用方案（尚未启用）

固定的用户入口可指向不可变、内容寻址的数据版本，避免时间戳相同内容重复建库。每个分片应记录schema、原始输入SHA、规则/共享依赖SHA、Git提交及dirty来源、父版本、来源hostname/用户机器标签、生成时间与本机路径。来源机器标签不是复用许可或规则身份。

增量编译需要明确路线/event的依赖图：局部规则变化仅重放受影响分片；共享规则、实例状态、停车历史、曝光隔离或split规则变化必须扩大重算范围，再全局校验成员、引用与配额。旧版本内容保持不变，新manifest记录分片复用链。该机制和自动清理仍待独立实现验证，本次没有更改生产数据格式或旧训练合同。

# Phase4：只使用现有数据的开发训练入口

用户明确要求不重新采集。停止CARLA补采方向；下面只读取已有`lead_data`，重新生成当前v54自动规则题库，不启动仿真、不生成新RGB、不改标签或提示词。无需提供LEAD/CARLA采集环境。

## 开训范围调整

此前首轮准入把五条关键边的YES/NO支持全部作为硬门；现有完整v52池仅缺`U-E5/proceed/readiness/YES`。缺这个标签意味着当前数据不能监督模型学会该条件，不意味着其它可靠样本不能开展开发训练。

新增显式`existing_data_development`模式，仅容许这一个已知支持格为空。它始终保留实际计数0和`unsupported_capabilities=["U-E5/proceed/YES"]`，成功状态为`admitted_existing_data_development`。不能把它写成放行能力已补齐、独立准确率认证或联合G0通过。`strict`仍是原入口默认值，仍阻断该缺格。其它关键格缺失继续阻断。

十事件缺失、训练/留出物理组冲突、固定等量评测失败、来源/合同漂移、两模式题序不同及空间预算不足继续阻断。训练计划每事件1024次/轮；val/test依各事件最小可用题数固定无放回等量。YES:NO不要求等量。本模式没有增设逐题人工审计门。跨任务完整联合G0仍须实际Phase3/Action/Phase4用途索引验收，单任务开发准入不代替它。

## 服务器下一步：现有数据转当前规则库，仅CPU

同步本次发布后，在现有AutoMoT目录运行一次。原始文件、旧正式库、旧权重都保留。服务器旧v52全池与当前v54生产合同不兼容，因此本次确实需要用已有原始数据生成新版题库；不是再次采集数据。无需重跑readiness，也不用旧fit。当前v54父库若已存在，可以用其有效原生资产另建轻量视图，不应重写旧SHA；此入口采用明确的新输出，避免自动误选旧版本。

```bash
cd /home/cruser1/lda/AutoMoT
conda activate automot
bash ./sync_automot.sh

MODEL_DIR=checkpoints/Qwen3.5-4B \
DATA_ROOT=lead_data \
REPLAY_WORKERS=2 \
OUTPUT_DIR="checkpoints/phase4_existing_$(date +%Y%m%d_%H%M%S_%N)" \
bash qwen3vl_local/audit_joint/server_phase4_existing.sh
```

使用480源`release_sources_v54_existing_data_20261013.json`。脚本依次完成新候选清单、当前原生gzip回放、共享RGB4规则库、自动留出、短两图/四图视图、首轮空间预算与开发准入；不训练。轮询保留20GiB，净写盘约12GiB上限；不是硬磁盘配额，不保证其它任务并行写盘仍能完成。建议与Phase3三轮错开执行，避免争用有限磁盘。若空间守护停止，交回失败包，不删除旧实验。

完成后交回脚本打印的：

- `handoff/fullflow_report.zip`（≤30,000,000字节）
- `handoff/fullflow_package.json`（大小、SHA及成员核验）

成功应为`admitted_existing_data_development`，报告继续显示U-E5能力缺口；失败包会保留具体阶段和其它阻断。只需交两个文件，不上传原始数据或权重。全池CPU时间取决于服务器，本机小库验证不代表完整池耗时。

## 该包通过后的固定首轮入口

下列命令只适用于本次正式CPU包已通过的输出目录。这里先把完整下一步写清楚，不要求现在同时启动GPU。将`P4_PREPARED`替换成上述实际输出目录：

```bash
P4_PREPARED=checkpoints/phase4_existing_实际时间戳
P4_TRAIN_OUT="checkpoints/phase4_existing_epoch_$(date +%Y%m%d_%H%M%S_%N)"
(
for P4_RGB in 2 4; do
  AUTOMATIC_COVERAGE_POLICY=existing_data_development \
  DATASET="$P4_PREPARED/view" \
  TEACHER_REFERENCES="$P4_PREPARED/references4" \
  FIRST_EPOCH_BUDGET="$P4_PREPARED/first_epoch_budget.json" \
  DATA_ROOT=lead_data MODEL_DIR=checkpoints/Qwen3.5-4B \
  RGB_MODE="$P4_RGB" OUTPUT_DIR="$P4_TRAIN_OUT" \
  bash qwen3vl_local/sft_new_loop_phase4/rgb_short/run_automatic.sh || exit "$?"
done
)
```

默认自动选择四张GPU；显式选卡时在上述训练调用前增加`GPU_IDS=0,1,2,3`。两模式顺序各一轮10240次呈现，固定val选优、最终检查点重载和固定test；不能以该命令示例冒称服务器已跑完或实际消费已验收。开发模式准入JSON保存在本次OUTPUT_DIR，包含覆盖限制。训练回传范围在CPU正式池通过后按实际产物固定，不要求重建该题库。

Phase3三轮配置、已经通过的首轮结论及数据复用路径不变。新发布仅更新源码清单指向，P3无需重建题库。

## 本机验证

完整Phase3/Phase4/audit_joint回归2478项通过，108.66秒。覆盖严格模式继续阻断、开发模式只容许指定缺格、其它关键格/缺事件/留出/隔离/预算/漂移继续阻断、原始与最终过滤分别诊断、失败仍打包以及训练入口显式传递模式。先前一次测试命令误用不存在的Phase3/tests目录，未执行测试；纠正路径后的完整日志保留，不将那次算作通过。

复用已有两个v54父库，仅新建视图，两图/四图各191+53=244条真实train ID和标签配对，父库21/13项输入绑定不变。小库留出为空，开发模式仍blocked；这不是正式池准入。当前480源匹配，原生缺模型prepare退出2并生成550字节双成员包；包SHA/CRC核验通过。真实Phase3已完成首轮包的三轮准入仍verified。无新RGB目视、全路线重放、CARLA或GPU运行，未修改生产标签/阈值/提示词。

证据：[phase4_existing_data_verification_20261013.json](phase4_existing_data_verification_20261013.json)；完整产物保留在`checkpoints/phase4_existing_development_20261013/`。

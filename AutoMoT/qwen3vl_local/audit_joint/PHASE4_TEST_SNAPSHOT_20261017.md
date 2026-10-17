# v81：RGB2训练和重载通过，固定test快照读取失败

收到phase4_epoch_report(3).zip，3270100字节、30成员，ZIP/成员SHA/CRC全部通过，带(1)副本相同，只计一次。563份源码逐项匹配v80。ZIP SHA cce299c90a7a78ce9304ef3861cd5fce069236e62d1aff4a066a8ddcc3d1b88d。证据留本机checkpoints/phase4_v80_received_20261017。

## 进度和此次失败

实际运行checkpoints/phase4_restart_20261017_162511。旧125707_138502059目录及权重已由用户删除，不能再给那个目录的接续命令；本次重训得到的权重必须保留。

- 视图迁移533.21秒成功；复用8614条路线缓存，没有重放老师。
- RGB2完整一轮，四rank实际10240次呈现逐条匹配原计划，十事件各1024。训练4416.45秒（73分36秒），val32.85秒。
- val370题，十事件各37，独立核对raw解析、配对身份、固定成员顺序和混淆矩阵。316/370=85.41%，YES18/66=27.27%，NO298/304=98.03%，全NO基线82.16%。与上轮统计相同。
- 四卡重载384.96秒通过：370题逐题结果与训练val包内容完全一致，prediction_mismatch_ids=[]、raw_text_differences=0；回执绑定本次checkpoint、优化器和案例SHA。权重未随ZIP上传，服务器报告核验通过，不宣称本地已读取权重。
- 随后固定test在加载validated_data/rgb2.receipt.json时，rank1报file changed while hashing；尚未加载test模型，test未完成，RGB4未启动。旧torchrun参数歧义已通过真实重载验证关闭。

旧hash只比较打开文件前与读完后的Path.stat，且错误不输出变化字段。因此当前证据不能证明是真实并发写动，还是打开挂载文件时刷新元数据；更不能归因训练、显卡、标签或NCCL。源码复查未找到正常复用阶段写入既有receipt的路径。无需重训、重载或建库来诊断。

## 本轮修正与验证边界

io.file_sha改在打开文件后，以实际文件描述符fstat建立读取基准，结束同时比较描述符及逻辑路径的device/inode/size/mtime/ctime和实际读取长度。打开时的元数据刷新不再被混同读中修改；读中改写、替换、软链接改指和时间戳变化仍拒绝。异常记录各字段前后值；不删mtime/ctime，不盲目重试、不跳过SHA。合成“open刷新mtime但字节未变”反例旧代码拒绝、新实现通过；这证明实现边界改善，**不是服务器根因已确证或已远程修复通过**。

phase4_continue支持本次exit1且已有成功reload的状态：核对完整采样与验证成员、重载报告/案例/原checkpoint SHA、当前adapter全部资产和optimizer字节后，跳过已完成的RGB2训练与重载。先CPU读取并验证现有RGB2快照，通过后执行RGB2固定test，再RGB4单轮/val/reload/test。缺快照拒绝隐式重建；首次失败即停，旧workflow/handoff保持，新诊断含snapshot_check日志。未调整冻结学生/teacher/提示词/采样/输入时间窗口和v69缓存适配器，当前view和checkpoint合同一致，不迁移、不重做原生验证证明。

186项相关回归14.73秒通过，含真实四CPU worker快照读取、四CPU分布式训练/评估、两CPU torchrun、哈希边界、缓存漂移拒绝、成功reload跳过与资产篡改拒绝、快照失败不启动GPU、打包及存储引用核验。首批42项含于186内，不相加。没有本轮服务器GPU结果，不承诺挂载文件若真实仍被修改也能继续。

## 质量结论

事件等量已验证训练1024/event、val37/event；固定test计划18/event尚未执行，RGB4尚无实际消费。UE4 YES0/29、UE6 YES4/22仍是模型效果问题，不能因为本次执行故障忽略。UE5 proceed YES仍无监督支持；UE2 return和UE7 complete评估覆盖不足。多事件val仅1物理组/单一NO，因此总体高分不能视作所有条件可靠。

自动规则数据仍可用于既定开发训练，不需要逐题人工审计；本包没有新的确定改标或提示词依据。本轮未新增RGB目视、未改变生产规则，不把此前小样本视觉复核冒充全源认证。先完成现有checkpoint的test及RGB4对照，不直接续七轮，不重做Phase3/readiness/batch选型/老师回放。

## 同步v81后仅接续当前目录

按现有sync_automot.sh的clone+rsync方式同步发布提交，不能在旧v80上执行下面命令期待新逻辑；不要使用服务器遗留.git判断同步。代码在一次运行期间保持不变。

```bash
export P4_RUN=checkpoints/phase4_restart_20261017_162511
export P4_PRODUCTION_CACHE=checkpoints/p4_verified_routes_v69
python -m qwen3vl_local.audit_joint.phase4_continue \
  --root "$P4_RUN" \
  --data-root lead_data \
  --model-dir checkpoints/Qwen3.5-4B
```

默认空闲四卡、B2/累积4，先CPU快照检查，再test和RGB4。前置核验如果失败，交终端报错；执行中失败交新包，不删除任何输出、不重复执行或重新训练绕过。

```bash
cat "$P4_RUN/continuation.json"
tail -n 40 "$P4_RUN/continuation_snapshot_check.log"
tail -n 40 "$P4_RUN/continuation_rgb2_test.log"
tail -n 40 "$P4_RUN/rgb4.log"
```

未到达的阶段尚无对应日志。成功或失败交回handoff_continued/phase4_epoch_report.zip和handoff_continued/phase4_epoch_package.json；最终ZIP仍<=30000000字节。请保留整个phase4_restart_20261017_162511，尤其epoch_000/adapter_model.safetensors、training_state.pt、view_upgrade、validated_data；ZIP不能恢复权重。

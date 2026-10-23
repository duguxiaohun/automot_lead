# Base 输入绑定清单 ctime 漂移恢复

远程四卡探针在 common.validate 对 input_bindings.json 散列时退出。用户日志显示dev、inode、size、mtime和读取字节数一致，仅ctime在读取中变化；之后两次严格读取均稳定并匹配原receipt。这证明文件当前与回执一致，不能证明最初元信息变化的根因已消除。仅建议手动重试并不足以修复启动流程。

新增phase3_base_hash_retry.py，仅为bounded base的common.validate中、由原common_receipt绑定的input_bindings.json提供一次有界恢复。首次严格散列因仅ctime变化而拒绝时，追加两次完整严格读取；恢复期间文件身份/大小/mtime/ctime须保持固定，两个SHA均须匹配原回执。最多三次完整读取，不重试第二次不稳定，不吞I/O或其它异常。内容修改、恢复mtime掩盖写入、替换文件/软链接、截断、持续ctime变化均拒绝。恢复开始及成功写stderr并标注rank，正常文件只读一次。

通用io.file_sha、冻结原生源清单及旧SHA不修改，原生验证仍执行。保护仅在bounded base的CPU准入、check和worker作用域安装并恢复；其它文件及single／abc不启用。新增助手源码与策略写入base合同，并通过兼容快照内容绑定、下发到真实子进程。旧base ready不能冒充当前源码准入，需重新cpu base；原生pool生成合同未变，原有兼容池通过来源检查后复用。失败缓存仍拒绝复用，不自动迁移未知数据。

本轮112项不同相关CPU测试通过（54.82秒），包括实际chmod造成ctime变化、真实内容写入/替换反例、作用域恢复、真实兼容子进程中的原生common.validate、采样/消费/完成验收与五组收尾。子进程测试只有index结构检查用合成替身，真实散列和receipt验证未替换。CPU测试不等于远程文件系统或真实四卡验收。完整证据只留本机checkpoints/phase3_hash_retry_review_20261023；轻量回执phase3_hash_retry_verification_20261023.json。未本轮正式全池、GPU或Phase4操作。现有ready文件字节未改，但base的旧源码合同需要更新；single／abc绑定源码保持。

## 当前远程重试

适用于用户已确认的状态：仅独立_probe失败，正式run尚未启动。先同步，再执行：

```bash
bash ./sync_automot.sh
bash qwen3vl_local/audit_joint/phase3_compare.sh cpu base &&
bash qwen3vl_local/audit_joint/phase3_compare.sh check base &&
bash qwen3vl_local/audit_joint/phase3_compare.sh train base
```

cpu需要重新验证来源并投影三轮；不是跳过准入，也不保证瞬时完成。train内部按prepare→四卡probe→train→summarize→test执行，探针失败不会训练。原独立_probe失败目录保留，新正式输出仍为phase3_compare_base/run_base_bounded_v3。不要重复运行已存在输出的probe base，不删除产物、改SHA或用verify把未完成任务标成完成。

如果已经出现正式run目录，不适用上述直接启动步骤，应先核对其状态；已有运行中的任务不能热更新源码。旧版本已完成权重须使用匹配的旧快照验收，本修复不自动迁移已完成训练合同。

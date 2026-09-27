# v23 语义基线的 ESTALE 工程维护

用户明确要求保留已修复的工程bug，不把它与未经效果验证的提示词/标定实验一同回退。
当前修订 `v23_io1` 的语义基线仍为v23，来源提交仍为 `b433aa605`；这不是v24效果晋升。

## 代码审查

从已审查的 `93102db1d` 提取文件发布修复，应用到v23构建器和并行扫描器。
只有 `build_dataset.py`、`parallel_scan.py`、`source_mapping.py` 三个快照文件变化：
前两个使用 `publish_file` / `atomic_write_text`，后一个同步新的实际mapping身份。
原始v23快照仍保留；新修订增加filesystem.py外部依赖SHA绑定。
实时Phase3源码也加入同一I/O修复，并将filesystem.py加入mapping源码哈希。

提示词、轨迹阈值/窗口、主要动作优先级、采样、split规划、所有标注JSON/JSONL均与v23快照字节一致。
没有引入v24完整训练池、smooth_cap或INVALID来源内配额重分配。
工程维护同样不能修改旧run的合同；即使输出语义未变，源码身份变化仍需校验/重建索引并新run。

Action外层 `_prepare_artifact` 的ready标记、内容校验、锁内续发及保留完成缓存一直保留，
本次补回上轮误撤的Phase3内部文件发布恢复。三条Action入口通过稳定登记共同使用修订。

## 验收证据

`test_phase3_io_backport.py` 的15项CPU检查通过：真实v23与修订构建器同一小样本配对，
比较全部产物内容及manifest（仅归一输出目录，受控使用同一mapping身份）；
六个文件发布点分别注入rename前失败/已成功却返回ESTALE，加上无故障路径，结果一致。
另检查67个冻结文件仅上述3个变动，原始文件身份表保持一致。
配对测试不读取真实全生产池，不代替训练机挂载或GPU效果验收。

## 部署与恢复边界

用户日志中的 `.candidate-*` 临时路径、157行入口及直接rename失败与本机当前ready恢复实现不同，
训练机须同步完整Action/Phase3相关修改及 `phase3_releases`，不能只替换提示词或stable.json。
同步后在AutoMoT目录执行 `python -m qwen3vl_local.action_prior.phase3_release show`，
应看到 `release=v23_io1`、`semantic_release=v23`。
同来源、同修订、ready/哈希完整的完成缓存可免扫描续发；旧脚本遗留的随机临时目录不保证可自动恢复。
不要删除 `.prepare.lock` 或手工改manifest/hash。有限重试耗尽仍明确失败并保留完成产物，
持续挂载故障需要训练机恢复存储服务；不承诺原始数据读取和训练权重I/O全程免故障。

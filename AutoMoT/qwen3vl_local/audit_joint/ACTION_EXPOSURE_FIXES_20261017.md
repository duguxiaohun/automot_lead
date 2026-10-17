# v72：连续段 action 补偿与运行缓存曝光绑定

每卡 B2 选型保持：Phase3 四卡、累积2、每更新16题；Phase4 四卡、累积4、每更新32题，均开启梯度检查点。不追加 batch benchmark 或服务器性能验收。

## 两项修正

共享训练采样器按候选段内的临时计数，累计每题实际减少的归一化 action 欠额：`min(1, max(0, target-count))/max(1,target)`。段内达到目标后不再给该 action 加分；候选间临时计数独立。少见题／锚点、连续段长度、尾部完整拼接优先级不变，同锚点的稀缺 action 机会补偿保留。它仍是软补偿，不保证全 epoch 全局最优。

反例四段分别为4A4B、4A4B、8A、8B，预算16、目标8A8B：原代码在两种步长均复现4A12B；修正后20个seed、两种步长均8A8B、两段完整8题、16题全新。下一epoch优先剩余16题。原有transition补偿、同锚点稀有action和6/6/2/2尾部拼接回归保持。

Phase4运行快照每次准备／复用都重新调用正式 `exposure_inventory()`，核对view中的完整曝光文件SHA清单，并把文件映射及排序后的物理group集合写入快照绑定。读取前后均检查，新增文件也通过动态枚举检测，不依赖源码合同的固定文件名单。曝光变化拒绝旧view；即便直接读缓存或使用reviewed数据，也执行检查。不由此推断历史训练已发生泄漏。

## 兼容性与使用

训练采样策略升为 `event_equal_contiguous_unseen_first_v4`，拒绝v1/v2/v3曝光history混用；学生运行快照升v2，旧快照不得改SHA继续使用。正式入口采用新OUTPUT_DIR准备新快照。

`phase4_view_upgrade` 新增已知v71源码支持，同时保留v61/v70。升级完整核验原生父库、两模式成员、排除项及RGB身份sidecar，再写新view；不覆盖旧view、不用于恢复旧checkpoint。曝光清单本身变化仍拒绝升级，不自动忽略新增开发曝光。

已选B2的正式首轮入口和参数沿用 [SAMPLING_RUNTIME_FIXES_20261016.md](SAMPLING_RUNTIME_FIXES_20261016.md) 中的命令，使用新OUTPUT_DIR；它可自动升级已知旧view。没有必要再运行显存/吞吐比较。这里不启动服务器训练，也没有收到新服务器view成功回执。

`phase4_load.py`、`phase4_production_cache.py` 及原生producer/teacher源码字节不变，已完成的v69路线缓存仍可复用；无需重建原生库或重放老师。

## 本轮验证

两个反例先新增回归，旧代码4项均失败；修正后327项相关CPU回归通过（26.23秒）。含真实本地reviewed小库的v71描述符fixture升级、两RGB成员一致、四个真实CPU worker禁止native loader仍成功读取缓存；库存文件增删改/清空、模拟曝光变化与原始loader同样拒绝、读取期间group变化拒绝；旧history、未知升级源码、坏缓存及磁盘余量规则保持。

没有新增目标4B/GPU/服务器整池训练。Phase3跨进程续训、批量生成评估、共享RGB视觉编码去重仍未实现；Phase4推荐入口仍是首轮流程，不能重复启动当作续训；两种RGB的评估成员可不同。

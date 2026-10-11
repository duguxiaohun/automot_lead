# v50发布后：服务器先交CPU与正式资产盘点包

当前先执行本页，不直接执行旧七轮训练，也不直接启动全量回放。v50生产接线已完成；服务器正式资产的版本、范围和空间仍待确认。后续建库/一轮训练模板保留在 [NEXT_TESTS](NEXT_TESTS_20261011.md) B–D，满足其前提后再用。

## 1. 同步到独立源码目录

使用服务器现有的 `sync_automot.sh`，源仓库为 **duguxiaohun/automot_lead，main**。沿用该脚本已经验证的参数，保留旧实验目录/数据/权重。进入同步出的 `AutoMoT/` 应用目录（当前目录应包含 `qwen3vl_local/`）。应用目录遗留的Bosch `.git` 与该分发无关，不用它判断本次同步成功。

本次内置 `release_sources_v50_20261011.json` 源文件SHA清单；交接程序会逐文件比对。`source_status=matched`仅表示与内置清单一致，远程提交来源仍以同步脚本实际clone的主仓库为准。

## 2. 运行一次CPU交接

激活服务器已有 `automot` 环境，在新的应用目录运行：

```bash
MODEL_DIR=checkpoints/Qwen3.5-4B \
  bash qwen3vl_local/audit_joint/server_readiness.sh
```

基座不在默认路径时，仅替换为现有的真实位置；不下载模型。checkpoint根目录不同可再设置 `CHECKPOINTS_ROOT=/真实路径`。默认输出为一个新的 `checkpoints/server_readiness_v50_时间戳/`；重跑生成新目录，不覆盖原报告。

脚本依次执行相关CPU回归、本地模型/vendor环境preflight、checkpoint元数据盘点、发布源码SHA比对及打包。禁用core；pytest使用已有20GiB余量守护。不会启动训练、全量标定、下载、清理或复制production。preflight不是四卡内核/性能测试。

CPU测试/环境检查失败也会继续生成诊断包，脚本最终退出2。不要仅凭退出2认定数据坏了；把报告交回即可。`source_status=mismatch`须先处理同步，不能据测试通过绕过它。这里的 `training_ready=null` 是预期：盘点不替代正式题库准入。metadata里的counts/task是**未认证的原文件声明**，不是通过v50加载器后的统计。

## 3. 交回两个文件

脚本最后打印准确路径。只需提供：

- `checkpoints/server_readiness_v50_时间戳/report.zip`：**不超过30,000,000字节**，含源码SHA核对、资产元数据/路径/大小、磁盘余量、tests/preflight日志与退出码及包内SHA回执。
- 同目录 `package.json`：ZIP大小、SHA及核验结果。

日志单文件最多保留末尾2MB，截断会明确记录；元数据盘点最多2000条，截断/读失败会标明incomplete。不读取训练索引行或权重内容，不带adapter、基座、RGB、production及原始split。目录软链接不递归；若资产在其它挂载点或通过目录软链接存放，请同时提供真实目录，并用 `CHECKPOINTS_ROOT` 指向该根再次收集，不把“未扫描”解释为丢失。

本包采用独立的 `server_readiness_v1` 格式，复验命令为：

```bash
python -m qwen3vl_local.audit_joint.server_readiness \
  --verify checkpoints/server_readiness_v50_实际时间戳/report.zip
```

不要用G0的 `verify-package` 验证这种CPU盘点包；它不包含G0捕获回执。ZIP的verified只证明包内字节一致，不证明资产可训练。若脚本在打包前被中断，则交回已有tests/preflight日志和最后报错；不要上传整个checkpoints目录。

## 4. 收到包后的下一阶段

先确定真实数据根目录、Phase3正式索引/variant、旧资产版本、候选路线范围与磁盘预算，再给出v50正式池生成命令。正式池已有且匹配v50时，可直接进行两模式自动规则留出/CPU采样核验；旧版资产不能改hash迁就新版加载器。

核对十事件train/val/test及关键条件支持后，才用新基座初始化两模式各完整一轮并做保存重载。之后交回每组run.json（含固定评测清单）、data_admission、逐题val输出、采样/逐rank执行日志、检查点身份和重载报告；沿用G0交接白名单打包，不含权重。详细GPU命令见NEXT_TESTS的D段。旧U-E1/U-E7拟合无需重复。

本地标定与RGB审计可同时继续，双方以相同生产源码SHA为界。若后续审计需要改生产规则，先登记新版本与影响，不能让远程正在运行的实验混入新的老师/提示词代码。

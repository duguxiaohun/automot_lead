# Qwen 本地代码运行兼容性复核

范围：递归扫描 qwen3vl_local 的 Python/Shell，重点复核当前 single、ABC token、ABC trajectory 和公开 compare/geometry 入口。不是全部历史实验的逐程序运行认证，也不是4B GPU或CARLA在线验收。

## 本次修复

- single先使用显式PYTHON、激活虚拟环境/Conda，再回退项目.venv与PATH python/python3。无需使用.venv。入口打印实际解释器。
- single正式流水线移除不必要的顶层ijson导入；只有独立流式审计需要ijson。启动前检查缺包/二进制导入错误，不自动安装或升级。
- single CPU失败可在无probe/run时重试all，旧失败指针保存在cpu_attempts。拒绝对已有probe/run重建准入；复用失败不覆盖原ready。修复自定义native-ready输出目录，GPU子进程清除旧分布式环境变量。
- 四个公开Shell（phase3_compare、geometry_study_abc、phase3_abc_refine、action_expert_ablation/geometry_abc）使用共享runtime_env.sh继承选择的解释器。ABC入口统一在应用根运行，避免默认数据/模型路径跟随调用目录。
- 新ABC token的流式prepare/all确实需要ijson，已前移依赖检查；ABC trajectory检查其所需环境。所有安装提示指向实际解释器，CARLA提示使用项目匹配的wheel。
- 之前single图像路径修复保留：允许已验证成员经过外部场景软链接，仍拒绝绝对路径/父目录跳转/无效链接/非文件，并校验内容与来源。

## 不使用.venv

在服务器已有Conda环境或系统Python依赖完整时直接执行原公开入口。PYTHON显式设置具有最高优先级；如果此前设置过PYTHON，它仍优先于新激活的Conda。

冻结历史入口继续保持其字节/来源合同。如果需要使用它们而不希望其历史.venv优先规则选错环境，用外层适配器：

```bash
bash qwen3vl_local/run_with_env.sh qwen3vl_local/audit_joint/phase3_base_default.sh status
```

该例只读查看已固定的base默认选择；不启动base实验。旧入口具体可用子命令仍以它的帮助和原文档为准。它不改冻结源码，也不伪造旧ready来源。

## 路径扫描的区别

递归扫描覆盖1096个Python、118个Shell；Python按原始字节解析以正确处理UTF-8 BOM，Shell使用bash -n，均无语法失败。路径候选逐项按当前入口、冻结旧入口、输出隔离、来源快照/交付包安全边界分类。不能把所有is_relative_to删除：输出不得覆盖输入，以及交付包不得逃出自己的目录，仍是正确检查。

base_dense/provenance维护逻辑路径与resolved mount证据，支持外部路线挂载；geometry原生来源同样绑定目录目标和内容。没有修改原生来源校验、标签或阈值。

暂停的phase3_base_prompt_experiment仍含旧的物理root图片限制，已列为未启用实验的已知兼容性问题。本轮遵守暂停base审计/改进决定，没有恢复或修改该实验。若以后重新启用，必须单独修复并重新绑定其计划，不能使用当前计划假称已兼容。

## 验证边界

本轮整批覆盖single/single_v3、ABC token、ABC trajectory及公开入口，包含真实小Qwen、四Gloo通信、外部软链接准备/重载、跨工作目录、带空格路径、环境选择和失败接续测试。最终无Git发布副本197项通过（29.29秒，258条现有依赖警告），五个真实Shell/模块帮助入口从项目目录外调用通过。详见本机checkpoints/phase3_single_portability_20261024发布回执，不与历史批数累加。

single最终源码全池CPU再次通过：204910有效锚点、35231计划输入真实解码、三轮3200更新。容量/标签/采样未改变；不代表远程数据必然相同。CPU全池来源+小模型测试不能保证未知GPU、NCCL、磁盘或外部中断永不出错，真实四卡probe仍位于正式训练之前。

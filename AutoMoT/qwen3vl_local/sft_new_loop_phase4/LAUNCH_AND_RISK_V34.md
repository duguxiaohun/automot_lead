# Phase4 v34：启动路径、依赖一致性与因果风险窗口

v33 只读审计指出的源码不匹配与两个入口缺陷均成立。本轮保留当前 Phase3 修改，按最终源码构建新的 v34 产物；不覆盖 v33，不修改旧 manifest 哈希。未新增训练标签或执行 GPU/CARLA 训练。

## 启动修复

`train.py`、`preflight.py`、`launch.py` 均禁用 argparse 参数缩写。shell 提前拒绝路径参数的完整写法和前缀缩写，包括 `--datas=/tmp/other`、`--data-r`、`--model-d`、`--output-d`，也覆盖分离的参数值写法。路径统一来自 DATASET/MODEL_DIR/DATA_ROOT/OUTPUT_DIR 环境变量；直接 Python 入口同样拒绝缩写，避免只在 shell 层防护。

`run_full_pipeline.sh` 的 `check`、`preflight`、`host-preflight` 都在检查完成后退出，不读取 `best.json`、不选择评估 GPU、不评估遗留 adapter。回归测试特意放入旧 best.json 验证这条控制流；这是进程边界测试，不是假称主机已加载模型成功。

默认数据目录更新为 `checkpoints/phase4_v34_full/data4`，RGB_MODE=2 使用 data2。默认 4B 基座权重仍缺失；主机检查必须明确报告 MODEL_DIR，不能把逻辑 trainable=true 当作本机已经可训。

## 源码与数据一致性

v33 记录的 Phase3/build_dataset.py SHA 为 `b7c717528872dfeaa7e0af821ffe7eea921daa9a12cd8fdd078d22c36481150d`，本轮开始时为 `21ea65e49fdc266eaef38b510b7494ba9016bd2076ff299ab9accf1a8dd6e4aa`。当前差异增加了 Phase3 额外开发物理组的 train-only 处理。本轮未撤销或改写该文件；Phase4 的实际物理划分要通过重扫逐组比较，而不是根据代码差异猜测变化。

`check_contract` 现在在报错中给出来源类别、文件名、保存 SHA 和当前 SHA。完整构建后，再次运行默认 2/4 图 `train.sh check`，并在最后验证所有绑定源码/资产/依赖与构建时相同。验收记录包含 UTC 时间和完整合同摘要；其后若工作树再变化，旧通过记录不代表新源码已验收。

## 风险登记与逐题核对

新增 13 个实际看过的训练曝光组和 11 处精确车辆不连续窗口，绑定相邻帧 RGB SHA；选中但未看的验证路线未加入曝光集。相机图像只确认可见不连续，不推断仿真删除原因。

新记录显式使用 `manual_scope=causal_history_window`，检查人工题整个因果历史包络，包括两张图之间的间隔。窗口之外不因同一路线登记了异常而自动撤回题目。旧记录的整路线复审承诺保持；没有定位帧的风险仍按整条路线未解决处理。审核卡、标注编译、逐题投影和加载验证采用一致的范围判断；已有显式逐帧uncertain/excluded判断在跨窗口审核段中继续保留，不因当前帧在窗口外而消失；命中窗口仍须真实逐帧复审，不能自动变成 NO。弱老师继续检查全部当前/历史源证据和 STOP 停稳证明，不只看模型输入的 2/4 张图。

对旧 v33 题目逐题检查：每模式，本批训练曝光路线中有 18 道训练题、3 道待审题，新增窗口与这些题目的完整因果证据重叠数均为 0。不能因此声称整个弱老师准确，也不能仅凭源数据异常指控当前标签已污染。详见新产物的 `new_risk_supervision_impact.json`。

本轮实际复看 2 条既有训练路线的 4 张原图：BlockedIntersection/Town13 69_0 的99/100帧、DynamicObjectCrossing/Town07 Scenario3_1 的24/25帧。均支持可见不连续登记，不提供释放标签。无新增完整序列、独立 RGB 或人工监督。用户报告的13序列/2003帧联系表筛查和39张原图属于继承工作；累计203/278格达到三个ID筛查数量，仍余215条选定序列，18格源不足，筛查不等于逐题认证。

## 使用与验收边界

```bash
# 在仓库根目录运行；检查数据与默认七轮采样，不训练。
PYTHON=/tmp/automot-qwen35-env/bin/python RGB_MODE=2 \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh check
PYTHON=/tmp/automot-qwen35-env/bin/python RGB_MODE=4 \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh check

# 需要已经安装完整本地模型；该命令本身不训练。
PYTHON=/tmp/automot-qwen35-env/bin/python MODEL_DIR=/path/to/local/Qwen3.5-4B \
  bash AutoMoT/qwen3vl_local/sft_new_loop_phase4/train.sh host-preflight
```

完整验证/测试继续遍历独立参考集，不为事件1:1丢弃测试题；已有逐事件、事件宏平均、缺事件列表及老师一致率分别报告。训练的呈现等额与每题重复上限保持，因此大题池不等于默认七轮充分覆盖。UE2/UE4绕行、RE2进入、UE7推进等监督与六事件独立评估缺口未被本轮代码修复；Action/CARLA在线接管仍未验收。正式十事件闭环条件尚未满足。

完整产物位于 `AutoMoT/checkpoints/phase4_v34_full`。最终核验结果见 `thirty_fourth_audit_verification_20261006.json`，源码验收时点见产物中的 `final_source_acceptance.json`。

## 最终实际结果

878项专项通过；最终默认启动检查在所有构建和验证之后再次执行，2/4图各七轮均通过。源码验收时点为 `2026-10-06T05:38:04.461361+00:00`，合同摘要为 `131518f66e22ab8f6032a3f070e1b00d08e270c648ad940839915279583cd5db`；当前默认主机检查唯一报告本地4B模型缺失。没有下载模型或启动GPU训练。

| 模式 | 训练题 | 弱题 | 每轮呈现 | 七轮不同题 world1/world4 | 池覆盖比例 | RGB核验题数 |
|---|---:|---:|---:|---:|---:|---:|
| 2图 | 75373 | 74930 | 400 | 705/705 | 0.9353% | 76371 |
| 4图 | 75373 | 74930 | 400 | 705/705 | 0.9353% | 76371 |

原443人工训练题逐行保持，val273/test186/review539逐字节保持；18个UE1旧静止正例保留，运动UE1 readiness YES与弱restrict仍为0。新增窗口没有给任何题目制造YES或NO。

每模式留出老师参考为val8345/test7952，仅衡量老师一致率；盲审961卡/56路线未填，严格批准0。弱数据trainable=true、formal_data_ready=false。原v33启动失败的证据与原产物保留，新结论只适用于v34及记录的源码版本。

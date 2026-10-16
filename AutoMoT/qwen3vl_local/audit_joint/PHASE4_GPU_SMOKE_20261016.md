# Phase4 GPU独立冒烟测试

v64服务器迁移180秒退出124，160.6秒已复用协议表20万次，但全池校验和视图迁移未完成。它不是GPU错误，也不能据此报告GPU吞吐。继续调大迁移超时无法满足先短时确认硬件/模型路径的需求。

新增phase4_gpu_smoke将运行诊断与整库准入分开。transport不读取任何数据或模型：每卡CUDA矩阵乘、NCCL all_reduce数值核验。model先做同样检查，再从原生train.jsonl前100行中读取第一条YES/NO样本；原生源码合同校验、实际所需2/4图解码与SHA检查后，使用正式short_row/four_row与encode_batch，GC、LoRA16/32、vision off，一次DDP前反向及AdamW。输入两图[t-2,t]、四图[t-3,t-2,t-1,t]。原样本相同图的SHA必须匹配，新增四图时刻只记录当前读取身份，不伪称新标签认证。

这是运行诊断：每卡重复同一条训练样本，不是十事件均衡训练，不检验整库成员SHA、完整老师证据、动作均衡、采样连续性或模型质量；所有summary明确full_pool_admitted=false。不生成学生view，不保存权重，不允许用结果替代正式数据准入或推荐batch容量。

同步本次白名单代码后，在服务器AutoMoT目录、venv中先执行：

```bash
python -m qwen3vl_local.audit_joint.phase4_gpu_smoke \
  --check transport --gpus 0,1,2,3 --timeout 60 \
  --output "checkpoints/p4_cuda_smoke_$(date +%Y%m%d_%H%M%S_%N)"
```

四rank全部transport_passed且exit0后，再执行单组模型测试：

```bash
python -m qwen3vl_local.audit_joint.phase4_gpu_smoke \
  --check model --gpus 0,1,2,3 --rgb-mode 2 \
  --native checkpoints/phase4_v57_20261014_232157_762998141/data4 \
  --data-root lead_data --model-dir checkpoints/Qwen3.5-4B \
  --timeout 180 \
  --output "checkpoints/p4_model_smoke_rgb2_$(date +%Y%m%d_%H%M%S_%N)"
```

父进程限时涵盖torchrun启动、模型加载及一次更新；另有少量进程清理时间。每阶段实时输出，30秒调用栈仅是进度诊断。失败就停止，没有trial矩阵。输出只含worker.log、各rank报告及summary.json；交summary.json，失败再交worker.log。两图成功后才把rgb-mode改4，使用新输出目录。模型加载本身也可能超过180秒，届时按阶段日志处理，不宣称模型不支持GPU。

本地验证包括两种输入选样/时刻、原图变化拒绝、非train拒绝、只读第一行不扫描整库、无模型/数据时transport仍可启动、缺rank失败、既有真实进程超时/信号清理及CPU/Gloo梯度测试。目标4B权重本机缺失，未做本地CUDA/模型实测。

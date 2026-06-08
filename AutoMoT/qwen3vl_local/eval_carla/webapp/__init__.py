"""eval_carla.webapp 包入口。

这里不在 import 时创建 Flask app 配置；真正的 eval_base、scenario 映射由
app.py 的 main() 在命令行参数解析后注入，方便同一个 app.py 指向不同结果目录。
"""

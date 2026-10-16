# 本地运行契约

复用 Python `C:\venv_bisenet\Scripts\python.exe`，PaddlePaddle 3.4.0，RTX5060 8GB，FP32。不安装新框架。数据只读 `C:\Users\user\algolab_data`；新文件仅写本波次目录。

启动前原样执行下面的 GPU 和源码见证。stdout 必须有 `SOURCE_WITNESS` 及 `GPU_WITNESS`，返回码必须是0。该检查只验证运行环境，不替代每个候选的正式训练入口 smoke。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='D:\论文复现\experiments\segformer_balance_wave6_20261016\src\PaddleSeg-2.10.0'
& 'C:\venv_bisenet\Scripts\python.exe' -c "import paddle,paddleseg; paddle.seed(42); paddle.set_device('gpu:0'); x=paddle.randn([4,32,25,25]); m=paddle.nn.Conv2D(32,32,3,padding=1); y=m(x); y.square().mean().backward(); print('SOURCE_WITNESS',paddleseg.__file__); print('GPU_WITNESS',paddle.__version__,list(y.shape),float(y.mean()),paddle.device.cuda.get_device_name())"
```

正式队列：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\venv_bisenet\Scripts\python.exe' 'D:\论文复现\experiments\segformer_balance_wave6_20261016\experiment_queue.py' --run
```

队列先复核原B0固定权重和数据哈希，再做全部smoke，然后串行18k训练、验证选模、冻结、独立val.py和精确边界评估、参数及速度测量。`queue.lock` 防止重复运行。子进程使用新的PYTHONPATH，覆盖旧editable安装；日志与真实退出码逐项保存。系统盘的原数据、框架和权重只读使用。

# RTX5060 Paddle GPU后端诊断（2026-10-24）

## 已确认

- RTX5060 为 Blackwell，Compute Capability 12.0 / SM120。
- 驱动：596.21，CUDA driver API 13.2。
- 本机已安装 CUDA Toolkit 13.4。
- cuDNN DLL来自 Python NVIDIA包，Paddle运行时可读到 cuDNN 9.9。
- Paddle 3.2.2、3.4.0 和官方 PaddleX Windows 50系列 wheel 3.0.0.dev20250717 均能完成最小 Conv2D GPU反向，但 PSPNet、DeepLabV3P、Fast-SCNN 在复杂 Conv/BatchNorm 反向阶段失败。
- Paddle 3.3.1 官方 cu129 Windows wheel 同样能完成最小 Conv2D反向，但 Fast-SCNN smoke 仍在 Conv2dGrad/cuDNN 失败。
- FLAGS_conv2d_disable_cudnn、FLAGS_cudnn_batchnorm_spatial_persistent=0 和确定性/搜索相关 flags 未消除失败。

## 根因边界

问题不再是缺少 DLL 或单纯版本号。当前 Windows GPU wheels 虽然能识别 SM120 并加载 cuDNN，但目标网络使用的复杂 cuDNN 反向 kernel 在 RTX5060/SM120 上仍不稳定。官方 PaddleX 文档也注明 Windows 50系列 wheel 存在训练相关已知问题。

## 保留证据

所有 smoke 日志和真实 exit JSON 保留在本目录的 logs/results 中；正式数据、模型和 test 结果未被改写。

## 可行修复路径

1. 使用带 SM120 编译的 Linux Paddle wheel（需要 WSL2/Linux/服务器）；
2. 或在具备 CUDA 13.x、Visual Studio/CMake 的环境中从 Paddle 源码以 SM120 重新编译；
3. 当前 Windows环境不再继续堆叠 flags，也不把 CPU结果混入RTX5060表格。

import sys, runpy, paddle
paddle.set_flags({'FLAGS_conv2d_disable_cudnn': True})
runpy.run_path(r'D:\论文复现\external\PaddleSeg-2.10.0\tools\train.py', run_name='__main__')

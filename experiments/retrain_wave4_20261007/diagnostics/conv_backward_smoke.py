import paddle, os
print('paddle', paddle.__version__, 'cuda', paddle.version.cuda())
paddle.set_device('gpu:0')
x=paddle.randn([1, 3, 400, 400], dtype='float32'); x.stop_gradient=False
conv=paddle.nn.Conv2D(3,64,3,padding=1)
y=conv(x); loss=paddle.mean(y*y); print('forward', list(y.shape), float(loss)); loss.backward(); print('backward_ok')

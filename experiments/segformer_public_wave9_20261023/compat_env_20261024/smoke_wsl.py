import paddle
print("VERSION", paddle.__version__, "CUDA", paddle.is_compiled_with_cuda(), "DEVICE", paddle.device.get_device())
x = paddle.randn([1, 3, 32, 32])
c = paddle.nn.Conv2D(3, 4, 3)
y = c(x)
y.mean().backward()
print("BACKWARD_OK")

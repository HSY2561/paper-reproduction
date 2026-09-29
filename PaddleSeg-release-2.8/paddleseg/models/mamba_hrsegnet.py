import math
from typing import Any
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, KaimingNormal, Assign
from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init


# --------------------------------------------------------
# VSSM / Mamba 相关模块实现 (已修复和完善)
# --------------------------------------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class VSSBlock(nn.Layer):
    """
    Vision State Space Block
    """

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Layer = nn.LayerNorm,
            ssm_d_state: int = 16,
            ssm_dt_rank: Any = "auto",
            ssm_ratio=2.0,
            **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ssm_d_state = ssm_d_state
        self.ssm_dt_rank = math.ceil(hidden_dim / 16) if ssm_dt_rank == "auto" else ssm_dt_rank
        self.ssm_ratio = ssm_ratio

        self.norm = norm_layer(hidden_dim)

        self.ss2d = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            dt_rank=self.ssm_dt_rank,
            d_conv=3,
            expand=ssm_ratio,
            **kwargs
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # 输入x的格式: (B, H, W, C)
        shortcut = x
        x = self.norm(x)
        x = self.ss2d(x)
        x = shortcut + self.drop_path(x)
        return x


class SS2D(nn.Layer):
    """
    2D-Selective-Scan module
    """

    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.0,
            dt_rank="auto",
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # 确保是整数
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 用于卷积和SSM参数的投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias_attr=False)
        self.conv2d = nn.Conv2D(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias_attr=False
        )
        self.act = nn.Silu()

        # x_proj 只为 dt, B, C 生成参数
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias_attr=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias_attr=True)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias_attr=False)

        # SSM state matrix A 和 D
        A = paddle.arange(1, self.d_state + 1, dtype='float32').reshape([1, -1])
        A = paddle.tile(A, [self.d_inner, 1])
        # 使用Assign从张量初始化
        self.A_log = self.create_parameter(shape=A.shape, default_initializer=Assign(paddle.log(A)))
        self.D = self.create_parameter(shape=[self.d_inner], default_initializer=Constant(value=1.0))

    def forward(self, x):
        B, H, W, C = x.shape

        x_in = self.in_proj(x)
        x1, x2 = x_in.chunk(2, axis=-1)

        x_conv = self.conv2d(x1.transpose([0, 3, 1, 2])).transpose([0, 2, 3, 1])
        x_conv = self.act(x_conv)

        y = self.s6(x_conv)

        y = y * x2
        y = self.out_proj(y)

        return y

    def s6(self, x):
        A = -paddle.exp(self.A_log.astype('float32'))
        D = self.D.astype('float32')

        x_proj = self.x_proj(x)
        delta, B_proj, C_proj = x_proj.split([self.dt_rank, self.d_state, self.d_state], axis=-1)
        delta = self.dt_proj(delta)

        y_list = []
        for i in range(4):
            if i == 1:
                x_i, delta_i, B_i, C_i = [paddle.flip(t, axis=[2]) for t in [x, delta, B_proj, C_proj]]
            elif i == 2:
                x_i, delta_i, B_i, C_i = [paddle.flip(t, axis=[1]) for t in [x, delta, B_proj, C_proj]]
            elif i == 3:
                x_i, delta_i, B_i, C_i = [paddle.flip(t, axis=[1, 2]) for t in [x, delta, B_proj, C_proj]]
            else:
                x_i, delta_i, B_i, C_i = x, delta, B_proj, C_proj

            y_scan = self.selective_scan(x_i, delta_i, A, B_i, C_i, D)

            if i == 1: y_scan = paddle.flip(y_scan, axis=[2])
            if i == 2: y_scan = paddle.flip(y_scan, axis=[1])
            if i == 3: y_scan = paddle.flip(y_scan, axis=[1, 2])

            y_list.append(y_scan)

        return sum(y_list)

    @staticmethod
    def selective_scan(u, delta, A, B, C, D):
        # 修正: 更改变量名以避免覆盖
        B_sh, H, W, d_inner = u.shape
        N = A.shape[1]

        deltaA = paddle.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)

        h = paddle.zeros((B_sh, H, d_inner, N), dtype=deltaA.dtype)
        y_list = []

        for i in range(W):
            h = deltaA[:, :, i] * h + deltaB[:, :, i] * u[:, :, i].unsqueeze(-1)
            y = h @ C[:, :, i].unsqueeze(-1)
            y_list.append(y.squeeze(-1))

        y = paddle.stack(y_list, axis=2)

        return y + u * D


# --------------------------------------------------------
# HrSegNet 核心模块修改
# --------------------------------------------------------

class MambaSegBlock(nn.Layer):
    """
    在低分辨率路径中集成了Mamba模块的SegBlock
    """

    def __init__(self,
                 base=32,
                 stage_index=1):
        super(MambaSegBlock, self).__init__()
        low_res_channels = base * int(math.pow(2, stage_index))

        self.h_conv1 = nn.Sequential(
            nn.Conv2D(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU()
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU()
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv2D(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU()
        )

        if stage_index == 1:
            self.l_downsample = nn.Sequential(
                nn.Conv2D(in_channels=base, out_channels=low_res_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2D(low_res_channels),
            )
        else:
            self.l_downsample = nn.Sequential(
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                nn.Conv2D(in_channels=base, out_channels=low_res_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2D(low_res_channels),
            )

        self.mamba_block1 = VSSBlock(hidden_dim=low_res_channels, ssm_d_state=16)
        self.mamba_block2 = VSSBlock(hidden_dim=low_res_channels, ssm_d_state=16)

        self.l2h_conv1 = nn.Conv2D(in_channels=low_res_channels, out_channels=base, kernel_size=1, stride=1, padding=0)
        self.l2h_conv2 = nn.Conv2D(in_channels=low_res_channels, out_channels=base, kernel_size=1, stride=1, padding=0)
        self.l2h_conv3 = nn.Conv2D(in_channels=low_res_channels, out_channels=base, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]

        out_h1 = self.h_conv1(x)
        out_l1_down = self.l_downsample(x)
        out_l1 = self.mamba_block1(out_l1_down.transpose([0, 2, 3, 1])).transpose([0, 3, 1, 2])
        out_l1_i = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True)
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.mamba_block2(out_l1.transpose([0, 2, 3, 1])).transpose([0, 3, 1, 2])
        out_l2_i = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = F.relu(out_l2)
        out_l3_i = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3

        return out_hl3


# --------------------------------------------------------
# 主模型和分割头 (保持不变)
# --------------------------------------------------------

class SegHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super(SegHead, self).__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.relu = nn.ReLU()
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2D(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2DTranspose(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        self.conv = nn.Conv2D(in_channels=interplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out


@manager.MODELS.add_component
class MambaHrSegNet(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 base=16,
                 num_classes=2,
                 pretrained=None):
        super(MambaHrSegNet, self).__init__()
        self.base = base
        self.num_classed = num_classes
        self.pretrained = pretrained

        self.stage1 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=base // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(base // 2),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2D(in_channels=base // 2, out_channels=base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU(),
        )

        self.seg1 = MambaSegBlock(base=base, stage_index=1)
        self.seg2 = MambaSegBlock(base=base, stage_index=2)
        self.seg3 = MambaSegBlock(base=base, stage_index=3)

        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.head = SegHead(inplanes=base, interplanes=base, outplanes=num_classes)

        self.init_weight()

    def forward(self, x):
        logit_list = []
        h, w = paddle.shape(x)[2:]

        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)
        last_out = self.head(hrseg3_out)

        if self.training:
            seghead1_out = self.aux_head1(hrseg1_out)
            seghead2_out = self.aux_head2(hrseg2_out)
            logit_list = [last_out, seghead1_out, seghead2_out]
        else:
            logit_list = [last_out]

        logit_list = [F.interpolate(logit, size=(h, w), mode='bilinear', align_corners=True) for logit in logit_list]
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for m in self.sublayers():
                if isinstance(m, nn.Conv2D):
                    param_init.kaiming_normal_init(m.weight)
                elif isinstance(m, (nn.BatchNorm2D, nn.LayerNorm)):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)
                elif isinstance(m, nn.Linear):
                    TruncatedNormal(std=.02)(m.weight)
                    if m.bias is not None:
                        Constant(value=0)(m.bias)


# # --- 模型测试 ---
# if __name__ == "__main__":
#     with paddle.no_grad():
#         model = MambaHrSegNet(base=16, num_classes=2)
#         model.eval()
#
#         x = paddle.randn([1, 3, 256, 256])
#
#         print("正在测试模型前向传播...")
#         try:
#             out = model(x)
#             print("前向传播成功!")
#             print("输出张量形状:", out[0].shape)
#         except Exception as e:
#             print(f"前向传播时发生错误: {e}")
#             import traceback
#
#             traceback.print_exc()

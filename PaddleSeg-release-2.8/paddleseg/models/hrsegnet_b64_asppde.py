import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init


# =================================================================================
# ASPP 模块代码 (无变化)
# =================================================================================
class ASPPModule(nn.Layer):
    """ASPP的单个分支"""

    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2D(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Layer):
    """Atrous Spatial Pyramid Pooling (ASPP)"""

    def __init__(self, in_channels, out_channels=256, rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )
        self.aspp_modules = nn.LayerList([
            ASPPModule(in_channels, out_channels, rate) for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(in_channels, out_channels, 1, stride=1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )
        total_channels = out_channels * (len(rates) + 2)
        self.conv_out = nn.Sequential(
            nn.Conv2D(total_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = paddle.shape(x)[2:]
        x1 = self.conv1x1(x)
        branches = [x1]
        for aspp_module in self.aspp_modules:
            branches.append(aspp_module(x))
        x_pool = self.global_avg_pool(x)
        x_pool = F.interpolate(x_pool, size=size, mode='bilinear', align_corners=False)
        branches.append(x_pool)
        concat = paddle.concat(branches, axis=1)
        return self.conv_out(concat)


# =================================================================================
# 轻量化且可配置的最终模型
# =================================================================================
@manager.MODELS.add_component
class HrSegNet_Lite(nn.Layer):
    """
    一个轻量化、可配置的HrSegNet版本，集成了ASPP和特征融合解码器。
    旨在性能和计算成本之间取得平衡。
    """

    def __init__(self,
                 in_channels=3,
                 num_classes=2,
                 # --- 以下是影响模型大小的关键参数，可在yml中配置 ---
                 base=64,  # 编码器基础通道数
                 aspp_rates=(1, 12, 24, 48),  # ASPP空洞率
                 aspp_out_channels=128,  # ASPP输出通道数 (推荐128)
                 decoder_channels=64,  # 解码器内部通道数 (推荐64)
                 # ----------------------------------------------------
                 pretrained=None
                 ):
        super(HrSegNet_Lite, self).__init__()

        print("=" * 50)
        print(f"Initializing HrSegNet_Lite model...")
        print(f"  - Encoder base channels: {base}")
        print(f"  - ASPP out_channels: {aspp_out_channels}")
        print(f"  - Decoder channels: {decoder_channels}")
        print("=" * 50)

        self.base = base
        self.pretrained = pretrained

        # --- 编码器部分 (Encoder) ---
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
        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)

        self.aspp = ASPP(
            in_channels=base,
            out_channels=aspp_out_channels,
            rates=list(aspp_rates)
        )

        # --- 轻量化解码器 (Lite Decoder) ---
        self.low_level_conv = nn.Sequential(
            nn.Conv2D(base, 48, 1, bias_attr=False),
            nn.BatchNorm2D(48),
            nn.ReLU()
        )

        # 融合后的特征处理模块（已大幅轻量化）
        self.last_conv = nn.Sequential(
            nn.Conv2D(aspp_out_channels + 48, decoder_channels, kernel_size=3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(decoder_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # --- 分割头 ---
        self.head = SegHead(inplanes=decoder_channels, interplanes=base, outplanes=num_classes)
        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)

        self.init_weight()

    def forward(self, x):
        h, w = paddle.shape(x)[2:]

        # 编码过程
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)

        # 特征提取与融合
        high_level_features = self.aspp(hrseg3_out)
        low_level_features = self.low_level_conv(hrseg2_out)

        high_level_features = F.interpolate(
            high_level_features,
            size=paddle.shape(low_level_features)[2:],
            mode='bilinear',
            align_corners=False
        )

        fused_features = paddle.concat([high_level_features, low_level_features], axis=1)
        refined_features = self.last_conv(fused_features)

        # 输出
        last_out = self.head(refined_features)

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
                elif isinstance(m, nn.BatchNorm2D):
                    param_init.constant_init(m.weight, value=1)
                    param_init.constant_init(m.bias, value=0)


# =================================================================================
# 依赖的模块 (SegBlock, SegHead)
# =================================================================================
class SegBlock(nn.Layer):
    def __init__(self,
                 base=32,
                 stage_index=1):
        super(SegBlock, self).__init__()
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
            self.l_conv1 = nn.Sequential(
                nn.Conv2D(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index == 2:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                nn.Conv2D(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index == 3:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                nn.Conv2D(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
                nn.ReLU(),
                nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)),
                          out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        else:
            raise ValueError("stage_index must be 1, 2 or 3")
        self.l_conv2 = nn.Sequential(
            nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)),
                      out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
            nn.ReLU()
        )
        self.l_conv3 = nn.Sequential(
            nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)),
                      out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(base * int(math.pow(2, stage_index))),
            nn.ReLU()
        )
        self.l2h_conv1 = nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)
        self.l2h_conv2 = nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)
        self.l2h_conv3 = nn.Conv2D(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)

    def forward(self, x):
        size = x.shape[2:]
        out_h1 = self.h_conv1(x)
        out_l1 = self.l_conv1(x)
        out_l1_i = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True)
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1
        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        out_l2_i = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2
        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        out_l3_i = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3
        return out_hl3


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

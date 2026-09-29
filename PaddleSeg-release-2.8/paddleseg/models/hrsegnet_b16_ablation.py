import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager, param_init
from paddleseg.utils import utils


class ASPPModule(nn.Layer):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Layer):
    def __init__(self, in_channels, out_channels=64, rates=(1, 6, 12, 18)):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
        )
        self.aspp_modules = nn.LayerList(
            [ASPPModule(in_channels, out_channels, rate) for rate in rates]
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
        )
        total_channels = out_channels * (len(rates) + 2)
        self.conv_out = nn.Sequential(
            nn.Conv2D(total_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        size = paddle.shape(x)[2:]
        branches = [self.conv1x1(x)]
        for branch in self.aspp_modules:
            branches.append(branch(x))
        pooled = self.global_avg_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        branches.append(pooled)
        return self.conv_out(paddle.concat(branches, axis=1))


class SegBlock(nn.Layer):
    def __init__(self, base=16, stage_index=1):
        super().__init__()
        self.h_conv1 = nn.Sequential(
            nn.Conv2D(base, base, 3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU(),
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2D(base, base, 3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU(),
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv2D(base, base, 3, stride=1, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU(),
        )

        low_channels = base * int(math.pow(2, stage_index))
        if stage_index == 1:
            self.l_conv1 = nn.Sequential(
                nn.Conv2D(base, low_channels, 3, stride=2, padding=1),
                nn.BatchNorm2D(low_channels),
                nn.ReLU(),
            )
        elif stage_index == 2:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                nn.Conv2D(base, low_channels, 3, stride=2, padding=1),
                nn.BatchNorm2D(low_channels),
                nn.ReLU(),
            )
        elif stage_index == 3:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2D(kernel_size=3, stride=2, padding=1),
                nn.Conv2D(base, low_channels, 3, stride=2, padding=1),
                nn.BatchNorm2D(low_channels),
                nn.ReLU(),
                nn.Conv2D(low_channels, low_channels, 3, stride=2, padding=1),
                nn.BatchNorm2D(low_channels),
                nn.ReLU(),
            )
        else:
            raise ValueError("stage_index must be 1, 2 or 3")

        self.l_conv2 = nn.Sequential(
            nn.Conv2D(low_channels, low_channels, 3, stride=1, padding=1),
            nn.BatchNorm2D(low_channels),
            nn.ReLU(),
        )
        self.l_conv3 = nn.Sequential(
            nn.Conv2D(low_channels, low_channels, 3, stride=1, padding=1),
            nn.BatchNorm2D(low_channels),
            nn.ReLU(),
        )
        self.l2h_conv1 = nn.Conv2D(low_channels, base, 1)
        self.l2h_conv2 = nn.Conv2D(low_channels, base, 1)
        self.l2h_conv3 = nn.Conv2D(low_channels, base, 1)

    def forward(self, x):
        size = x.shape[2:]
        out_h1 = self.h_conv1(x)
        out_l1 = self.l_conv1(x)
        out_l1_i = F.interpolate(out_l1, size=size, mode="bilinear", align_corners=True)
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        out_l2_i = F.interpolate(out_l2, size=size, mode="bilinear", align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        out_l3_i = F.interpolate(out_l3, size=size, mode="bilinear", align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3
        return out_hl3


class SegHead(nn.Layer):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2D(inplanes)
        self.relu = nn.ReLU()
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2D(inplanes, interplanes, 3, stride=1, padding=1),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2DTranspose(
                    inplanes,
                    interplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.BatchNorm2D(interplanes),
                nn.ReLU(),
            )
        self.conv = nn.Conv2D(interplanes, outplanes, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        return self.conv(x)


class _HrSegNetB16Variant(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        base=16,
        num_classes=2,
        use_aspp=False,
        use_lite_decoder=False,
        aspp_rates=(1, 6, 12, 18),
        aspp_out_channels=64,
        decoder_channels=64,
        low_level_channels=48,
        pretrained=None,
    ):
        super().__init__()
        self.use_aspp = use_aspp
        self.use_lite_decoder = use_lite_decoder
        self.pretrained = pretrained

        self.stage1 = nn.Sequential(
            nn.Conv2D(in_channels, base // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(base // 2),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2D(base // 2, base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(base),
            nn.ReLU(),
        )
        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)

        high_level_channels = base
        if use_aspp:
            self.aspp = ASPP(in_channels=base, out_channels=aspp_out_channels, rates=aspp_rates)
            high_level_channels = aspp_out_channels
        else:
            self.aspp = None

        if use_lite_decoder:
            self.low_level_conv = nn.Sequential(
                nn.Conv2D(base, low_level_channels, 1, bias_attr=False),
                nn.BatchNorm2D(low_level_channels),
                nn.ReLU(),
            )
            self.last_conv = nn.Sequential(
                nn.Conv2D(
                    high_level_channels + low_level_channels,
                    decoder_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(decoder_channels),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            head_inplanes = decoder_channels
        else:
            self.low_level_conv = None
            self.last_conv = None
            head_inplanes = high_level_channels

        self.head = SegHead(inplanes=head_inplanes, interplanes=base, outplanes=num_classes)
        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
            return
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.kaiming_normal_init(layer.weight)
            elif isinstance(layer, nn.BatchNorm2D):
                param_init.constant_init(layer.weight, value=1)
                param_init.constant_init(layer.bias, value=0)

    def forward(self, x):
        h, w = paddle.shape(x)[2:]
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)

        high_level = self.aspp(hrseg3_out) if self.use_aspp else hrseg3_out
        if self.use_lite_decoder:
            low_level = self.low_level_conv(hrseg2_out)
            high_level = F.interpolate(
                high_level,
                size=paddle.shape(low_level)[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = paddle.concat([high_level, low_level], axis=1)
            refined = self.last_conv(fused)
            last_out = self.head(refined)
        else:
            last_out = self.head(high_level)

        if self.training:
            seghead1_out = self.aux_head1(hrseg1_out)
            seghead2_out = self.aux_head2(hrseg2_out)
            logits = [last_out, seghead1_out, seghead2_out]
        else:
            logits = [last_out]
        return [
            F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=True)
            for logit in logits
        ]


@manager.MODELS.add_component
class HrSegNetB16ASPP(_HrSegNetB16Variant):
    def __init__(self, **kwargs):
        super().__init__(use_aspp=True, use_lite_decoder=False, **kwargs)


@manager.MODELS.add_component
class HrSegNetB16Decoder(_HrSegNetB16Variant):
    def __init__(self, **kwargs):
        super().__init__(use_aspp=False, use_lite_decoder=True, **kwargs)


@manager.MODELS.add_component
class HrSegNetB16A(_HrSegNetB16Variant):
    def __init__(self, **kwargs):
        super().__init__(use_aspp=True, use_lite_decoder=True, **kwargs)

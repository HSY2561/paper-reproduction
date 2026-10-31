"""W11 four-cell decoder ablation, isolated from the existing SegFormer path.

M1 changes the complete decoder. M2 is the same shallow-detail residual in E2
and E3. The original SegFormer and same-channel Aligned implementation remain
unchanged; no unused original head is registered in M1 models.
"""
import paddle
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from .segformer import SegFormer
from .sfnet import AlignedModule
from .backbones.mit_b0_s4ffn768 import load_exact_state


class NarrowAlignedModule(AlignedModule):
    """Two input widths, with the original SFNet flow_warp semantics unchanged."""
    def __init__(self, low_channels=128, high_channels=64, flow_channels=32):
        nn.Layer.__init__(self)
        self.down_h = nn.Conv2D(high_channels, flow_channels, 1, bias_attr=False)
        self.down_l = nn.Conv2D(low_channels, flow_channels, 1, bias_attr=False)
        self.flow_make = nn.Conv2D(flow_channels * 2, 2, 3, padding=1,
                                   bias_attr=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_origin = h_feature
        size = paddle.shape(low_feature)[2:]
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode='bilinear',
                                  align_corners=True)
        flow = self.flow_make(paddle.concat([h_feature, low_feature], axis=1))
        return self.flow_warp(h_feature_origin, flow, size=size)


class ShallowDetailRestore(nn.Layer):
    """5,490 registered values: spatial gate and signed detail residual."""
    def __init__(self):
        super().__init__()
        self.detail = layers.ConvBNReLU(32, 16, 1, bias_attr=False)
        self.semantic = layers.ConvBNReLU(128, 16, 1, bias_attr=False)
        self.gate = nn.Conv2D(32, 1, 1, bias_attr=True)
        self.depthwise = layers.ConvBNReLU(16, 16, 3, groups=16,
                                          bias_attr=False)
        self.output = layers.ConvBN(16, 128, 1, bias_attr=False)
        self.alpha = self.create_parameter(
            shape=[1], dtype='float32',
            default_initializer=nn.initializer.Constant(0.1))

    def forward(self, raw_c1, feature):
        detail = self.detail(raw_c1)
        semantic = self.semantic(feature)
        gate = F.sigmoid(self.gate(paddle.concat([detail, semantic], axis=1)))
        residual = self.output(self.depthwise(detail))
        return feature + self.alpha * gate * residual


@manager.MODELS.add_component
class SegFormerComplementarity(SegFormer):
    """Fixed-width W11 four cells controlled solely by ``use_m1/use_m2``.

    E0/E2 retain all A0 head state keys, E1/E3 share all M1 keys, and both
    M2 cells use ``detail_restore.*``. All cells share ``backbone.*`` and
    ``linear_pred.*`` shapes for explicit component initialization.
    """
    def __init__(self, num_classes, backbone, embedding_dim=128,
                 align_corners=False, pretrained=None, use_m1=False,
                 use_m2=False):
        if embedding_dim != 128 or list(backbone.feat_channels) != [32, 64, 160, 256]:
            raise ValueError('W11 requires width128 and [32,64,160,256] features')
        if num_classes != 2:
            raise ValueError('W11 protocol fixes two Crack500 classes')
        if use_m1:
            # Calling SegFormer.__init__ here would register an unused old head.
            nn.Layer.__init__(self)
            self.backbone = backbone
            self.num_classes = num_classes
            self.align_corners = align_corners
            self.pretrained = None
            self.dropout = nn.Dropout2D(0.1)
            self.m1_c1 = layers.ConvBNReLU(32, 128, 1, bias_attr=False)
            self.m1_c2 = layers.ConvBNReLU(64, 32, 1, bias_attr=False)
            self.m1_c3 = layers.ConvBNReLU(160, 32, 1, bias_attr=False)
            self.m1_c4 = layers.ConvBNReLU(256, 32, 1, bias_attr=False)
            self.m1_deep_fuse = layers.ConvBNReLU(96, 64, 1, bias_attr=False)
            self.m1_aligned = NarrowAlignedModule(128, 64, 32)
            self.m1_out_fuse = layers.ConvBNReLU(192, 128, 1, bias_attr=False)
            self.linear_pred = nn.Conv2D(128, num_classes, 1, bias_attr=True)
        else:
            # Preserve original A0 construction, state keys and E0 forward path.
            super().__init__(num_classes=num_classes, backbone=backbone,
                             embedding_dim=128, align_corners=align_corners,
                             pretrained=None, balance_modules=['aligned'])
        self.use_m1 = bool(use_m1)
        self.use_m2 = bool(use_m2)
        if self.use_m2:
            self.detail_restore = ShallowDetailRestore()
        if pretrained is not None:
            load_exact_state(self, pretrained)
        self.pretrained = pretrained

    def _decode_a0(self, features):
        """A0 operations needed before dropout, preserving original tensor order."""
        c1, c2, c3, c4 = features
        c1_size = c1.shape[2:]
        p4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4.shape[2], c4.shape[3]])
        p3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3.shape[2], c3.shape[3]])
        p3 = F.interpolate(p3, size=c1_size, mode='bilinear',
                           align_corners=self.align_corners)
        p2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2.shape[2], c2.shape[3]])
        p2 = F.interpolate(p2, size=c1_size, mode='bilinear',
                           align_corners=self.align_corners)
        p1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1.shape[2], c1.shape[3]])
        p4 = self.balance_blocks['aligned']([p1, p4])
        return self._fuse_multiscale((p4, p3, p2, p1))

    def _decode_m1(self, features):
        c1, c2, c3, c4 = features
        low = self.m1_c1(c1)
        p2 = self.m1_c2(c2)
        p3 = F.interpolate(self.m1_c3(c3), size=c2.shape[2:], mode='bilinear',
                           align_corners=self.align_corners)
        p4 = F.interpolate(self.m1_c4(c4), size=c2.shape[2:], mode='bilinear',
                           align_corners=self.align_corners)
        high = self.m1_deep_fuse(paddle.concat([p2, p3, p4], axis=1))
        aligned = self.m1_aligned([low, high])
        return self.m1_out_fuse(paddle.concat([low, aligned], axis=1))

    def forward(self, x):
        if not self.use_m1 and not self.use_m2:
            return super().forward(x)
        features = self.backbone(x)
        feature = self._decode_m1(features) if self.use_m1 else self._decode_a0(features)
        if self.use_m2:
            feature = self.detail_restore(features[0], feature)
        logits = self.linear_pred(self.dropout(feature))
        return [F.interpolate(logits, size=x.shape[2:], mode='bilinear',
                              align_corners=self.align_corners)]

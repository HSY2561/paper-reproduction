"""Controlled Aligned + WTConv repair for the VOCdevkit3000 audit.

The original C3 candidate replaced SegFormer's 1024->256 fusion mixer with a
1024->128 bottleneck.  This variant keeps the original Aligned fusion path and
adds one 256-channel residual WTConv after it.  The residual projection is
zero-initialized so the new block starts as the C1 function, which isolates the
effect of the added block from a random residual perturbation.
"""

import paddle
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager
from paddleseg.models.segformer_simplify import SegFormerSimplify
from paddleseg.utils import utils


class IdentityInitWTConv(nn.Layer):
    """Low/high-frequency residual block with an identity initial state."""

    def __init__(self, channels):
        super().__init__()
        self.low = nn.AvgPool2D(2, 2)
        self.up = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.high = nn.Conv2D(
            channels, channels, 5, padding=2, groups=channels, bias_attr=False)
        self.project = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.norm = nn.BatchNorm2D(channels)
        self.relu = nn.ReLU()
        # Start from the original fused feature and learn the residual gradually.
        self.project.weight.set_value(paddle.zeros_like(self.project.weight))

    def forward(self, x):
        low = self.up(self.low(x))
        low = F.interpolate(low, size=x.shape[2:], mode='bilinear',
                            align_corners=False)
        residual = self.project(self.high(x) + low)
        return x + self.relu(self.norm(residual))


@manager.MODELS.add_component
class SegFormerJointFix(SegFormerSimplify):
    """Aligned fusion followed by a full-width residual WTConv."""

    def __init__(self, num_classes, backbone, embedding_dim=256,
                 align_corners=False, pretrained=None, alignment='full'):
        super().__init__(
            num_classes=num_classes,
            backbone=backbone,
            embedding_dim=embedding_dim,
            align_corners=align_corners,
            pretrained=None,
            decoder_kind='concat',
            alignment=alignment,
            fusion_layout='standard')
        if alignment != 'full':
            raise ValueError('SegFormerJointFix requires full Aligned fusion')
        self.wtconv = IdentityInitWTConv(embedding_dim)
        if pretrained is not None:
            # Diagnostic warm-start only.  The new WTConv parameters keep their
            # identity initialization while all C1-compatible parameters are
            # loaded from the frozen Aligned checkpoint.
            utils.load_pretrained_model(self, pretrained)

    def forward(self, x):
        raw = self.backbone(x)
        p1, p2, p3, p4 = self._native_projections(raw)
        high = self.balance_blocks['aligned']([p1, p4])
        fused = self.linear_fuse(paddle.concat([
            high,
            self._resize(p3, p1.shape[2:]),
            self._resize(p2, p1.shape[2:]),
            p1,
        ], axis=1))
        fused = self.wtconv(fused)
        logits = self.linear_pred(self.dropout(fused))
        return [self._resize(logits, x.shape[2:])]

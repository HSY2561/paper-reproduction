"""W13 single-module and Aligned-paired fusion candidates.

Each candidate is inserted once after the four-branch fusion. The paired
variant retains the original Aligned C1/C4 flow before the module; the single
variant uses ordinary C4 resize. These are controlled replacements, not a
claim of reproducing a full external network.
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddleseg.cvlibs import manager
from paddleseg.models.segformer_simplify import SegFormerSimplify


class RepDWFusion(nn.Layer):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.reduce = nn.Conv2D(in_channels, channels, 1, groups=4, bias_attr=False)
        self.dw3 = nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False)
        self.dw1 = nn.Conv2D(channels, channels, 1, groups=channels, bias_attr=False)
        self.pw = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn = nn.BatchNorm2D(channels)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.reduce(x)
        return self.act(self.bn(self.pw(self.dw3(x) + self.dw1(x))))


class GhostFusion(nn.Layer):
    def __init__(self, in_channels, channels, ratio=2):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.primary = nn.Sequential(nn.Conv2D(in_channels, hidden, 1, bias_attr=False), nn.BatchNorm2D(hidden), nn.ReLU())
        self.cheap = nn.Sequential(nn.Conv2D(hidden, hidden, 3, padding=1, groups=hidden, bias_attr=False), nn.BatchNorm2D(hidden), nn.ReLU())
        self.project = nn.Conv2D(hidden * 2, channels, 1, bias_attr=False)
        self.bn = nn.BatchNorm2D(channels)
    def forward(self, x):
        y = self.primary(x); z = self.cheap(y)
        return self.bn(self.project(paddle.concat([y, z], axis=1)))


class PDCFusion(nn.Layer):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.reduce = nn.Conv2D(in_channels, channels, 1, groups=4, bias_attr=False)
        self.proj = nn.Conv2D(channels * 4, channels, 1, bias_attr=False)
        self.bn = nn.BatchNorm2D(channels)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.reduce(x)
        up = F.pad(x[:, :, :-1, :], [0, 0, 1, 0]); down = F.pad(x[:, :, 1:, :], [0, 0, 0, 1])
        left = F.pad(x[:, :, :, :-1], [1, 0, 0, 0]); right = F.pad(x[:, :, :, 1:], [0, 1, 0, 0])
        d = paddle.concat([x-up, x-down, x-left, x-right], axis=1)
        return self.act(self.bn(self.proj(d)))


class MSCAFusion(nn.Layer):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.reduce = nn.Conv2D(in_channels, channels, 1, groups=4, bias_attr=False)
        self.branches = nn.LayerList()
        for k in (3, 5, 7):
            self.branches.append(nn.Sequential(
                nn.Conv2D(channels, channels, (1, k), padding=(0, k//2), groups=channels, bias_attr=False),
                nn.Conv2D(channels, channels, (k, 1), padding=(k//2, 0), groups=channels, bias_attr=False)))
        self.project = nn.Conv2D(channels * 3, channels, 1, bias_attr=False)
        self.bn = nn.BatchNorm2D(channels)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.reduce(x)
        y = paddle.concat([branch(x) for branch in self.branches], axis=1)
        return self.act(self.bn(self.project(y)))


@manager.MODELS.add_component
class SegFormerPair(SegFormerSimplify):
    def __init__(self, num_classes, backbone, embedding_dim=128, align_corners=False,
                 pretrained=None, alignment='none', pair_module='repdw'):
        if alignment not in ('none', 'full'):
            raise ValueError('W13 alignment must be none or full')
        if pair_module not in ('repdw', 'ghost', 'pdc', 'msca'):
            raise ValueError('Unknown W13 pair module')
        super().__init__(num_classes=num_classes, backbone=backbone, embedding_dim=embedding_dim,
                         align_corners=align_corners, pretrained=None, decoder_kind='concat',
                         alignment=alignment, fusion_layout='standard')
        self.pair_module = pair_module
        block = {'repdw': RepDWFusion, 'ghost': GhostFusion, 'pdc': PDCFusion, 'msca': MSCAFusion}[pair_module]
        # Replace the original 4W->W fusion convolution, rather than adding
        # a post-fusion block. This is the measured compute/parameter tradeoff.
        self._sub_layers.pop('linear_fuse', None)
        self.pair_block = block(embedding_dim * 4, embedding_dim)
        if pretrained is not None:
            # W13 configs use backbone.pretrained for ImageNet; a complete model
            # checkpoint is intentionally not loaded into this changed head.
            raise ValueError('W13 model pretrained must be null; initialize backbone explicitly')
    def forward(self, x):
        raw = self.backbone(x)
        projected = self._native_projections(raw)
        p1, p2, p3, p4 = projected
        if self.alignment_kind == 'full':
            high = self.balance_blocks['aligned']([p1, p4])
        else:
            high = F.interpolate(p4, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners)
        fused = self.pair_block(paddle.concat([high, F.interpolate(p3, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners),
                                               F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=self.align_corners), p1], axis=1))
        logits = self.linear_pred(self.dropout(fused))
        return [F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners)]

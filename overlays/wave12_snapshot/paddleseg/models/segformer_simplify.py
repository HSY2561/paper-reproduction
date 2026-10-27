"""W12 deletion/replacement decoders; original SegFormer/SFNet are unchanged.

``distributed`` is a parameter-preserving linear execution reorder, not a new
trainable block. ``native_sum`` replaces the factorized concat mixer with four
direct native-scale projections and one shared BN/ReLU. No unused legacy head
is instantiated in that path.
"""
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from .segformer import SegFormer, MLP
from .sfnet import AlignedModule
from .backbones.mit_b0_s4ffn768 import load_exact_state


class SimplifyAligned(AlignedModule):
    """Retain SFNet warp coordinates while replacing only the flow predictor.

    Coarse flow is predicted at the actual C2 feature shape. Its values are
    parameterized in the final C1 target-grid units; resizing the flow therefore
    adds no amplitude multiplication. It is a new predictor, not a claim of
    equivalence to fine-grid flow.
    """
    def __init__(self, channels, kind='full'):
        if kind not in ('full', 'flow32', 'flow16', 'dwflow32', 'coarseflow32'):
            raise ValueError('Unknown alignment replacement: ' + str(kind))
        flow_channels = channels // 2 if kind == 'full' else (16 if kind == 'flow16' else 32)
        if kind == 'dwflow32':
            nn.Layer.__init__(self)
            self.down_h = nn.Conv2D(channels, flow_channels, 1, bias_attr=False)
            self.down_l = nn.Conv2D(channels, flow_channels, 1, bias_attr=False)
            self.flow_make = nn.Sequential(
                nn.Conv2D(2 * flow_channels, 2 * flow_channels, 3, padding=1,
                          groups=2 * flow_channels, bias_attr=False),
                nn.Conv2D(2 * flow_channels, 2, 1, bias_attr=False))
        else:
            super().__init__(channels, flow_channels)
        self.kind = kind

    def predict_flow(self, low_feature, high_feature, coarse_size=None):
        target_size = paddle.shape(low_feature)[2:]
        low = self.down_l(low_feature)
        high = self.down_h(high_feature)
        if self.kind == 'coarseflow32':
            if coarse_size is None:
                raise ValueError('coarseflow32 requires the actual raw C2 spatial shape')
            low = F.interpolate(low, size=coarse_size, mode='bilinear', align_corners=True)
            high = F.interpolate(high, size=coarse_size, mode='bilinear', align_corners=True)
        else:
            high = F.interpolate(high, size=target_size, mode='bilinear', align_corners=True)
        flow = self.flow_make(paddle.concat([high, low], axis=1))
        if self.kind == 'coarseflow32':
            flow = F.interpolate(flow, size=target_size, mode='bilinear', align_corners=True)
        return flow

    def forward(self, features, coarse_size=None):
        low, high = features
        flow = self.predict_flow(low, high, coarse_size)
        return self.flow_warp(high, flow, size=paddle.shape(low)[2:])


@manager.MODELS.add_component
class SegFormerSimplify(SegFormer):
    """Controlled W12 decoder deletion/replacement family.

    ``embedding_dim=128, decoder_kind='concat', alignment='full',
    fusion_layout='standard'`` has exactly the existing A0 state keys/shapes and
    calls the unchanged original forward. Width and predictor variants do not
    add a second decoder. Native sum has no alignment or registered concat head.
    """
    def __init__(self, num_classes, backbone, embedding_dim=128,
                 align_corners=False, pretrained=None, decoder_kind='concat',
                 alignment='full', fusion_layout='standard'):
        if list(backbone.feat_channels) != [32, 64, 160, 256]:
            raise ValueError('W12 requires MiT-B0 feature widths [32,64,160,256]')
        if num_classes != 2 or embedding_dim not in (64, 80, 96, 112, 128, 160, 192, 256):
            raise ValueError('W12 requires two classes and an explicitly supported width')
        if decoder_kind not in ('concat', 'native_sum'):
            raise ValueError('Unknown decoder_kind: ' + str(decoder_kind))
        if fusion_layout not in ('standard', 'distributed'):
            raise ValueError('Unknown fusion_layout: ' + str(fusion_layout))
        if alignment not in ('none', 'full', 'flow32', 'flow16', 'dwflow32', 'coarseflow32'):
            raise ValueError('Unknown alignment: ' + str(alignment))
        if decoder_kind == 'native_sum':
            if alignment != 'none' or fusion_layout != 'standard':
                raise ValueError('native_sum deletes alignment and concat; no extra path is permitted')
            nn.Layer.__init__(self)
            self.backbone = backbone
            self.num_classes = num_classes
            self.align_corners = align_corners
            self.pretrained = None
            self.linear_c4 = MLP(256, embedding_dim)
            self.linear_c3 = MLP(160, embedding_dim)
            self.linear_c2 = MLP(64, embedding_dim)
            self.linear_c1 = MLP(32, embedding_dim)
            self.sum_norm = layers.SyncBatchNorm(embedding_dim)
            self.sum_relu = nn.ReLU()
            self.dropout = nn.Dropout2D(0.1)
            self.linear_pred = nn.Conv2D(embedding_dim, num_classes, 1)
            self.balance_modules = ()
            self.balance_blocks = nn.LayerDict()
        else:
            # Instantiate only the original live concat path. Add exactly one
            # selected alignment predictor rather than wrapping an existing one.
            super().__init__(num_classes=num_classes, backbone=backbone,
                             embedding_dim=embedding_dim, align_corners=align_corners,
                             pretrained=None, balance_modules=[])
            if alignment != 'none':
                self.balance_blocks['aligned'] = SimplifyAligned(embedding_dim, alignment)
                self.balance_modules = ('aligned',)
        self.embedding_dim = embedding_dim
        self.decoder_kind = decoder_kind
        self.alignment_kind = alignment
        self.fusion_layout = fusion_layout
        if pretrained is not None:
            load_exact_state(self, pretrained)
        self.pretrained = pretrained

    def _native_projections(self, features):
        values = []
        for number, feature in enumerate(features, 1):
            projected = getattr(self, 'linear_c' + str(number))(feature)
            values.append(projected.transpose([0, 2, 1]).reshape(
                [0, 0, feature.shape[2], feature.shape[3]]))
        return values

    def _resize(self, feature, size):
        return F.interpolate(feature, size=size, mode='bilinear',
                             align_corners=self.align_corners)

    def _distributed_decode(self, projected, raw_features):
        p1, p2, p3, p4 = projected
        target = p1.shape[2:]
        # Flow must see the unchanged projections, never the remapped channels.
        if self.alignment_kind != 'none':
            aligned = self.balance_blocks['aligned']
            flow = aligned.predict_flow(p1, p4, raw_features[1].shape[2:])
        weights = self.linear_fuse._conv.weight
        width = self.embedding_dim
        mixed = []
        for index, feature in enumerate((p4, p3, p2, p1)):
            block = weights[:, index * width:(index + 1) * width, :, :]
            native = F.conv2d(feature, block, bias=None)
            if index == 0 and self.alignment_kind != 'none':
                native = aligned.flow_warp(native, flow, size=paddle.shape(p1)[2:])
            elif index != 3:
                native = self._resize(native, target)
            mixed.append(native)
        # Same branch sum as the original concat pointwise transform. Floating
        # reduction ordering can differ, and is measured rather than hidden.
        feature = mixed[0] + mixed[1] + mixed[2] + mixed[3]
        return self.linear_fuse._relu(self.linear_fuse._batch_norm(feature))

    def forward(self, x):
        if (self.decoder_kind == 'concat' and self.fusion_layout == 'standard'
                and self.alignment_kind != 'coarseflow32'):
            return super().forward(x)
        raw = self.backbone(x)
        projected = self._native_projections(raw)
        p1, p2, p3, p4 = projected
        if self.decoder_kind == 'native_sum':
            feature = (self._resize(p4, p1.shape[2:]) +
                       self._resize(p3, p1.shape[2:]) +
                       self._resize(p2, p1.shape[2:]) + p1)
            feature = self.sum_relu(self.sum_norm(feature))
        elif self.fusion_layout == 'distributed':
            feature = self._distributed_decode(projected, raw)
        else:
            high = self.balance_blocks['aligned']([p1, p4], coarse_size=raw[1].shape[2:])
            feature = self.linear_fuse(paddle.concat(
                [high, self._resize(p3, p1.shape[2:]),
                 self._resize(p2, p1.shape[2:]), p1], axis=1))
        logits = self.linear_pred(self.dropout(feature))
        return [self._resize(logits, x.shape[2:])]


def fold_plain_state_to_native_sum(plain_state, width):
    """Fold a same-width no-Alignment concat head into native-sum initial state.

    Uses the [C4,C3,C2,C1] fusion block order. The calculation is CPU NumPy
    float64 followed by casting to original tensor dtype. This only establishes
    initial functional proximity, not equality of the subsequent optimization.
    """
    if any(key.startswith('balance_blocks.') for key in plain_state):
        raise ValueError('Fold requires the no-alignment plain decoder state')
    required = 'linear_fuse._conv.weight'
    if required not in plain_state:
        raise ValueError('Plain concat fusion weight is absent')
    array = plain_state[required].numpy().astype('float64')
    if list(array.shape) != [width, 4 * width, 1, 1]:
        raise ValueError('Unexpected fusion width')
    result = {key: value.detach().clone() for key, value in plain_state.items()
              if key.startswith(('backbone.', 'linear_pred.'))}
    for block_index, stage in enumerate((4, 3, 2, 1)):
        prefix = 'linear_c' + str(stage) + '.proj.'
        weight, bias = plain_state[prefix + 'weight'], plain_state[prefix + 'bias']
        block = array[:, block_index * width:(block_index + 1) * width, 0, 0]
        folded_weight = weight.numpy().astype('float64') @ block.T
        folded_bias = bias.numpy().astype('float64') @ block.T
        result[prefix + 'weight'] = paddle.to_tensor(folded_weight, dtype=weight.dtype)
        result[prefix + 'bias'] = paddle.to_tensor(folded_bias, dtype=bias.dtype)
    for suffix in ('weight', 'bias', '_mean', '_variance'):
        result['sum_norm.' + suffix] = plain_state['linear_fuse._batch_norm.' + suffix].detach().clone()
    return result

# The SegFormer code was heavily based on https://github.com/NVlabs/SegFormer
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/NVlabs/SegFormer#license

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class MLP(nn.Layer):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.proj(x)
        return x


@manager.MODELS.add_component
class SegFormer(nn.Layer):
    """
    The SegFormer implementation based on PaddlePaddle.

    The original article refers to
    Xie, Enze, et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." arXiv preprint arXiv:2105.15203 (2021).

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        embedding_dim (int): The MLP decoder channel dimension.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature.
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 embedding_dim,
                 align_corners=False,
                 pretrained=None,
                 decoder_variant='base',
                 balance_modules=None):
        super(SegFormer, self).__init__()

        self.pretrained = pretrained
        self.align_corners = align_corners
        self.decoder_variant = decoder_variant
        self.backbone = backbone
        self.num_classes = num_classes
        self.balance_modules = tuple(balance_modules or [])
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.backbone.feat_channels

        if decoder_variant == 'ham_decoder':
            from paddleseg.models.segnext import LightHamHead
            # The official head consumes raw C2/C3/C4 at C2 (1/8) resolution.
            # Do not instantiate the unused MLP decoder or upsample to C1.
            self.ham_decoder = LightHamHead(
                in_channels=[c2_in_channels, c3_in_channels, c4_in_channels],
                num_classes=num_classes, channels=128, ham_channels=128,
                dropout_rate=0.1, align_corners=align_corners, num_groups=8,
                ham_kwargs={'MD_S': 1, 'MD_R': 32, 'TRAIN_STEPS': 6,
                            'EVAL_STEPS': 7, 'RAND_INIT': False})
            # Materialize the official fixed-basis buffer before checkpoint
            # loading, so reconstruction preserves exactly the saved basis.
            nmf = self.ham_decoder.hamburger.ham
            nmf.register_buffer('bases', nmf._build_bases(1, 1, 128, 32))
            self.init_weight()
            return

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        # These variants wrap mature PaddleSeg modules around the original
        # SegFormer feature path.  The wrappers keep the baseline tensor
        # contract intact and are selected only through the config value.
        if decoder_variant == 'emau_context':
            from paddleseg.models.emanet import EMAU
            self.context_refine = EMAU(c4_in_channels,
                                       max(8, c4_in_channels // 8),
                                       stage_num=3,
                                       momentum=0.1)
            self.context_fuse = layers.ConvBNReLU(c4_in_channels * 2,
                                                   c4_in_channels,
                                                   kernel_size=1,
                                                   bias_attr=False)
        elif decoder_variant == 'dual_context':
            from paddleseg.models.layers.attention import DualAttentionModule
            self.context_refine = DualAttentionModule(c4_in_channels,
                                                      c4_in_channels)

        self.dropout = nn.Dropout2D(0.1)
        self.linear_fuse = layers.ConvBNReLU(in_channels=embedding_dim * 4,
                                             out_channels=embedding_dim,
                                             kernel_size=1,
                                             bias_attr=False)

        self.linear_pred = nn.Conv2D(embedding_dim,
                                     self.num_classes,
                                     kernel_size=1)
        # Wave 6 adapters instantiate existing PaddleSeg blocks.  Empty
        # balance_modules keeps the original architecture and state keys.
        self.balance_blocks = nn.LayerDict()
        for name in self.balance_modules:
            if name == 'se':
                from paddleseg.models.backbones.mobilenetv3 import SEModule
                block = SEModule(embedding_dim, reduction=4)
            elif name == 'polarized':
                from paddleseg.models.layers.polaried_self_attention import PolarizedSelfAttentionModule
                block = PolarizedSelfAttentionModule(embedding_dim, embedding_dim)
            elif name == 'gc':
                from paddleseg.models.gcnet import GlobalContextBlock
                block = GlobalContextBlock(c4_in_channels, c4_in_channels, 1 / 16)
            elif name == 'cca':
                from paddleseg.models.ccnet import CrissCrossAttention
                block = CrissCrossAttention(c4_in_channels)
            elif name == 'pam':
                from paddleseg.models.layers.attention import PAM
                block = PAM(c4_in_channels)
            elif name == 'cam':
                from paddleseg.models.layers.attention import CAM
                block = CAM(c4_in_channels)
            elif name == 'nonlocal':
                from paddleseg.models.layers.nonlocal2d import NonLocal2D
                block = NonLocal2D(c4_in_channels, reduction=4)
            elif name == 'msca':
                from paddleseg.models.backbones.mscan import SpatialAttention
                block = SpatialAttention(embedding_dim)
            elif name == 'encoding':
                from paddleseg.models.encnet import EncModule
                block = EncModule(c4_in_channels, num_codes=16)
            elif name == 'glore':
                from paddleseg.models.glore import GruModule
                block = GruModule(c4_in_channels, num_state=64, num_node=32,
                                  normalize=True)
            elif name == 'aspp_sep':
                block = layers.ASPPModule((1, 2, 4), embedding_dim,
                                          embedding_dim, align_corners,
                                          use_sep_conv=True, image_pooling=False)
            elif name == 'psp_pool':
                block = layers.PPModule(embedding_dim, embedding_dim,
                                        (1, 2, 3, 6), True, align_corners)
            elif name == 'aligned':
                from paddleseg.models.sfnet import AlignedModule
                block = AlignedModule(embedding_dim, embedding_dim // 2)
            elif name == 'uafm_channel':
                block = layers.UAFM_ChAtten(embedding_dim, embedding_dim,
                                            embedding_dim)
            elif name == 'gated_shape':
                from paddleseg.models.gscnn import GatedSpatailConv2d
                block = GatedSpatailConv2d(embedding_dim, embedding_dim)
                self.balance_gate = nn.Conv2D(embedding_dim, 1, 1)
            elif name == 'fast_fusion':
                from paddleseg.models.fast_scnn import FeatureFusionModule
                if embedding_dim != 128:
                    raise ValueError('Official FastSCNN fusion requires width128')
                block = FeatureFusionModule(embedding_dim, embedding_dim,
                                             embedding_dim, align_corners)
            else:
                raise ValueError('Unknown official balance module: ' + name)
            if name in self.balance_blocks:
                raise ValueError('Duplicate balance module: ' + name)
            self.balance_blocks[name] = block
        # Official PaddleSeg modules used as controlled decoder ablations.
        if decoder_variant in ('uafm', 'uafm_ppcontext'):
            self.detail_fusion = layers.UAFM_SpAtten(embedding_dim, embedding_dim, embedding_dim, resize_mode='bilinear')
        if decoder_variant in ('ppcontext', 'uafm_ppcontext'):
            from paddleseg.models.pp_liteseg import PPContextModule
            self.context_refine = PPContextModule(embedding_dim, embedding_dim // 4, embedding_dim, (1, 3, 6))

        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        feats = self.backbone(x)
        if self.decoder_variant == 'ham_decoder':
            logit = self.ham_decoder(feats)
            return [F.interpolate(logit, size=x.shape[2:], mode='bilinear',
                                  align_corners=self.align_corners)]
        c1, c2, c3, c4 = feats

        for name in self.balance_modules:
            if name in ('gc', 'cca', 'pam', 'cam', 'nonlocal', 'glore'):
                c4 = self.balance_blocks[name](c4)
            elif name == 'encoding':
                _, c4 = self.balance_blocks[name](c4)

        if self.decoder_variant == 'emau_context':
            c4_context = self.context_refine(c4)
            c4 = self.context_fuse(paddle.concat([c4, c4_context], axis=1))
        elif self.decoder_variant == 'dual_context':
            c4 = self.context_refine(c4)

        ############## MLP decoder on C1-C4 ###########
        c1_shape = c1.shape
        c2_shape = c2.shape
        c3_shape = c3.shape
        c4_shape = c4.shape

        _c4 = self.linear_c4(c4).transpose([0, 2, 1]).reshape(
            [0, 0, c4_shape[2], c4_shape[3]])
        for name in self.balance_modules:
            if name in ('aspp_sep', 'psp_pool'):
                _c4 = self.balance_blocks[name](_c4)
        _c4_native = _c4
        _c4 = F.interpolate(_c4,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).transpose([0, 2, 1]).reshape(
            [0, 0, c3_shape[2], c3_shape[3]])
        _c3 = F.interpolate(_c3,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).transpose([0, 2, 1]).reshape(
            [0, 0, c2_shape[2], c2_shape[3]])
        _c2 = F.interpolate(_c2,
                            size=c1_shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).transpose([0, 2, 1]).reshape(
            [0, 0, c1_shape[2], c1_shape[3]])

        if 'aligned' in self.balance_modules:
            _c4 = self.balance_blocks['aligned']([_c1, _c4_native])

        _c = self.linear_fuse(paddle.concat([_c4, _c3, _c2, _c1], axis=1))
        for name in self.balance_modules:
            if name in ('se', 'polarized', 'msca'):
                _c = self.balance_blocks[name](_c)
            elif name == 'uafm_channel':
                _c = self.balance_blocks[name](_c1, _c)
            elif name == 'gated_shape':
                _c = self.balance_blocks[name](_c, self.balance_gate(_c1))
            elif name == 'fast_fusion':
                _c = self.balance_blocks[name](_c, _c4_native)
        if self.decoder_variant in ('uafm', 'uafm_ppcontext'):
            _c = self.detail_fusion(_c1, _c)
        if self.decoder_variant in ('ppcontext', 'uafm_ppcontext'):
            _c = _c + self.context_refine(_c)

        logit = self.dropout(_c)
        logit = self.linear_pred(logit)
        return [
            F.interpolate(logit,
                          size=x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
        ]

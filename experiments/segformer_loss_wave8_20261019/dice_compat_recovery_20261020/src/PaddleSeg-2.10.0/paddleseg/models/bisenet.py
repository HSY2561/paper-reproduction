# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models import layers


@manager.MODELS.add_component
class BiSeNetV2(nn.Layer):
    """
    The BiSeNet V2 implementation based on PaddlePaddle.

    The original article refers to
    Yu, Changqian, et al. "BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation"
    (https://arxiv.org/abs/2004.02147)

    Args:
        num_classes (int): The unique number of target classes.
        lambd (float, optional): A factor for controlling the size of semantic branch channels. Default: 0.25.
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 lambd=0.25,
                 align_corners=False,
                 in_channels=3,
                 pretrained=None,
                 context_type='none',
                 fusion_type='bga',
                 shape_stream=False,
                 point_refine=False):
        super().__init__()

        C1, C2, C3 = 64, 64, 128
        db_channels = (C1, C2, C3)
        C1, C3, C4, C5 = int(C1 * lambd), int(C3 * lambd), 64, 128
        sb_channels = (C1, C3, C4, C5)
        mid_channels = 128

        self.db = DetailBranch(in_channels, db_channels)
        self.sb = SemanticBranch(in_channels, sb_channels,
                                 context_type='none' if context_type in ('ocr', 'ocr_high') else context_type)

        if fusion_type == 'uafm_sp':
            self.bga = layers.UAFM_SpAtten(mid_channels, mid_channels,
                                           mid_channels, resize_mode='bilinear')
        else:
            self.bga = BGA(mid_channels, align_corners)
        self.aux_head1 = SegHead(C1, C1, num_classes)
        self.aux_head2 = SegHead(C3, C3, num_classes)
        self.aux_head3 = SegHead(C4, C4, num_classes)
        self.aux_head4 = SegHead(C5, C5, num_classes)
        self.head = SegHead(mid_channels, mid_channels, num_classes)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()

        # Integration only: reuse PaddleSeg OCRNet's published OCR operators.
        # Construct additions after the original initialization so C0 and the
        # shared backbone initialization remain unchanged for a fixed seed.
        self.use_ocr = context_type in ('ocr', 'ocr_high')
        self.ocr_high = context_type == 'ocr_high'
        self.use_shape = shape_stream
        self.use_point = point_refine
        if self.use_ocr:
            from paddleseg.models.ocrnet import SpatialGatherBlock, SpatialOCRModule
            self.ocr_gather = SpatialGatherBlock(C5, num_classes)
            self.ocr_context = SpatialOCRModule(C5, C5 // 2, C5)

        if self.use_shape:
            # GSCNNHead's three-stage shape stream, with its official blocks.
            # Adaptations: use the 1/4 detail feature (64 channels), guide it
            # with BiSeNetV2's semantic stages, and concatenate its edge signal
            # with the existing BGA feature. No ASPP or GSCNN decoder is used.
            from paddleseg.models.gscnn import GatedSpatailConv2d
            from paddleseg.models.backbones.resnet_vd import BasicBlock
            self.shape_guides = nn.LayerList([
                nn.Conv2D(c, 1, 1) for c in (C3, C4, C5)
            ])
            self.shape_res = nn.LayerList([
                BasicBlock(c, c, stride=1) for c in (64, 32, 16)
            ])
            self.shape_reduce = nn.LayerList([
                nn.Conv2D(a, b, 1) for a, b in ((64, 32), (32, 16), (16, 8))
            ])
            self.shape_gates = nn.LayerList([
                GatedSpatailConv2d(c, c) for c in (32, 16, 8)
            ])
            self.shape_edge = nn.Conv2D(8, 1, 1, bias_attr=False)
            self.shape_fusion = layers.ConvBNReLU(mid_channels + 1, mid_channels, 1)

        if self.use_point:
            # Reuse the published PaddleSeg PointRend point head with the
            # existing BiSeNetV2 coarse mask and 1/4 detail feature.
            from paddleseg.models.pointrend import PointHead
            self.point_head = PointHead(
                num_classes=num_classes,
                in_channels=[64],
                out_channels=64,
                in_index=[0],
                num_fcs=3,
                scale_factor=2,
                subdivision_steps=3,
                subdivision_num_points=8196,
                dropout_ratio=0)
            # Paddle 3.4 returns int64 spatial shapes; the upstream PointHead
            # stores this scalar as int32 and concat rejects mixed dtypes.
            self.point_head.subdivision_num_points = (
                self.point_head.subdivision_num_points.astype('int64'))

    def forward(self, x):
        if self.use_shape or self.use_point:
            dfm = x
            for index, layer in enumerate(self.db.convs):
                dfm = layer(dfm)
                if index == 4:
                    if self.use_shape:
                        shape_feature = dfm
                    if self.use_point:
                        point_feature = dfm
        else:
            dfm = self.db(x)
        feat1, feat2, feat3, feat4, sfm = self.sb(x)
        if self.use_ocr and not self.ocr_high:
            # Existing supervised auxiliary logits provide OCR's soft regions.
            # Reuse this result for aux loss; do not run the BN/dropout twice.
            logit4 = self.aux_head4(feat4)
            regions = self.ocr_gather(sfm, logit4)
            sfm = self.ocr_context(sfm, regions)
        fused = self.bga(dfm, sfm)
        if self.ocr_high:
            # Controlled placement change: identical OCR operators, but
            # gather context on the 1/8 fused feature rather than 1/32.
            logit4 = self.aux_head4(feat4)
            high_res_regions = F.interpolate(
                logit4, fused.shape[2:], mode='bilinear',
                align_corners=self.align_corners)
            regions = self.ocr_gather(fused, high_res_regions)
            fused = self.ocr_context(fused, regions)
        if self.use_shape:
            for guide, res, reduce, gate, semantic in zip(
                    self.shape_guides, self.shape_res, self.shape_reduce,
                    self.shape_gates, (feat2, feat3, feat4)):
                guide_map = F.interpolate(guide(semantic), shape_feature.shape[2:],
                                          mode='bilinear', align_corners=self.align_corners)
                shape_feature = gate(reduce(res(shape_feature)), guide_map)
            edge_logit = self.shape_edge(shape_feature)
            edge_for_fusion = F.interpolate(F.sigmoid(edge_logit), fused.shape[2:],
                                            mode='bilinear', align_corners=self.align_corners)
            fused = self.shape_fusion(paddle.concat([fused, edge_for_fusion], axis=1))
        logit = self.head(fused)

        if self.use_point:
            point_output = self.point_head([point_feature], [logit])
            if not self.training:
                return [F.interpolate(point_output[0],
                                      x.shape[2:],
                                      mode='bilinear',
                                      align_corners=self.align_corners)]

        if not self.training:
            logit_list = [logit]
        else:
            logit1 = self.aux_head1(feat1)
            logit2 = self.aux_head2(feat2)
            logit3 = self.aux_head3(feat3)
            if not self.use_ocr:
                logit4 = self.aux_head4(feat4)
            logit_list = [logit, logit1, logit2, logit3, logit4]
            if self.use_shape:
                # Official BCELoss(edge_label=True) consumes raw edge logits.
                logit_list.append(edge_logit)

        logit_list = [
            F.interpolate(logit,
                          x.shape[2:],
                          mode='bilinear',
                          align_corners=self.align_corners)
            for logit in logit_list
        ]

        if self.use_point:
            # Official PointCrossEntropyLoss consumes (point logits, coords).
            logit_list.append(point_output)

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)
        else:
            for sublayer in self.sublayers():
                if isinstance(sublayer, nn.Conv2D):
                    param_init.kaiming_normal_init(sublayer.weight)
                elif isinstance(sublayer, (nn.BatchNorm, nn.SyncBatchNorm)):
                    param_init.constant_init(sublayer.weight, value=1.0)
                    param_init.constant_init(sublayer.bias, value=0.0)


class StemBlock(nn.Layer):

    def __init__(self, in_dim, out_dim):
        super(StemBlock, self).__init__()

        self.conv = layers.ConvBNReLU(in_dim, out_dim, 3, stride=2)

        self.left = nn.Sequential(
            layers.ConvBNReLU(out_dim, out_dim // 2, 1),
            layers.ConvBNReLU(out_dim // 2, out_dim, 3, stride=2))

        self.right = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.fuse = layers.ConvBNReLU(out_dim * 2, out_dim, 3)

    def forward(self, x):
        x = self.conv(x)
        left = self.left(x)
        right = self.right(x)
        concat = paddle.concat([left, right], axis=1)
        return self.fuse(concat)


class ContextEmbeddingBlock(nn.Layer):

    def __init__(self, in_dim, out_dim):
        super(ContextEmbeddingBlock, self).__init__()

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.bn = layers.SyncBatchNorm(in_dim)

        self.conv_1x1 = layers.ConvBNReLU(in_dim, out_dim, 1)
        self.add = layers.Add()
        self.conv_3x3 = nn.Conv2D(out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        gap = self.gap(x)
        bn = self.bn(gap)
        conv1 = self.add(self.conv_1x1(bn), x)
        return self.conv_3x3(conv1)


class GatherAndExpansionLayer1(nn.Layer):
    """Gather And Expansion Layer with stride 1"""

    def __init__(self, in_dim, out_dim, expand):
        super().__init__()

        expand_dim = expand * in_dim

        self.conv = nn.Sequential(layers.ConvBNReLU(in_dim, in_dim, 3),
                                  layers.DepthwiseConvBN(in_dim, expand_dim, 3),
                                  layers.ConvBN(expand_dim, out_dim, 1))
        self.relu = layers.Activation("relu")

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class GatherAndExpansionLayer2(nn.Layer):
    """Gather And Expansion Layer with stride 2"""

    def __init__(self, in_dim, out_dim, expand):
        super().__init__()

        expand_dim = expand * in_dim

        self.branch_1 = nn.Sequential(
            layers.ConvBNReLU(in_dim, in_dim, 3),
            layers.DepthwiseConvBN(in_dim, expand_dim, 3, stride=2),
            layers.DepthwiseConvBN(expand_dim, expand_dim, 3),
            layers.ConvBN(expand_dim, out_dim, 1))

        self.branch_2 = nn.Sequential(
            layers.DepthwiseConvBN(in_dim, in_dim, 3, stride=2),
            layers.ConvBN(in_dim, out_dim, 1))

        self.relu = layers.Activation("relu")

    def forward(self, x):
        return self.relu(self.branch_1(x) + self.branch_2(x))


class DetailBranch(nn.Layer):
    """The detail branch of BiSeNet, which has wide channels but shallow layers."""

    def __init__(self, in_channels, feature_channels):
        super().__init__()

        C1, C2, C3 = feature_channels

        self.convs = nn.Sequential(
            # stage 1
            layers.ConvBNReLU(in_channels, C1, 3, stride=2),
            layers.ConvBNReLU(C1, C1, 3),
            # stage 2
            layers.ConvBNReLU(C1, C2, 3, stride=2),
            layers.ConvBNReLU(C2, C2, 3),
            layers.ConvBNReLU(C2, C2, 3),
            # stage 3
            layers.ConvBNReLU(C2, C3, 3, stride=2),
            layers.ConvBNReLU(C3, C3, 3),
            layers.ConvBNReLU(C3, C3, 3),
        )

    def forward(self, x):
        return self.convs(x)


class SemanticBranch(nn.Layer):
    """The semantic branch of BiSeNet, which has narrow channels but deep layers."""

    def __init__(self, in_channels, feature_channels, context_type='none'):
        super().__init__()
        C1, C3, C4, C5 = feature_channels

        self.stem = StemBlock(in_channels, C1)

        self.stage3 = nn.Sequential(GatherAndExpansionLayer2(C1, C3, 6),
                                    GatherAndExpansionLayer1(C3, C3, 6))

        self.stage4 = nn.Sequential(GatherAndExpansionLayer2(C3, C4, 6),
                                    GatherAndExpansionLayer1(C4, C4, 6))

        self.stage5_4 = nn.Sequential(GatherAndExpansionLayer2(C4, C5, 6),
                                      GatherAndExpansionLayer1(C5, C5, 6),
                                      GatherAndExpansionLayer1(C5, C5, 6),
                                      GatherAndExpansionLayer1(C5, C5, 6))

        self.ce = ContextEmbeddingBlock(C5, C5)
        if context_type == 'ppm':
            self.context = layers.PPModule(C5, C5, (1, 2, 3, 6), True,
                                           False)
        elif context_type == 'aspp':
            self.context = layers.ASPPModule(
                (1, 6, 12, 18), C5, C5, False,
                use_sep_conv=True, image_pooling=True)
        else:
            self.context = nn.Identity()

    def forward(self, x):
        stage2 = self.stem(x)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5_4 = self.stage5_4(stage4)
        fm = self.context(self.ce(stage5_4))
        return stage2, stage3, stage4, stage5_4, fm


class BGA(nn.Layer):
    """The Bilateral Guided Aggregation Layer, used to fuse the semantic features and spatial features."""

    def __init__(self, out_dim, align_corners):
        super().__init__()

        self.align_corners = align_corners

        self.db_branch_keep = nn.Sequential(
            layers.DepthwiseConvBN(out_dim, out_dim, 3),
            nn.Conv2D(out_dim, out_dim, 1))

        self.db_branch_down = nn.Sequential(
            layers.ConvBN(out_dim, out_dim, 3, stride=2),
            nn.AvgPool2D(kernel_size=3, stride=2, padding=1))

        self.sb_branch_keep = nn.Sequential(
            layers.DepthwiseConvBN(out_dim, out_dim, 3),
            nn.Conv2D(out_dim, out_dim, 1), layers.Activation(act='sigmoid'))

        self.sb_branch_up = layers.ConvBN(out_dim, out_dim, 3)

        self.conv = layers.ConvBN(out_dim, out_dim, 3)

    def forward(self, dfm, sfm):
        db_feat_keep = self.db_branch_keep(dfm)
        db_feat_down = self.db_branch_down(dfm)
        sb_feat_keep = self.sb_branch_keep(sfm)

        sb_feat_up = self.sb_branch_up(sfm)
        sb_feat_up = F.interpolate(sb_feat_up,
                                   db_feat_keep.shape[2:],
                                   mode='bilinear',
                                   align_corners=self.align_corners)

        sb_feat_up = F.sigmoid(sb_feat_up)
        db_feat = db_feat_keep * sb_feat_up

        sb_feat = db_feat_down * sb_feat_keep
        sb_feat = F.interpolate(sb_feat,
                                db_feat.shape[2:],
                                mode='bilinear',
                                align_corners=self.align_corners)

        return self.conv(db_feat + sb_feat)


class SegHead(nn.Layer):

    def __init__(self, in_dim, mid_dim, num_classes):
        super().__init__()

        self.conv_3x3 = nn.Sequential(layers.ConvBNReLU(in_dim, mid_dim, 3),
                                      nn.Dropout(0.1))

        self.conv_1x1 = nn.Conv2D(mid_dim, num_classes, 1, 1)

    def forward(self, x):
        conv1 = self.conv_3x3(x)
        conv2 = self.conv_1x1(conv1)
        return conv2

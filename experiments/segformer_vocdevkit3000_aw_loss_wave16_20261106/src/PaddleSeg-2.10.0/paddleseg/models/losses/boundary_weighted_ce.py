import paddle
from paddle import nn
from paddle.nn import functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class BoundaryWeightedCrossEntropyLoss(nn.Layer):
    """Cross entropy with a differentiable weight on label transitions."""

    def __init__(self, edge_weight=1.0, ignore_index=255):
        super().__init__()
        self.edge_weight = float(edge_weight)
        self.ignore_index = int(ignore_index)

    def forward(self, logits, labels):
        labels_for_edge = paddle.cast(labels != self.ignore_index, 'float32')
        labels_for_edge = paddle.unsqueeze(labels_for_edge, 1)
        labels_for_edge = paddle.cast(labels, 'float32') * labels_for_edge

        local_max = F.max_pool2d(
            labels_for_edge, kernel_size=3, stride=1, padding=1)
        local_min = -F.max_pool2d(
            -labels_for_edge, kernel_size=3, stride=1, padding=1)
        edge_mask = paddle.cast((local_max - local_min) > 0, 'float32')
        edge_mask = paddle.squeeze(edge_mask, axis=1)

        logits_nhwc = paddle.transpose(logits, [0, 2, 3, 1])
        per_pixel = F.cross_entropy(
            logits_nhwc,
            labels,
            axis=-1,
            ignore_index=self.ignore_index,
            reduction='none')
        valid = paddle.cast(labels != self.ignore_index, 'float32')
        weights = valid * (1.0 + self.edge_weight * edge_mask)
        return paddle.sum(per_pixel * weights) / paddle.clip(
            paddle.sum(weights), min=1.0)

"""Haar-wavelet convolution blocks for the SegFormer crack models.

This module is deliberately separate from the historical ``wtconv``
candidate.  The historical candidate uses average pooling and a depthwise
convolution; ``WTConvWavelet`` implements an explicit, invertible 2-D Haar
analysis/synthesis path instead:

    x -> {LL, LH, HL, HH} -> learned sub-band filtering -> IWT -> residual

The Haar filters are implemented with tensor slicing rather than a learned
convolution.  This makes the transform deterministic, keeps the filters out
of the optimizer, and avoids backend-specific grouped-convolution weight
layouts.  Odd spatial sizes are padded by copying the last row/column and
cropped back after synthesis, so the block is shape-preserving.
"""

import paddle
from paddle import nn


class HaarDWT2D(nn.Layer):
    """Single-level orthonormal 2-D Haar analysis transform.

    The returned sub-bands are concatenated in the channel order
    ``[LL, LH, HL, HH]``.  A copy-pad is used only when height or width is
    odd; the original spatial size is returned as metadata for IWT cropping.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _copy_pad(x):
        h = int(x.shape[2])
        w = int(x.shape[3])
        if h % 2:
            x = paddle.concat([x, x[:, :, -1:, :]], axis=2)
        if w % 2:
            x = paddle.concat([x, x[:, :, :, -1:]], axis=3)
        return x, (h, w)

    def forward(self, x):
        x, original_hw = self._copy_pad(x)
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        # The 1/2 factor makes the four filters orthonormal and gives an
        # exactly invertible transform under the matching synthesis formulas.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return paddle.concat([ll, lh, hl, hh], axis=1), original_hw


class HaarIWT2D(nn.Layer):
    """Matching inverse Haar synthesis transform with odd-size cropping."""

    def __init__(self):
        super().__init__()

    def forward(self, bands, original_hw):
        c4 = int(bands.shape[1])
        if c4 % 4:
            raise ValueError("HaarIWT2D expects four equally sized sub-bands")
        c = c4 // 4
        ll, lh, hl, hh = paddle.split(bands, num_or_sections=4, axis=1)

        x00 = (ll + lh + hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll - lh - hl + hh) * 0.5

        # Arrange [row-even/odd, col-even/odd] into the original pixel grid.
        # The reshape/transpose sequence is differentiable and works for
        # dynamic Paddle tensors in both eager training and inference.
        stacked = paddle.stack([x00, x01, x10, x11], axis=-1)
        stacked = paddle.reshape(
            stacked,
            [int(stacked.shape[0]), c, int(stacked.shape[2]), int(stacked.shape[3]), 2, 2],
        )
        stacked = paddle.transpose(stacked, [0, 1, 2, 4, 3, 5])
        out = paddle.reshape(
            stacked,
            [int(stacked.shape[0]), c, int(stacked.shape[2]) * 2, int(stacked.shape[4]) * 2],
        )
        return out[:, :, : original_hw[0], : original_hw[1]]


class WTConvWavelet(nn.Layer):
    """Lightweight WTConv with explicit Haar DWT/IWT.

    ``LL`` carries low-frequency/context information while ``LH``, ``HL``
    and ``HH`` carry horizontal, vertical and diagonal details.  A grouped
    depthwise filter acts independently on each sub-band, followed by fixed
    IWT synthesis.  IWT reconstructs a full-resolution feature;
    a depthwise local refinement and a pointwise projection then form the
    residual output.
    """

    def __init__(self, channels, levels=1, detail_kernel=3):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if levels <= 0:
            raise ValueError("levels must be positive")
        if detail_kernel % 2 == 0:
            raise ValueError("detail_kernel must be odd")
        self.channels = int(channels)
        self.levels = int(levels)
        self.dwt = HaarDWT2D()
        self.iwt = HaarIWT2D()

        # One depthwise filter per channel and per Haar sub-band at each
        # level.  The fixed DWT/IWT add no parameters.  We intentionally do
        # not add a dense 4C->4C mixer here: the standard lightweight WTConv
        # path keeps sub-band filtering depthwise and lets IWT perform the
        # fixed cross-band synthesis.  Recursing over LL gives a genuine
        # multi-level wavelet path when levels > 1; the default one-level path
        # is the lightest complete transform for the current decoder budget.
        self.subband_dw = nn.LayerList()
        self.wavelet_scale = nn.ParameterList()
        for _ in range(self.levels):
            self.subband_dw.append(
                nn.Conv2D(
                    self.channels * 4,
                    self.channels * 4,
                    detail_kernel,
                    padding=detail_kernel // 2,
                    groups=self.channels * 4,
                    bias_attr=False,
                )
            )
            self.wavelet_scale.append(
                self.create_parameter(
                    [1, self.channels * 4, 1, 1],
                    default_initializer=nn.initializer.Constant(0.1),
                )
            )
        self.base_dw = nn.Conv2D(
            self.channels,
            self.channels,
            detail_kernel,
            padding=detail_kernel // 2,
            groups=self.channels,
            bias_attr=False,
        )
        self.base_scale = self.create_parameter(
            [1, self.channels, 1, 1],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.local_dw = nn.Conv2D(
            self.channels,
            self.channels,
            3,
            padding=1,
            groups=self.channels,
            bias_attr=False,
        )
        self.out_proj = nn.Conv2D(self.channels, self.channels, 1, bias_attr=False)
        self.out_norm = nn.BatchNorm2D(self.channels)
        self.act = nn.ReLU()

    def forward(self, x):
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                "WTConvWavelet expected {} channels, got {}".format(
                    self.channels, int(x.shape[1])
                )
            )
        current = x
        high_bands = []
        shapes = []
        for level in range(self.levels):
            bands, original_hw = self.dwt(current)
            bands = self.subband_dw[level](bands)
            bands = bands * self.wavelet_scale[level]
            current, high = paddle.split(bands, [self.channels, self.channels * 3], axis=1)
            high_bands.append(high)
            shapes.append(original_hw)

        # Reconstruct from the coarsest LL back to the original resolution.
        for level in range(self.levels - 1, -1, -1):
            current = self.iwt(
                paddle.concat([current, high_bands[level]], axis=1),
                shapes[level],
            )
        reconstructed = current
        local = self.base_scale * self.base_dw(x)
        local = local + self.local_dw(reconstructed)
        return x + self.act(self.out_norm(self.out_proj(local)))


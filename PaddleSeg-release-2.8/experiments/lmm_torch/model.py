import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(kernel_size=3, sigma=1.0, kernel_value=0.9):
    center = (kernel_size - 1) / 2.0
    y, x = np.ogrid[:kernel_size, :kernel_size]
    kernel = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    kernel *= kernel_value
    return kernel.astype("float32")


def _laplacian_kernel_2d(kernel_size=3, kernel_value=0.9):
    if kernel_size != 3:
        raise ValueError("Only 3x3 Laplacian kernel is supported.")
    kernel = np.array(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype="float32",
    )
    return kernel * kernel_value


def _build_depthwise_kernel_bank(channels, kernel_2d):
    kernel = np.expand_dims(np.expand_dims(kernel_2d, axis=0), axis=0)
    kernel = np.tile(kernel, (channels, 1, 1, 1))
    return torch.from_numpy(kernel)


class SEM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class EEM(nn.Module):
    def __init__(self, ch_in, ch_out, kernel=3, groups=1, reduction=16):
        super().__init__()
        if ch_in % groups != 0:
            raise ValueError(f"ch_in={ch_in} must be divisible by groups={groups}")
        self.groups = groups
        self.register_buffer(
            "gk",
            _build_depthwise_kernel_bank(ch_in, _gaussian_kernel_2d(kernel, sigma=1.0, kernel_value=0.9)),
        )
        self.register_buffer(
            "lk",
            _build_depthwise_kernel_bank(ch_in, _laplacian_kernel_2d(kernel, kernel_value=0.9)),
        )

        half = ch_out // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, half, kernel_size=1, groups=2),
            nn.PReLU(num_parameters=half, init=0.05),
            nn.InstanceNorm2d(half),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_in, half, kernel_size=1, groups=2),
            nn.PReLU(num_parameters=half, init=0.05),
            nn.InstanceNorm2d(half),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(half, ch_out, kernel_size=1, groups=2, bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out),
        )
        self.sem1 = SEM(ch_out, reduction=reduction)
        self.sem2 = SEM(ch_out, reduction=reduction)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.03)

    def forward(self, x):
        dog = F.conv2d(x, self.gk, padding=1, groups=self.groups)
        log = F.conv2d(dog, self.lk, padding=1, groups=self.groups)
        dog = self.conv1(dog - x)
        log = self.conv2(log)
        tot = self.conv3(dog * log)
        return self.prelu(x + self.sem2(x) + tot + self.sem1(tot))


class PFM(nn.Module):
    def __init__(self, ch_in, ch_out, ch_out_3x3e, pool_ch_out, eem_ch_out, reduction, shortcut=False):
        super().__init__()
        self.use_shortcut = shortcut
        self.reducer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, groups=2, bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.03),
            nn.GroupNorm(4, ch_out),
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, groups=2, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, ch_out),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out_3x3e, kernel_size=3, padding=1, groups=2, bias=False),
            nn.PReLU(num_parameters=ch_out_3x3e, init=0.0),
            nn.GroupNorm(4, ch_out_3x3e),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ch_out, pool_ch_out, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, pool_ch_out),
        )
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ch_in, ch_out + ch_out_3x3e, kernel_size=1, groups=4, bias=False),
                nn.GroupNorm(4, ch_out + ch_out_3x3e),
            )
        self.eem = EEM(ch_out, eem_ch_out, kernel=3, groups=ch_out, reduction=reduction[0])
        self.sem1 = SEM(ch_out + ch_out_3x3e, reduction=reduction[1])
        self.sem2 = SEM(ch_out + ch_out_3x3e, reduction=reduction[1])
        self.prelu = nn.PReLU(num_parameters=ch_out + ch_out_3x3e, init=0.03)

    def forward(self, x):
        x1 = self.reducer(x)
        b1 = self.b1(x1)
        b2 = self.b2(x1 + b1)
        b3 = self.b3(x1)
        eem = self.eem(x1)
        y1 = torch.cat([x1 + b1 + b3 + eem, b2], dim=1)
        x_skip = self.shortcut(x) if self.use_shortcut else x
        return self.prelu(x_skip + y1 + self.sem1(y1) + self.sem2(x_skip))


class PDAM(nn.Module):
    def __init__(self, ch_in, ch_out, reduction, dropout):
        super().__init__()
        half = ch_out // 2
        self.conv1a = nn.Sequential(
            nn.Conv2d(ch_in[0], half, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, half),
        )
        self.conv1b = nn.Sequential(
            nn.Conv2d(ch_in[0], half, kernel_size=1, groups=2, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, half),
        )
        self.conv2a = nn.Sequential(
            nn.Conv2d(ch_in[1], half, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=half, init=-0.01),
            nn.GroupNorm(4, half),
        )
        self.conv2b = nn.Sequential(
            nn.Conv2d(ch_in[1], half, kernel_size=1, groups=2, bias=False),
            nn.PReLU(num_parameters=half, init=-0.01),
            nn.GroupNorm(4, half),
        )
        self.conv3a = nn.Sequential(
            nn.Conv2d(ch_in[2], half, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=half, init=-0.01),
            nn.GroupNorm(4, half),
        )
        self.conv3b = nn.Sequential(
            nn.Conv2d(ch_in[2], half, kernel_size=1, groups=4, bias=False),
            nn.PReLU(num_parameters=half, init=-0.01),
            nn.GroupNorm(4, half),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, groups=2, bias=False),
            nn.PReLU(num_parameters=ch_out, init=0.01),
            nn.GroupNorm(4, ch_out),
        )
        self.dropout = nn.Dropout2d(dropout)
        self.sem = SEM(ch_in[0], reduction=reduction)

    def forward(self, x, x1, x2):
        x0 = self.sem(x)
        x0 = torch.cat([self.conv1a(x + x0), self.conv1b(x + x0)], dim=1)
        x1 = torch.cat([self.conv2a(x1), self.conv2b(x1)], dim=1)
        x2 = torch.cat([self.conv3a(x2), self.conv3b(x2)], dim=1)
        x3 = self.dropout(F.softmax(x1 * x2, dim=-1))
        return self.conv4(x0 + x1 + x2 + x3)


class FRCM(nn.Module):
    def __init__(self, ch_ins, ch_out, n_sides=11):
        super().__init__()
        self.reducers = nn.ModuleList([nn.Conv2d(ch_in, ch_out, kernel_size=1) for ch_in in ch_ins])
        self.gn = nn.GroupNorm(1, ch_out)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.1)
        self.fused = nn.Conv2d(ch_out * n_sides, ch_out, kernel_size=1)

    def forward(self, img_shape, sides):
        late_sides = []
        for feature, conv in zip(sides, self.reducers):
            side = F.interpolate(conv(feature), size=img_shape, mode="bilinear", align_corners=True)
            late_sides.append(self.gn(self.prelu(side)))
        fused = self.prelu(self.fused(torch.cat(late_sides, dim=1)))
        late_sides.append(fused)
        return late_sides


class LMMNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.prelayer = nn.Conv2d(in_channels, 96, kernel_size=3, padding=1, bias=False)
        self.sem1 = SEM(96, reduction=24)

        self.pfm1 = PFM(96, 32, 64, 32, 32, [8, 24])
        self.pfm2 = PFM(96, 32, 64, 32, 32, [8, 24])
        self.pfm3 = PFM(96, 32, 96, 32, 32, [8, 32], shortcut=True)
        self.pfm4 = PFM(128, 32, 96, 32, 32, [8, 32])
        self.pfm5 = PFM(128, 32, 96, 32, 32, [8, 32])
        self.pfm6 = PFM(128, 64, 128, 64, 64, [16, 48], shortcut=True)
        self.pfm7 = PFM(192, 64, 128, 64, 64, [16, 48])
        self.pfm8 = PFM(192, 64, 128, 64, 64, [16, 48])
        self.pfm9 = PFM(192, 64, 128, 64, 64, [16, 48])

        self.conv1 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=1, groups=4, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, 128),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=3, padding=2, dilation=2, groups=2, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, 128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=4, dilation=4, groups=4, bias=False),
            nn.ReLU(inplace=True),
            nn.GroupNorm(4, 64),
        )
        self.sem2 = SEM(320, reduction=80)

        self.pdam4 = PDAM([320, 192, 192], 128, 64, 0.0125)
        self.pdam3 = PDAM([128, 192, 192], 128, 32, 0.0125)
        self.pdam2 = PDAM([128, 128, 128], 96, 32, 0.025)
        self.pdam1 = PDAM([96, 96, 96], 64, 12, 0.05)

        self.frcm = FRCM([96, 96, 128, 128, 128, 192, 192, 192, 192, 64, 320], ch_out=2)

        self.sem3 = SEM(88, reduction=11)
        self.lastlayer = nn.Sequential(
            nn.Conv2d(88, 64, kernel_size=3, padding=1, groups=4, bias=False),
            nn.PReLU(num_parameters=64, init=-0.01),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=64, init=0.0),
            nn.GroupNorm(4, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False),
            nn.PReLU(num_parameters=num_classes, init=0.0),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.3)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.1)
        self._init_weights()

    def forward(self, x):
        img_shape = x.shape[2:]
        x1 = self.prelayer(x)
        x = x1 + self.sem1(x1)

        i1 = self.pfm1(x)
        i2 = self.pfm2(i1)
        x = self.dropout1(self.max_pool(i2))

        i3 = self.pfm3(x)
        i4 = self.pfm4(i3)
        i5 = self.pfm5(i4)
        x = self.dropout2(self.max_pool(i5))

        i6 = self.pfm6(x)
        i7 = self.pfm7(i6)
        i8 = self.pfm8(i7)
        i9 = self.pfm9(i8)
        x = self.dropout3(self.max_pool(i9))

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = torch.cat([x1, x2, x3], dim=1)
        x2 = self.sem2(x1)

        x = F.interpolate(x1 + x2, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.pdam4(x, i9, i8)
        x = self.pdam3(x, i7, i6)
        x = self.dropout3(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.pdam2(x, i5, i4)
        x = self.dropout2(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.pdam1(x, i2, i1)
        x = self.dropout1(x)

        sides = self.frcm(img_shape, [i1, i2, i3, i4, i5, i6, i7, i8, i9, x, x1])
        x = torch.cat([x] + sides, dim=1)
        x = self.lastlayer(x + self.sem3(x))
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

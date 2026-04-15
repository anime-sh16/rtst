import torch.nn as nn


class DWConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        stride,
        padding=None,
        bias=False,
        export_mode=False,
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2 if padding is None else padding,
            groups=in_channels,
            padding_mode="zeros" if export_mode else "reflect",
            bias=bias,
        )

    def forward(self, x):
        return self.conv2d(x)


class PWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, bias=False):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=bias,
        )

    def forward(self, x):
        return self.conv2d(x)


class SENet(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class DownSample(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, bias=False, export_mode=False
    ):
        super().__init__()

        out_channels_2 = out_channels
        out_channels_1 = out_channels_2 // 2

        self.downsample = nn.Sequential(
            DWConvLayer(
                in_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=bias,
                export_mode=export_mode,
            ),
            PWConvLayer(in_channels, out_channels_1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            DWConvLayer(
                out_channels_1,
                kernel_size=3,
                stride=stride,
                bias=bias,
                export_mode=export_mode,
            ),
            PWConvLayer(out_channels_1, out_channels_2, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels_2),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.downsample(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion_factor,
        kernel_size,
        stride,
        bias=False,
        se_attention_bool=False,
        export_mode=False,
    ):
        super().__init__()
        hidden_dim = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        self.conv_path_1 = nn.Sequential(
            PWConvLayer(in_channels, hidden_dim, stride=1, bias=bias),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            DWConvLayer(
                hidden_dim,
                kernel_size,
                stride=stride,
                bias=bias,
                export_mode=export_mode,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        if se_attention_bool:
            self.se_attention = SENet(hidden_dim)
        self.conv_path_2 = nn.Sequential(
            PWConvLayer(hidden_dim, out_channels, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.conv_path_1(x)
        if hasattr(self, "se_attention"):
            out = self.se_attention(out)
        out = self.conv_path_2(out)
        if self.use_residual:
            out = out + identity
        return out


class UpSample(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, bias=False, export_mode=False
    ):
        super().__init__()

        out_channels_1 = in_channels // 2

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="bilinear", align_corners=False),
            DWConvLayer(
                in_channels, kernel_size=3, stride=1, bias=bias, export_mode=export_mode
            ),
            PWConvLayer(in_channels, out_channels_1, stride=1, bias=bias),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU6(inplace=True),
            DWConvLayer(
                out_channels_1,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
                export_mode=export_mode,
            ),
            PWConvLayer(out_channels_1, out_channels, stride=1, bias=True),
        )

    def forward(self, x):
        return self.upsample(x)


class TransformationNetworkV2(nn.Module):
    def __init__(self, se_attention_bool=False, export_mode=False):
        super().__init__()
        self.downsample = DownSample(
            in_channels=3, out_channels=64, stride=2, export_mode=export_mode
        )
        self.upsample = UpSample(
            in_channels=64, out_channels=3, stride=2, export_mode=export_mode
        )
        self.residual_path = nn.Sequential(
            InvertedResidualBlock(
                in_channels=64,
                out_channels=64,
                expansion_factor=3,
                kernel_size=3,
                stride=1,
                bias=False,
                se_attention_bool=se_attention_bool,
                export_mode=export_mode,
            ),
            InvertedResidualBlock(
                in_channels=64,
                out_channels=64,
                expansion_factor=3,
                kernel_size=3,
                stride=1,
                bias=False,
                se_attention_bool=se_attention_bool,
                export_mode=export_mode,
            ),
            InvertedResidualBlock(
                in_channels=64,
                out_channels=64,
                expansion_factor=3,
                kernel_size=3,
                stride=1,
                bias=False,
                se_attention_bool=se_attention_bool,
                export_mode=export_mode,
            ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.downsample(x)
        x = self.residual_path(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        return x

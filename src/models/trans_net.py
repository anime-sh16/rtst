import torch
import torch.nn as nn


class TransformationNetwork(nn.Module):
    def __init__(self, norm_layer_type=nn.BatchNorm2d, export_mode=False):
        super().__init__()

        self.downsample_block = Downsample(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=2,
            norm_layer_type=norm_layer_type,
            export_mode=export_mode,
        )
        self.upsample_block = UpsampleV2(
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=2,
            norm_layer_type=norm_layer_type,
            export_mode=export_mode,
        )

        self.residual_conn = nn.Sequential(
            ResidualBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                norm_layer_type=norm_layer_type,
                export_mode=export_mode,
            ),
            ResidualBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                norm_layer_type=norm_layer_type,
                export_mode=export_mode,
            ),
            ResidualBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                norm_layer_type=norm_layer_type,
                export_mode=export_mode,
            ),
            ResidualBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                norm_layer_type=norm_layer_type,
                export_mode=export_mode,
            ),
            ResidualBlock(
                128,
                128,
                kernel_size=3,
                stride=1,
                norm_layer_type=norm_layer_type,
                export_mode=export_mode,
            ),
        )

    def forward(self, x):
        x = self.downsample_block(x)
        x = self.residual_conn(x)
        x = self.upsample_block(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        export_mode=False,
    ):
        super().__init__()
        self.reflection_pad = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.reflection_pad,
            padding_mode="zeros" if export_mode else "reflect",
            bias=bias,
        )

    def forward(self, x) -> torch.Tensor:
        return self.conv2d(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_type=nn.BatchNorm2d,
        export_mode=False,
    ):
        super().__init__()
        self.conv_path = nn.Sequential(
            ConvLayer(
                in_channels, out_channels, kernel_size, stride, export_mode=export_mode
            ),
            norm_layer_type(out_channels),
            nn.ReLU(inplace=True),
            ConvLayer(
                out_channels, out_channels, kernel_size, stride, export_mode=export_mode
            ),
            norm_layer_type(out_channels),
        )
        if in_channels != out_channels:
            self.identity_path = nn.Sequential(
                ConvLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    export_mode=export_mode,
                ),
                norm_layer_type(out_channels),
            )
        else:
            self.identity_path = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) -> torch.Tensor:
        return self.relu(self.conv_path(x) + self.identity_path(x))


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_type=nn.BatchNorm2d,
        export_mode=False,
    ):
        super().__init__()

        out_channels_3 = out_channels
        out_channels_2 = out_channels_3 // 2
        out_channels_1 = out_channels_2 // 2

        self.downsample = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels_1,
                kernel_size=9,
                stride=1,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_1),
            nn.ReLU(inplace=True),
            ConvLayer(
                out_channels_1,
                out_channels_2,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_2),
            nn.ReLU(inplace=True),
            ConvLayer(
                out_channels_2,
                out_channels_3,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.downsample(x)


class TransposeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.pad,
            output_padding=stride - 1,
            bias=bias,
        )

    def forward(self, x) -> torch.Tensor:
        return self.conv2d(x)


class UpsampleConvLayer(nn.Module):
    """Upsample + Conv replacement for TransposeConvLayer to avoid checkerboard artifacts."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=False,
        export_mode=False,
    ):
        super().__init__()
        if export_mode:
            self.upsample = nn.Upsample(
                scale_factor=stride, mode="bilinear", align_corners=False
            )
        else:
            self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")
        self.conv2d = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=bias,
            export_mode=export_mode,
        )

    def forward(self, x) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv2d(x)
        return x


class UpsampleV2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_type=nn.BatchNorm2d,
        export_mode=False,
    ):
        super().__init__()
        out_channels_1 = in_channels // 2
        out_channels_2 = out_channels_1 // 2
        out_channels_3 = out_channels

        self.upsample = nn.Sequential(
            UpsampleConvLayer(
                in_channels,
                out_channels_1,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_1),
            nn.ReLU(inplace=True),
            UpsampleConvLayer(
                out_channels_1,
                out_channels_2,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_2),
            nn.ReLU(inplace=True),
            ConvLayer(
                out_channels_2,
                out_channels_3,
                kernel_size=9,
                stride=1,
                bias=True,
                export_mode=export_mode,
            ),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.upsample(x)
        return torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_type=nn.BatchNorm2d,
        export_mode=False,
    ):
        super().__init__()

        out_channels_1 = in_channels // 2
        out_channels_2 = out_channels_1 // 2
        out_channels_3 = out_channels

        self.upsample = nn.Sequential(
            TransposeConvLayer(
                in_channels,
                out_channels_1,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_1),
            nn.ReLU(inplace=True),
            TransposeConvLayer(
                out_channels_1,
                out_channels_2,
                kernel_size,
                stride,
                export_mode=export_mode,
            ),
            norm_layer_type(out_channels_2),
            nn.ReLU(inplace=True),
            ConvLayer(
                out_channels_2,
                out_channels_3,
                kernel_size=9,
                stride=1,
                bias=True,
                export_mode=export_mode,
            ),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.upsample(x)
        return torch.sigmoid(x)

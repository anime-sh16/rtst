import torch
import torch.nn as nn


class TransformationNetwork(nn.Module):
    def __init__(self, norm_layer_type=nn.BatchNorm2d):
        super().__init__()

        self.downsample_block = Downsample(
            in_channels=3,
            out_channels=128,
            kernel_size=3,
            stride=2,
            norm_layer_type=norm_layer_type,
        )
        self.upsample_block = Upsample(
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=2,
            norm_layer_type=norm_layer_type,
        )

        self.residual_conn = nn.Sequential(
            ResidualBlock(
                128, 128, kernel_size=3, stride=1, norm_layer_type=norm_layer_type
            ),
            ResidualBlock(
                128, 128, kernel_size=3, stride=1, norm_layer_type=norm_layer_type
            ),
            ResidualBlock(
                128, 128, kernel_size=3, stride=1, norm_layer_type=norm_layer_type
            ),
            ResidualBlock(
                128, 128, kernel_size=3, stride=1, norm_layer_type=norm_layer_type
            ),
            ResidualBlock(
                128, 128, kernel_size=3, stride=1, norm_layer_type=norm_layer_type
            ),
        )

    def forward(self, x):
        x = self.downsample_block(x)
        x = self.residual_conn(x)
        x = self.upsample_block(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.reflection_pad = kernel_size // 2
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.reflection_pad,
            padding_mode="reflect",
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
    ):
        super().__init__()
        self.conv_path = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            norm_layer_type(out_channels),
            nn.ReLU(inplace=True),
            ConvLayer(out_channels, out_channels, kernel_size, stride),
            norm_layer_type(out_channels),
        )
        if in_channels != out_channels:
            self.identity_path = nn.Sequential(
                ConvLayer(in_channels, out_channels, kernel_size, stride),
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
    ):
        super().__init__()

        out_channels_3 = out_channels
        out_channels_2 = out_channels_3 // 2
        out_channels_1 = out_channels_2 // 2

        self.downsample = nn.Sequential(
            ConvLayer(in_channels, out_channels_1, kernel_size=9, stride=1),
            norm_layer_type(out_channels_1),
            nn.ReLU(inplace=True),
            ConvLayer(out_channels_1, out_channels_2, kernel_size, stride),
            norm_layer_type(out_channels_2),
            nn.ReLU(inplace=True),
            ConvLayer(out_channels_2, out_channels_3, kernel_size, stride),
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


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        norm_layer_type=nn.BatchNorm2d,
    ):
        super().__init__()

        out_channels_1 = in_channels // 2
        out_channels_2 = out_channels_1 // 2
        out_channels_3 = out_channels

        self.upsample = nn.Sequential(
            TransposeConvLayer(in_channels, out_channels_1, kernel_size, stride),
            norm_layer_type(out_channels_1),
            nn.ReLU(inplace=True),
            TransposeConvLayer(out_channels_1, out_channels_2, kernel_size, stride),
            norm_layer_type(out_channels_2),
            nn.ReLU(inplace=True),
            TransposeConvLayer(
                out_channels_2, out_channels_3, kernel_size=9, stride=1, bias=True
            ),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.upsample(x)
        return torch.sigmoid(x)

import torch
import torch.nn as nn
from descriptors import Descriptor


class EffConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True
    ):
        super(EffConv2d, self).__init__()
        if kernel_size >= 5:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (1, kernel_size),
                    stride=stride,
                    padding=(0, padding),
                    bias=bias,
                ),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    (kernel_size, 1),
                    stride=stride,
                    padding=(padding, 0),
                    bias=bias,
                ),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, stride=1, bias=True
    ):
        super(DeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class PyramidMaxout(nn.Module):
    def __init__(self, configs, out_channels):
        super(PyramidMaxout, self).__init__()
        conv_configs = configs["conv_configs"]
        kernel_levels = configs["kernel_levels"]
        self.layers = nn.ModuleList(
            [
                EffConv2d(
                    **conv_configs,
                    out_channels=out_channels,
                    kernel_size=2 * i - 1,
                    padding=i - 1
                )
                for i in kernel_levels
            ]
        )

    def forward(self, x):
        x = torch.cat([torch.unsqueeze(layer(x), 0) for layer in self.layers], dim=0)
        x = torch.max(x, dim=0)
        return x[0]


class Decoder(nn.Module):
    def __init__(self, decoder_configs):
        super(Decoder, self).__init__()
        conv_configs = decoder_configs["conv_configs"]
        input_size = decoder_configs["input_size"]
        output_size = decoder_configs["output_size"]
        layer_counter = 0

        # this is the number of layers of kernel size 2 that will be used to match output size
        filler_layer_counter = 0

        while input_size <= output_size - (
            conv_configs["kernel_size"] - 1 - conv_configs["stride"]
        ):
            input_size = (conv_configs["kernel_size"] - 1) + (
                input_size - 1
            ) * conv_configs["stride"]
            layer_counter += 1

        while input_size < output_size:
            filler_layer_counter += 1
            input_size += 1

        self.layers = nn.ModuleList(
            [DeConv2d(**conv_configs) for _ in range(layer_counter)]
            + [DeConv2d(kernel_size=2, stride=1) for _ in range(filler_layer_counter)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            print("decoder layer", x.shape)
        return x


class SE(nn.Module):
    def __init__(self, snow_mask_configs):
        super(SE, self).__init__()
        self.pyramid_maxout = PyramidMaxout(snow_mask_configs, out_channels=1)
        # just one parameter for now
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.pyramid_maxout(x)
        x = self.prelu(x)
        x = torch.clamp(x, min=0, max=1)
        return x


class AE(nn.Module):
    def __init__(self, abberation_configs):
        super(AE, self).__init__()
        self.pyramid_maxout = PyramidMaxout(abberation_configs, out_channels=3)
        # just one parameter for now
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.pyramid_maxout(x)
        x = self.prelu(x)
        return x


class Rt(nn.Module):
    def __init__(self, trans_recovery_configs):
        super(Rt, self).__init__()
        snow_mask_configs = trans_recovery_configs["snow_mask_configs"]
        abberation_configs = trans_recovery_configs["abberation_configs"]
        decoder_configs = trans_recovery_configs["decoder_configs"]
        self.decoder = Decoder(decoder_configs)
        self.se = SE(snow_mask_configs)
        self.ae = AE(abberation_configs)

    def forward(self, ft, x):
        ft = self.decoder(ft)
        z = self.se(ft)
        a = self.ae(ft)
        z_mask = torch.where(z >= 1, 0, z)

        y = (x - (a * z_mask)) / (1 - z_mask)
        fc = y * torch.norm(z) * a

        return y, fc


class TranRecovery(nn.Module):
    def __init__(self, dilation_pyrimand_configs, trans_recovery_configs, data_configs):
        super(TranRecovery, self).__init__()

        self.Dt = Descriptor(
            dilation_pyramid_configs=dilation_pyrimand_configs,
            data_configs=data_configs,
        )

        ft_shape = self.Dt.get_output_dims()
        print("ft_shape", ft_shape)
        trans_recovery_configs["decoder_configs"]["in_channels"] = ft_shape[1]
        trans_recovery_configs["decoder_configs"]["out_channels"] = ft_shape[1]
        trans_recovery_configs["decoder_configs"]["input_size"] = ft_shape[2]
        trans_recovery_configs["decoder_configs"]["output_size"] = data_configs[
            "sample_shape"
        ]

        self.Rt = Rt(trans_recovery_configs)

    def forward(self, x):
        ft = self.Dt(x)
        y, fc = self.Rt(ft, x)
        return y, fc

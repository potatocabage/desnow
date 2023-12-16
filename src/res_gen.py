import torch
import torch.nn as nn
from descriptors import Descriptor
from trans_recovery import Decoder, EffConv2d


class PyramidSum(nn.Module):
    def __init__(self, pyramid_sum_configs):
        super(PyramidSum, self).__init__()
        self.pyramid_sum_configs = pyramid_sum_configs
        conv_configs = pyramid_sum_configs["conv_configs"]
        kernel_levels = pyramid_sum_configs["kernel_levels"]
        self.layers = nn.ModuleList(
            [
                EffConv2d(
                    **conv_configs,
                    out_channels=3,
                    kernel_size=2 * (i + 1) - 1,
                    padding=i
                )
                for i in range(kernel_levels)
            ]
        )

    def forward(self, x):
        out = torch.zeros_like(x)
        outs = [layer(x) for layer in self.layers]
        out = torch.sum(torch.stack(outs), dim=0)
        return out


class Rr(nn.Module):
    def __init__(self, res_gen_configs):
        super(Rr, self).__init__()
        self.res_gen_configs = res_gen_configs
        self.need_decoder = res_gen_configs["need_decoder"]
        if self.need_decoder:
            self.decoder = Decoder(res_gen_configs["decoder_configs"])
        self.pyramid_sum = PyramidSum(res_gen_configs["pyramid_sum_configs"])

    def forward(self, x):
        if self.need_decoder:
            x = self.decoder(x)
        x = self.pyramid_sum(x)
        return x


class ResGen(nn.Module):
    def __init__(self, dilation_pyramid_configs, res_gen_configs, data_configs):
        super(ResGen, self).__init__()
        self.res_gen_configs = res_gen_configs
        self.Dr = Descriptor(
            dilation_pyramid_configs=dilation_pyramid_configs, data_configs=data_configs
        )

        fr_shape = self.Dr.get_output_dims()
        print("fr_shape", fr_shape)
        if fr_shape[2] < data_configs["sample_shape"]:
            res_gen_configs['need_decoder'] = True
            print("ResGen needs decoder")
            res_gen_configs["decoder_configs"]["conv_configs"]["in_channels"] = fr_shape[1]
            res_gen_configs["decoder_configs"]["conv_configs"]["out_channels"] = fr_shape[1]
            res_gen_configs["decoder_configs"]["input_size"] = fr_shape[2]
            res_gen_configs["decoder_configs"]["output_size"] = data_configs["sample_shape"]
        else:
            res_gen_configs["need_decoder"] = False

        res_gen_configs["pyramid_sum_configs"]["conv_configs"][
            "in_channels"
        ] = fr_shape[1]

        self.Rr = Rr(res_gen_configs)

    def forward(self, fc):
        fr = self.Dr(fc)
        r = self.Rr(fr)
        return r

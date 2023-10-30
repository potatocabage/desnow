from trans_recovery import *
from res_gen import *


class Desnow(nn.Module):
    def __init__(self, model_configs, data_configs):
        super(Desnow, self).__init__()

        self.model_configs = model_configs
        self.data_configs = data_configs
        self.dilation_pyramid_configs = model_configs["dilation_pyramid_configs"]
        self.trans_recovery_configs = model_configs["trans_recovery_configs"]
        self.res_gen_configs = model_configs["res_gen_configs"]

        self.trans_recovery = TranRecovery(
            dilation_pyrimand_configs=self.dilation_pyramid_configs,
            trans_recovery_configs=self.trans_recovery_configs,
            data_configs=self.data_configs,
        )
        self.res_gen = ResGen(
            dilation_pyramid_configs=self.dilation_pyramid_configs,
            res_gen_configs=self.res_gen_configs,
            data_configs=self.data_configs,
        )

    def forward(self, x):
        y_snow_free, fc = self.trans_recovery(x)
        r = self.res_gen(fc)
        y = y_snow_free + r
        return y

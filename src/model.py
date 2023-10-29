from descriptors import Descriptor
from trans_recovery import *


class Desnow(nn.Module):
    def __init__(self, model_configs, data_configs):
        super(Desnow, self).__init__()

        self.model_configs = model_configs
        self.data_configs = data_configs
        self.dilation_pyramid_configs = model_configs["dilation_pyramid_configs"]
        self.trans_recovery_configs = model_configs["trans_recovery_configs"]

        self.Dt = Descriptor(self.dilation_pyramid_configs, self.data_configs)
        self.Dr = Descriptor(self.dilation_pyramid_configs, self.data_configs)
        self.trans_recovery

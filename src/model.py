from trans_recovery import *
from res_gen import *
import torch.nn as nn


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
        # print('sample_shape', x.shape)
        # print(type(x))
        y_snow_free, fc, z_hat = self.trans_recovery(x)
        r = self.res_gen(fc)
        y_hat = y_snow_free + r
        return y_snow_free, y_hat, z_hat


class CustomLoss(nn.Module):
    def __init__(self, loss_configs):
        super(CustomLoss, self).__init__()
        self.loss_configs = loss_configs
        self.pool_levels = loss_configs["pool_levels"]
        self.z_lambda = loss_configs["z_lambda"]
        self.layers = nn.ModuleList([nn.MaxPool2d(2**i, 2**i, padding=(2**i)//2) 
                                     for i in range(self.pool_levels+1)])
        # l2 regularization is done by weight_decay in optimizer

    def calc_loss(self, gen, true):
        loss = 0
        for layer in self.layers:
            loss += torch.norm(layer(true) - layer(gen))**2
        return loss

    def forward(self, y, z, y_snow_free, y_hat, z_hat):
        y_snow_free_loss = self.calc_loss(y_snow_free, y)
        y_hat_loss = self.calc_loss(y_hat, y)
        z_hat_loss = self.calc_loss(z_hat, z)

        return y_snow_free_loss + y_hat_loss + self.z_lambda*z_hat_loss       
        
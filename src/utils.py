import torch
import torch.nn as nn
from configs import open_configs


# def conv_feature_concat(features, dilation_range):
#     cat_features = torch.empty(0)
#     for i in range(dilation_range):
#         conv = nn.Conv2d(
#             features.shape(1), features.shape(1), kernel_size=2**i, dilation=2**i
#         )
#         cat_features = torch.cat((cat_features, conv(features)), 1)

#     return cat_features

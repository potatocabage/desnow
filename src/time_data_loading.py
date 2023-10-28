from PIL import Image
import os
import numpy as np
import torch
from configs import open_configs
from load_data import load_data
from format_data import format_data
from matplotlib import pyplot as plt
from time import time

n = 1000
print("NORM BY IMAGE")
data_configs, training_configs, dilation_range = open_configs()
print("configs oppened")
training_datasets = load_data(data_configs)
print("data loaded")
data = torch.utils.data.ConcatDataset(training_datasets)
print("data concatenated")

start = time()
for i in range(1000):
    print(data[i][0].shape)
    # print(data[i][0])
    print(torch.max(data[i][0]))
    print(torch.min(data[i][0]))
    print(torch.max(data[i][1]))
    print(torch.min(data[i][1]))
    # plt.imshow(data[i,0], interpolation='nearest')
    # plt.show()
    x = data[i]
end = time()
print(f"getting {n} images took {end - start} sec")

print("NORM BY ITEM")
data_configs["norm_by_image"] = False
training_datasets = load_data(data_configs)
print("data loaded")
data = torch.utils.data.ConcatDataset(training_datasets)
print("data concatenated")

start = time()
for i in range(1000):
    print(data[i][0].shape)
    # print(data[i][0])
    print(torch.max(data[i][0]))
    print(torch.min(data[i][0]))
    print(torch.max(data[i][1]))
    print(torch.min(data[i][1]))
    # plt.imshow(data[i,0], interpolation='nearest')
    # plt.show()
    x = data[i]
end = time()
print(f"getting {n} images took {end - start} sec")

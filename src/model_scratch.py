from descriptors import *
from configs import open_configs
from load_data import load_data
from format_data import format_data
from trans_recovery import *
from model import Desnow
import torch

if __name__ == "__main__":
    data_configs, training_configs, model_configs = open_configs()
    print("configs oppened")
    training_datasets = load_data(data_configs)
    print("data loaded")
    data = torch.utils.data.ConcatDataset(training_datasets)
    print("data concatenated")

    model = Desnow(model_configs=model_configs, data_configs=data_configs)
    model.eval()

    with torch.no_grad():
        out = model(data[0][0].unsqueeze(0))
    print(out.shape)

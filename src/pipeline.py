from configs import open_configs
from load_data import load_data
from format_data import format_data
from time import time


def pipeline():
    data_configs, training_configs, model_configs = open_configs()

    training_datasets = load_data(data_configs)

    training_dataloader = format_data(training_datasets, training_configs)

    return training_dataloader


if __name__ == "__main__":
    pipeline()
    print("Done!")

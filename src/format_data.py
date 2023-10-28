import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np


def format_data(training_datasets, training_configs):
    training_dataset = ConcatDataset(training_datasets)
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )

    return training_dataloader

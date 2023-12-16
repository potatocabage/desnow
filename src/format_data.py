import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np


def format_data(training_datasets, val_datasets, test_l_datasets, test_m_datasets, test_s_datasets, training_configs):
    
    training_dataset = ConcatDataset(training_datasets)
    val_datasets = ConcatDataset(val_datasets)
    test_l_datasets = ConcatDataset(test_l_datasets)
    test_m_datasets = ConcatDataset(test_m_datasets)
    test_s_datasets = ConcatDataset(test_s_datasets)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )
    val_dataloader = DataLoader(
        val_datasets,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )
    test_l_dataloader = DataLoader(
        test_l_datasets,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )
    test_m_dataloader = DataLoader(
        test_m_datasets,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )
    test_s_dataloader = DataLoader(
        test_s_datasets,
        batch_size=training_configs["batch_size"],
        shuffle=True,
        num_workers=training_configs["num_workers"],
    )

    return training_dataloader, val_dataloader, test_l_dataloader, test_m_dataloader, test_s_dataloader

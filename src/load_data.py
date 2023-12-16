import os
import numpy as np
from PIL import Image
from dataset import *
from time import time
from multiprocessing import Pool
import random


def load_data(data_configs):
    sample_shape = data_configs["sample_shape"]
    norm_by_image = data_configs["norm_by_image"]
    dataset_workers = data_configs["dataset_workers"]
    num_train_val_images = data_configs["num_train_val_images"]
    num_test_images = data_configs["num_test_images"]
    train_val_split = data_configs["train_val_split"]

    path = data_configs["data_path"]
    train_path = os.path.join(path, "training/")
    train_syn_path = os.path.join(train_path, "synthetic/")
    train_gt_path = os.path.join(train_path, "gt/")
    train_mask_path = os.path.join(train_path, "mask/")
    test_path = os.path.join(path, "test/")
    test_l_path = os.path.join(test_path, "Snow100K-L/")
    test_m_path = os.path.join(test_path, "Snow100K-M/")
    test_s_path = os.path.join(test_path, "Snow100K-S/")
    test_l_syn_path = os.path.join(test_l_path, "synthetic/")
    test_l_gt_path = os.path.join(test_l_path, "gt/")
    test_l_mask_path = os.path.join(test_l_path, "mask/")
    test_m_syn_path = os.path.join(test_m_path, "synthetic/")
    test_m_gt_path = os.path.join(test_m_path, "gt/")
    test_m_mask_path = os.path.join(test_m_path, "mask/")
    test_s_syn_path = os.path.join(test_s_path, "synthetic/")
    test_s_gt_path = os.path.join(test_s_path, "gt/")
    test_s_mask_path = os.path.join(test_s_path, "mask/")

    train_datasets = load_dataset_from_path(train_syn_path,
        train_gt_path, train_mask_path, sample_shape, norm_by_image,
        dataset_workers, num_train_val_images)
    val_datasets = train_datasets[round(train_val_split * (1-len(train_datasets))):]
    train_datasets = train_datasets[:round(train_val_split * len(train_datasets))]

    test_l_datasets = load_dataset_from_path(test_l_syn_path, test_l_gt_path, test_l_mask_path,
        sample_shape, norm_by_image, dataset_workers, num_test_images, full_image=True)
    test_m_datasets = load_dataset_from_path(test_m_syn_path, test_m_gt_path, test_m_mask_path,
        sample_shape, norm_by_image, dataset_workers, num_test_images, full_image=True)
    test_s_datasets = load_dataset_from_path(test_s_syn_path, test_s_gt_path, test_s_mask_path,
        sample_shape, norm_by_image, dataset_workers, num_test_images, full_image=True)

    return train_datasets, val_datasets, test_l_datasets, test_m_datasets, test_s_datasets


def load_dataset_from_path(syn_path, gt_path, mask_path, sample_shape, norm_by_image, dataset_workers, num_images, full_image=False):

    datasets = []
    print(len(os.listdir(syn_path)))
    start = time()

    # don't want to trigger if statement for every image
    if full_image:
        Dataset = FullImageDataset
    elif norm_by_image:
        Dataset = ImageNormDataset
    else:
        Dataset = ItemNormDataset

    if dataset_workers != 1:
        print("parallelizing with {} workers".format(dataset_workers))
        with Pool(dataset_workers) as p:
            args = [
                (
                    os.path.join(syn_path, syn),
                    os.path.join(gt_path, gt),
                    os.path.join(mask_path, mask),
                    sample_shape,
                )
                for syn, gt, mask in zip(
                    os.listdir(syn_path)[0:num_images],
                    os.listdir(gt_path)[0:num_images],
                    os.listdir(mask_path)[0:num_images],
                )
            ]
            datasets = p.starmap(Dataset, args)
    else:
        for syn, gt, mask in zip(
            os.listdir(syn_path)[0:num_images],
            os.listdir(gt_path)[0:num_images],
            os.listdir(mask_path)[0:num_images],
        ):
            # start = time()
            datasets.append(
                Dataset(
                    os.path.join(syn_path, syn),
                    os.path.join(gt_path, gt),
                    os.path.join(mask_path, mask),
                    sample_shape,
                )
            )
    end = time()
    print(f"Loading {len(datasets)} took {end - start} seconds")

    return datasets




# # Get a list of all the .jpg files in the directory
# files = [f for f in os.listdir(path) if f.endswith(".jpg")]

# # Initialize empty numpy arrays to hold the images
# num_images = len(files)
# image_size = (256, 256)
# rgb_channels = 3
# images = np.zeros(
#     (num_images, image_size[0], image_size[1], rgb_channels), dtype=np.uint8
# )

# # Loop through the files and load each image into the numpy array
# for i, file in enumerate(files):
#     image = Image.open(os.path.join(path, file))
#     images[i] = np.array(image)

# # Print the shape of the numpy array to verify it has the correct dimensions
# print(images.shape)

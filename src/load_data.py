import os
import numpy as np
from PIL import Image
from dataset import *
from time import time
from multiprocessing import Pool


def load_data(data_configs):
    sample_shape = data_configs["sample_shape"]
    norm_by_image = data_configs["norm_by_image"]
    dataset_workers = data_configs["dataset_workers"]
    num_images = data_configs["num_images"]

    path = data_configs["data_path"]
    train_path = os.path.join(path, "training/")
    train_syn_path = os.path.join(train_path, "synthetic/")
    train_gt_path = os.path.join(train_path, "gt/")

    training_datasets = []
    print(len(os.listdir(train_syn_path)))
    start = time()

    # don't want to trigger if statement for every image
    if norm_by_image:
        Dataset = ImageNormDataset
    else:
        Dataset = ItemNormDataset

    if dataset_workers != 1:
        print("parallelizing with {} workers".format(dataset_workers))
        with Pool(dataset_workers) as p:
            args = [
                (
                    os.path.join(train_syn_path, syn),
                    os.path.join(train_gt_path, gt),
                    sample_shape,
                )
                for syn, gt in zip(
                    os.listdir(train_syn_path)[0:num_images],
                    os.listdir(train_gt_path)[0:num_images],
                )
            ]
            training_datasets = p.starmap(Dataset, args)
    else:
        for syn, gt in zip(
            os.listdir(train_syn_path)[0:num_images],
            os.listdir(train_gt_path)[0:num_images],
        ):
            # start = time()
            training_datasets.append(
                Dataset(
                    os.path.join(train_syn_path, syn),
                    os.path.join(train_gt_path, gt),
                    sample_shape,
                )
            )
    end = time()
    print(f"Loading {len(training_datasets)} took {end - start} seconds")
    return training_datasets


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

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import cv2
import torchvision.transforms as transforms

# assume that all images are in RBG format
norm_transform = transforms.Compose(
    # [transforms.PILToTensor(), transforms.Normalize([0, 0, 0], [1, 1, 1])]
    [
        # normalize to [0, 1]
        transforms.ToTensor(),
        # normalize to [-1, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

PIL_transform = transforms.Compose(
    [
        transforms.PILToTensor(),
    ]
)


# # this dataset will norm the entire image at init time
# class ImageNormDataset(Dataset):
#     def __init__(self, syn_img_path, gt_img_path, data_shape):
#         syn_img = Image.open(syn_img_path)
#         self.syn_img = norm_transform(syn_img)

#         gt_img = Image.open(gt_img_path)
#         self.gt_img = norm_transform(gt_img)

#         self.data_shape = data_shape

#     def __len__(self):
#         return (self.syn_img.shape[1] - self.data_shape) * (
#             self.syn_img.shape[2] - self.data_shape
#         )

#     def __getitem__(self, index):
#         # each item is a 3x64x64 image by sliding a mask across the image
#         return (
#             self.syn_img[
#                 :,
#                 index % self.data_shape : (index % self.data_shape) + self.data_shape,
#                 index // self.data_shape : (index // self.data_shape) + self.data_shape,
#             ],
#             self.gt_img[
#                 :,
#                 index % self.data_shape : (index % self.data_shape) + self.data_shape,
#                 index // self.data_shape : (index // self.data_shape) + self.data_shape,
#             ],
#         )


# this dataset will norm the entire image at init time
class ImageNormDataset(Dataset):
    def __init__(self, syn_img_path, gt_img_path, snow_mask_path, data_shape):
        syn_img = Image.open(syn_img_path)
        self.syn_img = transforms.ToTensor()(syn_img).to(torch.float16)
        # self.syn_img = (self.syn_img - torch.min(self.syn_img)) * (
        #     1.0 / (torch.max(self.syn_img) - torch.min(self.syn_img))
        # )
        self.syn_img = (self.syn_img - torch.min(self.syn_img)) / (
            torch.max(self.syn_img) - torch.min(self.syn_img)
        )

        self.syn_img = self.syn_img * 2.0 - 1.0

        gt_img = Image.open(gt_img_path)
        self.gt_img = transforms.ToTensor()(gt_img)
        self.gt_img = (self.gt_img - torch.min(self.gt_img)) / (
            torch.max(self.gt_img) - torch.min(self.gt_img)
        )
        self.gt_img = self.gt_img * 2.0 - 1.0

        snow_mask = Image.open(snow_mask_path)
        self.snow_mask = transforms.ToTensor()(snow_mask)
        self.snow_mask = (self.snow_mask - torch.min(self.syn_img)) / (
            torch.max(self.syn_img) - torch.min(self.syn_img)
        )

        self.data_shape = data_shape
        self.img_shape = self.syn_img.shape

    def __len__(self):
        return (self.syn_img.shape[1] - self.data_shape) * (
            self.syn_img.shape[2] - self.data_shape
        )

    def __getitem__(self, index):
        # each item is a 3x64x64 image by sliding a mask across the image
        # print(index % self.data_shape, (index % self.data_shape) + self.data_shape)
        # print(index // self.data_shape, (index // self.data_shape) + self.data_shape)
        # index = index % ((self.img_shape[1] - self.data_shape) * (self.img_shape[2] - self.data_shape))
        # print('image shape', self.img_shape)
        # print('syn shape', self.syn_img.shape)
        # print('data shape', self.data_shape)
        # print(index)
        # print(self.__len__())
        # print(index // (self.img_shape[2]-self.data_shape), (index // (self.img_shape[2]-self.data_shape)) + self.data_shape)
        # print(index % (self.img_shape[2]-self.data_shape), (index % (self.img_shape[2]-self.data_shape)) + self.data_shape)
        
        return (
            self.syn_img[
                :,
                index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
                index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
            ],
            self.gt_img[
                :,
                index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
                index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
            ],
            self.snow_mask[
                :,
                index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
                index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
            ],
            )


# # this dataset will norm each 64x64 patch of the image at get_item time
# class ItemNormDataset(Dataset):
#     def __init__(self, syn_img_path, gt_img_path, data_shape):
#         # self.syn_img = PIL_transform(Image.open(syn_img_path)).float()

#         # self.gt_img = PIL_transform(Image.open(gt_img_path)).float()

#         # self.data_shape = data_shape

#         self.syn_img = cv2.cvtColor(cv2.imread(syn_img_path), cv2.COLOR_BGR2RGB)
#         self.gt_img = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2RGB)
#         self.data_shape = data_shape

#     def __len__(self):
#         return (self.syn_img.shape[0] - self.data_shape) * (
#             self.syn_img.shape[1] - self.data_shape
#         )

#     def __getitem__(self, index):
#         # each item is a 3x64x64 image by sliding a mask across the image
#         # return (
#         #     norm_transform(
#         #         self.syn_img[
#         #             index % self.data_shape : (index % self.data_shape)
#         #             + self.data_shape,
#         #             index // self.data_shape : (index // self.data_shape)
#         #             + self.data_shape,
#         #             :,
#         #         ]
#         #     ),
#         #     norm_transform(
#         #         self.gt_img[
#         #             index % self.data_shape : (index % self.data_shape)
#         #             + self.data_shape,
#         #             index // self.data_shape : (index // self.data_shape)
#         #             + self.data_shape,
#         #             :,
#         #         ]
#         #     ),
#         # )

#         syn_item = self.syn_img[
#             index % self.data_shape : (index % self.data_shape) + self.data_shape,
#             index // self.data_shape : (index // self.data_shape) + self.data_shape,
#             :,
#         ]
#         gt_item = self.gt_img[
#             index % self.data_shape : (index % self.data_shape) + self.data_shape,
#             index // self.data_shape : (index // self.data_shape) + self.data_shape,
#             :,
#         ]

#         syn_item = cv2.normalize(
#             syn_item,
#             None,
#             alpha=-1,
#             beta=1,
#             norm_type=cv2.NORM_MINMAX,
#             dtype=cv2.CV_32F,
#         )
#         gt_item = cv2.normalize(
#             gt_item, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
#         )

#         syn_item = torch.from_numpy(syn_item).float().permute(2, 0, 1)
#         gt_item = torch.from_numpy(gt_item).float().permute(2, 0, 1)

#         return syn_item, gt_item
#         # return (syn_item - torch.mean(syn_item)) * (
#         #     2.0 / (torch.max(syn_item) - torch.min(syn_item))
#         # ), (gt_item - torch.mean(gt_item)) * (
#         #     2.0 / (torch.max(gt_item) - torch.min(gt_item))
#         # )
#         # return (syn_item - torch.mean(syn_item)) * (1.0 / (torch.std(syn_item))), (
#         #     gt_item - torch.mean(gt_item)
#         # ) * (1.0 / (torch.std(gt_item)))


# this dataset will norm each 64x64 patch of the image at get_item time
class ItemNormDataset(Dataset):
    def __init__(self, syn_img_path, gt_img_path, snow_mask_path, data_shape):
        syn_img = Image.open(syn_img_path)
        self.syn_img = transforms.ToTensor()(syn_img).to(torch.float16)
        gt_img = Image.open(gt_img_path)
        self.gt_img = transforms.ToTensor()(gt_img).to(torch.float16)
        snow_mask = Image.open(snow_mask_path)
        self.snow_mask = transforms.ToTensor()(snow_mask)
        self.data_shape = data_shape

    def __len__(self):
        return (self.syn_img.shape[1] - self.data_shape) * (
            self.syn_img.shape[2] - self.data_shape
        )

    def __getitem__(self, index):
        # each item is a 3x64x64 image by sliding a mask across the image

        syn_item = self.syn_img[
            :,
            index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
            index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
        ]
        gt_item = self.gt_img[
            :,
            index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
            index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
        ]
        snow_mask_item = self.snow_mask[
            :,
            index // (self.img_shape[2]-self.data_shape) : (index // (self.img_shape[2]-self.data_shape)) + self.data_shape,
            index % (self.img_shape[2]-self.data_shape) : (index % (self.img_shape[2]-self.data_shape)) + self.data_shape,
        ]

        syn_item = (syn_item - torch.min(syn_item)) / (
            torch.max(syn_item) - torch.min(syn_item)
        )
        gt_item = (gt_item - torch.min(gt_item)) / (
            torch.max(gt_item) - torch.min(gt_item)
        )
        snow_mask_item = (snow_mask_item - torch.min(snow_mask_item)) / (
            torch.max(snow_mask_item) - torch.min(snow_mask_item)
        )
        syn_item = syn_item * 2.0 - 1.0
        gt_item = gt_item * 2.0 - 1.0
        snow_mask_item = snow_mask_item * 2.0 - 1.0

        return syn_item, gt_item, snow_mask_item
    
class FullImageDataset(Dataset):
    def __init__(self, syn_img_path, gt_img_path, snow_mask_path, sample_shape):
        # sample shape is not used and only to keep API the same
        syn_img = Image.open(syn_img_path)
        self.syn_img = transforms.ToTensor()(syn_img).to(torch.float16)
        gt_img = Image.open(gt_img_path)
        self.gt_img = transforms.ToTensor()(gt_img).to(torch.float16)
        snow_mask = Image.open(snow_mask_path)
        self.snow_mask = transforms.ToTensor()(snow_mask)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # each item is a 3x64x64 image by sliding a mask across the image

        syn_item = self.syn_img
        gt_item = self.gt_img

        syn_item = (syn_item - torch.min(syn_item)) / (
            torch.max(syn_item) - torch.min(syn_item)
        )
        gt_item = (gt_item - torch.min(gt_item)) / (
            torch.max(gt_item) - torch.min(gt_item)
        )
        snow_mask_item = (snow_mask_item - torch.min(syn_item)) / (
            torch.max(syn_item) - torch.min(syn_item)
        )
        syn_item = syn_item * 2.0 - 1.0
        gt_item = gt_item * 2.0 - 1.0
        snow_mask_item = snow_mask_item * 2.0 - 1.0

        return syn_item, gt_item, snow_mask_item

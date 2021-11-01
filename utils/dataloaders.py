# -*- coding: utf-8 -*-
# @File : dataloader.py
# @Author: Runist
# @Time : 2021/10/30 8:36
# @Software: PyCharm
# @Brief: Data input function

import os
import torch.utils.data as data
from PIL import Image
from utils import transforms as tr
from utils.helpers import path_sort
import glob
import cv2 as cv
import random
import numpy as np
import math
import re
import torch


def get_image_path(data_dir):
    """
    Dataset structure
    /
    ├───Train
    │     ├─── A(image_l)
    │     ├─── B(image_r)
    │     └─── label
    ├───Val
    │     ├─── A(image_l)
    │     ├─── B(image_r)
    │     └─── label
    Args:
        data_dir: The root of data directory.

    Returns: data_path

    """
    data_path = []

    image_files_l = glob.glob("{}/A/*.jpg".format(data_dir))
    image_files_r = glob.glob("{}/B/*.jpg".format(data_dir))
    label_files = glob.glob("{}/OUT/*.jpg".format(data_dir))

    image_files_l = path_sort(image_files_l)
    image_files_r = path_sort(image_files_r)
    label_files = path_sort(label_files)

    for image_l, image_r, label in zip(image_files_l, image_files_r, label_files):
        data_path.append({"image_l": image_l, "image_r": image_r, "label": label})

    return data_path


def cdd_loader(image_l_path, image_r_path, label_path, aug):
    """
    Image loader.
    Args:
        image_l_path: Image path of A
        image_r_path: Image path of B
        label_path: Image path of label
        aug: Whether use data augmentation

    Returns: image_l, image_r, label

    """
    image_l = Image.open(image_l_path)
    image_r = Image.open(image_r_path)
    label = Image.open(label_path)

    sample = {'image_l': image_l, 'image_r': image_r, 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image_l'], sample['image_r'], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, data_path, aug=False):
        self.data_path = data_path
        self.aug = aug

    def __getitem__(self, index):
        image_l_path = self.data_path[index]['image_l']
        image_r_path = self.data_path[index]['image_r']
        label_path = self.data_path[index]['label']

        return cdd_loader(image_l_path, image_r_path, label_path, self.aug)

    def __len__(self):
        return len(self.data_path)


def get_data_loader(data_dir, batch_size, aug=False):
    """
    get torch  DataLoader
    Args:
        data_dir: The root of data directory.
        batch_size: The number image of one batch
        aug: Whether use data augmentation

    Returns: torch DataLoader

    """
    data_path = get_image_path(data_dir)
    dataset = CDDloader(data_path, aug=aug)

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=aug,
                                         num_workers=8)

    return loader

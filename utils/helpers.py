# -*- coding: utf-8 -*-
# @File : helpers.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: Some util function

import torch.utils.data
import torch.nn as nn
import numpy as np
import random
from utils.losses import jaccard_loss, dice_loss, hybrid_loss
from models.Models import SNUNet_ECAM
import os
import shutil


def seed_torch(seed):
    """
    Set random seed.
    Args:
        seed: number

    Returns: None

    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def remove_dir_and_create_dir(dir_name):
    """
    Remove directory and create directory.
    Args:
        dir_name: Name of the directory

    Returns: None

    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    ind = np.arange(256, dtype=np.uint8)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """

    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def path_sort(path):
    sort_dict = {}
    sort_path = []
    for p in path:
        file_name = os.path.split(p)[1]
        sort_dict.update({p: file_name})

    sort_dict = sorted(sort_dict.items(), key=lambda x: x[1])
    for k in sort_dict:
        sort_path.append(k[0])

    return sort_path


def get_criterion(args):
    """
    Get loss function.
    Args:
        args: External pass parameter object

    Returns: reference of loss function

    """
    if args.loss_function == 'hybrid':
        criterion = hybrid_loss
    if args.loss_function == 'bce':
        criterion = nn.CrossEntropyLoss()
    if args.loss_function == 'dice':
        criterion = dice_loss
    if args.loss_function == 'jaccard':
        criterion = jaccard_loss

    return criterion


def load_model(args, device):
    """
    Load model
    Args:
        args: External pass parameter object
        device: torch device object

    Returns: model

    """
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = SNUNet_ECAM(args.n_channel, in_ch=3, out_ch=args.num_classes, method=args.up_method)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    return model

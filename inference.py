# -*- coding: utf-8 -*-
# @File : inference.py
# @Author: Runist
# @Time : 2021/10/30 8:36
# @Software: PyCharm
# @Brief: Inference script

from utils.parser import parser
from utils.helpers import remove_dir_and_create_dir, label_to_color_image
from utils.transforms import SingleImageResizeImage, SingleImageNormalize, SingleImageToTensor, SingleImageToNumpy
from utils.dataloaders import get_image_path
import torchvision.transforms as transforms
import torch
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import re
import random
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    parser.add_argument("--inference_h_size", type=int, default=256, help='The height of model input.')
    parser.add_argument("--inference_w_size", type=int, default=256, help='The width of model input.')
    parser.add_argument("--inference_padding_h_size", type=int, default=0, help='The padding height of model input.')
    parser.add_argument("--inference_padding_w_size", type=int, default=0, help='The padding width of model input.')
    args = parser.parse_args()
    just_label = True       # It will output single channel label image if set True, or color image of prediction

    h = args.inference_h_size
    w = args.inference_w_size
    pad_h = args.inference_padding_h_size
    pad_w = args.inference_padding_w_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(777)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    remove_dir_and_create_dir(args.pred_dir)

    image_transforms = transforms.Compose([
        SingleImageResizeImage(mode="RGB", size=(h, w), padding=(pad_h, pad_w)),
        SingleImageNormalize(),
        SingleImageToTensor()
    ])

    label_transforms = transforms.Compose([
        SingleImageResizeImage(mode="L", size=(h, w), padding=(pad_h, pad_w)),
        SingleImageToNumpy()
    ])

    data_path = get_image_path(args.dataset_val_dir)

    path = './weights/tutorial/epoch=399_miou=0.9343_recall=0.9601_precision=0.9708.pt'  # the path of the model
    model = torch.load(path)

    model.eval()
    with torch.no_grad():
        tbar = tqdm(range(len(data_path)))
        for i in tbar:
            image_l_path = data_path[i]["image_l"]
            image_r_path = data_path[i]['image_r']
            label_path = data_path[i]['label']

            image_id = os.path.split(image_r_path)[1][:-4]

            image_l = Image.open(image_l_path)
            image_r = Image.open(image_r_path)
            label = Image.open(label_path)

            image_l = image_transforms(image_l)
            image_r = image_transforms(image_r)
            label = label_transforms(label)
            label = label[pad_h//2:h+pad_h//2, pad_w//2:w+pad_w//2]

            image_l = image_l.to(dev)
            image_r = image_r.to(dev)

            cd_pred = model(image_l, image_r)

            cd_pred = torch.argmax(cd_pred, 1)
            cd_pred = cd_pred.cpu().numpy()
            cd_pred = cd_pred.squeeze()
            cd_pred = cd_pred[pad_h//2:h+pad_h//2, pad_w//2:w+pad_w//2]

            if just_label:
                file_path = '{}/{}.png'.format(args.pred_dir, image_id)
                cv.imwrite(file_path, cd_pred)
            else:
                image_l = image_l.data.cpu().numpy()
                image_l = np.squeeze(image_l, axis=0)
                image_l = np.transpose(image_l, (1, 2, 0)).astype(np.float32) * 255

                image_r = image_r.data.cpu().numpy()
                image_r = np.squeeze(image_r, axis=0)
                image_r = np.transpose(image_r, (1, 2, 0)).astype(np.float32) * 255

                image_l = image_l[pad_h//2:h+pad_h//2, pad_w//2:w+pad_w//2]
                image_r = image_r[pad_h//2:h+pad_h//2, pad_w//2:w+pad_w//2]

                label = label_to_color_image(label)
                result = label_to_color_image(cd_pred)

                image_l = cv.cvtColor(image_l, cv.COLOR_RGB2BGR)
                image_r = cv.cvtColor(image_r, cv.COLOR_RGB2BGR)
                image = np.vstack((np.hstack((image_l, image_r)), np.hstack((label, result))))

                file_path = '{}/{}.png'.format(args.pred_dir, image_id)
                cv.imwrite(file_path, image)

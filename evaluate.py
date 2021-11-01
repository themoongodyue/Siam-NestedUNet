# -*- coding: utf-8 -*-
# @File : evaluate.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: Evaluate scripts

import cv2 as cv
import glob
import numpy as np
from utils.metrics import generate_matrix
from utils.parser import args
from utils.helpers import path_sort
import os


if __name__ == '__main__':

    labels_path = glob.glob("{}/OUT/*.jpg".format(args.dataset_val_dir))
    preds_path = glob.glob("{}/*.png".format(args.pred_dir))

    labels_path = path_sort(labels_path)[:91]
    preds_path = path_sort(preds_path)

    total_cm = np.zeros((args.num_classes,)*2)

    for label_path, pred_path in zip(labels_path, preds_path):
        label_id = os.path.splitext(os.path.split(label_path)[1])[0]
        pred_id = os.path.splitext(os.path.split(pred_path)[1])[0]
        if label_id != pred_id:
            raise Exception("Picture group order is mismatch.")

        label = cv.imread(label_path, cv.IMREAD_UNCHANGED)
        pred = cv.imread(pred_path, cv.IMREAD_UNCHANGED)

        # label = np.array(label / 255, dtype=np.int64)     # If label need remap, please uncomment it.

        h, w = pred.shape[:2]
        label = cv.resize(label, (w, h), interpolation=cv.INTER_NEAREST)

        pred = np.expand_dims(pred, axis=-1)
        label = np.expand_dims(label, axis=-1)

        cm = generate_matrix(label, pred, args.num_classes)
        total_cm += cm

    print(total_cm)

    sum_over_col = np.sum(total_cm, axis=0).astype(float)
    sum_over_row = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    sum_over_row = np.where(
        sum_over_row > 0,
        sum_over_row,
        np.ones_like(sum_over_row))

    sum_over_col = np.where(
        sum_over_col > 0,
        sum_over_col,
        np.ones_like(sum_over_col))

    ious = cm_diag / denominator
    recalls = cm_diag / sum_over_row
    precisions = cm_diag / sum_over_col

    print('Recall for each class:')
    for i, recall in enumerate(recalls):
        print('    class {}: {:f}'.format(i, recall))

    print('Precision for each class:')
    for i, precision in enumerate(precisions):
        print('    class {}: {:f}'.format(i, precision))

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        print('    class {}: {:f}'.format(i, iou))

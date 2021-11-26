# -*- coding: utf-8 -*-
# @File : parser.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: Configure script

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0,1,2,3', help='Select gpu device.')

parser.add_argument('--h_size', type=int, default=256, help='The height of model input.')
parser.add_argument('--w_size', type=int, default=256, help='The width of model input.')
parser.add_argument("--padding_h_size", type=int, default=0, help='The padding height of model input.')
parser.add_argument("--padding_w_size", type=int, default=0, help='The padding height of model input.')
parser.add_argument('--n_channel', type=int, default=16, help='The hidden channel of model.')
parser.add_argument('--up_method', type=str, default="bilinear", choices=["bilinear", "transpose"],
                    help='The method of up sample.')
parser.add_argument('--num_classes', type=int, default=1+1, help='The number of class.')
parser.add_argument('--epochs', type=int, default=400, help='The number of training epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Number of examples per batch.')
parser.add_argument('--learn_rate_init', type=float, default=4e-4,
                    help='Initial value of cosine annealing learning rate.')
parser.add_argument('--learn_rate_end', type=float, default=1e-6,
                    help='End value of cosine annealing learning rate.')
parser.add_argument('--loss_function', type=str, default='hybrid',
                    help='The loss function for Siam-NestedUnet.')

parser.add_argument('--dataset_train_dir', type=str,
                    default="/home/ubuntu/data/subset/train",
                    help='The directory containing the train data.')
parser.add_argument('--dataset_val_dir', type=str,
                    default="/home/ubuntu/data/subset/val",
                    help='The directory containing the val data.')
parser.add_argument('--weights_dir', type=str, default="./weights/tutorial_40",
                    help='The directory of saving weights.')
parser.add_argument('--log_dir', type=str, default="./weights/tutorial_40/log",
                    help='The directory of saving weights.')
parser.add_argument('--pred_dir', type=str, default='/home/ubuntu/code/Siam-NestedUNet/output',
                    help='The directory of the predict image.')

args = parser.parse_args()

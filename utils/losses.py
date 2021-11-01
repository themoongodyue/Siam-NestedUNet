# -*- coding: utf-8 -*-
# @File : losses.py
# @Author: Runist
# @Time : 2021/9/26 10:56
# @Software: PyCharm
# @Brief: Model loss function

from utils.parser import args
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=0., size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, y_pred, y_true):

        logpt = F.log_softmax(y_pred)

        y_true = torch.argmax(y_true, dim=-1)
        y_true = torch.unsqueeze(y_true, dim=-1)
        logpt = logpt.gather(1, y_true)
        pt = Variable(logpt.data.exp())
        # y_true的第1维是通道维度，也即预测是哪一类的标签。gather取axis=1的索引，相当于y_true的值替换了列索引
        # 为了取到具体的值，还需要行索引，根据gather的算法，y_true中的每个值本身的位置，提供行索引。如果dim=0
        # 则本身的值替换行索引，本身的值所在的位置，提供列索引

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(y_pred, y_true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        y_true: a tensor of shape [B*H*W, C].
        y_pred: a tensor of shape [B*H*W, C]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    prob = F.softmax(y_pred, dim=1)
    y_true = y_true.type(y_pred.type())
    # 分开前景，背景求和是为了减轻类别不平衡的情况
    intersection = torch.sum(prob * y_true, dim=0)
    cardinality = torch.sum(prob + y_true, dim=0)
    loss = (2. * intersection / (cardinality + eps)).mean()

    return 1 - loss


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return 1 - jacc_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-7, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.eps = eps

    def forward(self, logits, true):
        """Computes the Tversky loss [1].
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (self.alpha * fps) + (self.beta * fns)
        tversky_loss = (num / (denom + self.eps)).mean()
        return 1 - tversky_loss


def hybrid_loss(y_pred, y_true):
    """Calculating the loss"""

    num_classes = y_pred.shape[1]

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(alpha=[1] * num_classes, gamma=2.0)
    focal = FocalLoss(gamma=0, alpha=None)

    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)

    y_pred = y_pred.reshape(-1, num_classes)
    y_true = y_true.reshape(-1,)

    valid_index = y_true < 255

    y_pred = y_pred[valid_index]
    y_true = y_true[valid_index]

    # one hot
    y_true = (torch.arange(num_classes).cuda() == torch.unsqueeze(y_true, dim=1)).long()

    bce = focal(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    loss = bce + dice

    return loss

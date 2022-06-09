import torch
import numpy as np
import pandas as pd


def single_hist(pred, mask, n_class=11):
    valid_filter = (mask >= 0) & (mask < n_class)
    hist = np.bincount(
        n_class * mask[valid_filter].astype(int) + pred[valid_filter],
        minlength=n_class**2
    ).reshape(n_class, n_class)
    return hist


def add_hist(hist, preds, masks, n_class=11):
    for pred, mask in zip(preds, masks):
        hist += single_hist(pred.flatten(), mask.flatten(), n_class)

    return hist


def segmentation_metrics(hist):
    """
    Parameters:
        preds (tensor): predicted masks in shape (B, H, W)
        masks (tensor): ground truth masks in shape (B, H, W)
        n_class (int): number of classes (11)

    Returns:
        acc (float): total accuracy
        acc_cls (float): mean accuracy for classes
        iou: (List[float]): IoU per class
        mean_iou (float): mean IoU for classes
    """

    acc = np.diag(hist).sum() / hist.sum()

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1) # TP / (TP + FN) = Recall
    acc_cls = np.nanmean(acc_cls)

    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)) # TP / TP + FN + FP
    miou = np.nanmean(iou)

    return acc, acc_cls, iou, miou


import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def forward(self, pred, target):
        '''
        pred: (B, C, H, W)
        target: (B, H, W)

        Binary Dice Loss
        '''

        pred = pred.squeeze() # (B, H, W)
        pred = torch.sigmoid(pred) # (B, H, W)

        intersection = torch.sum(pred * target, (1, 2))
        cardinality = torch.sum(pred + target, (1, 2))

        dice_score = (2.0 * intersection) / (cardinality + self.eps)
        return torch.mean(1.0 - dice_score)


def binary_dice_loss(pred, target):
    return BinaryDiceLoss()(pred, target)


def bce_with_logits_loss(pred, target):
    pred = pred.squeeze()
    return nn.BCEWithLogitsLoss()(pred, target)


def combined_loss(pred, target, dice_coef=0.25, bce_coef=0.75):
    '''
    pred: (B, 1, H, W)
    target: (B, H, W)
    '''
    dice = binary_dice_loss(pred, target)
    bce = bce_with_logits_loss(pred, target)
    combined = dice_coef * dice + bce_coef * bce
    return combined, dice, bce

if __name__ == '__main__':
    pred = torch.randn(8, 1, 224, 224)
    target = torch.ones(8, 224, 224)
    loss = combined_loss(pred, target, num_classes=2)
    print(loss)

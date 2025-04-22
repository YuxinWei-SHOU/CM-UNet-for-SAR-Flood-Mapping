import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# only bce loss

class BCEOnlyLoss(nn.Module):
    def __init__(self):
        super(BCEOnlyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        # Compute binary cross-entropy loss
        bce_loss = self.bce_loss(y_pred, y_true)
        return bce_loss

def FCCDN_loss_without_seg(scores, labels):
    scores = scores.squeeze(1) if len(scores.shape) > 3 else scores
    labels = labels.squeeze(1) if len(labels.shape) > 3 else labels

    # Use BCEOnlyLoss to compute the loss
    criterion = BCEOnlyLoss()

    # Compute the loss
    bce_loss = criterion(scores, labels)

    # Return the loss value (single loss term)
    return bce_loss


# diceloss+bceloss
class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.00001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)

        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred.to(dtype=torch.float32), y_true)


class dice_bce_loss(nn.Module):

    def __init__(self, beta=1, smooth=1e-5):
        super(dice_bce_loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = dice_loss()
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce_loss(y_pred, y_true)

        # Sigmoid activation for Dice Loss calculation
        y_pred = torch.sigmoid(y_pred)

        # Adjust dimension indices based on shape
        if y_pred.dim() == 4:  # Batch size, Channels, Height, Width
            dims = (0, 2, 3)
        elif y_pred.dim() == 3:  # Batch size, Height, Width
            dims = (0, 1, 2)
        else:
            raise ValueError(f"Unexpected dimension for y_pred: {y_pred.dim()}")

        # Dice Loss
        tp = torch.sum(y_true * y_pred, dim=dims)
        fp = torch.sum(y_pred, dim=dims) - tp
        fn = torch.sum(y_true, dim=dims) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) / (
                    (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        dice_loss = 1 - torch.mean(score)

        return dice_loss + bce_loss, dice_loss, bce_loss

def FCCDN_loss_without_seg_combine(scores, labels):
    scores = scores.squeeze(1) if len(scores.shape) > 3 else scores
    labels = labels.squeeze(1) if len(labels.shape) > 3 else labels

    criterion = dice_bce_loss()

    # Compute loss
    loss_change, diceloss, bceloss = criterion(scores, labels)

    return loss_change, diceloss, bceloss

class bce_focal_sar_loss(nn.Module):
    def __init__(self, gamma=2, smooth=1e-5):
        super(bce_focal_sar_loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce_loss(y_pred, y_true)

        # Sigmoid activation for Focal Loss calculation
        y_pred_sigmoid = torch.sigmoid(y_pred)

        # Adjust dimension indices based on shape
        if y_pred.dim() == 4:  # Batch size, Channels, Height, Width
            dims = (0, 2, 3)
        elif y_pred.dim() == 3:  # Batch size, Height, Width
            dims = (0, 1, 2)
        else:
            raise ValueError(f"Unexpected dimension for y_pred: {y_pred.dim()}")

        # Focal Loss
        focal_loss = -torch.mean(
            (1 - y_pred_sigmoid) ** self.gamma * y_true * torch.log(y_pred_sigmoid + self.smooth)
            + y_pred_sigmoid ** self.gamma * (1 - y_true) * torch.log(1 - y_pred_sigmoid + self.smooth)
        )

        # SAR Loss
        sar_feature_loss = (
            torch.mean(torch.abs(y_pred_sigmoid[:, 1:, :] - y_pred_sigmoid[:, :-1, :]))
            + torch.mean(torch.abs(y_pred_sigmoid[:, :, 1:] - y_pred_sigmoid[:, :, :-1]))
        )

        total_loss = focal_loss + bce_loss + 0.3 * sar_feature_loss

        return total_loss, focal_loss, bce_loss

def FCCDN_loss_without_seg_combine2(scores, labels):
    scores = scores.squeeze(1) if len(scores.shape) > 3 else scores
    labels = labels.squeeze(1) if len(labels.shape) > 3 else labels

    criterion = bce_focal_sar_loss()

    # Compute loss
    loss_change, focalloss, bceloss = criterion(scores, labels)

    return loss_change, focalloss, bceloss

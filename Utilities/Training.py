import torch
from torch import nn
import math

def get_lr_lambda(number_warmup_batches):
    def warmup(current_step: int):
        if current_step < number_warmup_batches:
            # print(current_step / number_warmup_batches ** 1.5)
            return current_step / number_warmup_batches ** 1.5
        else:
            # print(1/math.sqrt(current_step))
            return 1/math.sqrt(current_step)

    return warmup


class focal_loss(nn.Module):
    def __init__(self, weights, gamma=2, label_smoothing=0):
        super(focal_loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)
        self.weights = weights
        self.gamma = gamma

    def forward(self, pred, targets):
        ce = self.ce_loss(pred, targets)
        pt = torch.exp(-ce)

        loss_sum = torch.sum(((1-pt) ** self.gamma) * ce * self.weights[targets])
        norm_factor = torch.sum(self.weights[targets])
        return loss_sum/norm_factor


"""
Noise Detector
"""


class binary_focal_loss(nn.Module):

    def __init__(self, _alpha, _gamma):
        super(binary_focal_loss, self).__init__()
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = _alpha
        self.gamma = _gamma

    def forward(self, pred, targets):
        bce = self.BCE_loss(pred, targets)
        prob_correct = torch.exp(-bce)
        loss_unweighted = (1.0 - prob_correct)**self.gamma * bce
        loss_weighted = torch.where(targets == 1,
                           loss_unweighted * self.alpha,
                           loss_unweighted * (1-self.alpha))
        return torch.mean(loss_weighted)
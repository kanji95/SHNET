import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Dice_loss:
    def __call__(self, inputs, targets):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        assert -1 not in denominator
        loss = 1 - (numerator + 1) / (denominator + 1)
        assert not torch.any(torch.isnan(loss))
        return loss.sum()


class Loss:
    def __init__(self, args):

        self.args = args

        self.l1_loss = nn.SmoothL1Loss()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = Dice_loss()

    def __call__(self, inputs, targets):

        # import pdb; pdb.set_trace()
        loss = 0
        if "dice" in self.args.loss:
            loss += self.dice_loss(inputs, targets)
        if "l1" in self.args.loss:
            loss += self.l1_loss(inputs, targets).sum(dim=1).mean()
        if "bce" in self.args.loss:
            # c_indx = gt_uncty == 1
            # loss_unc = self.bce_loss(uncty, gt_uncty.float())
            # loss_c = self.bce_loss(inputs[c_indx], targets[c_indx])
            # loss += (1 - self.args.unc_prob)*loss_c + (self.args.unc_prob)*loss_unc
            loss += self.bce_loss(inputs, targets)
            # inputs = torch.clamp(inputs, 1e-7, 1 - 1e-7)
            # loss += (-0.7*targets*torch.log(inputs) -0.3*(1 - targets)*torch.log(1 - inputs)).mean()
        if loss == 0:
            raise Exception(f"{self.args.loss} loss not implemented!")
        return loss

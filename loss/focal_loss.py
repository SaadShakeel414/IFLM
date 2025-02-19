# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 22:28:26 2022

@author: Saad Shakeel
"""

import torch
import torch.nn as nn


# Support: ['FocalLoss']


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:00:37 2022

@author: Administrator
"""

import torch
import torch.nn as nn
# from .Modules_IFLM import Spatial_Channel_Feature_interaction, CL_Fea_interaction, ProjectorBlock, Local_Att_Conv_Block, SCIFM_R
from .Modules_IFLM import SCFIM, HCAM

# ---------------------------------- LResNet50E-IR network Begin ----------------------------------

class BlockIR(nn.Module):
    def __init__(self, inplanes, planes, stride, dim_match):
        super(BlockIR, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        if dim_match:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class LResNet_IFLM(nn.Module):

    def __init__(self, block, layers, filter_list, is_gray=False):
        self.inplanes = 64
        super(LResNet_IFLM, self).__init__()
        # input is (mini-batch,3 or 1,112,96)
        # use (conv3x3, stride=1, padding=1) instead of (conv7x7, stride=2, padding=3)
        if is_gray:
            self.conv1 = nn.Conv2d(1, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)  # gray
        else:
            self.conv1 = nn.Conv2d(3, filter_list[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filter_list[0])
        self.prelu1 = nn.PReLU(filter_list[0])
        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.SCFIM = SCFIM(512,r=4)
        self.HCAM = HCAM(512,r=4)

        self.sigmoid = nn.Sigmoid()
        self.dense = nn.Conv2d(128, 512, kernel_size=1, padding=0, bias=False)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(512 * 7 * 7),
            nn.Dropout(p=0.4),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),  # fix gamma ???
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(block(inplanes, planes, stride, False))
        for i in range(1, blocks):
            layers.append(block(planes, planes, stride=1, dim_match=True))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.HCAM(x)
        x = self.SCFIM(x)

        # x = self.SFIM(x)

        # x = self.SCFIM(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)


def LResNet50E_IR_IFLM(is_gray=False):
    filter_list = [64, 64, 128, 256, 512]
    layers = [3, 4, 14, 3]
    return LResNet_IFLM(BlockIR, layers, filter_list, is_gray)
# ---------------------------------- LResNet50E-IR network End ----------------------------------

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file ECA.py
# @author: wujiangu
# @date: 2023-12-23 18:28
# @description: ECA module

import math

import torch.nn as nn


# ECA_2D attention
class ECA_2D(nn.Module):
    """ECA_2D attention
    input: (B, C, H, W)
    output: (B, C, H, W)
    """

    def __init__(self, channel: int):
        super(ECA_2D, self).__init__()

        k_size = math.ceil(math.log(channel, 2) / 2 + 0.5)

        k_size = int(k_size)
        if k_size % 2 == 0:
            k_size += 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = self.sigmoid(y)

        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


# ECA 1D
class ECA_1D(nn.Module):
    """
    ECA_1D attention
    input: (B, C, L)
    output: (B, C, L)
    """

    def __init__(self, channel: int):
        super(ECA_1D, self).__init__()

        k_size = math.ceil(math.log(channel, 2) / 2 + 0.5)

        k_size = int(k_size)
        if k_size % 2 == 0:
            k_size += 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.transpose(-1, -2)

        y = self.conv(y)
        y = self.sigmoid(y)

        y = y.transpose(-1, -2)
        return x * y

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file generator.py
# @author: wujiangu
# @date: 2024-01-31 14:46
# @description: generator


import torch.nn as nn
import torch.nn.functional as F


class PhotometricGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, photo_size=(128, 128)):
        """Photometric Generator
        :in_channels: int
        :out_channels: int
        :photo_size: tuple
        """
        super().__init__()

        self.photo_size = photo_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # first 128 -> 32 * 32 * out_channels
        self.fc1 = nn.Linear(in_channels, 32 * 32 * out_channels)

        # 32 * 32 * out_channels -> 64 * 64 * out_channels by DeConv
        self.deconv1 = nn.ConvTranspose2d(
            out_channels,
            out_channels * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        # 64 * 64 * out_channels * 8 -> 128 * 128 * out_channels * 8 by DeConv
        self.deconv2 = nn.ConvTranspose2d(
            out_channels * 8,
            out_channels * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )

        # 128 * 128 * out_channels * 8 -> 128 * 128 * (out_channels * 8 * 10) by DeConv
        self.deconv3 = nn.ConvTranspose2d(
            out_channels * 8,
            out_channels * 8,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )

        self.deconv4 = nn.ConvTranspose2d(
            out_channels * 8,
            out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.out_channels, 32, 32)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)

        x = x.reshape(x.shape[0], -1)

        return x

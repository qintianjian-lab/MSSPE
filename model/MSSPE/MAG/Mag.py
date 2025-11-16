#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file Mag.py
# @author: wujiangu
# @date: 2023-12-23 18:49
# @description: mag module

import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class MagEX(nn.Module):
    """
    MagEX module 4 fc layers
    input: [B, L] batch_size, length
    output: [B, S] batch_size, feature_size
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.fc4(x)


if __name__ == "__main__":
    mag = MagEX(4, 128)
    torchinfo.summary(mag, (2, 4))

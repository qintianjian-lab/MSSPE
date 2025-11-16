#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file fusion.py
# @author: wujiangu
# @date: 2024-01-18 11:30
# @description: fusion
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

sys.path.insert(0, os.path.abspath("./"))

from model.MSSPE.generator import PhotometricGenerator


class Mlp(nn.Module):

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        in_dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, num_heads, attn_drop)
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(
            in_features=in_dim,
            hidden_features=int(in_dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Fusion(nn.Module):
    """
    Fusion: fuse the output of the 5 branches
    input: (batch_size, 5, 128)
    output: (batch_size, 128)
    method: transformer block (MAE)
    """

    def __init__(
        self,
        in_dim=128,
        out_dim=128,
        nhead=8,
        num_layers=3,
        mask_attention=0,
        mlp_ratio=4.0,
        attn_drop=0.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super(Fusion, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.mask_attention = mask_attention

        self.position_embedding = nn.Parameter(
            torch.randn(1, 5 + 1, in_dim), requires_grad=True
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim), requires_grad=True)

        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    in_dim,
                    num_heads=nhead,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder_prediction = nn.ModuleList(
            [
                PhotometricGenerator(in_dim, 5),
                nn.Linear(in_dim, 5),
                PhotometricGenerator(in_dim, 4),
                nn.Linear(in_dim, 4),
                nn.Linear(in_dim, 3584),
            ]
        )

    def forward(self, x, mask=None):
        # x: (batch_size, 5, 128)
        batch_size = x.shape[0]
        x = torch.cat((self.cls_token.expand(batch_size, -1, -1), x), dim=1)
        x = x + self.position_embedding

        # repeat mask: num_heads * (batch_size, 5, 5)
        mask = (
            mask.unsqueeze(0)
            .repeat(self.nhead, 1, 1, 1)
            .reshape(self.nhead * batch_size, 6, 6)
        )
        x = x.transpose(0, 1)
        for blk in self.encoder_blocks:
            x = blk(x, mask)

        decoder_res_list = []

        # decoder prediction
        for i in range(5):
            # x_i = x[i + 1]
            x_i = x[0]
            decoder_res = self.decoder_prediction[i](x_i)
            decoder_res_list.append(decoder_res)

        x = x.transpose(0, 1)

        cls_token = x[:, 0, :]
        return cls_token, decoder_res_list

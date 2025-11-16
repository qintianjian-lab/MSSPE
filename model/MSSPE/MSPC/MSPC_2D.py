#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file mspc.py
# @author: wujiangu
# @date: 2023-12-23 18:31
# @description: MSPC model

import os
import sys
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import DropPath

sys.path.append(os.path.abspath("./"))
from model.MSSPE.MSPC.module.ECA import ECA_2D


class PartialConv_2D(nn.Module):
    """Partial Convolution for photometric"""

    def __init__(
        self,
        channel: int,
        pc_conv_size: int = 3,
        pc_conv_size_add: int = 9,
        n_div=8,
    ):
        super(PartialConv_2D, self).__init__()

        self.n_div = n_div

        self.conv1 = torch.nn.Conv2d(
            in_channels=channel // n_div,
            out_channels=channel // n_div,
            kernel_size=pc_conv_size,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv2 = torch.nn.Conv2d(
            in_channels=channel // n_div,
            out_channels=channel // n_div,
            kernel_size=pc_conv_size + pc_conv_size_add,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv3 = torch.nn.Conv2d(
            in_channels=channel // n_div,
            out_channels=channel // n_div,
            kernel_size=pc_conv_size + pc_conv_size_add + pc_conv_size_add,
            stride=1,
            padding="same",
            bias=False,
        )

    def forward(self, x):
        channel = x.shape[1]
        x_div1 = x[:, : channel // self.n_div]
        x_div2 = x[:, channel // self.n_div : channel // self.n_div * 2]
        x_div3 = x[:, channel // self.n_div * 2 : channel // self.n_div * 3]
        x_remains = x[:, channel // self.n_div * 3 :]

        x_div1 = self.conv1(x_div1)
        x_div2 = self.conv2(x_div2)
        x_div3 = self.conv3(x_div3)

        x = torch.cat([x_div1, x_div2, x_div3, x_remains], dim=1)

        return x


class MLPBlock_2D(nn.Module):
    """MLP Block"""

    def __init__(
        self,
        channel: int,
        mlp_ratio: float,
        drop_path_rate: float,
        act_layer: torch.nn.Module,
        norm_layer: torch.nn.Module,
        pc_conv_size: int = 3,
        pc_conv_size_add: int = 9,
        n_div: int = 8,
        attention: str = "",
    ):
        super(MLPBlock_2D, self).__init__()

        mlp_hidden_channel = int(channel * mlp_ratio)

        mlp_layers: List[torch.nn.Module] = [
            ECA_2D(channel) if attention == "ECA_F" else nn.Identity(),
            nn.Conv2d(channel, mlp_hidden_channel, 1, bias=False),
            norm_layer(mlp_hidden_channel),
            act_layer(inplace=True) if type(act_layer) == nn.ReLU else act_layer(),
            nn.Conv2d(mlp_hidden_channel, channel, 1, bias=False),
            ECA_2D(channel) if attention == "ECA_B" else nn.Identity(),
        ]

        self.mlp = nn.Sequential(*mlp_layers)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        )

        self.spatial_mixing = PartialConv_2D(
            channel,
            pc_conv_size,
            pc_conv_size_add,
            n_div,
        )

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))

        return x


class BasicStage_2D(nn.Module):
    """Basic Stage"""

    def __init__(
        self,
        channel: int,
        mlp_ratio: float,
        drop_path_rate: list[float],
        act_layer: torch.nn.Module,
        norm_layer: torch.nn.Module,
        depth: int,
        pc_conv_size: int = 3,
        pc_conv_size_add: int = 9,
        n_div: int = 8,
        attention: str = "",
    ):
        super(BasicStage_2D, self).__init__()

        blocks: List[torch.nn.Module] = [
            MLPBlock_2D(
                channel=channel,
                mlp_ratio=mlp_ratio,
                drop_path_rate=drop_path_rate[i],
                pc_conv_size=pc_conv_size,
                pc_conv_size_add=pc_conv_size_add,
                n_div=n_div,
                act_layer=act_layer,
                norm_layer=norm_layer,
                attention=attention,
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PatchEmbed_2D(nn.Module):
    """Patch Embedding"""

    def __init__(
        self,
        conv_size: int,
        conv_stride: int,
        in_channel: int,
        out_channel: int,
        norm_layer: torch.nn.Module,
    ):
        super(PatchEmbed_2D, self).__init__()

        self.proj = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=conv_size,
            stride=conv_stride,
            bias=False,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging_2D(nn.Module):
    """Patch Merging"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        conv_size: int,
        conv_stride: int,
        norm_layer: torch.nn.Module,
    ):
        super(PatchMerging_2D, self).__init__()

        self.reduction = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=conv_size,
            stride=conv_stride,
            bias=False,
        )

        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.reduction(x)
        x = self.norm(x)
        return x


class MSPCNet_2D(nn.Module):
    """MPSCNet"""

    def __init__(
        self,
        in_chans: int = 5,
        embed_dim: int = 128,
        depth: list[int] = [1, 1, 3, 1],
        depth_scale: float = 5.0,
        mp_ratio: float = 2.0,
        patch_conv_size: int = 3,
        patch_conv_stride: int = 2,
        pc_conv_size: int = 3,
        pc_conv_size_add: int = 2,
        n_div: int = 8,
        merge_conv_size: int = 3,
        merge_conv_stride: int = 2,
        head_dim: int = 128,
        drop_path_rate: float = 0.3,
        norm_layer: str = "BN",
        act_layer: str = "RELU",
        attention: str = "ECA_F",
    ):
        """
        @param in_chans: input channel
        @param embed_dim: embedding dimension
        @param depth: stage depth
        @param depth_scale: stage depth scale
        @param mp_ratio: mlp ratio
        @param patch_conv_size: patch embedding conv size
        @param patch_conv_stride: patch embedding conv stride
        @param pc_conv_size_1: partial conv size 1
        @param pc_conv_size_2: partial conv size 2
        @param merge_conv_size: patch merging conv size
        @param merge_conv_stride: patch merging conv stride
        @param head_dim: head dimension
        @param drop_path_rate: drop path rate
        @param norm_layer: norm layer
        @param act_layer: activation layer
        @param attention: attention
        @param scconv_sr: scconv sr
        @param scconv_cr: scconv cr
        """
        super(MSPCNet_2D, self).__init__()

        depth = [int(x * depth_scale) for x in depth]
        if norm_layer == "BN":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "LN":
            norm_layer = nn.LayerNorm

        if act_layer == "RELU":
            act_layer = nn.ReLU
        elif act_layer == "GELU":
            act_layer = nn.GELU

        self.num_stage = len(depth)
        self.patch_embed = PatchEmbed_2D(
            conv_size=patch_conv_size,
            conv_stride=patch_conv_stride,
            in_channel=in_chans,
            out_channel=embed_dim,
            norm_layer=norm_layer,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        stages_list = []
        for i in range(self.num_stage):
            stage = BasicStage_2D(
                channel=embed_dim,
                mlp_ratio=mp_ratio,
                drop_path_rate=dpr[sum(depth[:i]) : sum(depth[: i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                pc_conv_size=pc_conv_size,
                pc_conv_size_add=pc_conv_size_add,
                n_div=n_div,
                attention=attention,
            )

            stages_list.append(stage)

            if i != self.num_stage - 1:
                stage = PatchMerging_2D(
                    in_channel=embed_dim,
                    out_channel=embed_dim,
                    conv_size=merge_conv_size,
                    conv_stride=merge_conv_stride,
                    norm_layer=norm_layer,
                )

                stages_list.append(stage)

        self.stages = nn.Sequential(*stages_list)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, head_dim, 1, bias=False),
            act_layer(),
            nn.Flatten(),
        )

    def forward(self, photo):
        feature = self.patch_embed(photo)
        feature = self.stages(feature)

        feature = self.head(feature)

        return feature

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file model.py
# @author: wujiangu
# @date: 2023-10-26 11:07
# @description: multi-scale partial convolution network for 2D photometric

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("./"))
from model.MSSPE.fusion import Fusion
from model.MSSPE.MAG.Mag import MagEX
from model.MSSPE.MSPC.MSPC_1D import MSPCNet_1D
from model.MSSPE.MSPC.MSPC_2D import MSPCNet_2D


# Unified model
class MSSPE(torch.nn.Module):
    def __init__(
        self,
        head_dim=128,
        spec_embed_dim=128,
        spec_depth=[1, 1, 2, 1],
        spec_depth_scale=1.0,
        spec_patch_conv_size=3,
        spec_patch_conv_stride=2,
        spec_pc_conv_size=5,
        spec_pc_conv_size_scale=9,
        spec_n_div=8,
        spec_merge_conv_size=3,
        spec_merge_conv_stride=2,
        spec_drop_path_rate=0.1,
        spec_attention="None",
        photo_embed_dim=128,
        photo_depth=[1, 1, 2, 1],
        photo_depth_scale=1.0,
        photo_patch_conv_size=3,
        photo_patch_conv_stride=2,
        photo_pc_conv_size=3,
        photo_pc_conv_size_add=9,
        photo_n_div=8,
        photo_merge_conv_size=3,
        photo_merge_conv_stride=2,
        photo_drop_path_rate=0.1,
        photo_attention="None",
        mask_use=[1, 1, 1, 1, 1],
        fusion_nhead=8,
        fusion_nlayer=3,
        fusion_mask_attention=0,
        fusion_mlp_ratio=4.0,
        fusion_attn_drop=0.0,
        fusion_drop=0.0,
        fusion_drop_path=0.0,
        num_classes=1,
    ):
        super(MSSPE, self).__init__()

        self.mask_use = mask_use

        if mask_use[-1] == 1:
            self.lamost_spec_model = MSPCNet_1D(
                in_chans=1,
                embed_dim=spec_embed_dim,
                depth=spec_depth,
                depth_scale=spec_depth_scale,
                patch_conv_size=spec_patch_conv_size,
                patch_conv_stride=spec_patch_conv_stride,
                pc_conv_size=spec_pc_conv_size,
                pc_conv_size_scale=spec_pc_conv_size_scale,
                n_div=spec_n_div,
                merge_conv_size=spec_merge_conv_size,
                merge_conv_stride=spec_merge_conv_stride,
                head_dim=head_dim,
                drop_path_rate=spec_drop_path_rate,
                attention=spec_attention,
            )
        else:
            self.lamost_spec_model = nn.Identity()

        if mask_use[0] == 1:
            self.sdss_photo_model = MSPCNet_2D(
                in_chans=5,
                embed_dim=photo_embed_dim,
                depth=photo_depth,
                depth_scale=photo_depth_scale,
                patch_conv_size=photo_patch_conv_size,
                patch_conv_stride=photo_patch_conv_stride,
                pc_conv_size=photo_pc_conv_size,
                pc_conv_size_add=photo_pc_conv_size_add,
                n_div=photo_n_div,
                merge_conv_size=photo_merge_conv_size,
                merge_conv_stride=photo_merge_conv_stride,
                head_dim=head_dim,
                drop_path_rate=photo_drop_path_rate,
                attention=photo_attention,
            )
        else:
            self.sdss_photo_model = nn.Identity()

        if mask_use[2] == 1:
            self.wise_photo_model = MSPCNet_2D(
                in_chans=4,
                embed_dim=photo_embed_dim,
                depth=photo_depth,
                depth_scale=photo_depth_scale,
                patch_conv_size=photo_patch_conv_size,
                patch_conv_stride=photo_patch_conv_stride,
                pc_conv_size=photo_pc_conv_size,
                pc_conv_size_add=photo_pc_conv_size_add,
                n_div=photo_n_div,
                merge_conv_size=photo_merge_conv_size,
                merge_conv_stride=photo_merge_conv_stride,
                head_dim=head_dim,
                drop_path_rate=photo_drop_path_rate,
                attention=photo_attention,
            )
        else:
            self.wise_photo_model = nn.Identity()

        if mask_use[1] == 1:
            self.sdss_mag_model = MagEX(5, head_dim)
        else:
            self.sdss_mag_model = nn.Identity()

        if mask_use[3] == 1:
            self.wise_mag_model = MagEX(4, head_dim)
        else:
            self.wise_mag_model = nn.Identity()

        self.head_dim = head_dim

        self.fusion = Fusion(
            in_dim=head_dim,
            out_dim=head_dim,
            nhead=fusion_nhead,
            num_layers=fusion_nlayer,
            mask_attention=fusion_mask_attention,
            mlp_ratio=fusion_mlp_ratio,
            attn_drop=fusion_attn_drop,
            drop=fusion_drop,
            drop_path=fusion_drop_path,
        )

        # classfier
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def freeze(self):
        # freeze all the parameters except the classifier
        for param in self.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(
        self,
        lamost_spec,
        sdss_photo,
        sdss_mag,
        wise_photo,
        wise_mag,
        mask,
    ):
        # mask: batch_size * 5
        batch_size = mask.shape[0]

        if self.mask_use[0] != 0:
            sdss_photo_feature = self.sdss_photo_model(sdss_photo).unsqueeze(1)
        else:
            sdss_photo_feature = torch.randn(batch_size, 1, self.head_dim).to(
                sdss_photo.device
            )

        if self.mask_use[2] != 0:
            wise_photo_feature = self.wise_photo_model(wise_photo).unsqueeze(1)
        else:
            wise_photo_feature = torch.randn(batch_size, 1, self.head_dim).to(
                wise_photo.device
            )
        if self.mask_use[1] != 0:
            sdss_mag_feature = self.sdss_mag_model(sdss_mag).unsqueeze(1)
        else:
            sdss_mag_feature = torch.randn(batch_size, 1, self.head_dim).to(
                sdss_mag.device
            )
        if self.mask_use[3] != 0:
            wise_mag_feature = self.wise_mag_model(wise_mag).unsqueeze(1)
        else:
            wise_mag_feature = torch.randn(batch_size, 1, self.head_dim).to(
                wise_mag.device
            )
        if self.mask_use[4] != 0:
            lamost_spec_feature = self.lamost_spec_model(lamost_spec).unsqueeze(1)
        else:
            lamost_spec_feature = torch.zeros(batch_size, 1, self.head_dim).to(
                lamost_spec.device
            )

        feature = torch.cat(
            [
                sdss_photo_feature,
                sdss_mag_feature,
                wise_photo_feature,
                wise_mag_feature,
                lamost_spec_feature,
            ],
            dim=1,
        )

        mask_scale = mask.unsqueeze(-1)

        feature = feature * mask_scale

        # insert a row to mask
        mask = torch.cat(
            [
                torch.ones(batch_size, 1).to(mask.device),
                mask,
            ],
            dim=1,
        )

        # mask to bool
        mask = mask.type(torch.bool)

        # reverse mask

        # mask (2 * 5) --> mask (5 * 5): multiply
        mask_1 = mask.unsqueeze(1).repeat(1, 6, 1)
        mask_2 = mask.unsqueeze(2).repeat(1, 1, 6)
        mask = mask_1 * mask_2

        # let Make the diagonal element True
        mask = mask + torch.eye(6).to(mask.device).type(torch.bool).unsqueeze(0).repeat(
            batch_size, 1, 1
        )

        mask = ~mask

        # fusion
        cls_token, decoder_res_list = self.fusion(feature, mask)

        # classifier
        res = self.classifier(cls_token)

        return res, decoder_res_list

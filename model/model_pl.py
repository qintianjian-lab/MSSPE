#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file model_pl.py
# @author: wujiangu
# @date: 2023-09-17 11:24
# @description: torch lightning model

import os
import sys

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath("./"))

from model.MSSPE.MSSPE import MSSPE


class Model_pl(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = MSSPE(
            **cfg["model_params"],
            mask_use=cfg["mask_use"],
            num_classes=len(self.cfg["pred_names"]),
        )

        # loss function: mse
        self.mean_mse_loss = torch.nn.MSELoss(reduction="mean")
        self.sum_mse_loss = torch.nn.MSELoss(reduction="sum")
        self.none_mse_loss = torch.nn.MSELoss(reduction="none")

        if "log_mode" in cfg and cfg["log_mode"] == "min":
            self.best_metric = 9999
        elif "log_mode" in cfg and cfg["log_mode"] == "max":
            self.best_metric = -9999
        else:
            self.best_metric = 9999
        self.loss_dict = {}

        self.stop_increase_epoch = 0
        self.start_full_train = False

        self.train_stage = "pretrain"
        self.pred_index = -1

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        return self.compute_step(batch, batch_idx, stage="train")

    def on_train_epoch_end(self):
        """ """
        self.stage_end(stage="train")

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        return self.compute_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        """ """
        self.stage_end(stage="val")

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:

        """
        return self.compute_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self):
        """ """
        self.stage_end(stage="test")

    def log_hyperparams(
        self, params: dict, on_step=False, on_epoch=True, prog_bar=False
    ):
        for k, v in params.items():
            if self.cfg["log_metric"] not in k:
                k = self.train_stage + "_" + k

            self.log(
                k,
                v,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
                sync_dist=True,
            )

    def compute_step(self, batch, batch_idx, stage="train"):
        """

        :param batch:
        :param batch_idx:
        :param stage:  (Default value = "train")

        """
        sdss_photo = batch["sdss_photo"]
        sdss_mag = batch["sdss_mag"]
        wise_photo = batch["wise_photo"]
        wise_mag = batch["wise_mag"]
        lamost_spec = batch["lamost_spec"]

        lamost_spec = lamost_spec[:, 0, :].unsqueeze(1)

        origin_data_list = [
            sdss_photo.clone(),
            sdss_mag.clone(),
            wise_photo.clone(),
            wise_mag.clone(),
            lamost_spec.clone(),
        ]

        mask = batch["mask"]

        mask_use = torch.tensor(self.cfg["mask_use"]).to(mask.device)

        mask_to_mask = torch.zeros_like(mask)

        mask = mask * mask_use

        # random mask if mask[i] (!= 0 > 2)
        mask_max_num = self.cfg["mask_max_num"]
        mask_loss_scale = mask.clone()

        if self.current_epoch < self.cfg["mask_epoch"] and stage != "test":
            for i in range(mask.shape[0]):
                mask_num = torch.sum(mask[i] != 0).item() - 1
                if mask_num >= 2:
                    mask_index = torch.where(mask[i] != 0)[0]
                    mask_index = torch.randperm(mask_index.shape[0] - 1)
                    mask_max_index_num = min(mask_num - 1, mask_max_num)
                    mask_index_num = torch.randint(
                        0, mask_max_index_num + 1, (1,)
                    ).item()
                    mask_index = mask_index[:mask_index_num]

                    mask[i][mask_index] = 0
                    mask_to_mask[i][mask_index] = 1

        # forward
        pred_list, decoder_res_list = self.model(
            lamost_spec, sdss_photo, sdss_mag, wise_photo, wise_mag, mask
        )

        loss = 0
        loss_dict = {}
        loss_dict[f"{stage}_generate_loss"] = 0

        if self.current_epoch < self.cfg["generate_epoch"] and stage != "test":
            for i in range(len(decoder_res_list)):
                if self.cfg["generate_loss_scale"][i] == 0:
                    continue
                name = self.cfg["data_item_list"][i]
                origin_data_list[i] = origin_data_list[i].reshape(
                    origin_data_list[i].shape[0], -1
                )

                # log when decoder_res_list[i] have NaN
                if torch.isnan(decoder_res_list[i]).sum() > 0:
                    print(f"NaN: {stage}_mse_{name}")
                    print(f"decoder_res_list[i]: {decoder_res_list[i]}")

                    # let nan to 99
                    decoder_res_list[i][torch.isnan(decoder_res_list[i])] = 99

                mse_decoder = self.none_mse_loss(
                    decoder_res_list[i], origin_data_list[i]
                )

                # mse_decoder = mse_decoder * mask_to_mask[:, i].unsqueeze(1)
                mse_decoder = mse_decoder * mask_loss_scale[:, i].unsqueeze(1)

                mse_decoder = torch.mean(mse_decoder)

                decoder_num = torch.sum(mask_loss_scale[:, i] != 0)
                if mask_loss_scale[:, i].sum() == 0:
                    mse_decoder = torch.tensor(0.0).to(mse_decoder.device)
                else:
                    mse_decoder = mse_decoder / decoder_num

                loss_dict[f"{stage}_generate_{name}_loss"] = mse_decoder
                loss_dict[f"{stage}_generate_loss"] += (
                    mse_decoder * self.cfg["generate_loss_scale"][i]
                )
            loss += loss_dict[f"{stage}_generate_loss"]

        loss_dict[f"{stage}_pred_loss"] = 0
        for i in range(len(self.cfg["pred_names"])):

            if self.pred_index != -1 and i != self.pred_index:
                continue

            pred_name = self.cfg["pred_names"][i]
            label = batch[pred_name]
            pred = pred_list[i]

            loss_pred = self.mean_mse_loss(pred, label)

            loss_dict[f"{stage}_{pred_name}_loss"] = loss_pred

            loss_dict[f"{stage}_pred_loss"] += (
                loss_pred * self.cfg["pred_loss_scale"][i]
            )

        # log loss
        loss += loss_dict[f"{stage}_pred_loss"]
        loss_dict[f"{stage}_loss"] = loss

        # log loss
        self.log_hyperparams(loss_dict, prog_bar=False, on_step=True)

        return loss

    def stage_end(self, stage="train"):
        """

        :param stage:  (Default value = "train")

        """

        pass

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg["lr"])
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cfg["T_0"],
                T_mult=self.cfg["T_mult"],
                eta_min=self.cfg["eta_min"],
            ),
            "name": "lr",
        }

        return [optimizer], [lr_scheduler]

    def set_train_stage(self, stage):
        """

        :param stage:

        """
        self.train_stage = stage

    def freeze_model(self):
        self.model.freeze()

    def set_pred_index(self, index):
        """

        :param index:

        """
        self.pred_index = index

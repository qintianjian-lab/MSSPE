#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file train.py
# @author: wujiangu
# @date: 2023-09-17 11:38
# @description: train

import os

import torch

from cfg.cfg import cfg
from dataset.all_dataset import AllDataset
from lightning.pytorch import Trainer
from model.model_pl import Model_pl
from torch.utils.data import DataLoader
from torchvision import transforms

torch.set_float32_matmul_precision("high")


def main():

    cfg["data_dir"] = os.path.join(cfg["data_dir_prefix"], cfg["data_dir"])

    sdss_transform = transforms.Compose(
        [
            transforms.Resize(cfg["image_size"], antialias=True),
            # normalize
            transforms.Normalize(
                mean=cfg["sdss_photo_mean"],
                std=cfg["sdss_photo_std"],
            ),
        ]
    )

    wise_transform = transforms.Compose(
        [
            transforms.Resize(cfg["image_size"], antialias=True),
            # normalize
            transforms.Normalize(
                mean=cfg["wise_photo_mean"],
                std=cfg["wise_photo_std"],
            ),
        ]
    )

    transform_dict = {
        "sdss_photo": sdss_transform,
        "sdss_mag": None,
        "wise_photo": wise_transform,
        "wise_mag": None,
        "lamost_spec": None,
    }

    train_dataset = AllDataset(
        dataset_path=os.path.join(cfg["data_dir"], "train"),
        transform_dict=transform_dict,
        data_item_list=cfg["data_item_list"],
        data_suffix_list=cfg["data_suffix_list"],
        sdss_mag_mean=cfg["sdss_mag_mean"],
        sdss_mag_std=cfg["sdss_mag_std"],
        wise_mag_mean=cfg["wise_mag_mean"],
        wise_mag_std=cfg["wise_mag_std"],
    )

    val_dataset = AllDataset(
        dataset_path=os.path.join(cfg["data_dir"], "val"),
        transform_dict=transform_dict,
        data_item_list=cfg["data_item_list"],
        data_suffix_list=cfg["data_suffix_list"],
        sdss_mag_mean=cfg["sdss_mag_mean"],
        sdss_mag_std=cfg["sdss_mag_std"],
        wise_mag_mean=cfg["wise_mag_mean"],
        wise_mag_std=cfg["wise_mag_std"],
    )

    test_dataset = AllDataset(
        dataset_path=os.path.join(cfg["data_dir"], "test"),
        transform_dict=transform_dict,
        data_item_list=cfg["data_item_list"],
        data_suffix_list=cfg["data_suffix_list"],
        sdss_mag_mean=cfg["sdss_mag_mean"],
        sdss_mag_std=cfg["sdss_mag_std"],
        wise_mag_mean=cfg["wise_mag_mean"],
        wise_mag_std=cfg["wise_mag_std"],
    )

    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    model_pl = Model_pl(cfg)

    trainer = Trainer(
        max_epochs=cfg["epochs"],
        log_every_n_steps=1,
        fast_dev_run=cfg["debug"],
        precision=cfg["precision"],
        strategy=(
            "ddp_find_unused_parameters_true"
            if cfg["device"] == "multi-gpu"
            else "auto"
        ),
        num_sanity_val_steps=0,
    )

    # train
    trainer.fit(model_pl, train_dataloader, val_dataloader)

    # 6. test
    if cfg["test"]:
        trainer.test(
            model_pl,
            test_dataloader,
            ckpt_path=(None),
        )


if __name__ == "__main__":
    main()

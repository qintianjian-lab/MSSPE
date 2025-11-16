#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @file all_dataset.py
# @author: wujiangu
# @date: 2023-12-24 21:32
# @description: all dataset for sdss_photo, sdss_mag, wise_photo, lamost_spec

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AllDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        label_name: str = "label.csv",
        index_column_name: str = "id",
        data_item_list: list = [
            "sdss_photo",
            "sdss_mag",
            "wise_photo",
            "wise_mag",
            "lamost_spec",
        ],
        data_suffix_list: list = [".npy", ".npy", ".npy", ".npy", ".npy"],
        transform_dict: dict = {
            "sdss_photo": None,
            "sdss_mag": None,
            "wise_photo": None,
            "wise_mag": None,
            "lamost_spec": None,
        },
        lamost_spec_length: int = 3584,
        sdss_mag_mean: list = None,
        sdss_mag_std: list = None,
        wise_mag_mean: list = None,
        wise_mag_std: list = None,
        use_generated_data: bool = 0,
    ):
        self.data_dir = os.path.join(dataset_path)
        self.transform_dict = transform_dict
        label_path = os.path.join(dataset_path, "label", label_name)
        self.label_pd = pd.read_csv(label_path, dtype={index_column_name: str})

        # fill nan with -99
        self.label_pd = self.label_pd.fillna(-99)
        # get column: coadd_id dtype

        self.data_item_dict = dict(zip(data_item_list, data_suffix_list))

        self.index_column = index_column_name
        self.lamost_spec_length = lamost_spec_length
        self.use_generated_data = use_generated_data

        self.sdss_mag_mean = torch.tensor(sdss_mag_mean) if sdss_mag_mean else None
        self.sdss_mag_std = torch.tensor(sdss_mag_std) if sdss_mag_std else None
        self.wise_mag_mean = torch.tensor(wise_mag_mean) if wise_mag_mean else None
        self.wise_mag_std = torch.tensor(wise_mag_std) if wise_mag_std else None

    def __len__(self):
        return len(self.label_pd)

    def __getitem__(self, idx):
        item_name = self.label_pd.iloc[idx][self.index_column]

        # add mask
        mask = torch.ones(len(self.data_item_dict))
        ret = {
            "basename": item_name,
        }

        for i, (item_dir, item_suffix) in enumerate(self.data_item_dict.items()):
            item_path = os.path.join(self.data_dir, item_dir, item_name + item_suffix)
            item_data = np.load(item_path)

            # transform item_data to int
            if np.all(item_data.astype(np.int32) == -99):
                mask[i] = 0
            elif np.any(item_data.astype(np.int32) == -99):
                mask[i] = 0

            item_data = torch.from_numpy(item_data).float()

            if self.transform_dict[item_dir] is not None:
                item_data = self.transform_dict[item_dir](item_data)

            if mask[i] == 0:
                item_data = torch.zeros_like(item_data)

            ret[item_dir] = item_data.float()

        if len(ret["lamost_spec"][0]) < self.lamost_spec_length:
            pad_length = self.lamost_spec_length - len(ret["lamost_spec"][0])
            pad_left = torch.zeros(2, pad_length // 2)
            pad_right = torch.zeros(2, pad_length - pad_length // 2)
            ret["lamost_spec"] = torch.cat(
                [pad_left, ret["lamost_spec"], pad_right], dim=1
            )

        # # normalize sdss_mag and wise_mag
        if (
            mask[1] != 0
            and self.sdss_mag_mean is not None
            and self.sdss_mag_std is not None
        ):
            ret["sdss_mag"] = (ret["sdss_mag"] - self.sdss_mag_mean) / self.sdss_mag_std
        if (
            mask[3] != 0
            and self.wise_mag_mean is not None
            and self.wise_mag_std is not None
        ):
            ret["wise_mag"] = (ret["wise_mag"] - self.wise_mag_mean) / self.wise_mag_std

        for col in self.label_pd.columns:
            if col in [self.index_column, "label"]:
                continue
            if self.label_pd[col].dtype not in [np.float64, np.int64]:
                continue

            ret[col] = self.label_pd.iloc[idx][col]

            if type(ret[col]) == np.float64:
                ret[col] = torch.tensor(ret[col]).float()

        ret["mask"] = mask

        return ret

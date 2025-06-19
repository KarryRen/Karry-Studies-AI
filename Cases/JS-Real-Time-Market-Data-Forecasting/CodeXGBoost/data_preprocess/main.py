# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 22:15
# @Author  : Karry Ren

""" The pipline for data preprocessing.

Download raw data from website, then run this pipline to get dataset.

"""

import os
import numpy as np
import pandas as pd

import CodeXGBoost.config as config
from utils import reduce_mem_usage

if __name__ == "__main__":
    # --- Read raw data & Describe ---- #
    raw_data_dir = "../../Data/train.parquet"
    raw_data_df = pd.read_parquet(f"{raw_data_dir}/partition_id=0/part-0.parquet")
    print(f"Read id={0}: {len(raw_data_df)} lines.")
    for i in range(1, 4):
        data_i_df = pd.read_parquet(f"{raw_data_dir}/partition_id={i}/part-0.parquet")
        raw_data_df = pd.concat([raw_data_df, data_i_df], ignore_index=True)
        print(f"Read id={i}: {len(data_i_df)} lines.")

    # ---- Step 1. Reduce memory ---- #
    raw_data_df = raw_data_df[config.DATA_COLUMNS]  # select columns
    raw_data_df = reduce_mem_usage(raw_data_df)  # change type

    # ---- Step 2. Nan Operation ---- #
    data_df = raw_data_df[raw_data_df["date_id"] >= config.SKIP_DATES].reset_index(drop=True)  # skip dates
    for col in config.DATA_COLUMNS[4:]:  # ffill to reduce nan
        if data_df[col].isna().any():
            data_df[col] = data_df.groupby("symbol_id")[col].ffill()
    data_df = data_df[config.SELECTED_COLUMNS]  # drop too much nan
    for col in config.SELECTED_COLUMNS[4:]:  # use mid to fill
        if data_df[col].isna().any():
            data_df[col] = data_df.groupby("symbol_id")[col].transform(lambda x: x.fillna(x.median()))

    # ---- Step 3. Normalization ---- #
    # change data type to float32
    data_df[config.SELECTED_COLUMNS[4:]] = data_df[config.SELECTED_COLUMNS[4:]].astype("float32")
    # z-score the feature columns
    mean = data_df[config.SELECTED_COLUMNS[4:-1]].mean()
    std = data_df[config.SELECTED_COLUMNS[4:-1]].std() + 1e-5
    data_df[config.SELECTED_COLUMNS[4:-1]] = (data_df[config.SELECTED_COLUMNS[4:-1]] - mean) / std

    # ---- Step 4. Gen downside mask ---- #
    data_df["downside_mask"] = 1.0  # all mask as 1
    data_df.loc[data_df["responder_6"] < 0.0, "downside_mask"] = 2.5
    data_df.loc[data_df["responder_6"] < -2.0, "downside_mask"] = 4.0

    # ---- Step 4. Build up final dataset ---- #
    # split train & valid
    dates = sorted(data_df["date_id"].unique())
    train_dates, valid_dates = dates[:-config.NUM_OF_VALID_DATES], dates[-config.NUM_OF_VALID_DATES:]
    data_df_train = data_df.loc[data_df["date_id"].isin(train_dates)]
    data_df_valid = data_df.loc[data_df["date_id"].isin(valid_dates)]
    # build samples by symbol
    os.makedirs("../../Data/dataset_xgboost/", exist_ok=True)
    # - for train data
    symbol_ids_train = sorted(data_df_train["symbol_id"].unique())
    train_line_num, train_symbol_num = len(data_df_train), len(symbol_ids_train)
    train_sample_num = train_line_num - train_symbol_num * (config.TIME_STEPS - 1)
    train_x, train_w = np.zeros((train_sample_num, config.TIME_STEPS, 64)), np.zeros((train_sample_num, 1))
    train_y, train_n = np.zeros((train_sample_num, 1)), np.zeros((train_sample_num, config.TIME_STEPS))
    train_sample_step = 0
    for symbol_id in symbol_ids_train:
        symbol_data_df = data_df_train[data_df_train["symbol_id"] == symbol_id].sort_values(by=["date_id", "time_id"])
        symbol_x_data = symbol_data_df[config.SELECTED_COLUMNS[4:-1]].values
        symbol_y_data = symbol_data_df[config.SELECTED_COLUMNS[-1]].values
        symbol_w_data = symbol_data_df[config.SELECTED_COLUMNS[3]].values * symbol_data_df["downside_mask"].values
        for i in range(config.TIME_STEPS - 1, len(symbol_data_df)):
            train_x[train_sample_step] = symbol_x_data[i - config.TIME_STEPS + 1:i + 1]
            train_y[train_sample_step] = symbol_y_data[i]
            train_w[train_sample_step] = symbol_w_data[i]

            train_sample_step += 1
        print(f"Train Processing symbol id: {symbol_id}, len={len(symbol_data_df)}, train_sample_step={train_sample_step}")
    assert not np.isnan(train_x).any(), "Train x has NaN values."
    assert not np.isnan(train_y).any(), "Train y has NaN values."
    assert not np.isnan(train_w).any(), "Train w has NaN values."
    assert not np.isnan(train_n).any(), "Train n has NaN values."
    np.savez("../../Data/dataset_xgboost/train", x=train_x, y=train_y, w=train_w, n=train_n)
    # - for valid data
    symbol_ids_valid = sorted(data_df_valid["symbol_id"].unique())
    valid_line_num, valid_symbol_num = len(data_df_valid), len(symbol_ids_valid)
    valid_sample_num = valid_line_num - valid_symbol_num * (config.TIME_STEPS - 1)
    valid_x, valid_w = np.zeros((valid_sample_num, config.TIME_STEPS, 64)), np.zeros((valid_sample_num, 1))
    valid_y, valid_n = np.zeros((valid_sample_num, 1)), np.zeros((valid_sample_num, config.TIME_STEPS))
    valid_sample_step = 0
    for symbol_id in symbol_ids_valid:
        symbol_data_df = data_df_valid[data_df_valid["symbol_id"] == symbol_id].sort_values(by=["date_id", "time_id"])
        symbol_x_data = symbol_data_df[config.SELECTED_COLUMNS[4:-1]].values
        symbol_y_data = symbol_data_df[config.SELECTED_COLUMNS[-1]].values
        symbol_w_data = symbol_data_df[config.SELECTED_COLUMNS[3]].values * symbol_data_df["downside_mask"].values
        for i in range(config.TIME_STEPS - 1, len(symbol_data_df)):
            valid_x[valid_sample_step] = symbol_x_data[i - config.TIME_STEPS + 1:i + 1]
            valid_y[valid_sample_step] = symbol_y_data[i]
            valid_w[valid_sample_step] = symbol_w_data[i]
            valid_sample_step += 1
        print(f"Valid Processing symbol id: {symbol_id}, len={len(symbol_data_df)}, valid_sample_step={valid_sample_step}")
    assert not np.isnan(valid_x).any(), "Valid x has NaN values."
    assert not np.isnan(valid_y).any(), "Valid y has NaN values."
    assert not np.isnan(valid_w).any(), "Valid w has NaN values."
    assert not np.isnan(valid_n).any(), "Valid n has NaN values."
    np.savez("../../Data/dataset_xgboost/valid", x=valid_x, y=valid_y, w=valid_w, n=valid_n)

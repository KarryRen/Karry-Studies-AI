# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 22:15
# @Author  : Karry Ren

""" The pipline for data preprocessing. """

import pandas as pd
import Code.config as config

from utils import describe_data, reduce_mem_usage, date_symbol_id_visual, describe_modeling_column

if __name__ == "__main__":
    # --- Read raw data & Describe ---- #
    raw_data_dir = "../../Data/train.parquet"
    raw_data_df = pd.read_parquet(f"{raw_data_dir}/partition_id=0/part-0.parquet")
    print(f"Read id={0}: {len(raw_data_df)} lines.")
    for i in range(1, 5):
        data_i_df = pd.read_parquet(f"{raw_data_dir}/partition_id={i}/part-0.parquet")
        raw_data_df = pd.concat([raw_data_df, data_i_df], ignore_index=True)
        print(f"Read id={i}: {len(data_i_df)} lines.")
    describe_data(raw_data_df)

    # ---- Step 1. Reduce memory ---- #
    raw_data_df = raw_data_df[config.DATA_COLUMNS]  # select columns
    raw_data_df = reduce_mem_usage(raw_data_df)  # change type

    # ---- Step 2. Nan Operation ---- #
    date_symbol_id_visual(raw_data_df)
    data_df = raw_data_df[raw_data_df["date_id"] >= config.SKIP_DATES].reset_index(drop=True)  # skip dates
    describe_data(data_df)
    describe_modeling_column(data_df[config.DATA_COLUMNS[4:]], False)
    print(data_df)
    for col in config.DATA_COLUMNS[4:]:
        if data_df[col].isna().any():
            data_df[col] = data_df.groupby("symbol_id")[col].ffill()
    describe_modeling_column(data_df[config.DATA_COLUMNS[4:]], True)

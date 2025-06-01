# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 21:51
# @Author  : Karry Ren

""" Describe the raw data information. """

import pandas as pd


def des_unique_info():
    """ Describe the unique information. """

    raw_data_dir = "../../Data/train.parquet"

    # --- Read the first part data ---- #
    data_0_df = pd.read_parquet(f"{raw_data_dir}/partition_id=0/part-0.parquet")
    date_df, symbol_df = data_0_df[["date_id"]], data_0_df[["symbol_id"]]
    print(f"Read {0}: {len(data_0_df)} lines.")

    # ---- Read the following part data ---- #
    for i in range(10):
        data_i_df = pd.read_parquet(f"{raw_data_dir}/partition_id={i}/part-0.parquet")
        date_df = pd.concat([date_df, data_i_df[["date_id"]]], ignore_index=True)
        symbol_df = pd.concat([symbol_df, data_i_df[["symbol_id"]]], ignore_index=True)
        print(f"Read {i}: {len(data_i_df)} lines.")

    # ---- Describe info ---- #
    print("************************************************")
    print(f"Total data: \t {len(date_df)} lines.")
    print(f"Unique date: \t {date_df['date_id'].nunique()}.")
    print(f"Unique symbol: \t {symbol_df['symbol_id'].nunique()}.")


if __name__ == "__main__":
    des_unique_info()

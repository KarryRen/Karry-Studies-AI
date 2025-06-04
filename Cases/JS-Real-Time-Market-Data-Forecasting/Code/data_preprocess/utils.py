# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 22:15
# @Author  : Karry Ren

""" Utils functions for data preprocessing.

ref. https://www.kaggle.com/code/yuanzhezhou/jane-street-baseline-lgb-xgb-and-catboost/notebook

"""

import pandas as pd
import numpy as np


def describe_data(df: pd.DataFrame) -> None:
    """ Describe the data information.

    :param df: Raw data dataframe.
    """

    # ---- Describe info ---- #
    print("\n**************** DATA DESCRIPTION ****************")
    print(f"Total data: \t {len(df)} lines.")
    print(f"Unique date: \t {df['date_id'].nunique()}.")
    print(f"Unique symbol: \t {df['symbol_id'].nunique()}.")
    print("**************************************************")


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ Reduce the memory usage of the dataframe.

    :param df: Dataframe to reduce.
    :return: Reduced dataframe.

    """

    print("\n************** REDUCING MEMORY **************")
    # memory_usage() 是 df 每列的内存使用量, sum 是对它们求和 (B=>KB=>MB)
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")
    for col in df.columns:  # 遍历每列的列名
        col_type = df[col].dtype  # 列名的 type
        if col_type != object and str(col_type) != "category":  # 不是 object 也就是说这里处理的是数值类型的变量
            c_min, c_max = df[col].min(), df[col].max()  # 求出这列的最大值和最小值
            if str(col_type)[:3] == "int":  # 如果是 int 类型的变量, 不管是 int8, int16, int32 还是 int64
                # 如果这列的取值范围是在 int8 的取值范围内, 那就对类型进行转换 (-128 到 127)
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                # 如果这列的取值范围是在 int16 的取值范围内, 那就对类型进行转换 (-32,768 到 32,767)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                # 如果这列的取值范围是在 int32 的取值范围内, 那就对类型进行转换 (-2,147,483,648 到 2,147,483,647)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                # 如果这列的取值范围是在 int64 的取值范围内, 那就对类型进行转换(-9,223,372,036,854,775,808 到 9,223,372,036,854,775,807)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:  # 如果是浮点数类型.
                # 如果数值在 float16 的取值范围内, 如果觉得需要更高精度可以考虑 float32
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    # 如果数值在 float32 的取值范围内，对它进行类型转换
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                # 如果数值在 float64 的取值范围内，对它进行类型转换
                else:
                    df[col] = df[col].astype(np.float64)
    # 计算一下结束后的内存
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%")
    print("*********************************************")
    return df


def date_symbol_id_visual(df: pd.DataFrame) -> None:
    """ Visualize the unique symbol id number of date.

    :param df: Dataframe to visualize.
    """

    print("\n*************** Date Symbol ID Data Visualization ***************")
    date_symbol_df = df[["date_id", "symbol_id"]]
    date_symbol_result = date_symbol_df.groupby("date_id")["symbol_id"].nunique().reset_index(name="unique_symbol_count")
    print(date_symbol_result)
    date_symbol_result.to_csv("date_unique_symbol_count.csv", index=False)
    print("*****************************************************************")


def describe_modeling_column(df: pd.DataFrame, fill_nan_way: str = None) -> None:
    """ Describe the data column to modeling (79 features with 1 label).

    :param df: Dataframe to describe.
    :param fill_nan_way: The way to fill nan.
    """

    print("\n**************** MODELING COLUMNS DES **************")
    desc = df.describe(percentiles=[0.5]).T
    desc["Has_NaN"], desc["Nan Num"], desc["NaN %"] = df.isna().any(), df.isna().sum(), df.isna().sum() / len(df)
    desc = desc[["max", "min", "50%", "Has_NaN", "Nan Num", "NaN %"]]
    desc.columns = ["Max", "Min", "Median", "Has_NaN", "Nan Num", "NaN %"]
    print(desc)
    desc.to_csv(f"modeling_columns_fill_nan_{fill_nan_way}.csv")
    print("****************************************************")

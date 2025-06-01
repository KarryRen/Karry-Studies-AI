# -*- coding: utf-8 -*-
# @Time    : 2025/6/1 21:32
# @Author  : Karry Ren

""""""

import pandas as pd

df = pd.read_parquet("../../Data/train.parquet/partition_id=0/part-0.parquet")
print(df.columns)

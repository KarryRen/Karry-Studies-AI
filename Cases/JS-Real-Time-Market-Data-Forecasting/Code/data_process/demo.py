# -*- coding: utf-8 -*-
# @Time    : 2025/5/23 15:56
# @Author  : Karry Ren

""""""

import pandas as pd

path = "../../Data"

data = pd.read_parquet(f"{path}/test.parquet/date_id=0/part-0.parquet").sort_values(by=["row_id", "date_id", "time_id"])
print(data)

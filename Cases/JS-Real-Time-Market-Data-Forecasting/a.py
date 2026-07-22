# -*- coding: utf-8 -*-
# @Time    : 2025/6/20 14:31
# @Author  : Karry Ren

""""""

import akshare as ak

dates = [x.strftime("%Y%m%d") for x in ak.tool_trade_date_hist_sina()["trade_date"]]
td = [x for x in dates if "20240101" <= x < "20250101"]
print(len(td))
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 16:04
# @Author  : Karry Ren

""" Download the translation dataset.
    - Source Language: English
    - Target Language: French
"""

import d2l.torch as d2l

d2l.DATA_HUB["fra-eng"] = (d2l.DATA_URL + "fra-eng.zip", "94646ad1522d915e7b0f9296181140edcf86a4f5")
d2l.download("fra-eng")

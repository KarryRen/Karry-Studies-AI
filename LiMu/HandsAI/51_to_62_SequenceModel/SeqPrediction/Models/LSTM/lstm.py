# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:41
# @Author  : Karry Ren

""" The lstm in pytorch. """

from torch import nn

# ---- Define the layer ---- #
lstm_layer = nn.LSTM(input_size=16, hidden_size=30, num_layers=1, batch_first=True)

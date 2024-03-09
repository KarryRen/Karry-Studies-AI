# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:41
# @Author  : Karry Ren

""" The lstm in pytorch. """

from torch import nn
import torch

# ---- Define the layer ---- #
lstm_layer = nn.LSTM(26, [128, 256], num_layers=2, batch_first=False)

if __name__ == '__main__':
    # forward demo, two state (h, c)

    state = (torch.zeros((2, 32, 128)), torch.zeros((2, 32, 128)))
    X = torch.rand(size=(35, 32, 26))
    Y, state_new = lstm_layer(X, state)
    print(Y.shape, state_new[0].shape, state_new[1].shape)

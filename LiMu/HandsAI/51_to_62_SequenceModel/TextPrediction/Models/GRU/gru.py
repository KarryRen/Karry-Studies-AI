# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 17:16
# @Author  : Karry Ren

""" The gru in pytorch. """

from torch import nn
import torch

# ---- Define the layer ---- #
rnn_layer = nn.GRU(26, 128, num_layers=2, batch_first=False)

if __name__ == '__main__':
    # forward demo

    state = torch.zeros((2, 32, 128))
    X = torch.rand(size=(35, 32, 26))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

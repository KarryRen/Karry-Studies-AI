# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 15:55
# @Author  : Karry Ren

""" The rnn in pytorch. """

from torch import nn
import torch

# ---- Define the layer ---- #
rnn_layer = nn.RNN(26, 128, num_layers=2, batch_first=False)

# ---- Forward Demo --- #
state = torch.zeros((2, 32, 128))
X = torch.rand(size=(35, 32, 26))
Y, state_new = rnn_layer(X, state)
print(Y.shape, state_new.shape)

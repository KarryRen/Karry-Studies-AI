# -*- coding: utf-8 -*-
# @Time    : 2024/4/4 17:42
# @Author  : Karry Ren

""" Dropout scratch. """

import torch


def dropout_layer(X: torch.Tensor, dropout: float):
    """ The dropout layer.

    :param X: input feature
    :param dropout: the dropout ratio

    """

    assert 0 <= dropout <= 1

    if dropout == 1:
        return torch.zeros_like(X)
    elif dropout == 0:
        return X
    else:
        mask = (torch.rand(X.shape) > dropout).float()  # use mask rather than slice to fasten !!
        return mask * X / (1.0 - dropout)

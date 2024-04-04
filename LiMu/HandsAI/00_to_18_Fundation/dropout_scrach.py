# -*- coding: utf-8 -*-
# @Time    : 2024/4/4 17:42
# @Author  : Karry Ren

""" Dropout scratch. """

import torch


def dropout_layer(X: torch.Tensor, dropout_ratio: float):
    """ The dropout layer.

    :param X: input feature
    :param dropout_ratio: the dropout ratio

    """

    # dropout ratio must be [0, 1]
    assert 0 <= dropout_ratio <= 1

    if dropout_ratio == 1:  # all zero
        return torch.zeros_like(X)
    elif dropout_ratio == 0:  # keep raw
        return X
    else:  # based on the function
        mask = (torch.rand(X.shape) > dropout_ratio).float()  # use mask rather than slice to fasten !!
        return mask * X / (1.0 - dropout_ratio)

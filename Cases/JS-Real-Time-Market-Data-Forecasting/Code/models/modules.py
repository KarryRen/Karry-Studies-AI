# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 14:04
# @Author  : Karry Ren

""" Modules for net. """

import torch
from torch import nn


class SELayer_2D(nn.Module):
    """ The 2D SELayer.

    Ref. https://github.com/hujie-frank/SENet

    """

    def __init__(self, channel: int, reduction: int = 16):
        super(SELayer_2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear_sq = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear_ex = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = input.size()
        feature = self.avg_pool(input).reshape(b, c)
        f_sq = self.linear_sq(feature)
        f_ex = self.linear_ex(f_sq)
        se_weight = f_ex.reshape(b, c, 1, 1)
        return input * se_weight

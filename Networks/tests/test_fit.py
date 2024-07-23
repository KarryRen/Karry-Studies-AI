# -*- coding: utf-8 -*-
# @author : KarryRen
# @time   : 2024/7/23 10:14

""" The test function of FiT. """

import torch
from finance_transformer import FiT


def test():
    # fix the seed of torch
    torch.manual_seed(0)
    # init the random feature
    feature = torch.randn(64, 5, 768)
    # init the model
    fit = FiT()
    # forward
    output = fit(feature)
    # check
    assert output, f"Error !"

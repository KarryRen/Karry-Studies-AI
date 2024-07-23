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
    feature = torch.randn(64, 5, 768, dtype=torch.float32)
    re_seq_feature = feature.clone()
    re_seq_feature[:, 0, :] = feature[:, 4, :]
    re_seq_feature[:, 4, :] = feature[:, 0, :]
    re_seq_feature[:, 1, :] = feature[:, 3, :]
    re_seq_feature[:, 3, :] = feature[:, 1, :]
    assert not (torch.all(re_seq_feature == feature))

    # init the model
    fit = FiT(
        input_size=768,
        dim=64,
        depth=1,
        heads=2,
        ff_hidden_dim=32
    )

    # forward
    output = fit(feature)  # shape=(64, 1)
    re_seq_output = fit(re_seq_feature)  # shape=(64, 1)

    # check
    assert torch.all(torch.abs(output.detach() - re_seq_output.detach()) < 1e-6), f"Error !"

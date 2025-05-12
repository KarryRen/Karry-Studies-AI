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
    features = torch.randn(64, 5, 768, dtype=torch.float32)
    masks = torch.randint(low=0, high=2, size=(64, 5), dtype=torch.long)
    # [[0, 1, 0, 0, 0],
    #  [1, 1, 1, 1, 1],
    #  [1, 0, 1, 1, 0]]
    re_seq_features = features.clone()
    re_seq_features[:, 0, :] = features[:, 4, :]
    re_seq_features[:, 4, :] = features[:, 0, :]
    re_seq_features[:, 1, :] = features[:, 3, :]
    re_seq_features[:, 3, :] = features[:, 1, :]
    assert not (torch.all(re_seq_features == features))
    re_seq_masks = masks.clone()
    re_seq_masks[:, 0] = masks[:, 4]
    re_seq_masks[:, 4] = masks[:, 0]
    re_seq_masks[:, 1] = masks[:, 3]
    re_seq_masks[:, 3] = masks[:, 1]
    assert not (torch.all(re_seq_masks == masks))

    # init the model
    fit = FiT(
        input_size=768,
        dim=64,
        depth=1,
        heads=2,
        ff_hidden_dim=32
    )

    # forward
    output = fit(features, masks)  # shape=(64, 1)
    re_seq_output = fit(re_seq_features, re_seq_masks)  # shape=(64, 1)

    # check
    assert torch.all(torch.abs(output.detach() - re_seq_output.detach()) < 1e-6), f"Error !"


def demo():
    # random the input
    features = torch.randn(64, 5, 768, dtype=torch.float32)
    masks = torch.randint(low=0, high=2, size=(64, 5), dtype=torch.long)

    # init the model
    fit = FiT(
        input_size=768,
        dim=64,
        depth=1,
        heads=2,
        ff_hidden_dim=32
    )

    # forward
    output = fit(features, masks)
    print(output)

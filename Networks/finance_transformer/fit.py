# -*- coding: utf-8 -*-
# @author : KarryRen
# @time   : 2024/7/22 16:45

""" The Finance Transformer (FiT), a new `Plug-and-Play` module for `Finance Panel Feature` Extraction.

Detail information can be found in `https://github.higgsasset.com/ResearchLab/FinBert/issues/20`.


"""

import torch
from torch import nn


class FinanceTransformer(nn.Module):
    """ The Finance Transformer (FiT), a new `Plug-and-Play` module for `Finance Panel Feature` Extraction. """

    def __init__(self, device: torch.device):
        """ Init function of the FiT.

        :param device: the computing device

        """

        super(FinanceTransformer, self).__init__()

        self.device = device

    def forward(self, panel_features: torch.Tensor) -> torch.Tensor:
        """ Forward function of FiT.


        :return:
            - y_pred, shape=(bs, label_len, 1)

        """

        print(self.device)
        out = panel_features
        return out

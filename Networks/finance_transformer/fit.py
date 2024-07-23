# -*- coding: utf-8 -*-
# @author : KarryRen
# @time   : 2024/7/22 16:45

""" FiT, A new Plug-and-Play module for Finance Cross-Section Feature Extraction. """

import torch
from torch import nn


# classes

class FiT(nn.Module):
    """ FiT, A new Plug-and-Play module for Finance Cross-Section Feature Extraction. """

    def __init__(self, ):
        """ Init function of the FiT.


        """

        super(FiT, self).__init__()

    def forward(self, panel_features: torch.Tensor) -> torch.Tensor:
        """ Forward function of FiT.


        :return:
            - y_pred, shape=(bs, label_len, 1)

        """

        print(self.device)
        out = panel_features
        return out

# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 13:07
# @Author  : Karry Ren

""" The neural network. """

import torch
from torch import nn

from Code.models.modules import SELayer_2D


class Multi_CNN(nn.Module):
    """ The CNN for multi-task with multi-scale detection """

    def __init__(self):
        """ The init function of Multi_CNN Net. """

        super(Multi_CNN, self).__init__()

        # ---- Part 1. Multi scale conv layer ---- #
        self.conv_small = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding="same"),
            nn.ReLU()
        )
        self.conv_large = nn.Sequential(
            nn.Conv2d(kernel_size=(5, 5), in_channels=3, out_channels=8, padding="same"),
            nn.ReLU()
        )

        # ---- Part 2. SE Module ---- #
        self.se_layer = SELayer_2D(channel=16, reduction=4)

        # ---- Part 3. Further Conv ---- #
        self.further_conv = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ---- Part 4. Multi task output ---- #
        self.fc_output = nn.Linear(in_features=32, out_features=1)
        self.fc_is_noise = nn.Linear(in_features=32, out_features=2)

    def forward(self, input: torch.Tensor) -> dict:
        """ The forward function of Multi CNN Net.

        :param input: the input tensor, shape=(bs, time_steps, h, w)

        :return: the output for multi-task

        """

        # ---- Step 1. Multi scale conv encoding ---- #
        f_small = self.conv_small(input)  # shape=(bs, 8, h, w)
        f_large = self.conv_large(input)  # shape=(bs, 8, h, w)
        feature = torch.cat((f_small, f_large), dim=1)  # shape=(bs, 16, h, w)

        # ---- Step 2. SE feature extraction ---- #
        f_se = self.se_layer(feature)  # shape=(bs, 16, h, w)

        # ---- Step 3. Future Conv ---- #
        f = self.further_conv(f_se)  # shape=(bs, 32, h-2, w-2)
        f = self.gap(f).reshape(f.shape[0], 32)  # shape=(bs, 32)

        # ---- Step 4. Get output & is_noise to return ---- #
        output = self.fc_output(f)
        is_noise = self.fc_is_noise(f)
        return {
            "output": output,
            "is_noise": is_noise
        }


if __name__ == "__main__":
    bath_size, time_steps, h, w = 64, 3, 8, 8
    input = torch.ones((bath_size, time_steps, h, w))
    model = Multi_CNN()
    output = model(input)
    print(output["output"].shape, output["is_noise"].shape)
    print(output["output"], output["is_noise"])

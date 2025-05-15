# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 10:24
# @Author  : Karry Ren

""" LeNet(). """

from torch import nn
import torch


class LeNet(nn.Module):
    """ The LeNet-5 model. """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        """ Initialized function of the LeNet model.

        :param num_classes: number of classes to classify.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def layer_summary(self, X_shape: tuple):
        """ Prints layer statistics.

        :param X_shape: shape of the input data.
        """

        # build up the input
        X = torch.randn(*X_shape)
        # for loop to print
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)

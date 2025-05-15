# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 16:38
# @Author  : Karry Ren


from torch import nn
import torch


class AlexNet(nn.Module):
    """ The AlexNet model. """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        """ Initialized function of the AlexNet model.

        :param in_channels: number of input channels.
        :param num_classes: number of classes to classify.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes))

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

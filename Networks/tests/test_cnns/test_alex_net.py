# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 16:44
# @Author  : Karry Ren

""" Test AlexNet(). """

from cnns import AlexNet


def demo():
    model = AlexNet()
    model.layer_summary((1, 3, 224, 224))


if __name__ == "__main__":
    demo()

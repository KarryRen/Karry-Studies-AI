# -*- coding: utf-8 -*-
# @Time    : 2025/5/15 16:30
# @Author  : Karry Ren

""" Test LeNet(). """

from cnns import LeNet


def demo():
    model = LeNet()
    model.layer_summary((1, 3, 28, 28))


if __name__ == "__main__":
    demo()

# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 23:00
# @Author  : Karry Ren

""" Test the autograd and detach. """

import torch

x = torch.range(0, 3, requires_grad=True)

u = x * x
y = u * x
y.sum().backward()
print(x.grad)

x.grad.zero_()

u = x * x
y = u.detach() * x
y.sum().backward()
print(x.grad)
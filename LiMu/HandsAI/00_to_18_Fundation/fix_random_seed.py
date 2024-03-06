# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 20:43
# @Author  : Karry Ren

""" How the random seed changing the init.
    Only the sequence matters
"""

import random
import os
import numpy as np
import torch

import torch.nn as nn


def fix_random_seed(seed=1029):
    """Fix the random seed to decrease the random of training.
    :param seed: the random seed
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_random_seed()

# model = nn.Sequential(
#     nn.Linear(2, 2),
#     nn.Linear(2, 2),
#     nn.Conv2d(2, 2, kernel_size=1)
# )
#
# print(model[0].weight)
# print(model[1].weight)
# print(model[2].weight)

# Parameter containing:
# tensor([[ 0.0088,  0.4801],
#         [ 0.5671, -0.5897]], requires_grad=True)
# Parameter containing:
# tensor([[-0.1656, -0.4606],
#         [-0.6994, -0.3886]], requires_grad=True)
# Parameter containing:
# tensor([[[[ 0.2588]],
#
#          [[ 0.3302]]],
#
#
#         [[[-0.1445]],
#
#          [[-0.3767]]]], requires_grad=True)

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Conv2d(2, 2, kernel_size=1),
    nn.Linear(2, 2),
    nn.Linear(2, 2)
)

print(model[0].weight)
print(model[1].weight)
print(model[2].weight)
print(model[3].weight)

# Parameter containing:
# tensor([[ 0.0088,  0.4801],
#         [ 0.5671, -0.5897]], requires_grad=True)
# Parameter containing:
# tensor([[[[-0.1656]],
#
#          [[-0.4606]]],
#
#
#         [[[-0.6994]],
#
#          [[-0.3886]]]], requires_grad=True)
# Parameter containing:
# tensor([[ 0.2588,  0.3302],
#         [-0.1445, -0.3767]], requires_grad=True)
# Parameter containing:
# tensor([[-0.6027, -0.4545],
#         [ 0.6119, -0.4824]], requires_grad=True)

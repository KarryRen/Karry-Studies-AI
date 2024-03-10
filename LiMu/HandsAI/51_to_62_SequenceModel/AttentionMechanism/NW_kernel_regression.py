# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 11:33
# @Author  : Karry Ren

""" The Simple Attention Mechanism Demo (Nadaraya-Watson kernel regression). """

import torch
from torch import nn
import matplotlib.pyplot as plt


# ---- Step 1. Generate the train & test data ---- #
def f(x):
    """ A simple function. """
    return 2 * torch.sin(x) + x ** 0.8


# - train
n_train = 60
x_train, _ = torch.sort(torch.rand(n_train) * 5)  # shape=(n_train)
y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # shape=(n_train)
# - test
x_test = torch.arange(0, 5, 0.1)  # shape=(n_test)
y_truth = f(x_test)  # shape=(n_train)


# ---- Step 2. Construct the model ---- #
def nw_kernel_regression(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """ The Nadaraya-Watson kernel regression Function.

    :param q: query, shape=(n_q)
    :param k: key, shape=(n_k)
    :param v: value, shape=(n_v)

    return:
        output, shape=(n_q)

    NOTE: n_k must equal to n_v, which means (k, v) pair
    """

    # ---- Step 1. Repeat q to len(k), shape =  ---- #
    q_repeat = q.repeat_interleave(len(k)).reshape((-1, len(k)))  # shape=(n_q, n_k)

    # ---- Step 2. Compute the attention weights, use Gaussian Kernel ---- #
    attention_weights = nn.functional.softmax(-(q_repeat - k) ** 2 / 2, dim=1)  # shape=(n_q, n_k)
    print(attention_weights.shape)

    # ---- Step 3. Weighted the value and get the output ---- #
    output = torch.matmul(attention_weights, v)

    return output


# ---- Step 3. Use the model to predict ---- #
y_pred = nw_kernel_regression(q=x_test, k=x_train, v=y_train)
plt.figure(figsize=(15, 6))
plt.plot(y_truth, label="y_true", color="g")
plt.plot(y_pred, label="y_pred", color="b")
plt.legend()
plt.show()

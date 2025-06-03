# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 13:37
# @Author  : Karry Ren

""" The metrics. """

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray = None, epsilon: float = 1e-10):
    """ :math:`R^2` (coefficient of determination) regression score function.
    :math:`R^2 = 1 - SSR/SST`.

    Best possible score is 1.0, and it can be NEGATIVE (because the model can be arbitrarily worse,
    it need not actually be the square of a quantity R).

    :param y_true: the label, shape=(num_of_samples)
    :param y_pred: the prediction, shape=(num_of_samples)
    :param weight: the weight of label, corr with sample, shape=(num_of_samples) CAN'T BE ALL ZERO !!
    :param epsilon: the epsilon to avoid 0 denominator

     return:
        - r2, a number shape=()

    """

    # ---- Step 1. Test the shape and weight ---- #
    assert y_true.shape == y_pred.shape, f"`y_true`, `y_pred` should have the SAME shape !"
    assert weight.shape == y_true.shape, f"`weight` should have the SAME shape as y_true&y_pred !"
    assert np.sum(weight) > 0, f"weight can't be all zero !"

    # ---- Step 2. Compute the SSR & SSE ---- #
    # compute: SSR = sum(weight * (y - y_hat)^2)), a number, shape=()
    ssr = np.sum(weight * ((y_true - y_pred) ** 2), axis=0, dtype=np.float32)
    # compute SST = sum((y - y_bar)^2)
    # - the weighted mean of y_true, shape=(num_of_samples)
    y_bar = np.sum(weight * y_true, axis=0, keepdims=True) / np.sum(weight, axis=0, keepdims=True)
    # - compute
    sst = np.sum(weight * ((y_true - y_bar) ** 2), axis=0, dtype=np.float32)

    # ---- Step 3. Compute and Return r2 = 1 - SSR/SST ---- #
    r2 = 1 - (ssr / (sst + epsilon))
    return r2


if __name__ == "__main__":
    y_true = np.array([1, 3, 3])
    y_pred = np.array([1, 2, 3])
    weight = np.array([1, 1, 0])
    r2 = r2_score(y_true=y_true, y_pred=y_pred, weight=weight)
    print("r2 = ", r2)

    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    print("acc = ", accuracy_score(y_true=y_true, y_pred=y_pred))
    print("precision = ", precision_score(y_true=y_true, y_pred=y_pred))
    print("f1 = ", f1_score(y_true=y_true, y_pred=y_pred))

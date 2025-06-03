# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 13:08
# @Author  : Karry Ren

""" The loss functions.

- MSE_Loss: mean squared error loss.
- CE_Loss: cross entropy loss.

"""

import torch
import torch.nn.functional as F


class MSE_Loss:
    """ Compute the MSE loss.

    loss = reduction((y_true - y_pred)^2)

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the MSE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in MSE_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weight: torch.Tensor):
        """ Call function of the MSE Loss.

        :param y_true: the true label of time series prediction, shape=(bs, 1)
        :param y_pred: the prediction, shape=(bs, 1)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 1. Compute the loss ---- #
        if self.reduction == "mean":
            # compute mse loss (`mean`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(weight * mse_sample_loss) / torch.sum(weight)  # weighted and mean
            batch_loss = mse_loss
        elif self.reduction == "sum":
            # compute mse loss (`sum`)
            mse_sample_loss = (y_pred - y_true) ** 2  # shape=(bs, 1)
            mse_loss = torch.sum(weight * mse_sample_loss)  # weighted and sum
            batch_loss = mse_loss
        else:
            raise TypeError(self.reduction)

        # ---- Step 2. Return loss ---- #
        return batch_loss


class CE_Loss:
    """ Compute the CE loss.

    loss = reduction(cross_entropy(y_true, y_pred))

    """

    def __init__(self, reduction: str = "mean"):
        """ Init function of the CE Loss.

        :param reduction: the reduction way of this loss, you have only 2 choices now:
            - `sum` for sum reduction
            - `mean` for mean reduction

        """

        assert reduction in ["sum", "mean"], f"Reduction in CE_Loss ERROR !! `{reduction}` is not allowed !!"
        self.reduction = reduction  # the reduction way

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, weight: torch.Tensor):
        """ Call function of the CE Loss.

        :param y_true: the true label of time series prediction, shape=(bs)
        :param y_pred: the prediction, shape=(bs, cls)
        :param weight: the weight indicates item meaningful or meaningless, shape=(bs, 1)

        return:
            - batch_loss: a Tensor number, shape=([])

        """

        # ---- Step 0. Test the weight shape & make the default weight ---- #
        assert weight.shape[0] == y_true.shape[0], "Weight should have the same length with y_true&y_pred !"

        # ---- Step 1. Compute the loss ---- #
        if self.reduction == "mean":
            # compute ce loss (`mean`)
            ce_sample_loss = F.cross_entropy(input=y_pred, target=y_true, reduction="none")  # (bs, 1)
            ce_loss = torch.sum(weight * ce_sample_loss) / torch.sum(weight)  # weighted and mean
            batch_loss = ce_loss
        elif self.reduction == "sum":
            # compute ce loss (`sum`)
            ce_sample_loss = F.cross_entropy(input=y_pred, target=y_true, reduction="none")
            ce_loss = torch.sum(weight * ce_sample_loss)  # weighted and sum
            batch_loss = ce_loss
        else:
            raise TypeError(self.reduction)

        # ---- Step 2. Return loss ---- #
        return batch_loss


if __name__ == "__main__":
    # An Example test two loss
    bs, T, D = 64, 3, 2
    y_true_mse = torch.zeros((bs, 1))
    y_true_ce = torch.zeros(bs, dtype=torch.long)
    weight = torch.ones((bs, 1))
    y_pred_mse = torch.ones((bs, 1))
    y_pred_ce = torch.ones((bs, 2))
    a = torch.ones((bs, T, D))
    rec_residuals = (torch.zeros((bs, T, D)), a, torch.zeros((bs, T, D)), torch.zeros((bs, T, D)))

    # ---- Test MSE_Loss ---- #
    loss_mse_sum = MSE_Loss(reduction="sum")
    l = loss_mse_sum(y_true=y_true_mse, y_pred=y_pred_mse, weight=weight)
    print(l)
    assert l == bs * 1
    loss_mse_mean = MSE_Loss(reduction="mean")
    l = loss_mse_mean(y_true=y_true_mse, y_pred=y_pred_mse, weight=weight)
    print(l)
    assert l == bs / weight.sum()

    # ---- Test CE_Loss ---- #
    y_pred_ce[:, 1] = -20
    loss_ce_sum = CE_Loss(reduction="sum")
    l = loss_ce_sum(y_true=y_true_ce, y_pred=y_pred_ce, weight=weight)
    print(l)
    loss_ce_mean = CE_Loss(reduction="mean")
    l = loss_ce_mean(y_true=y_true_ce, y_pred=y_pred_ce, weight=weight)
    print(l)

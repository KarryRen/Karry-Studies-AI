# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 16:38
# @Author  : Karry Ren

"""Realization of the linear_regression from scratch"""

import torch
import random
import matplotlib.pyplot as plt
from typing import List


def synthetic_data(w: torch.Tensor, b, num_examples):
    """Generate dataset y = Xw + b + noise.
    :param w: the weight, shape = (d)
    :param b: the bias, shape = 1
    :param num_examples, the num of examples

    return:
        X, the feature, shape = (num_examples, d)
        y, the label, shape = (num_examples, 1)

    """

    # ---- Step 1. Generate random X, shape=(num_examples, d) ---- #
    X = torch.normal(0, 1, (num_examples, len(w)))
    # ---- Step 2. Compute y = Xw + b
    y = torch.matmul(X, w) + b
    # ---- Step 3. Add noise ---- #
    y += torch.normal(0, 0.01, y.shape)

    return X, y.reshape((-1, 1))


def data_iter(batch_size: int, features: torch.Tensor, labels: torch.Tensor):
    """Read small batch data, and construct the dataloader.
    :param batch_size: the batch size
    :param features: the features
    :param labels: the labels

    """

    # ---- Step 1. Shuffle the data ---- #
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    # ---- Step 2 Construct the shuffled dataset ---- #
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        # generate an iter which can be used as for-loop
        yield features[batch_indices], labels[batch_indices]


class LinearReg:
    """The linear regression model."""

    def __init__(self, feature_dim: int):
        """Init the Model.
        :param feature_dim: the dimension of feature

        """

        self.w = torch.normal(0, 0.01, size=(feature_dim, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x: torch.Tensor):
        """Forward computing: output = xw + b.
        :param x: the input feature.

        """

        return torch.matmul(x, self.w) + self.b


def squared_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """The loss function.
    :param y_hat: the predicted y, shape = (bs, 1)
    :param y: the label, shape = (bs, 1)

    """

    return torch.sum((y_hat - y) ** 2 / 2)


def batch_sgd(params: List[torch.Tensor], lr: float, batch_size: int):
    """Batch SGD optimizing algorithm."""

    with torch.no_grad():
        for param in params:
            # update param
            param -= lr * param.grad / batch_size
            # zero the grad of param
            param.grad.zero_()


if __name__ == '__main__':
    BATCH_SIZE = 10
    LR = 0.03
    NUM_EPOCH = 10

    # ---- Step 1. Set the true w & b, then generate the dataset ---- #
    # here, we have two features
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # ---- See the data ---- #
    # here, we see the relationship between feature 1 and label, the positive linear
    # plt.figure()
    # plt.scatter(x=features[:, 0].detach().numpy(), y=labels[:, 0].detach().numpy())
    # plt.show()

    # ---- Step 2. Construct the dataloader ---- #
    data_loader = data_iter(BATCH_SIZE, features, labels)

    # ---- Step 3. Construct the Model & Loss Function & Optimizer ---- #
    model = LinearReg(feature_dim=len(true_w))
    loss = squared_loss
    optimizer = batch_sgd

    # ---- Step 4. Begin train & valid ---- #
    for epoch in range(NUM_EPOCH):
        # Train (Different epoch have different sequence to read data !!)
        for X, y in data_iter(BATCH_SIZE, features, labels):
            # 1. forward
            y_pred = model.forward(X)
            # 2. compute loss
            l = loss(y_pred, y)
            # 3. backward to compute grad
            l.backward()
            # 4. update the param
            optimizer([model.w, model.b], LR, BATCH_SIZE)
        # Valid
        with torch.no_grad():
            prediction = model.forward(features)
            train_l = loss(prediction, labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
            # plt.figure(figsize=(15, 9))
            # plt.scatter(x=range(len(labels)), y=labels[:])
            # plt.scatter(x=range(len(labels)), y=prediction[:], color="r")
            # plt.savefig(f"Epoch_{epoch}.png")

    print(f"the error of `w`: {((true_w - model.w.reshape(true_w.shape)) / true_w).data * 100} %")
    print(f"the error of `b`: {((true_b - model.b) / true_b).data * 100} %")

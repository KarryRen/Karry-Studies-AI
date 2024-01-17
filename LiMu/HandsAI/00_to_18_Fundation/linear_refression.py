# -*- coding: utf-8 -*-
# @Time    : 2024/1/14 18:06
# @Author  : Karry Ren

"""Realization of the linear_regression by some modules of torch."""

import torch
import torch.utils.data as data
from linear_regression_scrach import synthetic_data

if __name__ == '__main__':
    BATCH_SIZE = 10
    LR = 0.035
    NUM_EPOCH = 10

    # ---- Step 1. Set the true w & b, then generate the dataset ---- #
    # here, we have two features
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # ---- Step 2. Generate the dataset & dataloader ---- #
    dataset = data.TensorDataset(*(features, labels))
    data_loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---- Step 3. Construct the Model & Loss Function & Optimizer ---- #
    model = torch.nn.Linear(2, 1, bias=True)
    loss = torch.nn.MSELoss(reduction="mean")

    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

    # ---- Step 4. Begin train & valid ---- #
    for epoch in range(NUM_EPOCH):
        # Train (Different epoch have different sequence to read data !!)
        model.train()
        for X, y in data_loader:
            # 0. zero_grad, pytorch will accumulate the grad
            # MUST do this !
            optimizer.zero_grad()
            # 1. forward
            y_pred = model(X)
            # 2. compute loss
            l = loss(y_pred, y)
            # 3. backward to compute grad
            l.backward()
            # 4. update the param
            optimizer.step()
        # Valid the model
        model.eval()
        with torch.no_grad():
            prediction = model.forward(features)
            train_l = loss(prediction, labels)
            print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")

    print(f"the error of `w`: {((true_w - model.weight.reshape(true_w.shape)) / true_w).data * 100} %")
    print(f"the error of `b`: {((true_b - model.bias) / true_b).data * 100} %")

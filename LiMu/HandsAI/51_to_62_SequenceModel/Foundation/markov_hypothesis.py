# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 14:00
# @Author  : Karry Ren

""" Auto self-regression model based on the markov hypothesis.
    Just a demo for `sin` prediction.

Really hard to predict multi-steps.

"""

import torch
from torch.utils import data
from torch import nn
import numpy as np

# ---- Step 1. Create dataset ---- #
# - generate the data
T = 1000  # 1000 points
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))  # a sin function with gaussian noise
# - generate the dataset
tau = 4  # the tau in hypothesis is 4
features = torch.zeros((T - tau, tau))  # features.shape = (T - tau, tau)
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))  # labels.shape = (T - tau, 1)
# - construct the dataloader
batch_size, n_train = 16, 600
train_dataset = data.TensorDataset(features[:n_train], labels[:n_train])
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = data.TensorDataset(features[n_train:], labels[n_train:])
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---- Step 2. Create Model and Loss ---- #
model = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), 0.01)

# ---- Step 3. Train ---- #
epochs = 10
loss_one_epoch = []
for epoch in range(epochs):
    for X, y in train_dataloader:
        optimizer.zero_grad()
        pred = model(X)
        l = loss(pred, y)
        loss_one_epoch.append(l.item())
        l.backward()
        optimizer.step()
    print(f"epoch {epoch + 1}, loss: {np.mean(loss_one_epoch)}")

# ---- Step 4. Test the model ---- #
""" You have the following 2 ways to test the model:
    1. Just do the test in test dataset which means one step prediction.
    2. Do the multi-steps prediction. (The error is accumulating !)
"""
model.eval()
...

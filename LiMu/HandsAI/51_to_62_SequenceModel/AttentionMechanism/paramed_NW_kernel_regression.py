# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 13:08
# @Author  : Karry Ren

""" The Nadaraya-Watson kernel regression with param. """

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
y_test_truth = f(x_test)  # shape=(n_train)


# ---- Step 2. Construct the model ---- #
class NWKernelRegression(nn.Module):
    """ The Nadaraya-Watson kernel regression with param. """

    def __init__(self, **kwargs):
        """ Init the model. """

        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
        self.attention_weights = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ Forward function.

        :param q: the queries, shape=(n_q)
        :param k: the keys, shape=(n_k)
        :param v: the values, shape=(n_v)

        """

        # ---- Step 1. Repeat queries to (n_q, n_k)
        q_repeat = q.repeat_interleave(k.shape[-1]).reshape((-1, k.shape[-1]))

        # ---- Step 2. Compute the weight, shape=(n_q, n_k) ---- #
        attention_weights = nn.functional.softmax(-((q_repeat - k) * self.w) ** 2 / 2, dim=1)
        self.attention_weights = attention_weights

        # ---- Step 3. Weighted the value ---- #
        output = torch.matmul(attention_weights.unsqueeze(1), v.unsqueeze(-1)).reshape(-1)

        return output


# ---- Step 3. Train Model ---- #
net = NWKernelRegression()
loss = nn.MSELoss(reduction="none")
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = net(x_train, x_train, y_train)
    l = loss(y_pred, y_train)
    l.sum().backward()
    optimizer.step()
    print(f"epoch {epoch + 1}, loss {float(l.sum()):.6f}")

# ---- Step 4. Predict the Model ---- #
y_test_pred = net(x_test, x_train, y_train)
plt.figure(figsize=(15, 6))
plt.plot(y_test_truth.detach().numpy(), label="y_test_true", color="g")
plt.plot(y_test_pred.detach().numpy(), label="y_test_pred", color="b")
plt.legend()
plt.show()

# ---- Step 5. Save the image ---- #
print(net.w)
attention_weight = net.attention_weights.detach().numpy()
plt.imshow(net.attention_weights.detach().numpy())
plt.savefig("a.png")

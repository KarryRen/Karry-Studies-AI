# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 14:23
# @Author  : Karry Ren

""" In `main.py`, I write the train and test function.

Because the time limitation, I will not do the train and test,
  and just write the model in Models.

"""

from typing import List
import torch
from torch import nn
from TextDataset.text_dataset import SeqDataLoader
from Models.RNN.rnn_scratch import RNNModelScratch, get_params, init_rnn_state, rnn


# ---- Step 0. Some tool functions ---- #
def sgd(params: List[torch.Tensor], lr: float, batch_size: int):
    """ Batch SGD optimizing algorithm.

    :param params: the list of all params
    :param lr: the learning rate
    :param batch_size: the batch size

    """

    with torch.no_grad():
        for param in params:
            # update param
            param -= lr * param.grad / batch_size
            # zero the grad of param
            param.grad.zero_()


def grad_clipping(model, theta: int):
    """ The grad clipping, a training trick.

    :param model: the model
    :param theta: the clipping theta

    """

    # ---- Collect all params ---- #
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params

    # ---- Compute the norm ---- #
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))

    # ----- Clip the grid ---- #
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# ---- Step 1. Load data ---- #
batch_size, num_steps = 32, 35
train_loader = SeqDataLoader(batch_size, num_steps)
vocab = train_loader.vocab

# ---- Step 2. Define the model and loss function ---- #
hidden_size = 512
device = torch.device("cpu")
model = RNNModelScratch(len(vocab), hidden_size, device, get_params, init_rnn_state, rnn)
loss = nn.CrossEntropyLoss()
optimizer = sgd

# ---- Step 3. Train the model ---- #
num_epoch, lr = 500, 1
for epoch in range(num_epoch):
    state = None
    for X, Y in train_loader:
        # - each iter will init the state, or detach it
        if state is None:
            state = model.begin_state(batch_size)
        else:
            for s in state:
                s.detach_()
        # - forward computing
        label = Y.T.reshape(-1)  # reshape from (bs, time_steps) to (time_steps * bs)
        X, label = X.to(device), label.to(device)
        pred, state = model(X, state)  # shape=(time_steps * bs, vocab_size)
        l = loss(pred, label.long()).mean()  # compute the loss and mean
        # - backward and update params
        l.backward()
        grad_clipping(model, 1)
        optimizer(params=model.params, lr=lr, batch_size=1)


# --- Step 4. Predict the model ---- #
def predict(prefix: str, pred_steps: int, model, vocab, device: torch.device):
    """ Predict the sentence of prefix. A greate way. """

    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    for y in prefix[1:]:  # pre-fix and get the hidden state
        _, state = model(get_input(), state)
        outputs.append(vocab[y])

    for _ in range(pred_steps):  # for-loop to predict
        y, state = model(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))

    return "".join([vocab.idx_to_token[i] for i in outputs])

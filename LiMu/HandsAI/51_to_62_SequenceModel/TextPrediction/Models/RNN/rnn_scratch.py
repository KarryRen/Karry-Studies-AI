# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 14:53
# @Author  : Karry Ren

""" The RNN Model from scratch. """

import torch
import torch.nn.functional as F


def get_params(vocab_size: int, hidden_size: int, device: torch.device) -> list:
    """ Get the params for rnn. Just two parts:
        - The hidden state, 3 params.
        - The output, 2 params.

    :param vocab_size: the size of vocab, which is the input and output size
    :param hidden_size: the size of hidden layer
    :param device: the computing device

    return: the list of 5 params.

    """

    input_size = output_size = vocab_size

    # ---- Define the params for hidden layer ---- #
    W_xh = torch.randn(size=(input_size, hidden_size), device=device) * 0.01
    W_hh = torch.randn(size=(hidden_size, hidden_size), device=device) * 0.01
    b_h = torch.zeros(hidden_size, device=device)

    # ---- Define the params for output layer ---- #
    W_hq = torch.randn(size=(hidden_size, output_size), device=device) * 0.01
    b_q = torch.zeros(output_size, device=device)

    # ----- Make all params require grad ---- #
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size: int, hidden_size: int, device: torch.device) -> tuple:
    """ Init the state of rnn.

    :param batch_size: the batch size
    :param hidden_size: the size of hidden layer
    :param device: the computing device

    return: the tuple of init state (only one item for rnn).

    """

    return (torch.zeros((batch_size, hidden_size), device=device),)


def rnn(inputs: torch.Tensor, state: tuple, params: list) -> tuple:
    """ The forward function of rnn.

    :param inputs: the input feature, shape=(time_steps, bs, vocab_size).
    :param state: the hidden state of last timestep.
    :param params: the list of params

    return: the tuple:
        - output: the output of each step, shape=(time_steps, bs, output_size)
        - final_state: the final state, a tuple (just one item, shape=(bs, hidden_size))

    """

    # ---- Get the param and hidden_state ---- #
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state

    # ---- For loop the time_steps to get the output for each step ---- #
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # shape=(bs, hidden_size)
        Y = torch.mm(H, W_hq) + b_q  # shape=(bs, output_size)
        outputs.append(Y)
    output = torch.cat(outputs, dim=0)
    final_state = (H,)
    return output, final_state


class RNNModelScratch:
    """ The RNN Model from Scratch. """

    def __init__(self, vocab_size: int, hidden_size: int, device: torch.device,
                 get_params, init_state, forward_fn):
        """ Init the rnn model.

        :param vocab_size: the size of vocab (one hot dim)
        :param hidden_size: the hidden size
        :param device: the computing device
        :param get_params: the get param way
        :param init_state: the init state way
        :param forward_fn: the forward function

        """

        self.vocab_size, self.hidden_size = vocab_size, hidden_size
        self.device = device
        self.params = get_params(vocab_size, hidden_size, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X: torch.Tensor, state: tuple):
        """ Forwarding.

        :param X: the input feature, shape=(bs, time_steps)
        :param state: the init state

        """

        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size: int):
        """ Init the rnn model.

        :param batch_size: the batch size

        """

        return self.init_state(batch_size, self.hidden_size, self.device)


if __name__ == '__main__':
    hidden_size = 512
    device = torch.device("cpu")
    model = RNNModelScratch(28, hidden_size, device, get_params, init_rnn_state, rnn)

    batch_size, num_steps = 32, 35
    X = torch.randint(0, 10, (batch_size, num_steps))
    state = model.begin_state(batch_size)
    Y = model(X.to(device), state)
    print(Y[0].shape)

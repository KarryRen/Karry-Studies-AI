# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 14:53
# @Author  : Karry Ren

""" The RNN Model from scratch. """

import torch
import torch.nn.functional as F


def rnn(inputs: torch.Tensor, state: tuple, params: list) -> tuple:
    """ The forward function of rnn.

    :param inputs: the input feature, shape=(time_steps, bs, vocab_size).
    :param state: the hidden state of last timestep.
        - a tuple, just have one item, default is torch.zeros(bs, hidden_size)
    :param params: the list of params
        - W_xh, shape=(input_size, hidden_size)
        - W_hh, shape=(hidden_size, hidden_size)
        - b_h, shape=(hidden_size)
        - W_hq, shape=(hidden_size, output_size)
        - b_q, shape=(hidden_size)

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
        # core operation of rnn
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)  # shape=(bs, hidden_size)
        Y = torch.mm(H, W_hq) + b_q  # shape=(bs, output_size)
        outputs.append(Y)
    output = torch.cat(outputs, dim=0)
    final_state = (H,)
    return output, final_state

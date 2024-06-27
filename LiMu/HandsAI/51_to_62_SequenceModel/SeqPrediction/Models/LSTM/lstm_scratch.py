# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:35
# @Author  : Karry Ren

""" The LSTM Model from scratch. """

import torch


def lstm(inputs: torch.Tensor, state: tuple, params: list) -> tuple:
    """ The forward function of rnn.

    :param inputs: the input feature, shape=(time_steps, bs, vocab_size).
    :param state: the hidden state of last timestep.
        - a tuple, just have one item, default is torch.zeros(bs, hidden_size)
    :param params: the list of params
        - W_xi, shape=(input_size, hidden_size)
        - W_hi, shape=(hidden_size, hidden_size)
        - b_i, shape=(hidden_size)
        - W_xf, shape=(input_size, hidden_size)
        - W_fi, shape=(hidden_size, hidden_size)
        - b_f, shape=(hidden_size)
        - W_xo, shape=(input_size, hidden_size)
        - W_ho, shape=(hidden_size, hidden_size)
        - b_o, shape=(hidden_size)

        - W_hq, shape=(hidden_size, output_size)
        - b_q, shape=(hidden_size)

    return: the tuple:
        - output: the output of each step, shape=(time_steps, bs, output_size)
        - final_state: the final state, a tuple (just one item, shape=(bs, hidden_size))

    """

    # ---- Get the param and hidden_state ---- #
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state
    outputs = []

    # ---- For loop the time_steps to get the output for each step ---- #
    for X in inputs:
        # -- Step 1. Three Gates -- #
        # input gate
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        # forget gate
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        # output gate
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        # -- Step 2. C tilda computation -- #
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        # -- Step 3. C Computation -- #
        C = F * C + I * C_tilda
        # -- Step 4. New Hidden State -- #
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

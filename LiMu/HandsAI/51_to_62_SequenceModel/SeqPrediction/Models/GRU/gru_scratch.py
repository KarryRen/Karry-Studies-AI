# -*- coding: utf-8 -*-
# @Time    : 2024/3/8 17:16
# @Author  : Karry Ren

""" The GRU Model from scratch. """

import torch


def gru(inputs: torch.Tensor, state: tuple, params: list):
    """ The forward function of gru.

    :param inputs: the input feature, shape=(time_steps, bs, vocab_size).
    :param state: the hidden state of last timestep.
        - a tuple, just have one item, default is torch.zeros(bs, hidden_size)
    :param params: the list of params
        - W_xz, shape=(input_size, hidden_size)
        - W_hz, shape=(hidden_size, hidden_size)
        - b_z, shape=(hidden_size)
        - W_xr, shape=(input_size, hidden_size)
        - W_hr, shape=(hidden_size, hidden_size)
        - b_r, shape=(hidden_size)
        - W_xh, shape=(input_size, hidden_size)
        - W_hh, shape=(hidden_size, hidden_size)
        - b_h, shape=(hidden_size)
        - W_hq, shape=(hidden_size, output_size)
        - b_q, shape=(hidden_size, output_size)

    return: the tuple:
        - output: the output of each step, shape=(time_steps, bs, output_size)
        - final_state: the final state, a tuple (just one item, shape=(bs, hidden_size))

    """

    # ---- Get the param and hidden_state ---- #
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    # ---- For loop the time_steps to get the output for each step ---- #
    for X in inputs:
        # -- Step 1. Two Gates -- #
        # update gate
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        # reset gate
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        # -- Step 2. Compute the tilda hidden state -- #
        # tilda hidden state
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        # -- Step 3. New hidden state -- #
        # the new hidden state
        H = Z * H + (1 - Z) * H_tilda
        # the output
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

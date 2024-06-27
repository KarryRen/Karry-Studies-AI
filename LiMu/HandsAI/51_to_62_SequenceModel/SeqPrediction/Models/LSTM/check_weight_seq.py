# -*- coding: utf-8 -*-
# @Time    : 2024/6/27 17:00
# @Author  : Karry Ren

""" Eval the torch lstm model computing way. Check the weight sequence. """

import torch
from torch import nn

from lstm_scratch import LSTM

if __name__ == "__main__":
    # ---- Step 1. Fix the random seed ---- #
    torch.manual_seed(0)

    # ---- Step 2. Define the input data & params ---- #
    batch_size, lstm_layer_num, input_size, hidden_size = 512, 1, 16, 30
    input_tensor = torch.ones((512, 5, 16))  # shape=(bs, time_steps, d)
    init_state = (torch.zeros((lstm_layer_num, batch_size, hidden_size)), torch.zeros((lstm_layer_num, batch_size, hidden_size)))

    # ---- Step 3. Define the model ---- #
    lstm_torch = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_layer_num, batch_first=True)
    lstm_scratch = LSTM(input_size=input_size, hidden_size=hidden_size)

    # ---- Step 4. Set param value ---- #
    # get value
    weight_ih_l0, bias_ih_l0 = lstm_torch.weight_ih_l0, lstm_torch.bias_ih_l0
    weight_hh_l0, bias_hh_l0 = lstm_torch.weight_hh_l0, lstm_torch.bias_hh_l0
    # transpose
    weight_ih_l0, weight_hh_l0 = weight_ih_l0.T, weight_hh_l0.T
    # set value
    seq_try = [
        (4, 3, 2, 1), (4, 3, 1, 2), (4, 2, 3, 1), (4, 2, 1, 3), (4, 1, 2, 3), (4, 1, 3, 2),
        (3, 4, 2, 1), (3, 4, 1, 2), (3, 2, 3, 1), (3, 2, 1, 3), (3, 1, 2, 3), (3, 1, 3, 2),
        (2, 4, 3, 1), (2, 4, 1, 3), (2, 3, 4, 1), (2, 3, 1, 4), (2, 1, 4, 3), (2, 1, 3, 4),
        (1, 4, 3, 2), (1, 4, 2, 3), (1, 3, 4, 2), (1, 3, 2, 4), (1, 2, 4, 3), (1, 2, 3, 4)
    ]
    loss_array = torch.zeros((24, 3))
    for i, seq in enumerate(seq_try):
        lstm_scratch.W_xf, lstm_scratch.W_hf = (weight_ih_l0[:, hidden_size * (seq[0] - 1):hidden_size * seq[0]],
                                                weight_hh_l0[:, hidden_size * (seq[0] - 1):hidden_size * seq[0]])
        lstm_scratch.b_xf[0], lstm_scratch.b_hf[0] = (bias_ih_l0[hidden_size * (seq[0] - 1):hidden_size * seq[0]],
                                                      bias_hh_l0[hidden_size * (seq[0] - 1):hidden_size * seq[0]])
        lstm_scratch.W_xi, lstm_scratch.W_hi = (weight_ih_l0[:, hidden_size * (seq[1] - 1):hidden_size * seq[1]],
                                                weight_hh_l0[:, hidden_size * (seq[1] - 1):hidden_size * seq[1]])
        lstm_scratch.b_xi[0], lstm_scratch.b_hi[0] = (bias_ih_l0[hidden_size * (seq[1] - 1):hidden_size * seq[1]],
                                                      bias_hh_l0[hidden_size * (seq[1] - 1):hidden_size * seq[1]])
        lstm_scratch.W_xc, lstm_scratch.W_hc = (weight_ih_l0[:, hidden_size * (seq[2] - 1):hidden_size * seq[2]],
                                                weight_hh_l0[:, hidden_size * (seq[2] - 1):hidden_size * seq[2]])
        lstm_scratch.b_xc[0], lstm_scratch.b_hc[0] = (bias_ih_l0[hidden_size * (seq[2] - 1):hidden_size * seq[2]],
                                                      bias_hh_l0[hidden_size * (seq[2] - 1):hidden_size * seq[2]])
        lstm_scratch.W_xo, lstm_scratch.W_ho = (weight_ih_l0[:, hidden_size * (seq[3] - 1):hidden_size * seq[3]],
                                                weight_hh_l0[:, hidden_size * (seq[3] - 1):hidden_size * seq[3]])
        lstm_scratch.b_xo[0], lstm_scratch.b_ho[0] = (bias_ih_l0[hidden_size * (seq[3] - 1):hidden_size * seq[3]],
                                                      bias_hh_l0[hidden_size * (seq[3] - 1):hidden_size * seq[3]])

        # ---- Step 3. Forward computation & comparing ---- #
        y_torch, h_torch = lstm_torch(input=input_tensor, hx=init_state)
        y_scratch, h_scratch = lstm_scratch(input=input_tensor, hx=(init_state[0][0], init_state[1][0]))
        print(
            f"{i}, {seq}:     "
            f"{(y_torch - y_scratch).sum() / y_scratch.sum()}, "
            f"{(h_torch[0][0] - h_scratch[0]).sum() / h_scratch[0].sum()},"
            f" {(h_torch[1][0] - h_scratch[1]).sum() / h_scratch[1].sum()}"
        )
        loss_array[i][0], loss_array[i][1], loss_array[i][2] = (y_torch - y_scratch).sum(), (h_torch[0][0] - h_scratch[0]).sum(), (
                h_torch[1][0] - h_scratch[1]).sum()

    print(torch.abs(loss_array[:, 0]).argmin(), torch.abs(loss_array[:, 1]).argmin(), torch.abs(loss_array[:, 2]).argmin())

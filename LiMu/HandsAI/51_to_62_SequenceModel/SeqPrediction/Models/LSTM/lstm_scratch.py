# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:35
# @Author  : Karry Ren

""" The LSTM Model from scratch. """

import torch
from typing import Tuple
import math


class LSTM:
    """ The LSTM Model from scratch. """

    def __init__(self, input_size: int, hidden_size: int):
        """ Init of the scratch LSTM Model.
        :param input_size: The input size of the LSTM Model.
        :param hidden_size: The hidden size of the LSTM Model.

        NOTE:
            - All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
                where :math:`k = \frac{1}{\text{hidden\_size}}`

        """

        stdv = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size

        # ---- Define the params ---- #
        # - input gate
        self.W_xi = torch.Tensor(input_size, hidden_size).uniform_(-stdv, stdv)
        self.b_xi = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        self.W_hi = torch.Tensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        self.b_hi = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        # - forget gate
        self.W_xf = torch.Tensor(input_size, hidden_size).uniform_(-stdv, stdv)
        self.b_xf = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        self.W_hf = torch.Tensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        self.b_hf = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        # - output gate
        self.W_xo = torch.Tensor(input_size, hidden_size).uniform_(-stdv, stdv)
        self.b_xo = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        self.W_ho = torch.Tensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        self.b_ho = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        # - c tilda
        self.W_xc = torch.Tensor(input_size, hidden_size).uniform_(-stdv, stdv)
        self.b_xc = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)
        self.W_hc = torch.Tensor(hidden_size, hidden_size).uniform_(-stdv, stdv)
        self.b_hc = torch.Tensor(1, hidden_size).uniform_(-stdv, stdv)

    def __call__(self, input: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Forward of the scratch LSTM Model.

        :param input: The input to the LSTM Model.
        :param hx: The hidden state of the LSTM Model.

        :return:
            - outputs: shape=(bs, time_steps, hidden_size)
            - H: shape=(bs, hidden_size)
            - C: shape=(bs, hidden_size)

        """

        # ---- Get the hidden_state & shape ---- #
        (H, C) = hx
        bs, time_steps, input_size = input.shape  # get the shape
        outputs = torch.zeros(bs, time_steps, self.hidden_size)  # the output tensor

        # ---- For loop the time_steps to get the output for each step ---- #
        for t in range(time_steps):
            # Get the `X` of t
            X = input[:, t, :]  # shape=(bs, input_size)
            # Three gates operation:
            # - input gate
            I = torch.sigmoid((X @ self.W_xi) + self.b_xi + (H @ self.W_hi) + self.b_hi)  # shape=(bs, hidden_size)
            # - forget gate
            F = torch.sigmoid((X @ self.W_xf) + self.b_xf + (H @ self.W_hf) + self.b_hf)  # shape=(bs, hidden_size)
            # - output gate
            O = torch.sigmoid((X @ self.W_xo) + self.b_xo + (H @ self.W_ho) + self.b_ho)  # shape=(bs, hidden_size)
            # C tilda:
            C_tilda = torch.tanh((X @ self.W_xc) + self.b_xc + (H @ self.W_hc) + self.b_hc)  # shape=(bs, hidden_size)
            # C:
            C = F * C + I * C_tilda  # shape=(bs, hidden_size)
            # Hidden state:
            H = O * torch.tanh(C)  # shape=(bs, hidden_size)
            # Set back
            outputs[:, t, :] = H

        # ---- Return the output & state ---- #
        return outputs, (H, C)

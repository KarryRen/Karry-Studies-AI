# -*- coding: utf-8 -*-
# @Time    : 2024/3/10 14:16
# @Author  : Karry Ren

""" The most important part of attention mechanism (attention score). """

import torch
from torch import nn
import math


class AdditiveAttention(nn.Module):
    """ The Additive-Attention mechanism. """

    def __init__(self, key_size: int, query_size: int, hidden_size: int, dropout: float = 0.1, **kwargs):
        """ The init function of additive attention.

        :param key_size: the size of key
        :param query_size: the size of query
        :param hidden_size: the size of hidden layer
        :param dropout: the dropout ratio

        """

        super(AdditiveAttention, self).__init__(**kwargs)

        # ---- Three Params ---- #
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, 1, bias=False)

        # ---- Dropout for regulation ---- #
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        """ Forward function.

        :param queries: shape=(bs, n_q, q)
        :param keys: shape=(bs, n_k, k)
        :param values: shape=(bs, n_v, v)

        return:
            - output, shape=(bs, n_q, v)

        NOTE:
            - the n_k must = n_v
            - the dim q can != dim k
        """

        # ---- Step 1. Linear q & k ---- #
        queries = self.W_q(queries)  # shape=(bs, n_q, hidden_size)
        keys = self.W_k(keys)  # shape=(bs, n_k, hidden_size)

        # ---- Step 2. Add and tanh ---- #
        # Most Important Operation for this function, broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)  # shape=(bs, n_q, n_k, hidden_size)
        features = torch.tanh(features)

        # ---- Step 3. Get the score ---- #
        scores = self.w_v(features).squeeze(-1)  # shape=(bs, n_q, n_k)

        # ---- Step 4. Use the softmax to get the attention weight ---- #
        attention_weights = nn.functional.softmax(scores, dim=-1)  # shape=(bs, n_q, n_k)

        # ---- Step 5. Weighted the values ---- #
        output = torch.bmm(self.dropout(attention_weights), values)  # shape=(ns, n_q, v)
        return output


class DotProductAttention(nn.Module):
    """ The Scaled Dot-product Attention mechanism. """

    def __init__(self, dropout: float = 0.1, **kwargs):
        """ The init function of dot-product attention.

        :param dropout: the dropout rate.

        """
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        """ Forward function.

        :param queries: shape=(bs, n_q, d)
        :param keys: shape=(bs, n_k, d)
        :param values: shape=(bs, n_v, v)

        return:
            - output, shape=(bs, n_q, v)

        NOTE:
            - the n_k must = n_v
            - the dim q must = dim k
        """

        # ---- Step 1. Get the d ---- #
        d = queries.shape[-1]

        # ---- Step 2. Use the function to compute the attention score ---- #
        # shape=(bs, n_q, n_k)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # ---- Step 3. Get the attention weights ---- #
        attention_weights = nn.functional.softmax(scores, dim=-1)

        # ---- Step 4. Weighted the values ---- #
        output = torch.bmm(self.dropout(attention_weights), values)
        return output

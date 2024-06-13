# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 17:54
# @Author  : Karry Ren

""" The Encoder-Decoder Model with attention mechanism. """

import torch
from torch import nn

class Encoder(nn.Module):
    """ The Encoder (Same with Seq2Seq Model). """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0, **kwargs):
        """ Init of encoder.

        :param input_size: the dimension of input feature
        :param hidden_size: the hidden size
        :param num_layers: the num of layers
        :param dropout: the dropout rate

        """

        super(Encoder, self).__init__(**kwargs)

        # build the rnn for encoder
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, X: torch.Tensor, *args):
        """ Forward of encoder.

        :param X: the source seq feature, shape=(bs, num_steps, input_size)

        return:
            - output, shape=(time_steps, bs, hidden_size)
            - state, shape=(num_layers, bs, hidden_size)
        """

        # ---- Step 0. Transpose to steps first ---- #
        X = X.permute(1, 0, 2)  # shape=(num_steps, bs, input_size)

        # ---- Step 1. RNN computation ----- #
        output, state = self.rnn(X)
        # output.shape=(num_steps, bs, hidden_size)
        # state.shape=(num_layers, bs, hidden_size), if LSTM state is a tuple of (h, c)

        return output, state


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


class AttentionDecoder(nn.Module):
    """ The Decoder. """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0,
                 **kwargs):
        """ Init of decoder.

        :param input_size: the dimension of input feature
        :param hidden_size: the hidden size (MUST same as encoder hidden_size)
        :param num_layers: the num of layers
        :param output_size: the output size
        :param dropout: the dropout rate

        """

        super(AttentionDecoder, self).__init__(**kwargs)
        # attention function
        self.attention = AdditiveAttention(key_size=hidden_size, query_size=hidden_size, hidden_size=hidden_size)

        # build the rnn for decoder
        self.rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers, dropout=dropout)

        # build the dense for output
        self.dense = nn.Linear(hidden_size, output_size)

    def init_state(self, enc_outputs: tuple, *args):
        """ Get the init state of decoder from encoder_outputs.

        :param enc_outputs: the output of encoder (based ont the encoder)

        return:
            - outputs: the output of encoder, shape=(time_steps, bs, hidden_size)
            - hidden_state: the last state of encoder, a tuple

        """

        outputs, hidden_state = enc_outputs

        return outputs.permute(1, 0, 2), hidden_state

    def forward(self, X: torch.Tensor, state: torch.Tensor):
        """ The forward of decoder.

        :param X: the target seq feature, shape=(bs, time_steps, input_size)
        :param state: the init state of decoder, from `init_state()` function.

        return:
            - output, shape=(bs, time_steps, output_size)

        """

        # ---- Step 0. Transpose to steps first ---- #
        X = X.permute(1, 0, 2)  # shape=(num_steps, bs, input_size)

        # ---- Step 1. Get init state ---- #
        enc_outputs, hidden_state = state

        # ---- Step 2. For loop to compute  ---- #
        outputs = []
        for x in X:
            # set the latest hidden_state as the query
            query = torch.unsqueeze(hidden_state[-1], dim=1)  # shape=(bs, 1, hidden_size)
            # get the context, use the query to attention with the enc_outputs
            # the core difference between seq2seq model
            context = self.attention(query, enc_outputs, enc_outputs)  # shape=(bs, 1, hidden_size)
            # cat the context to x
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)  # shape=(bs, 1, input_size)
            # transpose x to (bs, 1, input_size) and do rnn operation
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            # out.shape=(1, bs, hidden_size)
            # hidden_state.shape=(1, bs, hidden_size), if LSTM state is a tuple of (h, c)
            outputs.append(out)

        # ---- Step 3. Get the outputs ---- #
        outputs = self.dense(torch.cat(outputs, dim=0))  # shape=(time_steps, bs, output_size)
        return outputs.permute(1, 0, 2)


class EncoderDecoder(nn.Module):
    """ The Encoder-Decoder Model Framework. """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        """ Init of the Encoder-Decoder Model.

        :param encoder: the encoder
        :param decoder: the decoder

        """
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X: torch.Tensor, dec_X: torch.Tensor, *args):
        """ Forward of the Encoder-Decoder Model.

        :param enc_X: the input feature of encoder
        :param dec_X: the input feature of decoder

        return: the decoder result

        """

        # ---- Step 1. Encoding the source feature ---- #
        enc_outputs = self.encoder(enc_X, *args)

        # ---- Step 2. Get the init state of decoder ---- #
        dec_state = self.decoder.init_state(enc_outputs, *args)

        # ---- Step 3. Get the output of decoder based on two inputs ---- #
        output = self.decoder(dec_X, dec_state)
        return output


if __name__ == '__main__':
    time_steps = 2
    bs, input_size, hidden_size, output_size = 32, 32, 64, 32
    enc_x = torch.randn((bs, time_steps, input_size))
    dec_x = torch.randn((bs, time_steps, input_size))

    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=1)
    decoder = AttentionDecoder(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=output_size)

    model = EncoderDecoder(encoder, decoder)

    y = model(enc_x, dec_x)
    print(y.shape)

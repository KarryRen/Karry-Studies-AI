# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 16:39
# @Author  : Karry Ren

""" The Framework of encoder-decoder model.
Take the rnn as an example.

"""

import torch
from torch import nn


class Encoder(nn.Module):
    """ The Encoder. """

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


class Decoder(nn.Module):
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

        super(Decoder, self).__init__(**kwargs)

        # build the rnn for decoder
        self.rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers, dropout=dropout)

        # build the dense for output
        self.dense = nn.Linear(hidden_size, output_size)

    def init_state(self, enc_outputs: tuple, *args):
        """ Get the init state of decoder from encoder_outputs.

        :param enc_outputs: the output of encoder (based ont the encoder)

        """

        return enc_outputs[1]

    def forward(self, X: torch.Tensor, state: torch.Tensor):
        """ The forward of decoder.

        :param X: the target seq feature, shape=(bs, time_steps, input_size)
        :param state: the init state of decoder, from `init_state()` function.

        return:
            - output, shape=(bs, time_steps, output_size)

        """

        # ---- Step 0. Transpose to steps first ---- #
        X = X.permute(1, 0, 2)  # shape=(num_steps, bs, input_size)

        # ---- Step 1. Repeat the state to each time_steps ---- #
        context = state[-1].repeat(X.shape[0], 1, 1)  # shape=(time_steps, bs, hidden_size)

        # ---- Step 2. Cat the context to X ---- #
        X_and_context = torch.cat((X, context), 2)  # shape=(time_steps, bs, hidden_size + input_size)

        # ---- Step 3. Decode the feature init by (state) ---- #
        output, state = self.rnn(X_and_context, state)
        # output.shape=(num_steps, bs, hidden_size)
        # state.shape=(num_layers, bs, hidden_size), if LSTM state is a tuple of (h, c)

        # ---- Step 4. Fully Connected to get the output ---- #
        output = self.dense(output).permute(1, 0, 2)  # shape=(bs, num_steps, hidden_size)
        return output


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
    bs, input_size, hidden_size, output_size = 3, 32, 64, 32
    enc_x = torch.randn((bs, time_steps, input_size))
    dec_x = torch.randn((bs, time_steps, input_size))

    encoder = Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=1)
    decoder = Decoder(input_size=input_size, hidden_size=hidden_size, num_layers=1, output_size=output_size)

    model = EncoderDecoder(encoder, decoder)

    y = model(enc_x, dec_x)
    print(y.shape)

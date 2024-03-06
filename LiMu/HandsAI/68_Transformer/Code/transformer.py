# -*- coding: utf-8 -*-
# @Time    : 2023/11/30 19:22
# @Author  : Karry Ren

""" The transformer network.
    Include the encoder and decoder, here are also train and predict.

"""

import math
import torch
from torch import nn
from typing import Optional
from modules import PositionalEncoding, TransformerEncoderBlock, TransformerDecoderBlock


class TransformerEncoder(nn.Module):
    """ The Transformer encoder. """

    def __init__(self, input_size: int, num_layers: int,
                 query_size: int, key_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, norm_shape: list,
                 ffn_num_hiddens: int, dropout: float, use_bias: bool = False):
        """ Init Transformer Encoder.

        :param input_size: the size of input vector
        :param num_layers: the num of encoder blocks
        :param query_size: the size of q
        :param key_size: the size of k
        :param value_size: the size of v
        :param num_hiddens: the d_model
        :param num_heads: the heads num of attention
        :param norm_shape: the norm shape
        :param ffn_num_hiddens: the hiddens of ffn input
        :param dropout: dropout ratio
        :param use_bias: use bias or not
        """

        super().__init__()
        self.num_hiddens = num_hiddens
        self.attention_weights = [None] * num_layers  # note the attention weights

        # ---- Part 1. Embedding ---- #
        self.embedding = nn.Linear(in_features=input_size, out_features=num_hiddens)
        # ---- Part 2. Positional Encoding ---- #
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        # ---- Part 3. Attention Blocks ---- #
        self.encoder_blocks = nn.Sequential()
        for i in range(num_layers):
            self.encoder_blocks.add_module("encoder block" + str(i),
                                           TransformerEncoderBlock(
                                               query_size=query_size, key_size=key_size, value_size=value_size,
                                               num_hiddens=num_hiddens, num_heads=num_heads,
                                               norm_shape=norm_shape, ffn_num_hiddens=ffn_num_hiddens,
                                               dropout=dropout, use_bias=use_bias))

    def forward(self, X: torch.Tensor, valid_lens: Optional[torch.Tensor] = None):
        """Compute of Encoder.
        :param X: the input, shape=(bs, seq, feature_dim)
        :param valid_lens: shape=(bs)
        :return: the encoded feature

        """

        # ---- Step 1. Embedding and Positional Encoding ---- #
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X)

        # ---- Step 2. For Loop Block Operations ---- #
        for i, encoder_block in enumerate(self.encoder_blocks):
            X = encoder_block(X, valid_lens)
            self.attention_weights[i] = encoder_block.multi_head_attention.attention.attention_weights

        return X


class TransformerDecoder(nn.Module):
    """ The Transformer Decoder. """

    def __init__(self, input_size: int, num_layers: int,
                 query_size: int, key_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, norm_shape: list,
                 ffn_num_hiddens: int, dropout: float):
        """ Init Transformer Encoder.

        :param input_size: the size of input vector
        :param num_layers: the num of encoder blocks
        :param query_size: the size of q
        :param key_size: the size of k
        :param value_size: the size of v
        :param num_hiddens: the d_model
        :param num_heads: the heads num of attention
        :param norm_shape: the norm shape
        :param ffn_num_hiddens: the hiddens of ffn input
        :param dropout: dropout ratio
        :param use_bias: use bias or not

        """

        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.attention_weights = [[None] * num_layers for _ in range(2)]  # shape=(num_layers, 2)

        # ---- Part 1. Embedding ---- #
        self.embedding = nn.Linear(in_features=input_size, out_features=num_hiddens)
        # ---- Part 2. Positional Encoding ---- #
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        # ---- Part 3. Attention Blocks ---- #
        self.decoder_blocks = nn.Sequential()
        for i in range(num_layers):
            self.decoder_blocks.add_module("decoder block" + str(i),
                                           TransformerDecoderBlock(
                                               decoder_block_idx=i,
                                               query_size=query_size, key_size=key_size, value_size=value_size,
                                               num_hiddens=num_hiddens, num_heads=num_heads,
                                               norm_shape=norm_shape, ffn_num_hiddens=ffn_num_hiddens, dropout=dropout))
        # ---- Part 4. FC ---- #
        self.dense = nn.Linear(in_features=num_hiddens, out_features=input_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        """
        :param X: The input of decoder.
            - training, shape=(bs, seq, size)
            - predict, when t, shape=(bs, t-1, size)
        :param state: the init state of decoder.

        """

        # ---- Step 1. Embedding and Positional Encoding ---- #
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        # ---- Step 2. For Loop Block Operations ---- #
        for i, decoder_block in enumerate(self.decoder_blocks):
            X, state = decoder_block(X, state)
            # note the `decoder` self-attention weight
            self.attention_weights[0][
                i] = decoder_block.multi_head_attention1.attention.attention_weights
            # note the `encoder-decoder` self-attention weight
            self.attention_weights[1][
                i] = decoder_block.multi_head_attention2.attention.attention_weights
        return torch.softmax(self.dense(X), dim=-1), state


class EncoderDecoder(nn.Module):
    """ The base class for the encoder--decoder architecture. """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """ Init the encoder and decoder.
        :param encoder: the encoder module
        :param decoder: the decoder module

        """

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        """ How to train. """

        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # return decoder output only, don't get the state
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps, save_attention_weights=False):
        """ How to predict. """

        batch = batch.to(device=device)
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [torch.expand_dims(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(torch.argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return torch.concat(outputs[1:], 1), attention_weights


if __name__ == '__main__':
    batch_size = 3
    num_steps = 5
    input_size = 3
    enc_x = torch.ones((batch_size, num_steps, input_size))
    dec_x = torch.ones((batch_size, num_steps, input_size))
    num_hiddens = 32
    q_size = k_size = v_size = num_hiddens
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    ffn_num_hiddens = 64
    norm_shape = [num_hiddens]

    encoder = TransformerEncoder(input_size=input_size, num_layers=num_layers,
                                 query_size=q_size, key_size=k_size, value_size=v_size,
                                 num_hiddens=num_hiddens, num_heads=num_heads,
                                 norm_shape=norm_shape,
                                 ffn_num_hiddens=ffn_num_hiddens,
                                 dropout=dropout, use_bias=False)
    decoder = TransformerDecoder(input_size=input_size, num_layers=num_layers,
                                 query_size=q_size, key_size=k_size, value_size=v_size,
                                 num_hiddens=num_hiddens, num_heads=num_heads, norm_shape=norm_shape,
                                 ffn_num_hiddens=ffn_num_hiddens, dropout=dropout)
    model = EncoderDecoder(encoder=encoder, decoder=decoder)

    y = model(enc_x, dec_x, torch.Tensor([3, 2, 4]))
    # print(model.predict_step(torch.Tensor((1, 3)), device=torch.device("cpu"), num_steps=4))

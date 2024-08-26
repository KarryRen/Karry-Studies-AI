# -*- coding: utf-8 -*-
# @Time    : 2023/11/30 19:28
# @Author  : Karry Ren

""" The modules of transformer.
Ref. d2l package.

"""

import torch
from torch import nn
import math
from typing import Optional


def masked_softmax(X: torch.Tensor, valid_lens: Optional[torch.Tensor] = None):
    """Perform softmax operation by masking elements on the LAST AXIS.
    Important in Decoder.

    :param X: the 3D attention score Matrix, shape=(bs, seq, seq) when do self-attention
    :param valid_lens: the valid lens, corresponding to the LAST AXIS of X
        - 1D: shape=(bs), when different seq in different batch have the different len
              different q in one batch use the same valid_len
        - 2D: shape=(bs,seq), seldom use
              different q in one batch use the different valid_len

    :return masked(opt.) softmax attention weight (bs, seq, seq)
    """

    def _sequence_mask(X, valid_len, value=0.):
        """Mask the attention score, replace the no valid score to `value`
        :param X: shape=(bs*seq, seq)
        :param valid_len: shape=(bs*seq)
        :param value: the mask value
        :return: the masked attention score, shape=(bs*seq, seq)
        """

        max_len = X.size(1)
        # get the valid range of each seq (bs*seq, seq), very clear !!
        valid_range = torch.arange(max_len, dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        # set the no valid range of X to value
        X[~valid_range] = value
        return X

    # if there is no valid_lens, do the softmax directly
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    # if there is valid lens, use the length to constrict the softmax
    else:
        shape = X.shape  # a tuple (bs, seq, seq)
        if valid_lens.dim() == 1:  # 1D situation, shape=(bs)
            # repeat the valid_len to (bs*seq) for example [1, 2] and bs = 2 to [1, 1, 2, 2]
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:  # 2D situation
            # reshape the valid len to (bs*seq)
            valid_lens = valid_lens.reshape(-1)
        # reshape the X to (bs*seq, seq)
        X = X.reshape(-1, shape[-1])
        # on the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X, valid_lens, value=-1e6)
        # reshape the X back to (bs, seq, seq)
        X = X.reshape(shape)
        return nn.functional.softmax(X, dim=-1)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout: float):
        """Init the Module.

        :param dropout: the dropout ratio
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                valid_lens: Optional[torch.Tensor] = None):
        """Follow the equation of scaled dot production attention in paper.

        :math:`softmax((QK^T) / sqrt(d_k))V`

        :param queries: shape=(bs, seq, q_size)
        :param keys: shape=(bs, seq, k_size=q_size)
        :param values: shape=(bs, seq, v_size)
        :param valid_lens: shape=(bs)
        :return: the attention_result (bs, seq, d_k)
        """

        # get the d_k
        d_k = keys.shape[-1]
        # attention_score = QK^T / sqrt(d_k), which are `MatMul` and `Scale` two steps
        # shape=(bs, seq, seq)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d_k)
        # attention_weights = softmax(mask(attention_scores)), which are `Mask(opt.)` and `Softmax` two steps
        # shape=(bs, seq, seq)
        self.attention_weights = masked_softmax(attention_scores, valid_lens)
        # use the dropout weight to cal weighted attention result, which is the `MatMul steps`
        # shape=(bs, seq, d_k)
        attention_result = torch.bmm(self.dropout(self.attention_weights), values)

        return attention_result


class MultiHeadAttention(nn.Module):
    """Multi-head attention Module."""

    def __init__(self, q_size: int, k_size: int, v_size: int,
                 num_hiddens: int, num_heads: int,
                 dropout: float, use_bias: bool = False):
        """Init the Multi-head attention Module.

        :param q_size: the size of q
        :param k_size: the size of k
        :param v_size: the size of v
        :param num_hiddens: the model dim: :math:`d_{model}`
        :param num_heads: the num of head
        :param dropout: the ratio of drop out
        :param bias: have bias or not in Linear Layers
        """

        super().__init__()
        self.num_heads = num_heads
        # the learnable `Linear Layer` of q, k, v attention
        self.W_q = nn.Linear(in_features=q_size, out_features=num_hiddens, bias=use_bias)
        self.W_k = nn.Linear(in_features=k_size, out_features=num_hiddens, bias=use_bias)
        self.W_v = nn.Linear(in_features=v_size, out_features=num_hiddens, bias=use_bias)
        self.W_o = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=use_bias)
        # the `Attention Mechanism`
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                valid_lens: Optional[torch.Tensor] = None):
        """Different from the equation of Multi-head attention in paper.
           As limu said, to fast the computation, this way is as `parallel` as possible.
           Rather than `for-loop` num_heads times to compute the different attention of different heads.
           And reduce the Linear parma.

        :param queries: shape=(bs, seq, q_size)
        :param keys: shape=(bs, seq, k_size)
        :param values: shape=(bs, seq, v_size)
        :param valid_lens: shape=(bs) or (bs, seq)

        :return: multi_head_attention_output (bs, seq, v_size)
        """

        # ---- Step 1. do the Linear Mapping ---- #
        # shape from (bs, seq, q/k/v_size) to (bs, seq, num_hiddens)
        # 本来这一步应该是有多少个头，就有多少个 W
        # 循环对每个头进行线性投影, 也即进行 n_heads 次投影
        # 构建出 n_heads 个特征 (num_hiddens -> num_hiddens // num_heads)
        # 但是这种循环太耗时, 因此实现的时候往往是先进行一个大的线性投影 (num_hiddens -> num_hiddens)
        # 把这个大线性投影看作多个头的集合, 然后再分开 (Step 2) 放在 bs 维度上做并行计算。
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)

        # ---- Step 2. split them to different head for parallel computation ---- #
        # shape from (bs, seq, num_hiddens) to (bs*num_heads, seq, num_hiddens/num_heads)
        queries = self.transpose_qkv(queries)
        keys = self.transpose_qkv(keys)
        values = self.transpose_qkv(values)
        # the valid_lens should be duplicated too, each head have the same valid_len
        # shape from (bs) to (bs*num_heads)
        if valid_lens is not None:
            # on axis 0, copy the first item (scalar or vector) for num_heads times
            # then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # ---- Step 3. do the Scaled Dot Product Attention, parallel ---- #
        output = self.attention(queries, keys, values, valid_lens)

        # ---- Step 4. concat all heads result ---- #
        # transpose the output from (bs*num_heads, seq, num_hiddens/num_heads) to
        # (bs, seq, num_hiddens)
        output_concat = self.transpose_output(output)
        multi_head_attention_output = self.W_o(output_concat)
        return multi_head_attention_output

    def transpose_qkv(self, X):
        """ Transposition for parallel computation of multiple attention heads.
        Avoid for-loop num_heads times computation.
        Transpose the num_heads dim to bs.

        :param X: the Q, K, V after Linear Mapping, shape=(bs, seq, num_hiddens)
        :return transposed_X, shape=(bs*num_heads, seq, q/k/v_size/num_heads)
        """

        # reshape to (bs, seq, num_heads, num_hiddens/num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # transpose the X to (bs, num_heads, seq, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # reshape to (bs*num_heads, seq, num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""

        # from (bs*num_heads, seq, num_hiddens/num_heads) to (bs, num_heads, seq, num_hiddens/num_heads)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        # from (bs, num_heads, seq, num_hiddens/num_heads) to (bs, seq, num_heads, num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)
        # from (bs, seq, num_heads, num_hiddens/num_heads) to (bs, seq, num_heads, num_hiddens)
        return X.reshape(X.shape[0], X.shape[1], -1)


class AddNorm(nn.Module):
    """ Add & Norm.
    The residual connection followed by layer normalization.
    """

    def __init__(self, norm_shape: list, dropout: float):
        """Init AddNorm Module.

        :param norm_shape: the layer norm shape

        :param dropout: the dropout ratio
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """ Add & LN.
        :param X: the skip connection of raw input
        :param Y: the processed input
        :return: LN(X + dropout(Y))
        """

        return self.ln(X + self.dropout(Y))


class PositionWiseFFN(nn.Module):
    """ The position-wise feed-forward network
        Two layers MLP.
    """

    def __init__(self, ffn_num_input: int, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(in_features=ffn_num_input, out_features=ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(in_features=ffn_num_hiddens, out_features=ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class TransformerEncoderBlock(nn.Module):
    """ The Transformer encoder block Module."""

    def __init__(self, query_size: int, key_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, norm_shape: list,
                 ffn_num_hiddens: int, dropout: float, use_bias: bool = False):
        """ Init Transformer Encoder Block.

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
        # ---- Part 1. the multi-head attention ---- #
        self.multi_head_attention = MultiHeadAttention(q_size=query_size, k_size=key_size, v_size=value_size,
                                                       num_hiddens=num_hiddens, num_heads=num_heads,
                                                       dropout=dropout, use_bias=use_bias)
        # ---- Part 2. residual add and LN ---- #
        self.addnorm1 = AddNorm(norm_shape=norm_shape, dropout=dropout)
        # ---- Part 3. FFN ---- #
        self.ffn = PositionWiseFFN(ffn_num_input=num_hiddens,
                                   ffn_num_hiddens=ffn_num_hiddens,
                                   ffn_num_outputs=num_hiddens)
        # ---- Part 4. residual add and LN ---- #
        self.addnorm2 = AddNorm(norm_shape=norm_shape, dropout=dropout)

    def forward(self, X: torch.Tensor, valid_lens: Optional[torch.Tensor] = None):
        """
        :param X: feature, shape=(bs, seq, num_hiddens)
        :param valid_lens: shape=(bs)
        :return: result after one transformer block, shape=(bs, seq, num_hiddens)
        """

        # ---- Step 1. Multi Heda Attention (self) ---- #
        multi_attentioned_X = self.multi_head_attention(queries=X, keys=X, values=X, valid_lens=valid_lens)
        # ---- Step 2. Add Raw and LN
        Y = self.addnorm1(X, multi_attentioned_X)
        # ---- Step 3. FFN ---- #
        ffn_Y = self.ffn(Y)
        # ---- Step 4. Add and LN
        output = self.addnorm2(Y, ffn_Y)

        return output


class PositionalEncoding(nn.Module):
    """ Positional encoding. Sin-Cos PE way. """

    def __init__(self, num_hiddens: int, dropout: float, max_len=1000):
        """ Init the PE module and PE array based on the function in paper.

        :param num_hiddens: the feature dim.
        :param dropout: the dropout ratio
        :param max_len: the max len, must be longer than seq

        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # create a long enough PE, must longer than seq
        self.PE = torch.zeros((1, max_len, num_hiddens))
        # compute the PE
        # shape=(max_len, 1)
        pos = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        # shape=(num_hiddens/2)
        div = torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        # expand dim, shape = (max_len, num_hiddens/2)
        X = pos / div
        # set value
        self.PE[:, :, 0::2] = torch.sin(X)  # the even feature of each pos
        self.PE[:, :, 1::2] = torch.cos(X)  # the odd feature of each pos

    def forward(self, X):
        """ Add PE. """

        X = X + self.PE[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class TransformerDecoderBlock(nn.Module):
    """ The i_th block of Decoder. """

    def __init__(self, decoder_block_idx: int,
                 query_size: int, key_size: int, value_size: int,
                 num_hiddens: int, num_heads: int, norm_shape: list,
                 ffn_num_hiddens: int, dropout):
        """Init Transformer Encoder Block.

        :param decoder_block_idx: the idx of decoder block
        :param query_size: the size of q
        :param key_size: the size of k
        :param value_size: the size of v
        :param num_hiddens: the d_model
        :param num_heads: the heads num of attention
        :param norm_shape: the norm shape
        :param ffn_num_hiddens: the hiddens of ffn input
        :param dropout: dropout ratio

        """

        super(TransformerDecoderBlock, self).__init__()
        # i is special in decoder
        self.i = decoder_block_idx
        # ---- Patt 1. the first attention block ---- #
        self.multi_head_attention1 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        # ---- Part 2. add & norm ---- #
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # ---- Patt 3. the second attention block ---- #
        self.multi_head_attention2 = MultiHeadAttention(
            query_size, key_size, value_size, num_hiddens, num_heads, dropout)
        # ---- Part 4. add & norm ---- #
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # ---- Part 5. FFN ---- #
        self.ffn = PositionWiseFFN(ffn_num_input=num_hiddens,
                                   ffn_num_hiddens=ffn_num_hiddens,
                                   ffn_num_outputs=num_hiddens)
        # ---- Part 6. Add Norm ---- #
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        """
        :param X: The input of Decoder.
            - training, shape=(bs, seq, feature)
        :param state: The state of encoder, a tuple
            - 0: enc_outputs (bs, seq, num_hiddens)
            - 1: enc_valid_lens (bs, )
            - 2: [None] * self.num_layers when training

        """

        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段, 输出序列的所有词元都在同一时间处理,
        # 因此 state[2][self.i] 初始化为 None。
        # 推理阶段, 输出序列是通过词元一个接着一个解码的, 不断地堆叠, state 不断改变
        # 因此 state[2][self.i] 包含着直到当前时间步第 i 个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if self.training:  # train 阶段构建掩码
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:  # predict 的时候就逐步向后走
            dec_valid_lens = None

        # ---- Step 1. Multi Heda Attention (self) (masked) ---- #
        multi_attentioned_X = self.multi_head_attention1(queries=X, keys=key_values, values=key_values,
                                                         valid_lens=dec_valid_lens)
        # ---- Step 2. Add & Norm ---- #
        Y = self.addnorm1(X, multi_attentioned_X)
        # ---- Step 3. Multi Heda Attention (with encoding out) ---- #
        multi_attentioned_Y = self.multi_head_attention2(queries=Y, keys=enc_outputs, values=enc_outputs,
                                                         valid_lens=enc_valid_lens)
        # ---- Step 4. Add & Norm ---- #
        Z = self.addnorm2(Y, multi_attentioned_Y)
        # ---- Step 5. FFN ---- #
        ffn_Z = self.ffn(Z)
        # ---- Step 6. Add & Norm ---- #
        output = self.addnorm3(Z, ffn_Z)
        return output, state


if __name__ == '__main__':
    num_hiddens = 64
    q_size = k_size = v_size = num_hiddens
    x = torch.ones((2, 4, 64))

    model = TransformerEncoderBlock(query_size=q_size, key_size=k_size, value_size=v_size,
                                    num_hiddens=num_hiddens, num_heads=8, norm_shape=[num_hiddens],
                                    ffn_num_hiddens=num_hiddens,
                                    dropout=0.1, use_bias=False)
    print(model(x).shape)

    pe = PositionalEncoding(num_hiddens=num_hiddens, dropout=0.1, max_len=10)

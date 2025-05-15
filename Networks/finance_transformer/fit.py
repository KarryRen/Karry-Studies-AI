# -*- coding: utf-8 -*-
# @author : KarryRen
# @time   : 2024/7/22 16:45

""" FiT, A new Plug-and-Play module for Finance Cross-Section Feature Extraction. """

import torch
from torch import nn
from einops import rearrange, repeat


class FeedForward(nn.Module):
    """ The FeedForward Module. Just is `2 layers MLP`. """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.):
        """ Init function of FeedForward.

        :param dim: the dimension of input feature
        :param hidden_dim: the hidden dimension
        :param dropout: the dropout ratio

        """

        super().__init__()

        # ---- A 2 layer MLP ---- #
        self.ff_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function of FeedForward.

        :param x: the input feature, shape=(bs, n, dim)

        :return: the ff(x), shape=x.shape

        """

        return self.ff_net(x)


class MultiHeadAttention(nn.Module):
    """ The Multi Head Attention Module with the self - `scale dot attention` mechanism. """

    def __init__(self, dim: int, heads: int, dropout: float = 0.):
        """ Init function of MultiHeadAttention.

        :param dim: the dimension of input feature
        :param heads: the number of heads
        :param dropout: the dropout ratio

        """

        super().__init__()

        # ---- Check and set the params --- #
        self.attn_weight = None  # just for visual
        assert dim % heads == 0, f"The dim must be divided by heads. Now `dim = {dim}`, `heads = {heads}` !"
        self.heads = heads

        # ---- The Weight of Q, K, V ---- #
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.scale = (dim // heads) ** 0.5

        # ---- The dropout of attention ---- #
        self.attn_dropout = nn.Dropout(dropout)

        # ---- Output weight ---- #
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward function of MultiHeadAttention.

        :param x: the input feature, will do the self multi head attention, shape=(bs, n, dim)
        :param mask: the mask of input feature, shape=(bs, n)

        :return out: feature after attention computing, shape=x.shape

        """

        # ---- Do the QKV weighting ---- #
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)  # shape=(bs, n, dim)

        # ---- Split to multi heads --- #
        q = rearrange(q, "bs n (h d) -> bs h n d", h=self.heads)  # shape=(bs, heads, n, dim//heads)
        k = rearrange(k, "bs n (h d) -> bs h n d", h=self.heads)  # shape=(bs, heads, n, dim//heads)
        v = rearrange(v, "bs n (h d) -> bs h n d", h=self.heads)  # shape=(bs, heads, n, dim//heads)

        # ---- Compute the `attn_score = (qk^T) / (sqrt(d_k <scale>))` ---- #
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / self.scale  # shape=(bs, heads, n, n)

        # ---- Mask the attention score ---- #
        mask = mask.unsqueeze(1)  # expand dim from (bs, n) to (bs, 1, n)
        mask = repeat(mask, "bs 1 n -> bs h n", h=self.heads)  # repeat shape from (bs, 1, n) to (bs, heads, n)
        mask = mask.unsqueeze(-2)  # expand dim from (bs, heads, n) to (bs, heads, 1, n)
        mask = repeat(mask, "bs h 1 n-> bs h s n", s=mask.shape[-1])  # repeat shape from (bs, heads, 1, n) to (bs, heads, n, n)
        attn_score[~(mask == 1)] = -1e6  # mask the value

        # ---- Compute the `attn_weight` and do dropout ---- #
        attn_weight = nn.functional.softmax(attn_score, dim=-1)  # shape=(bs, heads, n, n)
        attn_weight = self.attn_dropout(attn_weight)  # shape=(bs, heads, n, n)

        # ---- Compute the `out` ---- #
        out = torch.matmul(attn_weight, v)  # shape=(bs, heads, n, dim//heads)
        out = rearrange(out, "bs h n d -> bs n (h d)")  # shape=(bs, n, dim)
        out = self.w_o(out)
        return out


class Transformer(nn.Module):
    """ The transformer.  Ref https://arxiv.org/abs/1706.03762 """

    def __init__(self, dim: int, depth: int, heads: int, ff_hidden_dim: int, dropout: float = 0.):
        """ Init function of Transformer.

        :param dim: the dimension of input feature
        :param depth: the number of layers
        :param heads: the number of heads in `multi_attention_head`
        :param ff_hidden_dim: the hidden dim of FeedForward Module
        :param dropout: the dropout ratio.

        """

        super().__init__()

        # ---- The core layers ---- #
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim=dim, heads=heads, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim=dim, hidden_dim=ff_hidden_dim, dropout=dropout),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward function of Transformer.

        :param x: input feature, shape=(bs, n, dim)
        :param mask: the mask of input feature, shape=(bs, n)

        :return: out: the encoded feature, shape=(bs, n, dim)

        """

        # ---- Computing layer by layer ---- #
        for attn, norm_attn, ff, norm_ff in self.layers:
            x = norm_attn(attn(x, mask) + x)  # Add&Norm shape=(bs, n, dim)
            x = norm_ff(ff(x) + x)  # Add&Norm shape=(bs, n, dim)

        # ---- Get the output ---- #
        out = x
        return out


class FiT(nn.Module):
    """ FiT, A new Plug-and-Play module for Finance Cross-Section Feature Extraction. """

    def __init__(
            self, input_size: int, dim: int, depth: int, heads: int, ff_hidden_dim: int,
            pool: str = "fin", lp_dropout: float = 0., te_dropout: float = 0.
    ):
        """ Init function of the FiT.

        :param input_size: the input feature size
        :param dim: the dimension of feature after Linear Projection(LP) and the feature before the Transformer Encoder (TE)
        :param depth: the depth of TE, which means the number of transformer encoder layer
        :param heads: the number of head of TE
        :param ff_hidden_dim: the hidden dimension of Feed Forward
        :param pool: the pooling way after TE, you have two choice now:
            - `fin`: use the feature corresponding to the [fin] token
            - `mean`: mean all features after TE
        :param lp_dropout: the dropout ratio of LP
        :param te_dropout: the dropout ratio of TE

        """

        super().__init__()
        assert pool in ("fin", "mean"), f"`pool` type must be either cls (cls token) or mean (mean pooling), now is {pool}"
        self.pool = pool

        # ---- The linear projection ---- #
        self.linear_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, dim),
            nn.LayerNorm(dim)
        )

        # ---- The fin token ---- #
        self.fin_token = nn.Parameter(torch.randn(1, 1, dim))
        self.lp_dropout = nn.Dropout(lp_dropout)

        # ---- The transformer encoder ---- #
        self.transformer_encoder = Transformer(dim=dim, depth=depth, heads=heads, ff_hidden_dim=ff_hidden_dim, dropout=te_dropout)

        # ---- The mlp head ---- #
        self.mlp_head = nn.Linear(dim, 1)

    def forward(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward function of FiT.

        :param feature: the input feature of FiT, shape=(bs, n, input_size), `n` means
            there are `n` samples of feature
        :param mask: the mask of input feature, meaning which sample is valid or not,
            shape=(bs, n). value is `0 or 1` and 0 means not valid, 1 means valid.

        :return out: the output, shape=(bs, 1)

        """

        # ---- Step 1. Do the LP ---- #
        x = self.linear_projection(feature)  # shape=(bs, n, dim)
        bs, n, _ = x.shape

        # ---- Step 2. Concat the `fin_token` ---- #
        fin_tokens = repeat(self.fin_token, "1 1 d -> bs 1 d", bs=bs)  # repeat (1, 1, dim) to (bs, 1, dim)
        x = torch.cat((fin_tokens, x), dim=1)  # concat the token, shape=(bs, n+1, dim)
        mask = torch.cat((torch.ones(bs, 1), mask), dim=1).to(dtype=torch.int32)  # the mask of first token must be 1, shape=(bs, n+1)
        x = self.lp_dropout(x)  # do the drop out

        # ---- Step 4. Do the TE ---- #
        x = self.transformer_encoder(x, mask)  # shape=(bs, n+1, dim)

        # ---- Step 5. Do the summary ---- #
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]  # shape=(bs, dim)

        # ---- Step 5. Do the MLP transformation ---- #
        out = self.mlp_head(x)  # shape=(bs, 1)
        return out

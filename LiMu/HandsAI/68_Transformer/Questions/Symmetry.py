# -*- coding: utf-8 -*-
# @Time    : 2024/7/3 17:10
# @Author  : Karry Ren

""" Why Attention have Symmetry ?

For Attention function $f()$, $f(reverse(x)) == reverse(f(x))$ why ?
Here is a case !

"""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        encoder_output = self.encoder(src)
        return encoder_output


# Example usage
d_model = 512
nhead = 8
num_layers = 6
model = TransformerEncoder(d_model, nhead, num_layers)
model.eval()

# Generate some random input
batch_size = 32
seq_len = 10
src = torch.rand(batch_size, seq_len, d_model)

# Reverse the order of the input sequence
src_reversed = src.flip(1)

# Pass the original and reversed inputs through the model
output1 = model(src).flip(1)
output2 = model(src_reversed)

print(f"Output 1: {output1[:5, 0, :5]}")
print(f"Output 2: {output2[:5, 0, :5]}")
print(f"Difference: {torch.norm(output1 - output2)}")

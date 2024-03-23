# -*- coding:utf-8 -*-

"""
@date: 2024/3/23 下午2:41
@summary:
"""
import torch
import torch.nn as nn
from layers.attention_layer import AttentionLayer
from layers.norm_layer import NormLayer

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, layer_norm_eps, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionLayer(embed_dim, hidden_dim, num_heads, dropout)

        self.ffw = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = NormLayer(hidden_dim, layer_norm_eps)
        self.norm2 = NormLayer(hidden_dim, layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        :param x: shape=(bs, seq_len, dim)
        '''
        ############### attention + Add + Norm ###############
        att_out, _ = self.attention(x)
        att_out = self.dropout(att_out)
        add_norm_out = self.norm1(x + att_out)

        ############### FFW + Add + Norm ###############
        ffw_out = self.ffw(add_norm_out)
        ffw_out = self.dropout(ffw_out)
        add_norm_out = self.norm2(add_norm_out + ffw_out)

        return add_norm_out


class ExampleConfig():
    def __init__(self):
        self.num_heads = 4
        self.layer_norm_eps = 1e-5
        self.resid_pdrop = 0.1
        self.attention_dropout = 0.1
        self.hidden_dim = 12
        self.hidden_dropout = 0.1


def layernorm_sample():
    torch.manual_seed(999)
    x = torch.rand((3, 4, 6))
    normalized_shape = [4, 6]
    norm1 = NormLayer(normalized_shape)
    norm2 = NormLayer(normalized_shape)
    print(norm1(x))
    print(norm2(x))


def t_TransformerBlock():
    torch.manual_seed(999)
    config = ExampleConfig()
    trans = TransformerBlock(config.hidden_dim, config.hidden_dim, config.num_heads, config.layer_norm_eps, config.hidden_dropout)
    q = torch.rand((3, 4, config.hidden_dim))
    r = trans(q)
    print(q)
    print(r)


if __name__ == "__main__":
    # layernorm_sample()
    t_TransformerBlock()

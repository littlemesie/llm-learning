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
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = AttentionLayer(config)

        self.ffw = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = NormLayer(config)
        self.norm2 = NormLayer(config)
        self.dropout = nn.Dropout(config.hidden_dropout)

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
        self.hidden_size = 12
        self.dropout = 0.1
        self.hidden_dropout = 0.1


def layernorm_sample():
    torch.manual_seed(999)
    x = torch.rand((3, 4, 6))
    config = ExampleConfig()
    config.normalized_shape = [4, 6]
    norm1 = NormLayer(config)
    norm2 = NormLayer(config)
    print(norm1(x))
    print(norm2(x))


def t_TransformerBlock():
    torch.manual_seed(999)
    config = ExampleConfig()
    config.normalized_shape = config.hidden_size
    trans = TransformerBlock(config)
    q = torch.rand((3, 4, config.hidden_size))
    r = trans(q)
    print(q)
    print(r)


if __name__ == "__main__":
    # layernorm_sample()
    t_TransformerBlock()

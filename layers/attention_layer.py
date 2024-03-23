# -*- coding:utf-8 -*-

"""
@date: 2024/3/23 下午2:18
@summary:
"""
import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super(AttentionLayer, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 需要能被 num_heads 整除"

        self.embed_size = embed_dim  # embedding 层
        self.hidden_dim = hidden_dim  # 隐藏层
        self.num_heads = num_heads  # 多头数量
        self.head_dim = hidden_dim // num_heads  # 每个头的维度

        self.q_linear = nn.Linear(embed_dim, hidden_dim)
        self.k_linear = nn.Linear(embed_dim, hidden_dim)
        self.v_linear = nn.Linear(embed_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """注意力特征矩阵计算 attention_weights_martix = softmax(QK/dk)V"""
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 当掩码矩阵存在时，则进行替换填充，即将0值所对应的坐标赋值一个极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # 按最后一维softmax
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def split_heads(self, x):
        """拆分多头(3位转换成4维，即将embed_size维拆分成num_heads份，且每一份的维度为head_dim)"""
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    # 合并多头(多个4维张量合并成一个3维张量，相当于还原成输入向量x的size)
    def merge_heads(self, x):
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        return x

    def forward(self, x, mask=None):

        q = self.split_heads(self.q_linear(x))
        k = self.split_heads(self.k_linear(x))
        v = self.split_heads(self.v_linear(x))

        # 计算注意力权重矩阵以及特征矩阵
        output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        # 合并多头
        output = self.merge_heads(output)

        return output, attention_weights
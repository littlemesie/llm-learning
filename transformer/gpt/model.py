# -*- coding:utf-8 -*-

"""
@date: 2024/4/18 下午7:36
@summary:
"""
import os
import math
import torch
import logging
from torch import nn
import torch.nn.functional as F

class GPTEmbeddings(nn.Module):
    """
    Include embeddings from word and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.embed_dropout_prob)

    def forward(self, input_ids, position_ids=None):
        """
        :param input_ids:  输入序列的原始token id, shape: [src_len, batch_size]
        :param position_ids: 位置序列，本质就是 [0,1,2,3,...,src_len-1], shape: [1,src_len]
        :return: [src_len, batch_size, hidden_size]
        """
        token_embedding = self.word_embeddings(input_ids)
        # shape:[src_len,batch_size,hidden_size]

        if position_ids is None:  # 在实际建模时这个参数其实可以不用传值
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)

        positional_embedding = self.position_embeddings(position_ids).transpose(0, 1)
        # [src_len, 1, hidden_size]

        embeddings = token_embedding + positional_embedding
        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size]
        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]
        embeddings = self.dropout(embeddings)
        return embeddings


class GPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.resid_dropout = nn.Dropout(config.resid_dropout_prob)
        self.num_attention_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.attn_weights = None

    def forward(self, hidden_state, mask):
        bsz, seq_len, hidden_size = hidden_state.size()

        query = self.q_proj(hidden_state).view(bsz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        # (bsz, num_attention_heads, seq_len, head_size)
        key = self.k_proj(hidden_state).view(bsz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        # (bsz, num_attention_heads, seq_len, head_size)
        value = self.v_proj(hidden_state).view(bsz, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        # (bsz, num_attention_heads, seq_len, head_size)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_size)  # (bsz, num_attention_heads, seq_len, seq_len)

        if mask is not None:
            attn_weights += mask

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, value)  # (bsz, num_attention_heads, seq_len, head_size)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        attn_outputs = self.out_proj(attn_outputs)
        attn_outputs = self.resid_dropout(attn_outputs)

        self.attn_weights = attn_weights

        return attn_outputs, attn_weights

class GPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        if config.hidden_activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = getattr(F, config.hidden_activation)
        self.dropout = nn.Dropout(config.resid_dropout_prob)

    def forward(self, hidden_state):
        hidden_state = self.linear1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.linear2(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state

class GPTDecoder(nn.Module):
    """
     GPTDecoder is a stack of N decoder layers.
    """

    def __init__(self, config):
        super().__init__()
        self.attn = GPTAttention(config)
        self.mlp = GPTMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_state, mask):
        hidden_state = hidden_state + self.attn(self.norm1(hidden_state), mask)
        hidden_state = hidden_state + self.mlp(self.norm2(hidden_state))
        return hidden_state

def process_attention_mask(mask):
    if mask.dim() == 2:
        mask = mask[None, None, :, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    mask = (1.0 - mask) * (-1e8)
    return mask

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embeddings = GPTEmbeddings(config)
        self.decoder = nn.ModuleList([GPTDecoder(config) for _ in range(config.num_hidden_layers)])


    def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    position_ids=None):
            """

            :param input_ids:
            :param attention_mask:
            :param position_ids:
            :return:
            """
            embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
            mask = process_attention_mask(attention_mask)

            hidden_state = embedding_output

            for layer in self.decoder:
                hidden_state = layer(hidden_state, mask)

            return hidden_state
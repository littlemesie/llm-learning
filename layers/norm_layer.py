# -*- coding:utf-8 -*-

"""
@date: 2024/3/23 下午2:23
@summary:
"""
import torch
import torch.nn as nn

class NormLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(NormLayer, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape, self.eps = normalized_shape, eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def _mean(self, x):
        _shape = list(x.shape[:-len(self.normalized_shape)]) + [-1]
        _x = x.view(*_shape)
        mean = torch.sum(_x, dim=-1) / _x.shape[-1]
        for i in range(len(x.shape) - len(mean.shape)):
            mean = mean.unsqueeze(-1)
        return mean

    def forward(self, x):
        '''
        参考论文 https://arxiv.org/abs/1607.06450
        参考链接 https://blog.csdn.net/xinjieyuan/article/details/109587913
        pytorch文档 https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        :param x: shape=(bs, seq_len, dim)
        '''
        mean = self._mean(x)
        std = self._mean((x - mean).pow(2) + self.eps).pow(0.5)
        x = (x - mean) / std
        return self.weight * x + self.bias

# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import math
import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn

# weight1 = Parameter(torch.FloatTensor(60, 60, 32))
# weight2 = Parameter(torch.FloatTensor(60, 32, 32))
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def save_weight(self):
        pass

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adjacency, input_feature,similarity):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        support= torch.bmm(adjacency, input_feature)
        support2 = torch.bmm(similarity,support)
        output = torch.matmul(support2, self.weight)

        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.input_dim) + ' -> ' \
            + str(self.output_dim) + ')'


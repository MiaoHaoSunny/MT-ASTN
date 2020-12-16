import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape)
        # print(input.shape)
        # print(self.weight.shape)
        input = input.permute((1, 0))
        support = torch.mm(input, self.weight)
        # print(support.shape)
        # print(adj.shape)
        output = torch.spmm(adj.to(device), support)

        if self.bias is not None:
            output = output + self.bias
            output = output.permute((1, 0))
            # print('gcn output', output.shape)
            return output
        else:
            output = output.permute((1, 0))
            # print('gcn output', output.shape)
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

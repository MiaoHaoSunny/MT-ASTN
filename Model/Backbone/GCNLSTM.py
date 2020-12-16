import torch
import torch.nn as nn
from torch.autograd import Variable

from Model.Graph_process.gcn import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNLSTMCell(nn.Module):
    def __init__(self, in_feature, out_feature, dropout):
        super(GCNLSTMCell, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.gcn = GCN(self.in_feature+self.out_feature, 4*self.out_feature, self.dropout)

    def forward(self, node_features, adjacency, cur_state):
        h_cur, c_cur = cur_state
        # print(node_features.shape, h_cur.shape)
        combined_feature = torch.cat([node_features, h_cur], dim=1)
        # combined_adj = torch.cat([adjacency, adjacency], dim=1)
        combined_gcn = self.gcn(combined_feature, adjacency)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_gcn, self.out_feature, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.out_feature, 273)).to(device),
                Variable(torch.zeros(batch_size, self.out_feature, 273)).to(device))


# class GCNLSTMCell(nn.Module):
#
#     def __init__(self, nfeat, nhid, dropout):
#         super(GCNLSTMCell, self).__init__()
#         self.nfeat = nfeat
#         self.nhid = nhid
#         # self.nclass = nclass
#         self.dropout = dropout
#         self.gcn = GCN(self.nfeat, self.nhid, self.dropout)
#
#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
#
#         combined = torch.cat([input_tensor, h_cur], dim=1)
#         combined_gcn = self.gcn(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_gcn, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
#
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)
#
#         return h_next, c_next
#
#     def init_hidden(self, batch_size):
#         return (Variable(torch.zeros(batch_size, self.hidden_dim, self.nfeat)).cuda(),
#                 Variable(torch.zeros(batch_size, self.hidden_dim, self.nfeat)).cuda())
class GCNLSTM(nn.Module):
    def __init__(self, in_feature, out_feature, dropout, num_layers, batch_first=False, return_all_layers=False):
        super(GCNLSTM, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dropout = dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.in_feature if i==0 else self.out_feature[i-1]
            cell_list.append(GCNLSTMCell(cur_input_dim, self.out_feature[i], self.dropout))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, nodes_features, adjacency, hidden_state=None):
        if hidden_state is not None:
            if not isinstance(hidden_state, tuple):
                raise ValueError('Hidden state must be tuple')
            else:
                hidden_state = self._init_not_none_hidden(hidden_state)
        else:
            hidden_state = self._init_hidden(batch_size=nodes_features.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = nodes_features.size(1)
        cur_layer_input = nodes_features

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](node_features=cur_layer_input[:, t, :, :],
                                                 adjacency=adjacency[:, t, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    def _init_not_none_hidden(self, hidden):
        init_hidden = []
        for i in range(self.num_layers):
            init_hidden.append(hidden)
        return init_hidden

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# class GCNLSTM(nn.Module):
#     def __init__(self, nfeat, nhid, dropout, num_layers, batch_first=False, return_all_layers=False):
#         super(GCNLSTM, self).__init__()
#         nhid = self._extend_for_multilayer(nhid, num_layers)
#         # nclass = self._extend_for_multilayer(nclass, num_layers)
#         if not len(nhid)  == num_layers:
#             raise ValueError('Inconsistent list length')
#
#         self.nfeat = nfeat
#         self.nhid = nhid
#         # self.nclass = nclass
#         self.dropout = dropout
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.return_all_layers = return_all_layers
#
#         cell_list = []
#         for i in range(0, self.num_layers):
#             cur_input_dim = self.nfeat if i==0 else self.nhid[i-1]
#             cell_list.append(GCNLSTMCell(cur_input_dim, self.nhid[i], self.dropout))
#         self.cell_list = nn.ModuleList(cell_list)
#
#     def forward(self, input_tensor, hidden_state=None):
#         if hidden_state is not None:
#             if not isinstance(hidden_state, tuple):
#                 raise ValueError('Hidden state must be tuple')
#             else:
#                 hidden_state = self._init_not_none_hidden(hidden_state)
#             # raise NotImplementedError()
#         else:
#             hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
#
#         layer_output_list = []
#         last_state_list = []
#
#         seq_len = input_tensor.size(1)
#         cur_layer_input = input_tensor
#         # print(cur_layer_input.shape)
#
#         for layer_idx in range(self.num_layers):
#
#             h, c = hidden_state[layer_idx]
#             # print('h', h.shape, 'c', c.shape)
#             output_inner = []
#             for t in range(seq_len):
#
#                 h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
#                                                  cur_state=[h, c])
#                 output_inner.append(h)
#
#             layer_output = torch.stack(output_inner, dim=1)
#             cur_layer_input = layer_output
#
#             layer_output_list.append(layer_output)
#             last_state_list.append([h, c])
#
#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list = last_state_list[-1:]
#
#         return layer_output_list, last_state_list
#
#     def _init_hidden(self, batch_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.cell_list[i].init_hidden(batch_size))
#         return init_states
#
#     def _init_not_none_hidden(self, hidden):
#         init_hidden = []
#         for i in range(self.num_layers):
#             init_hidden.append(hidden)
#         return init_hidden
#
#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param

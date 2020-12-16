import torch.nn as nn
import torch
import torch.nn.functional as F
from Model.Graph_process.gcn_layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        # out = []
        out = None
        for i in range(x.size(0)):
            out_id = F.relu(self.gc1(x[i], adj[i]))
            out_id = F.dropout(out_id, self.dropout, training=self.training)
            if i == 0:
                out = out_id.view((1, out_id.size(0), out_id.size(1)))
            else:
                out = torch.cat((out, out_id.view((1, out_id.size(0), out_id.size(1)))), dim=0)
            # out.append(out_id)
        # out = torch.stack(out_id, dim=-1)
        # print('out', out.shape)
        # x = self.gc2(x, adj)
        return out

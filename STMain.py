import torch
import torch.nn as nn
import torch.functional as F

from Model.Backbone.GCNLSTM import *
from Model.Backbone.initial import ReverseLayerF
from Model.Backbone.Self_attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HSTN(nn.Module):
    def __init__(self, in_channels, in_features, out_channels, kernel_size, padding):
        super(HSTN, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # self.conv3d = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=(3, 3, 3), padding=(3//2, 3//2, 3//2),
        #                         bias=True)
        self.conv3d = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                kernel_size=self.kernel_size, padding=self.padding)
        self.gcn_lstm = GCNLSTM(in_feature=self.in_features, out_feature=[self.out_channels], dropout=0.1,
                                batch_first=True, num_layers=1, return_all_layers=False)
        self.relu = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=16, mode='nearest')
        # self.upsample1 = nn.Linear(6*self.out_channels*16, 6*self.out_channels*256)
        # self.upsample2 = nn.Linear(6*self.out_channels*1, 6*self.out_channels*256)

    def _graph_to_image(self, graph):
        graph1 = graph[:, :, :, 0:256]
        graph1 = graph1.view((graph1.size(0), graph1.size(1), graph1.size(2), 16, 16))
        graph2 = graph[:, :, :, 256:256+16]
        # graph2 = self.upsample1(graph2)
        graph2 = graph2.view((graph1.size(0), graph1.size(1), graph1.size(2), 4, 4))
        # # graph2 = self.upsample1(graph2.view())
        for i in range(graph2.size(1)):
            if i == 0:
                graph2_ = torch.unsqueeze((self.upsample1(graph2[:, i, :, :, :])), dim=1)
            else:
                graph2_ = torch.cat((graph2_, torch.unsqueeze((self.upsample1(graph2[:, i, :, :, :])), dim=1)), dim=1)
        graph2 = graph2_
        graph3 = graph[:, :, :, -1]
        # graph3 = self.upsample2(graph3)
        graph3 = graph3.view((graph1.size(0), graph1.size(1), graph1.size(2), 1, 1))
        for j in range(graph3.size(1)):
            if j == 0:
                graph3_ = torch.unsqueeze((self.upsample2(graph3[:, i, :, :, :])), dim=1)
            else:
                graph3_ = torch.cat((graph3_, torch.unsqueeze((self.upsample2(graph3[:, j, :, :, :])), dim=1)), dim=1)
        graph3 = graph3_
        graph = graph1 + graph2 + graph3
        return graph

    def forward(self, img, nodes, adj):
        # img = img.permute((0, 2, 1, 3, 4))
        img = self.conv3d(img)
        graph, _ = self.gcn_lstm(nodes, adj)
        graph_img = self._graph_to_image(graph=graph[0])
        graph_img = graph_img.permute((0, 2, 1, 3, 4))
        # print('img', img.shape, 'graph_img', graph_img.shape)
        out = torch.cat((img, graph_img), dim=1)
        out = self.relu(out)
        # print('img', img.shape, 'graph_img', graph_img.shape, 'out', out.shape)
        return out, graph[0]


class flow_private_encoder(nn.Module):
    def __init__(self):
        super(flow_private_encoder, self).__init__()
        self.hstn1 = HSTN(in_channels=2, in_features=256, out_channels=4, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn2 = HSTN(in_channels=8, in_features=4, out_channels=16, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn3 = HSTN(in_channels=32, in_features=16, out_channels=32, kernel_size=3, padding=(3//2, 3//2, 3//2))

    def forward(self, x, node, adj):
        out, graph = self.hstn1(x, node, adj)
        out, graph = self.hstn2(out, graph, adj)
        out, _ = self.hstn3(out, graph, adj)
        return out


class od_private_encoder(nn.Module):
    def __init__(self):
        super(od_private_encoder, self).__init__()
        # self.conv1plus1 = nn.Conv3d(in_channels=16*16, out_channels=2, kernel_size=1)
        self.hstn1 = HSTN(in_channels=2, in_features=256, out_channels=4, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn2 = HSTN(in_channels=8, in_features=4, out_channels=16, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn3 = HSTN(in_channels=32, in_features=16, out_channels=32, kernel_size=3, padding=(3//2, 3//2, 3//2))

    def forward(self, x, node, adj):
        out, graph = self.hstn1(x, node, adj)
        out, graph = self.hstn2(out, graph, adj)
        out, _ = self.hstn3(out, graph, adj)
        return out


class shared_encoder(nn.Module):
    def __init__(self):
        super(shared_encoder, self).__init__()
        self.hstn1 = HSTN(in_channels=2, in_features=256, out_channels=4, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn2 = HSTN(in_channels=8, in_features=4, out_channels=16, kernel_size=3, padding=(3//2, 3//2, 3//2))
        self.hstn3 = HSTN(in_channels=32, in_features=16, out_channels=32, kernel_size=3, padding=(3//2, 3//2, 3//2))

    def forward(self, x, node, adj):
        out, graph = self.hstn1(x, node, adj)
        out, graph = self.hstn2(out, graph, adj)
        out, _ = self.hstn3(out, graph, adj)
        return out


class STMultiTask(nn.Module):
    def __init__(self, queue=True):
        super(STMultiTask, self).__init__()
        self.conv1plus1 = nn.Conv3d(in_channels=16*16, out_channels=2, kernel_size=1)

        self.flow_private_encoder = flow_private_encoder()
        self.od_private_encoder = od_private_encoder()
        self.shared_encoder = shared_encoder()

        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc1', nn.Linear(3*64*16*16, 64))
        self.shared_encoder_pred_domain.add_module('relu1', nn.ReLU())
        self.shared_encoder_pred_domain.add_module('fc2', nn.Linear(64, 32))
        self.shared_encoder_pred_domain.add_module('relu2', nn.ReLU())
        self.shared_encoder_pred_domain.add_module('fc3', nn.Linear(32, 2))

        self.flow_private_pred_domain = nn.Sequential()
        self.flow_private_pred_domain.add_module('fc1', nn.Linear(3*64*16*16, 64))
        self.flow_private_pred_domain.add_module('relu1', nn.ReLU())
        self.flow_private_pred_domain.add_module('fc2', nn.Linear(64, 32))
        self.flow_private_pred_domain.add_module('relu2', nn.ReLU())
        self.flow_private_pred_domain.add_module('fc3', nn.Linear(32, 2))

        self.od_private_pred_domain = nn.Sequential()
        self.od_private_pred_domain.add_module('fc1', nn.Linear(3*64*16*16, 64))
        self.od_private_pred_domain.add_module('relu1', nn.ReLU())
        self.od_private_pred_domain.add_module('fc2', nn.Linear(64, 32))
        self.od_private_pred_domain.add_module('relu2', nn.ReLU())
        self.od_private_pred_domain.add_module('fc3', nn.Linear(32, 2))

        self.flow_private_decoder = nn.Conv3d(in_channels=128, out_channels=2, kernel_size=(1, 3, 3),
                                              padding=(0, 3//2, 3//2), stride=1, bias=True)

        self.od_private_decoder = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3),
                                            padding=(0, 3//2, 3//2), stride=1, bias=True)

        # self.MultiHeadAttention = nn.MultiheadAttention(256*16*16, 8)
        self.queue = queue

        self.flow_queue_tmp = torch.zeros((15*24, 64*16*16))
        self.register_buffer('flow_temporal_queue', self.flow_queue_tmp)

        self.register_buffer('flow_queue_ptr', torch.zeros(1, dtype=torch.long))

        self.od_queue_tmp = torch.zeros((15*24, 16*16*64))
        self.register_buffer('od_temporal_queue', self.od_queue_tmp)
        self.register_buffer('od_queue_ptr', torch.zeros(1, dtype=torch.long))

        # flow attention part
        self.proj1 = nn.Linear(in_features=64*16*16, out_features=64*16*16)
        self.context_vector = nn.Parameter(torch.randn(16*16*64, 1).float())
        self.softmax = nn.Softmax(dim=1)

        # od attention part
        self.proj2 = nn.Linear(in_features=64*16*16, out_features=64*16*16)
        self.od_context_vector = nn.Parameter(torch.randn(16*16*64, 1).float())

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, features):
    #     batch_size = features.size(0)
    #
    #     ptr = int(self.queue_ptr)
    #
    #     assert 384 % batch_size == 0
    #
    #     self.temporal_queue[ptr:ptr+batch_size] = features[:, 0].view(batch_size, -1)
    #
    #     ptr = (ptr + batch_size) % 384
    #     self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _od_dequeue_and_enqueue(self, features):
        batch_size = features.size(0)

        ptr = int(self.od_queue_ptr)
        assert 360 % batch_size == 0
        self.od_temporal_queue[ptr:ptr+batch_size] = features[:, 0].reshape(batch_size, -1)

        ptr = (ptr + batch_size) % 360
        self.od_queue_ptr[0] = ptr

    @torch.no_grad()
    def _flow_dequeue_and_enqueue(self, features):
        batch_size = features.size(0)

        ptr = int(self.flow_queue_ptr)

        assert 360 % batch_size == 0
        self.flow_temporal_queue[ptr:ptr+batch_size] = features[:, 0].reshape(batch_size, -1)

        ptr = (ptr + batch_size) % 360
        self.flow_queue_ptr[0] = ptr

    def temporal_attention(self, x):
        H = F.tanh(self.proj1(x))
        score = self.softmax(H.matmul(self.context_vector))
        # print(score)
        x = x.mul(score)
        x = torch.sum(x, dim=1)
        return x, score

    def od_temporal_attention(self, y):
        H = F.tanh(self.proj2(y))
        score1 = self.softmax(H.matmul(self.od_context_vector))
        y = y.mul(score1)
        # x_mul = torch.mul(x, score)
        y = torch.sum(y, dim=1)
        return y, score1

    def forward(self, flow, od, node_features, adjs, alpha):
        flow_result = []
        od_result = []

        batch = flow.shape[0]

        # od = od.permute((0, 2, 1, 3, 4))
        od = self.conv1plus1(od)

        flow, od = flow.float(), od.float()

        # flow modeling
        flow_private_feature = self.flow_private_encoder(flow, node_features, adjs)
        flow_private_feature_fc = flow_private_feature.view(batch, -1)
        print(flow_private_feature_fc.shape)
        flow_private_domain_label = self.flow_private_pred_domain(flow_private_feature_fc)
        flow_result.append(flow_private_feature)
        flow_result.append(flow_private_domain_label)

        flow_shared_feature = self.shared_encoder(flow, node_features, adjs)
        flow_result.append(flow_shared_feature)
        flow_shared_feature_fc = flow_shared_feature.view(batch, -1)
        reversed_flow_shared_feature = ReverseLayerF.apply(flow_shared_feature_fc, alpha)
        flow_domain_label = self.shared_encoder_pred_domain(reversed_flow_shared_feature)
        flow_result.append(flow_domain_label)

        flow_all_feature = flow_private_feature + flow_shared_feature
        # print(flow_all_feature.shape)
        flow_all_feature = flow_all_feature.permute(0, 2, 1, 3, 4)
        # print(flow_all_feature.shape)

        long_term_feature = self.flow_temporal_queue.clone().detach()
        long_term_feature = long_term_feature.view(1, -1, 16*16*64)
        long_term_feature_att_proj, long_term_feature_att = self.temporal_attention(long_term_feature)
        long_term_feature_att_proj = long_term_feature_att_proj.expand(batch, long_term_feature_att_proj.shape[-1])
        # print(long_term_feature_att_proj.shape)

        short_term_feature_att_proj, short_term_feature_att = self.temporal_attention(flow_all_feature.reshape(batch, 3, -1))

        flow_all_feature_att = torch.cat((long_term_feature_att_proj, short_term_feature_att_proj), dim=-1).\
            view((batch, -1, 1, 16, 16))
        # print(long_term_feature_att_proj.shape, short_term_feature_att_proj.shape, flow_all_feature_att.shape)

        flow_out = self.flow_private_decoder(flow_all_feature_att)
        # print(flow_out.shape)
        flow_out = flow_out.permute((0, 2, 1, 3, 4))
        # print(flow_out.shape)
        flow_result.append(flow_out)
        # flow_all_feature = flow_all_feature.permute(0, 2, 1, 3, 4)
        if self.queue == True:
            self._flow_dequeue_and_enqueue(flow_all_feature.reshape(batch, -1, 16*16*64))

        # od Modeling
        od_private_feature = self.od_private_encoder(od, node_features, adjs)
        od_private_feature_fc = od_private_feature.view(batch, -1)
        od_private_domain_label = self.od_private_pred_domain(od_private_feature_fc)
        od_result.append(od_private_feature)
        od_result.append(od_private_domain_label)

        od_shared_feature = self.shared_encoder(od, node_features, adjs)
        od_result.append(od_shared_feature)
        od_shared_feature_fc = od_shared_feature.view(batch, -1)
        reversed_od_shared_feature = ReverseLayerF.apply(od_shared_feature_fc, alpha)
        od_domain_label = self.shared_encoder_pred_domain(reversed_od_shared_feature)
        od_result.append(od_domain_label)

        od_all_features = od_private_feature + od_shared_feature

        od_long_term = self.od_temporal_queue.clone().detach()
        od_long_term = od_long_term.view(1, -1, 16*16*64)
        od_long_term_att_proj, od_long_term_att = self.od_temporal_attention(od_long_term)
        od_long_term_att_proj = od_long_term_att_proj.expand(batch, long_term_feature_att_proj.shape[-1])

        # od_all_features_ = od_all_features.reshape(batch, 6, -1)

        od_short_term_att_proj, od_short_term_att = self.od_temporal_attention(od_all_features.reshape(batch, 3, -1))
        od_all_features_att = torch.cat((od_long_term_att_proj, od_short_term_att_proj), dim=1).view((batch, -1, 1,
                                                                                                      16, 16))

        od_out = self.od_private_decoder(od_all_features_att)
        od_out = od_out.permute((0, 2, 1, 3, 4))
        od_result.append(od_out)

        # od_all_features = od_all_features.reshape(batch, -1, 16*16*64)
        # print('od_all', od_all_features.shape)
        if self.queue == True:
            self._od_dequeue_and_enqueue(od_all_features.reshape(batch, -1, 16*16*64))
        # print('done')
        return flow_result, od_result

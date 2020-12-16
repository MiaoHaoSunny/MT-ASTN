from torch.autograd import Function
import torch.nn as nn
import torch


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, common, specific):
        # batch_size = input1.size(0)
        # input1 = input1.view(batch_size, -1)
        # input2 = input2.view(batch_size, -1)
        #
        # input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        # input1_l2 = input1.div(input1_l2_norm.expand_as(input1)+1e-6)
        #
        # input2_l2_orm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        # input2_l2 = input2.div(input2_l2_orm.expand_as(input2)+1e-6)
        #
        # diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        if common.shape != specific.shape:
            raise ValueError('Common feature and specific feature must have same shape!!!')
        batch = common.shape[0]
        seq = common.shape[1]
        diff = 0.0
        for batch_idx in range(batch):
            # seq_list = []
            seq_loss = 0.0
            for seq_idx in range(seq):
                common_idx = common[batch_idx, seq_idx, :, :, :].view(1, -1)
                # common_idx = common_idx.permute(0, 2, 1)
                specific_idx = specific[batch_idx, seq_idx, :, :, :].view(1, -1)
                relation = common_idx.t().mm(specific_idx)
                seq_idx_loss = torch.mean(relation)
                seq_loss += seq_idx_loss
            diff += seq_loss/seq
        diff /= batch
        # specific = specific.view(specific.size(0), -1)
        # common = common.view(common.size(0), -1)
        # relation = specific.t().mm(common)
        # diff = torch.mean(relation)
        return diff

import torch
from torch import nn


class WeightedBCE(nn.Module):
    def __init__(self, pos, neg):
        """
        :param pos: list of 14 elements
        :param neg: list of 14 elements
        """
        super(WeightedBCE, self).__init__()
        self.pos = torch.FloatTensor(pos).cuda()
        self.neg = torch.FloatTensor(neg).cuda()

    def forward(self, output, target):
        """
        :param output: tensor [n, 14]
        :param target: tensor [n, 14]
        :return:
        """
        epsilon = 1e-10
        w_pos = self.neg / (self.pos + self.neg)
        w_neg = self.pos / (self.pos + self.neg)
        loss = w_pos * (target * torch.log(output+epsilon)) + \
               w_neg * ((1 - target) * torch.log(1 - output+epsilon))
        loss = torch.sum(loss, dim=1)
        return torch.neg(torch.mean(loss))

class UnWeightedBCE(nn.Module):
    def __init__(self):
        """
        :param pos: list of 14 elements
        :param neg: list of 14 elements
        """
        super(UnWeightedBCE, self).__init__()

    def forward(self, output, target):
        """
        :param output: tensor [n, 14]
        :param target: tensor [n, 14]
        :return:
        """
        epsilon = 1e-10
        loss = (target * torch.log(output+epsilon)) + \
               ((1 - target) * torch.log(1 - output+epsilon))
        loss = torch.sum(loss, dim=1)
        return torch.neg(torch.mean(loss))

import torch
from torch import nn
from collections import OrderedDict


class LinNet(nn.Module):

    def __init__(self, hidden_size):
        super(LinNet, self).__init__()
        self.FC_1 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        out = self.FC_1(x)
        return out


class MoCoV2(nn.Module):

    def __init__(self, encoder, Queue_Len=16384, T=0.07, m=0.99999, hidden_size=128):
        super(MoCoV2, self).__init__()
        self.Queue_Len = Queue_Len
        self.m = m
        self.Temp = T

        clf = nn.Sequential(OrderedDict([
            ('FC_1', nn.Linear(1000, 500)),
            ('ReLU', nn.ReLU(inplace=True)),
            ('FC_2', nn.Linear(500, hidden_size))
        ]))
        self.query_encoder = encoder(num_classes=hidden_size)
        self.query_encoder.fc = clf
        self.key_encoder = encoder(num_classes=hidden_size)
        self.key_encoder.fc = clf
        self.queue = torch.randn((self.Queue_Len, hidden_size))

# Change to .view()
    def InfoNCELoss(self, query, key):
        l_positive = torch.exp(torch.div(torch.bmm(query.unsqueeze(1), key.unsqueeze(-1)).unsqueeze(-1), self.T))
        l_negative = torch.sum(torch.exp(torch.div(torch.mm(query, torch.t(self.queue)), self.T)), dim=1)

        out = torch.mean(- torch.log(torch.div(l_positive, l_negative + l_positive)))
        return out

    def momentum_update(self):
        for theta_q, theta_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            theta_k.data = theta_k.data * self.m + theta_q.data * (1. - self.m)

# Queuing not good yet
    def requeue(self, keys):
        self.queue = torch.cat((self.queue, keys), 0)
        if self.queue.shape[0] > self.Queue_Len:
            self.queue = self.queue[keys.shape[0]:, :]

# NOT DONE! What to do with loss??
    def forward(self, query_im, key_im):

        query = self.query_encoder(query_im)
        normalize = nn.BatchNorm1d(num_features=query.shape[1])
        query = normalize(query)

        with torch.no_grad():
            self.momentum_update()
            key = self.key_encoder(key_im)
            key = normalize(key)

        loss = self.InfoNCELoss(query=query, key=key)

        self.requeue(keys=key)

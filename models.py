import torch
from torch import nn
from collections import OrderedDict
from config_parser import config_args
from torchvision.models.resnet import resnet50

class ResNet_Encoder(nn.Module):
    def __init__(self, num_classes, original_model):
        # original_model = resnet50()
        super(ResNet_Encoder, self).__init__()
        self.original_model = original_model
        num_ftrs = self.original_model.fc.in_features  # 2048

        # mlp head
        self.original_model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                               nn.ReLU(),
                                               nn.Linear(num_ftrs, num_classes)
                                               )

    def forward(self, x):
        x = self.original_model(x)
        return x




class LinNet(nn.Module):

    def __init__(self, hidden_size):
        super(LinNet, self).__init__()
        self.FC_1 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        out = self.FC_1(x)
        return out


class MoCoV2(nn.Module):

    def __init__(self, moco_args):

        super(MoCoV2, self).__init__()

        self.queue_len = moco_args['queue_len']
        self.m = moco_args['momentum']
        self.Temp = moco_args['temprature']
        self.num_classes = moco_args['num_classes']

        self.query_encoder = ResNet_Encoder(num_classes=self.num_classes)
        self.key_encoder = ResNet_Encoder(num_classes=self.num_classes)

        self.queue = torch.randn((self.queue_len, self.num_classes))

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
        if self.queue.shape[0] > self.queue_len:
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


if __name__ == '__main__':
    moco_args = config_args['moco_model']
    moco_model = MoCoV2(moco_args)
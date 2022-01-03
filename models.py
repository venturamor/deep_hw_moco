import torch
from torch import nn
from collections import OrderedDict
from config_parser import config_args
from torchvision.models.resnet import resnet18
import copy


class LinCls(nn.Module):
    def __init__(self, moco_args):
        super(LinCls, self).__init__()
        self.feat_dim = moco_args['feat_dim']
        self.num_classes = moco_args['num_classes']
        self.net = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.feat_dim, self.num_classes),
                                 nn.Softmax(dim=0)
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class ResNet_Encoder(nn.Module):
    def __init__(self, feat_dim, original_model):
        # original_model = resnet50()
        super(ResNet_Encoder, self).__init__()
        self.original_model = original_model
        # num_ftrs = original_model.fc.in_features  # 2048 for resnet50, 512 for resnet18
        num_ftrs = 512

        # mlp head
        self.original_model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                               nn.ReLU(),
                                               nn.Linear(num_ftrs, feat_dim)
                                               )

    def forward(self, x):
        x = self.original_model(x)
        return x


class MoCoV2(nn.Module):

    def __init__(self, moco_args):
        super(MoCoV2, self).__init__()

        self.queue_len = moco_args['K']
        self.feat_dim = moco_args['feat_dim']
        self.moco_args = moco_args
        original_model = resnet18()
        self.f_q = ResNet_Encoder(feat_dim=self.feat_dim, original_model=original_model)
        self.f_k = copy.deepcopy(self.f_q)

        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data.copy_(theta_q.data).detach()
            theta_k.requires_grad = False

        # self.queue = torch.randn((self.feat_dim, self.queue_len))
        self.register_buffer("queue", torch.randn(self.feat_dim, self.queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0).cuda().detach()

    @torch.no_grad()
    def momentum_update(self):
        m = self.moco_args['momentum']
        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data = theta_k.data * m + theta_q.data * (1. - m)
            theta_k.requires_grad = False

    # queue update
    @torch.no_grad()
    def requeue(self, k):
        """

        :param k:
        :param queue:
        :return:
        """
        self.queue = torch.cat((torch.t(self.queue), k), 0)
        return torch.t(self.queue[k.shape[0]:, :])

    def InfoNCELoss(self, q, k):
        T = self.moco_args['temperature']

        # logits
        # l_positive = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))  # Nx1
        # l_negative = torch.mm(q.view(N, C), queue.view(C, K))  # NxK
        l_positive = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_negative = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_positive, l_negative], dim=1) / T  # Nx(1+K)

        return logits

    def forward(self, im_q, im_k):
        """

        :param im_q:
        :param im_k:
        :return:
        """
        q = self.f_q(im_q)
        q = torch.nn.functional.normalize(q, dim=1)
        N = q.shape[0]
        with torch.no_grad():
            self.momentum_update()
            k = self.f_k(im_k)
            k = torch.nn.functional.normalize(k, dim=1).detach()

        logits = self.InfoNCELoss(q, k)
        labels = torch.zeros((N,), dtype=torch.long).cuda()
        self.queue = self.requeue(k).detach()
        return logits, labels


if __name__ == '__main__':
    moco_args = config_args['moco_model']
    moco_model = MoCoV2(moco_args)

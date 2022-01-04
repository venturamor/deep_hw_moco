import torch
from torch import nn
from config_parser import config_args
from torchvision.models import efficientnet_b1
import copy


class LinCls(nn.Module):
    def __init__(self, moco_args):
        super(LinCls, self).__init__()
        self.feat_dim = moco_args['feat_dim']
        self.num_classes = moco_args['num_classes']
        self.net = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim),
                                 nn.SiLU(),
                                 nn.Linear(self.feat_dim, self.num_classes),
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class Encoder(nn.Module):
    def __init__(self, feat_dim, original_model):
        super(Encoder, self).__init__()
        self.original_model = original_model
        num_ftrs = 1280

        # mlp head
        self.original_model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
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
        original_model = efficientnet_b1(pretrained=True)
        self.f_q = Encoder(feat_dim=self.feat_dim, original_model=original_model)
        self.f_k = copy.deepcopy(self.f_q)

        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data.copy_(theta_q.data).detach()
            theta_k.requires_grad = False

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
        self.queue = torch.t(torch.cat((torch.t(self.queue), k), dim=0))
        return self.queue[:, k.shape[0]:]

    def InfoNCELoss(self, q, k):
        T = self.moco_args['temperature']
        N, C = q.shape
        K = self.queue_len
        # logits
        l_positive = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(-1)  # Nx1
        l_negative = torch.mm(q.view(N, C), self.queue.view(C, K))  # NxK

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

        with torch.no_grad():
            self.momentum_update()
            k = self.f_k(im_k)
            k = torch.nn.functional.normalize(k, dim=1).detach()

        logits = self.InfoNCELoss(q, k)
        labels = torch.zeros((q.shape[0],), dtype=torch.long).cuda()
        self.queue = self.requeue(k).detach()
        return logits, labels


if __name__ == '__main__':
    moco_args = config_args['moco_model']
    moco_model = MoCoV2(moco_args)

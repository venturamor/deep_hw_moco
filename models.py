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

        self.queue_len = moco_args['K']
        self.num_classes = moco_args['num_classes']

        original_model = resnet50()
        self.f_q = ResNet_Encoder(num_classes=self.num_classes, original_model=original_model)
        self.f_k = ResNet_Encoder(num_classes=self.num_classes, original_model=original_model)

        self.queue = torch.randn((self.queue_len, self.num_classes))


if __name__ == '__main__':
    moco_args = config_args['moco_model']
    moco_model = MoCoV2(moco_args)

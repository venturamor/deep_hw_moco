from torchvision import transforms
import torch
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MOCO_V2_Transform():
    Transform = transforms.Compose([
                        transforms.RandomResizedCrop(
                            size=224),
                        transforms.RandomApply(transforms.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), p=0.75),
                        transforms.RandomApply(transforms.GaussianBlur(
                            kernel_size=(3, 3)), p=0.75),
                        transforms.ToTensor()
                        ])
    return Transform


class DoubleTransform:
    
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, im):
        return self.transform(im), self.transform(im)

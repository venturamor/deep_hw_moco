import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from config_parser import config_args


class Dataset_forMOCO(Dataset):

    def __init__(self, dataset_args, csv_df, transform_args, transform_flag=True):

        self.csv_df = csv_df
        self.data_path = dataset_args['data_path']
        self.noisy_labels = dataset_args['noisy_labels_x']  # for get_item
        self.transform_args = transform_args
        self.transform_flag = transform_flag
        self.aug_transform = self.data_augmentation()

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, item):
        img_path = os.path.join(self.data_path, self.csv_df['path'][item])
        # get image and label
        img = Image.open(img_path)
        label = self.csv_df[self.noisy_labels][item]
        # apply transformation if exists
        if self.transform_flag:
            img = self.aug_transform(img)
        # create pairs of image & label
        data = {'image': img, 'label': label}
        return data

    def data_augmentation(self):
        """
        transformations as described in MOCO original paper [Technical details section]
        :return:
        """
        gaussian_blur_ = self.transform_args['GaussianBlur']
        color_jitter_ = self.transform_args['ColorJitter']
        self.aug_transform = nn.Sequential(
                    transforms.PILToTensor(),
                    transforms.Normalize((0.5, 0., 0.5), (0.5, 0.5, 0.5)),
                    transforms.RandomResizedCrop(self.transform_args['SizeCrop']),  # 224 - as in moco orig
                    # strong color jitter - moco_v2
                    transforms.ColorJitter(brightness=color_jitter_['brightness'],
                                           contrast=color_jitter_['contrast'],
                                           saturation=color_jitter_['saturation'],
                                           hue=color_jitter_['hue']),
                    transforms.RandomHorizontalFlip(p=self.transform_args['RandomHorizontalFlip']),
                    transforms.RandomGrayscale(p=self.transform_args['RandomGrayscale']),
                    # blur - moco_v2
                    transforms.GaussianBlur(kernel_size=gaussian_blur_['kernel_size'],
                                            sigma=(gaussian_blur_['sigma_start'], gaussian_blur_['sigma_end']))
                    )

def get_csv_file(csv_path):
    csv_file = pd.read_csv(csv_path)
    train_df = csv_file[csv_file['is_valid'] == False]
    val_df = csv_file[csv_file['is_valid'] == True]

    return train_df, val_df


if __name__ == '__main__':

    # config
    dataset_args = config_args['dataset']
    transform_args = config_args['transform_augmentation ']


    train_df, val_df = get_csv_file(dataset_args['csv_path'])

    train_dataset = Dataset_forMOCO(
        dataset_args=dataset_args,
        transform_args=transform_args,
        csv_df= train_df)

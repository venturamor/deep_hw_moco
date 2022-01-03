import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from config_parser import config_args


class Dataset_forMOCO(Dataset):

    def __init__(self, dataset_args, csv_df, transform_flag=True, aug_transform=None):
        self.csv_df = csv_df
        self.data_path = dataset_args['data_path']
        self.noisy_labels = dataset_args['noisy_labels_x']  # for get_item
        self.aug_transform = aug_transform
        self.transform_flag = transform_flag

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, item):

        img_path = os.path.join(self.data_path, self.csv_df['path'][item])
        # get image and label
        img = Image.open(img_path)
        label = self.csv_df[self.noisy_labels][item]
        # apply transformation if exists
        if self.transform_flag:
            pil2tensor = transforms.PILToTensor()
            img_tensor = pil2tensor(img).float()
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            img1 = self.aug_transform(img_tensor)
            img2 = self.aug_transform(img_tensor)
        # create pairs of image & label
        data = {'image1': img1, 'image2': img2}
        return data


class Dataset_forLinCls(Dataset):
    def __init__(self, dataset_args, csv_df):
        self.csv_df = csv_df
        self.data_path = dataset_args['data_path']
        self.noisy_labels = dataset_args['noisy_labels_x']  # for get_item

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, item):
        img_path = os.path.join(self.data_path, self.csv_df['path'][item])
        # get image and label
        img = Image.open(img_path)
        label = self.csv_df[self.noisy_labels][item]

        # apply transformation if exists
        resize = transforms.Resize((300, 300))
        pil2tensor = transforms.PILToTensor()
        img_tensor = pil2tensor(img).float()
        img_tensor = resize(img_tensor)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        # create pairs of image & label
        data = {'image': img_tensor, 'label': label}
        return data


def data_augmentation(transform_args):
    """
    transformations as described in MOCO original paper [Technical details section]
    :param transform_args:
    :return:
    """
    gaussian_blur_ = transform_args['GaussianBlur']
    color_jitter_ = transform_args['ColorJitter']
    aug_transform = nn.Sequential(
        # transforms.PILToTensor(),
        transforms.Resize(300),
        transforms.RandomResizedCrop(transform_args['SizeCrop'], scale=(0.25, 1.0)),
        # 224 - orig MoCo, less here for computation speed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # strong color jitter - moco_v2
        transforms.RandomApply(torch.nn.ModuleList(
            [transforms.ColorJitter(brightness=color_jitter_['brightness'],
                                    contrast=color_jitter_['contrast'],
                                    saturation=color_jitter_['saturation'],
                                    hue=color_jitter_['hue']),
             transforms.GaussianBlur(kernel_size=gaussian_blur_['kernel_size'],
                                     sigma=(gaussian_blur_['sigma_start'], gaussian_blur_['sigma_end']))
             ]), p=transform_args['p_apply']),

        transforms.RandomHorizontalFlip(p=transform_args['RandomHorizontalFlip']),
        transforms.RandomGrayscale(p=transform_args['RandomGrayscale']),
        # blur - moco_v2
        # transforms.GaussianBlur(kernel_size=gaussian_blur_['kernel_size'],
        #                         sigma=(gaussian_blur_['sigma_start'], gaussian_blur_['sigma_end']))
    )

    return aug_transform


def get_csv_file(dataset_args):
    # t
    csv_file = pd.read_csv(dataset_args['csv_path'])
    # first division to train and valid
    train_df = csv_file[csv_file['is_valid'] == False]
    val_df = csv_file[csv_file['is_valid'] == True]
    train_df = train_df.sample(frac=1)
    val_df = val_df.sample(frac=1)

    # moco part
    index_moco_train = int(len(train_df) * dataset_args['moco_classifier_frac'])
    index_moco_val = int(len(val_df) * dataset_args['moco_classifier_frac'])

    train_df_moco = train_df[:index_moco_train]
    val_df_moco = val_df[:index_moco_val]

    # classifier part
    train_df_classifier = train_df[index_moco_train:]
    train_df_classifier = train_df_classifier.reset_index().iloc[:, 1:]
    val_df_classifier = val_df[index_moco_val:]

    # and test
    index_test = int(len(val_df) * dataset_args['test_frac'])
    test_df_moco = val_df_moco[:index_test]
    test_df_moco = test_df_moco.reset_index().iloc[:, 1:]
    test_df_classifier = val_df_classifier[:index_test]
    test_df_classifier = test_df_classifier.reset_index().iloc[:, 1:]

    # fix val
    val_df_moco = val_df_moco[index_test:]
    val_df_moco = val_df_moco.reset_index().iloc[:, 1:]
    val_df_classifier = val_df_classifier[index_test:]
    val_df_classifier = val_df_classifier.reset_index().iloc[:, 1:]

    moco_df = {'train': train_df_moco, 'val': val_df_moco, 'test': test_df_moco}
    classifier_df = {'train': train_df_classifier, 'val': val_df_classifier, 'test': test_df_classifier}

    return moco_df, classifier_df


if __name__ == '__main__':
    # config
    dataset_args = config_args['dataset']
    transform_args = config_args['transform_augmentation']

    moco_df, classifier_df = get_csv_file(dataset_args)

    aug_transform = data_augmentation(transform_args)
    train_dataset = Dataset_forMOCO(dataset_args=dataset_args, csv_df=moco_df['train'],
                                    transform_flag=True, aug_transform=aug_transform)

    train_dataset.__getitem__(1)

    print('done')

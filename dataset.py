import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


class Dataset_forMOCO(Dataset):

    def __init__(self, dataset_args, csv_df, transform_flag=True, aug_transform=None):
        self.csv_df = csv_df    # For get_item
        self.data_path = dataset_args['data_path']  # For get_item
        self.noisy_labels = dataset_args['noisy_labels_x']  # For get_item
        self.aug_transform = aug_transform  # Transform function for images
        self.transform_flag = transform_flag    # Technically should always be True

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, item):
        # Path to image
        img_path = os.path.join(self.data_path, self.csv_df['path'][item])
        # Get image
        img = Image.open(img_path)
        # Apply transformation if exists (again, should technically exist)
        if self.transform_flag:
            # Resize image to (300,300) before augmenting for computational reasons
            resize = transforms.Resize((300, 300))
            # Pil to Tensor
            pil2tensor = transforms.PILToTensor()
            img_tensor = pil2tensor(img).float()
            img_tensor = resize(img_tensor)
            # If image is grayscale, make 3-channel grayscale where channel 1=2=3
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            # Apply random transformation twice, creating 2 different transforms of same image
            img1 = self.aug_transform(img_tensor)
            img2 = self.aug_transform(img_tensor)
        # Create pair of the two images
        data = {'image1': img1, 'image2': img2}
        return data


class Dataset_forLinCls(Dataset):
    def __init__(self, dataset_args, csv_df):
        self.csv_df = csv_df    # For get_item
        self.data_path = dataset_args['data_path']  # For get_item
        self.noisy_labels = dataset_args['noisy_labels_x']  # For get_item

    def __len__(self):
        return self.csv_df.shape[0]

    def __getitem__(self, item):
        # Path to image
        img_path = os.path.join(self.data_path, self.csv_df['path'][item])
        # Get image and label
        img = Image.open(img_path)
        label = self.csv_df[self.noisy_labels][item]

        # Resize image to (300,300) before augmenting for computational speed
        resize = transforms.Resize((300, 300))
        # Pil to Tensor
        pil2tensor = transforms.PILToTensor()
        img_tensor = pil2tensor(img).float()
        img_tensor = resize(img_tensor)
        # If image is grayscale, make 3-channel grayscale where channel 1=2=3
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        # Create pair of image & label
        data = {'image': img_tensor, 'label': label}
        return data


def data_augmentation(transform_args):
    """
    Transformations as described in MoCo original paper [Technical details section]
    :param transform_args:
    :return: aug_transform:
    """
    gaussian_blur_ = transform_args['GaussianBlur']  # Gaussian blur args
    color_jitter_ = transform_args['ColorJitter']   # Color jitter args
    # Sequential of: RandomResizedCrop, Normalization, RandomApply of Color Jitter
    # and Gaussian Blur,Random Horizontal Flip, and Random grayscale with appropriate arguments
    aug_transform = nn.Sequential(
        transforms.RandomResizedCrop(transform_args['SizeCrop'], scale=(0.25, 1.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    )

    return aug_transform


def get_csv_file(dataset_args):
    # Read .csv file
    csv_file = pd.read_csv(dataset_args['csv_path'])
    # Divide .csv file to train and validation
    train_df = csv_file[csv_file['is_valid'] == False]
    val_df = csv_file[csv_file['is_valid'] == True]
    # Randomly shuffle train and validation dataframes
    train_df = train_df.sample(frac=1)
    val_df = val_df.sample(frac=1)

    # Create MoCo train and validation sets
    index_moco_train = int(len(train_df) * dataset_args['moco_classifier_frac'])
    index_moco_val = int(len(val_df) * dataset_args['moco_classifier_frac'])

    train_df_moco = train_df[:index_moco_train]
    train_df_moco = train_df_moco.reset_index().iloc[:, 1:]  # Reset indices
    val_df_moco = val_df[:index_moco_val]

    # Create linear classifier train and validation sets
    train_df_classifier = train_df[index_moco_train:]
    train_df_classifier = train_df_classifier.reset_index().iloc[:, 1:]  # Reset indices
    val_df_classifier = val_df[index_moco_val:]

    # Create MoCo and linear classifier test sets
    index_test = int(len(val_df) * dataset_args['test_frac'])
    test_df_moco = val_df_moco[:index_test]
    test_df_moco = test_df_moco.reset_index().iloc[:, 1:]  # Reset indices
    test_df_classifier = val_df_classifier[:index_test]
    test_df_classifier = test_df_classifier.reset_index().iloc[:, 1:]  # Reset indices

    # Minor fixes to validation sets
    val_df_moco = val_df_moco[index_test:]
    val_df_moco = val_df_moco.reset_index().iloc[:, 1:]  # Reset indices
    val_df_classifier = val_df_classifier[index_test:]
    val_df_classifier = val_df_classifier.reset_index().iloc[:, 1:]  # Reset indices

    # Create MoCo and linear classifier dataset dictionaries (train, val, test)
    moco_df = {'train': train_df_moco, 'val': val_df_moco, 'test': test_df_moco}
    classifier_df = {'train': train_df_classifier, 'val': val_df_classifier, 'test': test_df_classifier}

    return moco_df, classifier_df

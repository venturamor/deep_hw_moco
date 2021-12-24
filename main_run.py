import torch
import torch.nn as nn
import dataset
from dataset import Dataset_forMOCO, get_csv_file
from config_parser import config_args
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # config
    dataset_args = config_args['dataset']
    transform_args = config_args['transform_augmentation']
    dataloader_args = config_args['dataloader']
    moco_df, classifier_df = get_csv_file(dataset_args)


    train_dataset = Dataset_forMOCO(
        dataset_args=dataset_args,
        transform_args=transform_args,
        csv_df=moco_df['train'], transform_flag=True)

    train_dl = DataLoader(train_dataset,
                          batch_size=dataloader_args['batch_size'],
                          num_workers=dataloader_args['num_workers'],
                          shuffle=True)

    

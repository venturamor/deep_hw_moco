import torch
import torch.nn as nn
import torch.optim as optim

import dataset
from dataset import Dataset_forMOCO, get_csv_file, data_augmentation
from config_parser import config_args
from torch.utils.data import DataLoader
from models import MoCoV2
from trainer import Trainer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    # config
    dataset_args = config_args['dataset']
    moco_args = config_args['moco_model']
    transform_args = config_args['transform_augmentation']
    dataloader_args = config_args['dataloader']
    moco_df, classifier_df = get_csv_file(dataset_args)

    aug_transform = data_augmentation(transform_args)

    train_dataset = Dataset_forMOCO(dataset_args=dataset_args, csv_df=moco_df['train'],
                                    transform_flag=True, aug_transform=aug_transform)

    train_dl = DataLoader(train_dataset,
                          batch_size=dataloader_args['batch_size'],
                          num_workers=dataloader_args['num_workers'],
                          shuffle=True)

    val_dataset = Dataset_forMOCO(dataset_args=dataset_args, csv_df=moco_df['val'],
                                    transform_flag=True, aug_transform=aug_transform)

    val_dl = DataLoader(val_dataset,
                          batch_size=dataloader_args['batch_size'],
                          num_workers=dataloader_args['num_workers'],
                          shuffle=True)

    moco_model = MoCoV2(moco_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(moco_model,
                      aug_transform,
                      moco_model_args=moco_args,
                      optimizer=optim.SGD(moco_model.f_q.parameters(),
                                          lr=moco_args['optim']['lr'],
                                          momentum=moco_args['optim']['momentum'],
                                          weight_decay=moco_args['optim']['weight_decay']),
                      device=device)
    trainer.fit(train_dl, val_dl)



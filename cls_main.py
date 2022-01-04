from models import MoCoV2, LinCls
import torch
from dataset import Dataset_forLinCls, get_csv_file
from config_parser import config_args
from torch.utils.data import DataLoader
from torch import optim
from trainer import LinCls_Trainer
import os


if __name__ == '__main__':
    torch.cuda.empty_cache()
    dataset_args = config_args['dataset']
    moco_args = config_args['moco_model']
    LinCls_args = config_args['lin_cls']
    dataloader_args = config_args['dataloader']
    _, classifier_df = get_csv_file(dataset_args)

    train_dataset = Dataset_forLinCls(dataset_args=dataset_args, csv_df=classifier_df['train'])

    train_dl = DataLoader(train_dataset,
                          batch_size=LinCls_args['batch_size'],
                          num_workers=dataloader_args['num_workers'],
                          shuffle=True,
                          pin_memory=True)

    val_dataset = Dataset_forLinCls(dataset_args=dataset_args, csv_df=classifier_df['val'])

    val_dl = DataLoader(val_dataset,
                        batch_size=LinCls_args['batch_size'],
                        num_workers=dataloader_args['num_workers'],
                        shuffle=True,
                        pin_memory=True)

    moco_model = MoCoV2(moco_args)
    encoder = moco_model.f_q
    encoder.load_state_dict(torch.load(os.path.join(moco_args['log_path'], 'best_fq_model.pt')))
    Net = LinCls(moco_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = LinCls_Trainer(Net,
                             encoder,
                             LinCls_args,
                             optimizer=optim.Adam(Net.parameters(),
                                                  lr=LinCls_args['optim']['lr'],
                                                  weight_decay=LinCls_args['optim']['weight_decay']),
                             device=device)
    trainer.fit(train_dl, val_dl)

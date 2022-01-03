import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import pandas as pd
import numpy as np
import models
# torch.autograd.set_detect_anomaly(True)


class MoCo_Trainer:
    def __init__(self, moco_model, aug_transform, moco_model_args, optimizer=None, device=None):
        """
        Initialize the trainer.
        :param moco_model:
        :param aug_transform:
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.moco_model = moco_model
        self.f_q = moco_model.f_q
        self.f_k = moco_model.f_k
        self.optimizer = optimizer
        self.device = device
        self.aug_transform = aug_transform
        self.moco_model_args = moco_model_args
        self.queue_train = moco_model.queue.to(self.device)
        # self.queue_train = torch.nn.functional.normalize(self.queue_train, dim=0)
        self.queue_val = moco_model.queue.to(self.device)
        # self.queue_val = torch.nn.functional.normalize(self.queue_val, dim=0)
        self.moco_logs_path = moco_model_args['log_path']
        self.writer = SummaryWriter(self.moco_logs_path)

        if self.device:
            self.f_q.to(self.device)
            self.f_k.to(self.device)


    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader
            ):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_dev: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        """
        scaler = torch.cuda.amp.GradScaler()
        self.f_q.train()
        self.f_k.train()
        num_epochs = self.moco_model_args['num_epochs']
        val_step = self.moco_model_args['val_step']
        save_every = self.moco_model_args['save_every']
        best_val_loss = 1e8
        last_epoch = 0

        if self.moco_model_args['resume_run'] and os.path.exists(os.path.join(self.moco_logs_path, 'checkpoint.pt')):
            checkpoints = torch.load(os.path.join(self.moco_logs_path, 'checkpoint.pt'))
            self.f_k = checkpoints['fk_model']
            self.f_q = checkpoints['fq_model']
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            best_val_loss = checkpoints['best_val_loss']
            last_epoch = checkpoints['last_epoch']
        gamma = 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=50,
                                                    gamma=gamma)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(last_epoch, num_epochs):
            self.moco_model.train()
            startTime = time.time()
            train_epoch_loss = 0
            for batch_data in dl_train:
                # image, label = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
                images_q = batch_data['image1'].to(self.device)
                images_k = batch_data['image2'].to(self.device)

                with torch.cuda.amp.autocast():
                    logits, labels = self.moco_model(images_q, images_k)
                    loss = criterion(logits, labels)

                # with torch.autograd.detect_anomaly():
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                train_epoch_loss += float(loss)
                loss = None

            scheduler.step()
            endTime = time.time()
            torch.cuda.empty_cache()
            avg_train_loss = train_epoch_loss / len(dl_train)
            print('epoch {}, loss {}, epoch time {} seconds, time remaining {} hours'.format(
                epoch + 1,
                avg_train_loss,
                round(endTime - startTime, 2),
                round(num_epochs * (endTime - startTime) / 3600 - epoch * (endTime - startTime) / 3600, 2)
            ))
            self.writer.add_scalar(tag='Loss/train_loss', scalar_value=avg_train_loss, global_step=epoch + 1)

            if (epoch + 1) % val_step == 0:
                avg_val_loss = self.eval(dl_dev)
                self.writer.add_scalar(tag='Loss/val_loss', scalar_value=avg_val_loss, global_step=epoch + 1)
                print('epoch {}, val loss {}'.format(epoch + 1, avg_val_loss))
                if avg_val_loss < best_val_loss:
                    print('save new best model')
                    torch.save(self.f_q.state_dict(), os.path.join(self.moco_logs_path, 'best_fq_model.pt'))
                    torch.save(self.f_k.state_dict(), os.path.join(self.moco_logs_path, 'best_fk_model.pt'))
                    best_val_loss = avg_val_loss

            if (epoch + 1) % save_every == 0:
                checkpoints = {
                    'fq_model': self.f_q.state_dict(),
                    'fk_model': self.f_k.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'last_epoch': epoch + 1,
                }
                torch.save(checkpoints, os.path.join(self.moco_logs_path, 'checkpoint.pt'))

    @torch.no_grad()
    def eval(self, dl_val: DataLoader):
        """
        Args:
            dl_val: model to evaluate
        Returns: f1 score
        """
        scaler = torch.cuda.amp.GradScaler()
        self.moco_model.eval()
        val_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for val_data in dl_val:
                image1 = val_data['image1'].to(self.device)
                image2 = val_data['image2'].to(self.device)

                images_q = image1
                images_k = image2
                with torch.cuda.amp.autocast():
                    logits, labels = self.moco_model(images_q, images_k)
                    loss = criterion(logits, labels).detach()
                val_loss += float(loss)
                loss = None

            return val_loss / len(dl_val)


class LinCls_Trainer:
    def __init__(self, LinCls, encoder, LinCls_args, optimizer=None, device=None):
        """
        Initialize the trainer.
        :param moco_model:
        :param aug_transform:
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.LinCls = LinCls
        self.encoder = encoder
        self.optimizer = optimizer
        self.device = device
        self.LinCls_args = LinCls_args
        self.LinCls_logs_path = LinCls_args['log_path']
        self.writer = SummaryWriter(self.LinCls_logs_path)

        if self.device:
            self.LinCls.to(self.device)
            self.encoder.to(self.device)


    def fit(self,
            dl_train: DataLoader,
            dl_dev: DataLoader
            ):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_dev: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        """
        scaler = torch.cuda.amp.GradScaler()
        self.encoder.eval()
        num_epochs = self.LinCls_args['num_epochs']
        val_step = self.LinCls_args['val_step']
        save_every = self.LinCls_args['save_every']
        best_val_loss = 1e8
        best_val_acc = 0
        last_epoch = 0

        if self.LinCls_args['resume_run'] and os.path.exists(os.path.join(self.LinCls_logs_path, 'checkpoint.pt')):
            checkpoints = torch.load(os.path.join(self.LinCls_logs_path, 'checkpoint.pt'))
            self.LinCls = checkpoints['LinCls']
            self.optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
            best_val_loss = checkpoints['best_val_loss']
            last_epoch = checkpoints['last_epoch']
        gamma = 0.5
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=5,
                                                    gamma=gamma)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(last_epoch, num_epochs):
            self.LinCls.train()
            startTime = time.time()
            train_epoch_loss = 0
            avg_acc = 0
            for batch_data in dl_train:
                images, temp_labels = batch_data['image'].to(self.device), batch_data['label']
                onehot = pd.get_dummies(temp_labels).to_numpy()
                labels = np.argmax(onehot, axis=1)
                labels = torch.LongTensor(labels).to(self.device)
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        encoding = self.encoder(images)
                    logits = self.LinCls(encoding)
                    loss = criterion(logits, labels)
                pred = torch.argmax(logits, dim=1)
                avg_acc += torch.mean((pred == labels).type(torch.float))
                # with torch.autograd.detect_anomaly():
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                train_epoch_loss += float(loss)
                loss = None
            avg_acc /= len(dl_train)
            scheduler.step()
            endTime = time.time()
            torch.cuda.empty_cache()
            avg_train_loss = train_epoch_loss / len(dl_train)
            print('epoch: {}, loss: {}, train accuracy: {}, epoch time: {} seconds, time remaining: {} hours'.format(
                epoch + 1,
                avg_train_loss,
                avg_acc,
                round(endTime - startTime, 2),
                round(num_epochs * (endTime - startTime) / 3600 - epoch * (endTime - startTime) / 3600, 2)
            ))
            self.writer.add_scalar(tag='Loss/train_loss', scalar_value=avg_train_loss, global_step=epoch + 1)

            if (epoch + 1) % val_step == 0:
                avg_val_loss, avg_val_acc = self.eval(dl_dev)
                self.writer.add_scalar(tag='Loss/val_loss', scalar_value=avg_val_loss, global_step=epoch + 1)
                self.writer.add_scalar(tag='Acc/val_acc', scalar_value=avg_val_acc, global_step=epoch + 1)
                print('epoch: {}, val loss: {}, val acc: {}'.format(epoch + 1, avg_val_loss, avg_val_acc))
                if avg_val_loss < best_val_loss:
                    print('save new best model')
                    torch.save(self.LinCls.state_dict(), os.path.join(self.LinCls_logs_path, 'best_LinCls.pt'))
                    best_val_loss = avg_val_loss
                if avg_val_acc < best_val_acc:
                    print('save new best model')
                    torch.save(self.LinCls.state_dict(), os.path.join(self.LinCls_logs_path, 'best_LinCls.pt'))
                    best_val_loss = avg_val_loss

            if (epoch + 1) % save_every == 0:
                checkpoints = {
                    'LinCls': self.LinCls.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_acc': best_val_acc,
                    'last_epoch': epoch + 1,
                }
                torch.save(checkpoints, os.path.join(self.LinCls_logs_path, 'checkpoint.pt'))

    @torch.no_grad()
    def eval(self, dl_val: DataLoader):
        """
        Args:
            dl_val: model to evaluate
        Returns: f1 score
        """
        scaler = torch.cuda.amp.GradScaler()
        self.LinCls.eval()
        val_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        val_acc = 0
        with torch.no_grad():
            for val_data in dl_val:
                images, temp_labels = val_data['image'].to(self.device), val_data['label']
                onehot = pd.get_dummies(temp_labels).to_numpy()
                labels = np.argmax(onehot, axis=1)
                labels = torch.LongTensor(labels).to(self.device)
                with torch.cuda.amp.autocast():
                    encoding = self.encoder(images)
                    logits = self.LinCls(encoding)
                    pred = torch.argmax(logits, dim=1)
                    val_acc += torch.mean((pred == labels).type(torch.float))
                    loss = criterion(logits, labels).detach()
                val_loss += float(loss)
                loss = None
            return val_loss / len(dl_val), val_acc / len(dl_val)


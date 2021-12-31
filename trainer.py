import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
torch.autograd.set_detect_anomaly(True)
def InfoNCELoss(q, k, queue, criterion, moco_model_args, device):
    T = moco_model_args['temperature']
    N, C = q.shape
    K = queue.shape[0]

    # logits
    # l_positive = torch.bmm(q.view(N, 1, C), k.view(N, C, 1))  # Nx1
    # l_negative = torch.mm(q.view(N, C), queue.view(C, K))  # NxK
    l_positive = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    l_negative = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

    logits = torch.cat([l_positive, l_negative], dim=1)  # Nx(1+K)
    labels = torch.zeros((N, ), dtype=torch.long).to(device)
    loss = criterion(logits / T, labels)

    return loss


# queue update
def requeue(k, queue):
    """

    :param k:
    :param queue:
    :return:
    """
    queue = torch.cat((torch.t(queue), k), 0)
    return torch.t(queue[k.shape[0]:, :])


class Trainer:
    def __init__(self, moco_model, aug_transform, moco_model_args, optimizer=None, device=None):
        """
        Initialize the trainer.
        :param moco_model:
        :param aug_transform:
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """

        self.f_q = moco_model.f_q
        self.f_k = moco_model.f_k
        self.optimizer = optimizer
        self.device = device
        self.aug_transform = aug_transform
        self.moco_model_args = moco_model_args
        self.queue_train = moco_model.queue.to(self.device)
        self.queue_train = torch.nn.functional.normalize(self.queue_train, dim=0)
        self.queue_val = moco_model.queue.to(self.device)
        self.queue_val = torch.nn.functional.normalize(self.queue_val, dim=0)
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
        decayRate = 0.9
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                    step_size=5,
                                                    gamma=decayRate,
                                                    verbose=True)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(last_epoch, num_epochs):
            startTime = time.time()
            train_epoch_loss = 0

            for batch_data in dl_train:
                # image, label = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
                image = batch_data['image'].to(self.device)
                image_q = self.aug_transform(image)
                image_k = self.aug_transform(image)
                # encoders
                with torch.no_grad():
                    k = self.f_k(image_k)
                    # moment
                    self.momentum_update()

                q = self.f_q(image_q)

                k.detach()  # update only with momentum
                with torch.autograd.detect_anomaly():
                    self.optimizer.zero_grad()

                    # Zero gradients, perform a backward pass,
                    # and update the weights.
                    loss = InfoNCELoss(q, k, self.queue_train, criterion, self.moco_model_args, device=self.device)

                    loss.backward()

                    self.optimizer.step()

                    train_epoch_loss += loss.item()
                    # queue update
                    self.queue_train = requeue(k, self.queue_train)
            scheduler.step()
            endTime = time.time()

            avg_train_loss = train_epoch_loss / len(dl_train)
            print('epoch {}, loss {}, epoch time {} seconds, time remaining {} hours'.format(
                epoch + 1,
                avg_train_loss,
                round(endTime - startTime, 2),
                round(1000 * (endTime - startTime) / 3600 - epoch * (endTime - startTime) / 3600, 2)
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

    def eval(self, dl_val: DataLoader):
        """
        Args:
            dl_val: model to evaluate
        Returns: f1 score
        """
        val_loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for val_data in dl_val:
                # image_val, label_val = val_data['image'].to(self.device), val_data['label'].to(self.device)
                image_val = val_data['image'].to(self.device)
                image_q = self.aug_transform(image_val)
                image_k = self.aug_transform(image_val)
                # encoders
                k = self.f_k(image_k)
                q = self.f_q(image_q)
                #
                loss = InfoNCELoss(q, k, self.queue_val, criterion, self.moco_model_args, device=self.device)
                val_loss += loss.item()

                self.queue_val = requeue(k, self.queue_val.to(self.device))

            return val_loss / len(dl_val)

    def momentum_update(self):
        m = self.moco_model_args['momentum']
        for theta_q, theta_k in zip(self.f_q.parameters(), self.f_k.parameters()):
            theta_k.data = theta_k.data * m + theta_q.data * (1. - m)
            theta_k.requires_grad = False

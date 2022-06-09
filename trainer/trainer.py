import numpy as np
import torch
import os.path as osp
from tabulate import tabulate

from metric import MetricTracker, segmentation_metrics, add_hist

class Trainer:
    def __init__(self, model, loss_fn, optimizer, lr_scheduler, metrics,
                 config, device,
                 train_loader, val_loader=None, len_epoch=None):
        self.config = config
        self.logger = self.config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.metric_ftns = metrics
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = config['trainer']['epochs']
        self.start_epoch = 1
        self.len_epoch = len(self.train_loader)
        self.do_validation = self.val_loader is not None

        self.log_step = int(np.sqrt(train_loader.batch_size))
        self.save_period = config['trainer']['save_period']
        self.checkpoint_dir = config.model_dir

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        self.train_met = MetricTracker(mode='train')
        self.val_met = MetricTracker(mode='val')

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # [1] Train
            train_log_dict, val_log_dict = self._train_epoch(epoch)

            # [2] Convert metric log to dataframe
            self.train_met.update_by_dict(train_log_dict)
            self.val_met.update_by_dict(val_log_dict)

            train_log_df = self.train_met.result_df()
            val_log_df = self.val_met.result_df()

            train_log_df.index.name = "EPOCH {}".format(epoch)
            val_log_df.index.name = "EPOCH {}".format(epoch)

            # [3] Print log dataframe
            self._report_log(epoch, train_log_df, val_log_df)

            # [4] Save Checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)


    def _train_epoch(self, epoch):
        self.model.train()

        hist = np.zeros((self.config['n_class'], self.config['n_class']))
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, aux1, aux2 = self.model(data)
            loss, dice, bce = self.loss_fn(output, target.to(torch.float32))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            #  output = torch.argmax(output, dim=1).detach().cpu().numpy()
            output = torch.sigmoid(output).squeeze().detach().cpu().numpy()
            output = (output > 0.5) * 1
            target = target.detach().cpu().numpy()
            hist = add_hist(hist, output, target)

            if batch_idx % self.log_step == 0:
                self.logger.info('Epoch: {} [{}/{}] Loss: {}'.format(
                    epoch,
                    batch_idx,
                    self.len_epoch,
                    loss.item()
                ))

        total_loss /= self.len_epoch
        acc, acc_cls, iou, miou = segmentation_metrics(hist)
        train_log = {
            'loss': total_loss,
            'acc': acc,
            'acc_cls': acc_cls,
            'mIoU': miou
        }

        train_log.update(
            self.train_met.ious2dict(iou)
        )

        # Validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return train_log, val_log


    def _valid_epoch(self, epoch):
        self.model.eval()

        total_loss = 0.0
        hist = np.zeros((self.config['n_class'], self.config['n_class']))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss, dice, bce = self.loss_fn(output, target.to(torch.float32))

                total_loss += loss.item()
                #  output = torch.argmax(output, dim=1).detach().cpu().numpy()
                output = torch.sigmoid(output).detach().cpu().numpy()
                output = (output > 0.5) * 1
                target = target.detach().cpu().numpy()
                hist = add_hist(hist, output, target)

            total_loss /= len(self.val_loader)
            acc, acc_cls, iou, miou = segmentation_metrics(hist)

        val_log = {
            'loss': total_loss,
            'acc': acc,
            'acc_cls': acc_cls,
            'mIoU': miou
        }

        val_log.update(self.val_met.ious2dict(iou))

        return val_log


    def _report_log(self, epoch, train_log_df, val_log_df):
        self.logger.info("-"*30)
        self.logger.info(" "*12 + "EPOCH [{}]".format(epoch) + " "*10)
        self.logger.info("-"*30)
        self.logger.info(" "*10 + "[TRAIN LOG]" + " "*10)
        self.logger.info(tabulate(train_log_df, headers='keys', tablefmt='psql'))
        self.logger.info(" "*10 + "[VAL LOG]" + " "*10)
        self.logger.info(tabulate(val_log_df, headers='keys', tablefmt='psql'))


    def _resume_checkpoint(self, resume_path):
        pass

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_sate': self.optimizer.state_dict(),
            'config': self.config
        }

        fname = osp.join(str(self.checkpoint_dir), 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, fname)
        self.logger.info('Saving checkpoint: {} ...'.format(fname))


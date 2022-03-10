import os
import sys
import time
import glob
import numpy as np
import random
import torch
import darts_cnn.utils as utils
import logging
import argparse
import torch.nn as nn
# import darts_cnn.genotypes as genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
# from collections import namedtuple
from torch.autograd import Variable
from darts_cnn.model import NetworkImageNet as Network
import pickle
from darts_cnn.utils import mixup_data, mixup_criterion
import torchvision.transforms as transforms

from .train_class import Genotype
# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class EvalImageNet:

    def __init__(self, image_data_path='./', test_mode=False):

        self.data = image_data_path
        self.test_mode = test_mode
        if test_mode:  self.batch_size = 128
        else:  self.batch_size = 512
        self.learning_rate = 0.333
        self.momentum = 0.9
        self.weight_decay = 3e-5
        self.report_freq = 10 if test_mode else 100
        # self.gpu = 0
        self.init_channels = 48
        self.layers = 14
        self.model_path = 'saved_models'
        self.auxiliary = True
        self.auxiliary_weight = 0.4
        self.cutout = True
        self.cutout_length = 16

        self.drop_path_prob = 0
        self.seed = 0
        self.grad_clip = 5.
        self.validation_set = True
        self.CLASSES = 1000
        self.label_smooth = 0.1
        self.n_workers = 8

        # settings on mixup augmentation: added by Xingchen
        self.mixup = False  # Mixup augmentation is implemented in NB301
        self.mixup_alpha = 0.2  # default mixup alpha in configspace.json

    def main(self, arch, epochs=250, load_weights=False, seed=0, save='arch_weights/method/arch_id/', **kwargs):
        num_gpus = torch.cuda.device_count()
        print(f'Number of GPUs available={num_gpus}')

        # Set up save file and logging
        self.save = save
        utils.create_exp_dir(self.save)
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        self.arch = arch
        if self.test_mode:
            self.epochs = 1
        else:
            self.epochs = epochs if epochs is not None else 250
        self.load_weights = load_weights
        # self.gpu = gpu
        # self.train_portion = train_portion
        # if self.train_portion == 1:
        #     self.validation_set = False
        self.seed = seed

        print('Train class params')
        print('arch: {}, epochs: {}, load_weights: {}'
              .format(arch, epochs, load_weights,))

        # cpu-gpu switch
        if not torch.cuda.is_available():
            raise ValueError('no gpu device available')
        else:
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            # device = torch.device(self.gpu)
            cudnn.benchmark = True
            cudnn.enabled = True
            cudnn.deterministic = True
            # logging.info('gpu device = %d' % self.gpu)

        if isinstance(arch, str): genotype = eval("genotypes.%s" % arch)
        else:   genotype = arch

        model = Network(self.init_channels, self.CLASSES, self.layers, self.auxiliary, genotype)
        if num_gpus > 1:
            model = nn.DataParallel(model)
            model = model.cuda()
        else:
            model = model.cuda()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        criterion_smooth = CrossEntropyLabelSmooth(self.CLASSES, self.label_smooth)
        criterion_smooth = criterion_smooth.cuda()

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # data_dir = os.path.join(self.data, 'imagenet')
        data_dir = self.data
        traindir = os.path.join(data_dir, 'train')
        validdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_data = dset.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
            ]))
        valid_data = dset.ImageFolder(
            validdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.n_workers)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.n_workers)

        if self.load_weights:
            logging.info('loading saved weights')
            ml = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(os.path.join(self.save, 'weights.pt'), map_location=ml))
            logging.info('loaded saved weights')

            with open(os.path.join(self.save, 'arch_results'), 'rb') as file:
                arch_results = pickle.load(file)

            return arch_results, model

        else:

            def adjust_lr(optimizer, epoch):
                # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
                if self.epochs - epoch > 5:
                    lr = self.learning_rate * (self.epochs - 5 - epoch) / (self.epochs - 5)
                else:
                    lr = self.learning_rate * (self.epochs - epoch) / ((self.epochs - 5) * 5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr

            train_accs = []
            valid_accs = []
            valid_accs_5 = []
            # test_accs = []
            train_losses = []
            train_runtimes = []
            valid_runtimes = []
            lr = self.learning_rate
            for epoch in range(self.epochs):
                current_lr = adjust_lr(optimizer, epoch)
                logging.info('epoch %d lr %e', epoch, current_lr)
                if epoch < 5 and self.batch_size > 256:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (epoch + 1) / 5.0
                    logging.info('Warming-up Epoch: %d, LR: %e', epoch, lr * (epoch + 1) / 5.0)
                if num_gpus > 1:  model.module.drop_path_prob = self.drop_path_prob * epoch / self.epochs
                else: model.drop_path_prob = self.drop_path_prob * epoch / self.epochs

                train_st = time.time()
                train_acc, train_obj = self.train(train_queue, model, criterion_smooth, optimizer)
                logging.info('train_acc %f, train_loss %f', train_acc, train_obj)
                train_time = time.time() - train_st

                with torch.no_grad():
                    valid_st = time.time()
                    valid_acc, valid_acc_top5, valid_obj = self.infer(valid_queue, model, criterion)
                    valid_time = time.time() - valid_st
                    logging.info(f'valid_acc top 1: {valid_acc}. top 5: {valid_acc_top5}')

                valid_accs.append(valid_acc)
                valid_accs_5.append(valid_acc_top5)
                train_accs.append(train_acc)
                # test_accs.append(test_acc)
                train_losses.append(train_obj)
                train_runtimes.append(train_time)
                valid_runtimes.append(valid_time)

                if max(valid_accs) == valid_acc: is_best = True
                else: is_best = False

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc_top1': max(valid_accs),
                    'optimizer': optimizer.state_dict(),
                }, is_best, self.save)

                # utils.save(model, os.path.join(self.save, 'weights.pt'))

                if epoch % 10 == 0:
                    arch_results = {
                        'full_train_accs': train_accs,
                        'full_valid_accs': valid_accs,
                        'full_valid_accs5': valid_accs_5,
                        'full_train_losses': train_losses,
                        'full_train_runtime': train_runtimes,
                        'full_valid_runtime': valid_runtimes,
                    }
                    with open(os.path.join(self.save, 'arch_results'), 'wb') as file:
                        pickle.dump(arch_results, file)

            arch_results = {
                'full_train_accs': train_accs,
                'full_valid_accs': valid_accs,
                'full_valid_accs5': valid_accs_5,
                'full_train_losses': train_losses,
                'full_train_runtime': train_runtimes,
                'full_valid_runtime': valid_runtimes,
            }

            return arch_results, model

    def train(self, train_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):
            # device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if self.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += self.auxiliary_weight * loss_aux

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)

            objs.update(loss.item(), n)
            # print(prec1)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                logging.info('train step=%03d loss=%e acc=%f', step, objs.avg, top1.avg, )

        return top1.avg, objs.avg

    def infer(self, valid_queue, model, criterion,):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        # device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')

        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                logging.info('valid step=%03d loss=%e top1=%f top5=%f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, top5.avg, objs.avg

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
import darts_cnn.genotypes as genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple
from torch.autograd import Variable
from darts_cnn.model import NetworkCIFAR as Network
import pickle
from darts_cnn.utils import mixup_data, mixup_criterion

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class Train:

    def __init__(self, image_data_path='./', dataset='cifar10'):

        self.data = image_data_path
        self.batch_size = 96
        self.learning_rate = 0.025
        self.learning_rate_min = 1e-8
        self.momentum = 0.9
        self.weight_decay = 3e-4
        self.report_freq = 100
        self.gpu = 0
        self.init_channels = 32
        self.layers = 8
        self.model_path = 'saved_models'
        self.auxiliary = True
        self.auxiliary_weight = 0.4
        self.cutout = True
        self.cutout_length = 16

        self.drop_path_prob = 0.2
        self.seed = 0
        self.grad_clip = 5
        self.validation_set = True
        if dataset == 'cifar10':
            self.CIFAR_CLASSES = 10
        elif dataset == 'cifar100':
            self.CIFAR_CLASSES = 100
        else:
            raise NotImplementedError
        self.dataset = dataset

        # settings on mixup augmentation: added by Xingchen
        self.mixup = True  # Mixup augmentation is implemented in NB301
        self.mixup_alpha = 0.2      # default mixup alpha in configspace.json

    def main(self, arch, epochs=100, gpu=0, load_weights=False, train_portion=0.8, seed=0,
             save='arch_weights/method/arch_id/', mixup=True, **kwargs):
        if not mixup:
            self.mixup = False

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
        self.epochs = epochs if epochs is not None else 100
        self.load_weights = load_weights
        self.gpu = gpu
        self.train_portion = train_portion
        if self.train_portion == 1:
            self.validation_set = False
        self.seed = seed

        print('Train class params')
        print('arch: {}, epochs: {}, gpu: {}, load_weights: {}, train_portion: {}'
              .format(arch, epochs, gpu, load_weights, train_portion))

        # cpu-gpu switch
        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            torch.manual_seed(self.seed)
            device = torch.device('cpu')

        else:
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            device = torch.device(self.gpu)
            cudnn.benchmark = True
            cudnn.enabled = True
            cudnn.deterministic = True
            logging.info('gpu device = %d' % self.gpu)

        if isinstance(arch, str):
            genotype = eval("genotypes.%s" % arch)
        else:
            genotype = arch

        model = Network(self.init_channels, self.CIFAR_CLASSES, self.layers, self.auxiliary, genotype)
        model = model.to(device)

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if self.dataset == 'cifar10':
            train_transform, test_transform = utils._data_transforms_cifar10(self.cutout, self.cutout_length)
            train_data = dset.CIFAR10(root=self.data, train=True, download=True, transform=train_transform)
            test_data = dset.CIFAR10(root=self.data, train=False, download=True, transform=test_transform)
        elif self.dataset == 'cifar100':
            train_transform, test_transform = utils._data_transforms_cifar100(self.cutout, self.cutout_length)
            train_data = dset.CIFAR100(root=self.data, train=True, download=True, transform=train_transform)
            test_data = dset.CIFAR100(root=self.data, train=False, download=True, transform=test_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        if self.validation_set:
            split = int(np.floor(self.train_portion * num_train))
        else:
            split = num_train

        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0)

        if self.validation_set:
            valid_queue = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size // 2,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                pin_memory=True, num_workers=0)

        test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size // 2, shuffle=False, pin_memory=True, num_workers=0)

        if self.load_weights:
            logging.info('loading saved weights')
            ml = 'cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(torch.load(os.path.join(self.save, 'weights.pt'), map_location=ml))
            logging.info('loaded saved weights')

            with open(os.path.join(self.save, 'arch_results'), 'rb') as file:
                arch_results = pickle.load(file)

            return arch_results, model

        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs),
                                                                   eta_min=self.learning_rate_min)

            valid_accs = []
            test_accs = []
            train_losses = []
            train_runtimes = []
            valid_runtimes = []
            for epoch in range(self.epochs):
                scheduler.step()
                logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
                model.drop_path_prob = self.drop_path_prob * epoch / self.epochs
                train_st = time.time()
                train_acc, train_obj = self.train(train_queue, model, criterion, optimizer)
                logging.info('train_acc %f, train_loss %f', train_acc, train_obj)
                train_time = time.time() - train_st

                if self.validation_set:
                    with torch.no_grad():
                        valid_st = time.time()
                        valid_acc, valid_obj = self.infer(valid_queue, model, criterion)
                        valid_time = time.time() - valid_st
                        logging.info('valid_acc %f', valid_acc)
                else:
                    valid_acc, valid_obj = 0, 0
                    valid_time = 0

                with torch.no_grad():
                    test_acc, test_obj = self.infer(test_queue, model, criterion, test_data=True)
                    logging.info('test_acc %f', test_acc)

                utils.save(model, os.path.join(self.save, 'weights.pt'))

                valid_accs.append(valid_acc)
                test_accs.append(test_acc)
                train_losses.append(train_obj)
                train_runtimes.append(train_time)
                valid_runtimes.append(valid_time)

                if epoch % 10 == 0:
                    arch_results = {
                        'full_valid_accs': valid_accs,
                        'full_test_accs': test_accs,
                        'full_train_losses': train_losses,
                        'full_train_runtime': train_runtimes,
                        'full_valid_runtime': valid_runtimes,
                    }
                    with open(os.path.join(self.save, 'arch_results'), 'wb') as file:
                        pickle.dump(arch_results, file)

            arch_results = {
                'full_valid_accs': valid_accs,
                'full_test_accs': test_accs,
                'full_train_losses': train_losses,
                'full_train_runtime': train_runtimes,
                'full_valid_runtime': valid_runtimes,
            }

            return arch_results, model

    def train(self, train_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        # top5 = utils.AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):
            device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')

            if self.mixup:
                input, target_a, target_b, lam = mixup_data(input, target, self.mixup_alpha,
                                                            use_cuda=torch.cuda.is_available())
                input, target_a, target_b = map(Variable, (input, target_a, target_b))
                target = target.to(device)
                input = input.to(device)
                target_a = target_a.to(device)
                target_b = target_b.to(device)
            else:
                input = Variable(input).to(device)
                target = Variable(target).to(device)

            logits, logits_aux = model(input)
            if self.mixup: loss = mixup_criterion(criterion, logits, target_a, target_b, lam)
            else:  loss = criterion(logits, target)
            if self.auxiliary:
                if self.mixup: loss_aux = mixup_criterion(criterion, logits_aux, target_a, target_b, lam)
                else: loss_aux = criterion(logits_aux, target)
                loss += self.auxiliary_weight * loss_aux

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
            optimizer.step()

            if self.mixup: prec1 = utils.accuracy_mixedup(logits, target, target_a, target_b, lam)
            else: prec1 = utils.accuracy(logits, target, )[0]
            n = input.size(0)

            objs.update(loss.item(), n)
            # print(prec1)
            top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                logging.info('train step=%03d loss=%e acc=%f', step, objs.avg, top1.avg, )

        return top1.avg, objs.avg

    def infer(self, valid_queue, model, criterion, test_data=False):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')

        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).to(device)
            target = Variable(target, volatile=True).to(device)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)

            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                if not test_data:
                    logging.info('valid step=%03d loss=%e top1=%f top5=%f', step, objs.avg, top1.avg, top5.avg)
                else:
                    logging.info('test step=%03d loss=%e top1=%f top5=%f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg


# class Eval(Train):
#     def __init__(self, image_data_path='./'):
#         super(Eval, self).__init__(image_data_path=image_data_path)
#         # PC-DARTS hyperparams
#         self.init_channels = 36
#         self.epochs = 600
#         self.batch_size = 96
#         # self.drop_path_prob = 0.3
#         self.layers = 20
#         self.validation_set = False
#         self.mixup = False
#
#     def main(self, arch, epochs=600, gpu=0, load_weights=False, train_portion=1, seed=0,
#              save='arch_weights/method/arch_id/'):
#         super(Eval, self).main(arch, epochs, gpu, load_weights, train_portion, seed, save)

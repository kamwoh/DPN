import argparse
import logging
import os

import torch

import configs
from scripts import train_hashing

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s %(asctime)s: %(message)s',
                    datefmt='%d-%m-%y %H:%M:%S')

torch.backends.cudnn.benchmark = True
configs.default_workers = os.cpu_count()

parser = argparse.ArgumentParser(description='DPN')
parser.add_argument('--nbit', default=64, type=int, help='number of bits')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--epochs', default=200, type=int, help='training epochs')
parser.add_argument('--ds', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet', 'nuswide'], help='dataset')
parser.add_argument('--arch', default='alexnet', choices=['alexnet'], help='backbone name')

# loss related
parser.add_argument('--margin', default=1.0, type=float, help='dpn margin')
parser.add_argument('--sl', default=1.0, type=float, help='dpn loss scale')
parser.add_argument('--ce', default=0.0, type=float, help='ce loss scale')
parser.add_argument('--reg', default=0.0, type=float, help='regularization on the outputs from hash layer')

# update centroid
parser.add_argument('--centroid-update-rate', default=1.0, type=float,
                    help='ratio of training data to update centroid (0~1)')
parser.add_argument('--centroid-update-time', default=0, type=int,
                    help='number of epochs to update centroid (non-negative)')
parser.add_argument('--centroid-method', default='N', choices=['N', 'B', 'O'], help='N = sign of gaussian; '
                                                                                    'B = bernoulli; '
                                                                                    'O = optimize')

parser.add_argument('--seed', default=torch.randint(100000, size=()).item(), help='seed number; default: random')

args = parser.parse_args()

config = {
    'arch': args.arch,
    'arch_kwargs': {
        'nbit': args.nbit,
        'nclass': 0,  # will be updated below
        'pretrained': True,
        'freeze_weight': False,
    },
    'batch_size': args.bs,
    'dataset': args.ds,
    'multiclass': args.ds == 'nuswide',
    'dataset_kwargs': {
        'resize': 256 if args.ds in ['imagenet100', 'nuswide'] else 224,
        'crop': 224,
        'norm': 2,
        'evaluation_protocol': 1,  # only affect cifar10
        'reset': False,
        'separate_multiclass': False,
    },
    'optim': 'sgd',
    'optim_kwargs': {
        'lr': args.lr,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'nesterov': False,
        'betas': (0.9, 0.999)
    },
    'epochs': args.epochs,
    'scheduler': 'step',
    'scheduler_kwargs': {
        'step_size': int(args.epochs * 0.5),
        'gamma': 0.1,
        'milestones': '0.5,0.75'
    },
    'save_interval': 0,
    'eval_interval': 10,
    'tag': 'dpn',
    'seed': args.seed,

    'update_rate': args.centroid_update_rate,
    'update_time': args.centroid_update_time,

    'centroid_generation': args.centroid_method,
    'margin': args.margin,
    'sl': args.sl,
    'ce': args.ce,
    'reg': args.reg
}

config['arch_kwargs']['nclass'] = configs.nclass(config)
config['R'] = configs.R(config)

logdir = (f'logs/{config["arch"]}{config["arch_kwargs"]["nbit"]}_'
          f'{config["dataset"]}_{config["dataset_kwargs"]["evaluation_protocol"]}_'
          f'{config["epochs"]}_'
          f'{config["optim_kwargs"]["lr"]}_'
          f'{config["optim"]}_'
          f'{config["sl"]}_{config["ce"]}')

if config['tag'] != '':
    logdir += f'/{config["tag"]}_{config["seed"]}_'
else:
    logdir += f'/{config["seed"]}_'

# make sure no overwrite problem
count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

while os.path.isdir(logdir):
    count += 1
    logdir = orig_logdir + f'{count:03d}'

config['logdir'] = logdir

count = 0
orig_logdir = logdir
logdir = orig_logdir + f'{count:03d}'

train_hashing.main(config)

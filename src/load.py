from __future__ import division
import os 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

import pandas as pd
import argparse
import shutil
import numpy as np
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import OrderedDict
from torchvision.models.resnet import resnet18 as raw_resnet18
# from dy_models.dy_resnet import resnet18 as dy_resnet18
from dy_resnet import resnet18 as dy_resnet18
from datetime import datetime

# from main import train
from main import load_ckp
from main import train
parser = argparse.ArgumentParser(description='dynamic convolution')
parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.1, )
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--net-name', default='dy_resnet18')

args = parser.parse_args()
print(args)
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# args.device = 'cpu' if torch.cuda.is_available() else 'cpu'




if args.dataset == 'cifar10':
    numclasses=10
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

loaders ={
    trainloader,testloader
}
model = dy_resnet18(num_classes=numclasses)
print("plp")
# model.eval()
print("lla")


# print(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

ckm_path = '/home/varshittha/dynamic-convolution/src/checkpoint/current_checkpoint.pt'

model, optimizer, start_epoch, valid_loss_min = load_ckp(ckm_path, model, optimizer)
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(),"GPUs available.")
    model = nn.DataParallel(model.cuda(),device_ids=[2,0,1,3])

model.to(f'cuda:{model.device_ids[0]}')
print("model = ", model)
print("optimizer = ", optimizer)
print("start_epoch = ", start_epoch)
print("valid_loss_min = ", valid_loss_min)
print("valid_loss_min = {:.6f}".format(valid_loss_min))

trained_model = train(start_epoch,60, valid_loss_min, loaders, model, optimizer, "./checkpoint/current_checkpoint.pt", "./best_model/best_model.pt")
# trained_model.eval() 
"""
to set dropout and batch, normalization layers to evaluation mode before running inference.
 Failing to do this will yield inconsistent inference results 

"""
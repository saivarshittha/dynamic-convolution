from __future__ import division
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os 
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
elif args.dataset=='cifar100':
    numclasses=100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
loaders ={
    trainloader,testloader
}
if args.net_name=='dy_resnet18':
    # print('nnn')
    model = dy_resnet18(num_classes=numclasses)

    # print('i doubt')
elif args.net_name=='raw_resnet18':
    model = raw_resnet18(num_classes=numclasses)
elif args.net_name=='raw_vgg11':
    model = raw_vgg11(num_classes=numclasses)
elif args.net_name=='dy_vgg11':
    model = dy_vgg11(num_classes=numclasses)
if torch.cuda.device_count() > 1:
    print(torch.cuda.device_count(),"GPUs available.")
    model = nn.DataParallel(model.cuda(),device_ids=[2,0,1,3])
# model.to(f'cuda:{model.device_ids[0]}')
# model = nn.DataParallel(model.module,device_ids = [2,0,1,3])
model.to(f'cuda:{model.device_ids[0]}')
# args.device = f'cuda:{model.device_ids[0]}'
# model.to(args.device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
print(str(args))

def time_stamp():
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Time : ",dt_string)

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()


def adjust_lr(optimizer, epoch):
    if epoch in [args.epochs*0.5, args.epochs*0.75, args.epochs*0.85]:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
            lr = p['lr']
        print('Change lr:'+str(lr))

def train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, checkpoint_path, best_model_path):
    """
    Keyword arguments:
    start_epochs -- the real part (default 0.0)
    n_epochs -- the imaginary part (default 0.0)
    valid_loss_min_input
    loaders
    model
    optimizer
    criterion
    use_cuda
    checkpoint_path
    best_model_path
    
    returns trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input 
    best_val_acc   = 0.0
    time_stamp()
    
    for epoch in range(start_epochs, n_epochs+1):
        print('Hi!!')
        print("epoch = ",epoch)
        # initialize variables to monitor training and validation loss
        avg_loss = 0.0
        test_loss = 0.0
        train_acc  = 0.0
        val_acc    = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        adjust_lr(optimizer, epoch)
        for batch_idx, (data, target) in enumerate(trainloader):
            
            data, target = data.to(f'cuda:{model.device_ids[0]}'), target.to(f'cuda:{model.device_ids[0]}')
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            avg_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {}, loss{:.6f}, acc{}'.format(epoch, loss.item(), train_acc/len(trainloader.dataset)), end='')

            
        if args.net_name.startswith('dy'):
            model.module.update_temperature()

        

        ######################    
        # validate the model #
        ######################
        model.eval()
        correct = 0.0
        with torch.no_grad():
            for data,label in testloader:
                data, label = data.to(f'cuda:{model.device_ids[0]}'), label.to(f'cuda:{model.device_ids[0]}')
                output = model(data)
                test_loss += F.cross_entropy(output, label, reduction = 'sum').item()
                pred =  output.data.max(1, keepdim=True)[1]
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        test_loss/=len(testloader.dataset)    
        correct = int(correct)
        print('Test set:average loss: {:.4f}, accuracy{}'.format(test_loss, 100.*correct/len(testloader.dataset)))
        val_acc = correct/len(testloader.dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    
 
        # print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        #     epoch, 
        #     train_loss,
        #     test_loss
        #     ))
        
        # create checkpoint variable and add important data
        checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        
        # save checkpoint
        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        ## TODO: save the model if validation loss has decreased
        print('val_loss this time =',test_loss,'min = ',valid_loss_min)
        if test_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,test_loss))
            # save checkpoint as best model
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = test_loss
    print(epoch,'epoch ended')   
    print('Best acc{}'.format(best_val_acc))     
    # return trained model
    return model
print('YOLO')
checkpoint_path = '/home/varshittha/dynamic-convolution/src/checkpoint/current_checkpoint.pt'
best_model_path = '/home/varshittha/dynamic-convolution/src/best_model/best_model.pt'
start_epochs    = 0
n_epochs        = args.epochs
valid_loss_min_input = np.Inf
print("IOLO!")
trained_model = train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, checkpoint_path, best_model_path)

if __name__ == '__main__':
    checkpoint_path = '/home/varshittha/dynamic-convolution/src/checkpoint/current_checkpoint.pt'
    best_model_path = '/home/varshittha/dynamic-convolution/src/best_model/best_model.pt'
    start_epochs    = 0
    n_epochs        = args.epochs
    valid_loss_min_input = np.Inf
    print("IOLO!")
    trained_model = train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, checkpoint_path, best_model_path)

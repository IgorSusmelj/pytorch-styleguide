# import statements
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms, models
import numpy as np
import argparse
from tqdm import tqdm
import time

from tensorboardX import SummaryWriter

from utils import load_checkpoint, save_checkpoint, ensure_dir
from model import MyModel

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a network for image classification on cifar10.")
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Spcifies learing rate for optimizer. (default: 1e-3)')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_checkpoint',
        type=str,
        default='',
        help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs. (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loaders. (default: 32)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for data loader. (default: 8)'
    )
    
    opt = parser.parse_args()
    
    # add code for datasets (we always use train and validation/ test set)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='cifar10',
                                     train=True, 
                                     transform=train_transforms, 
                                     download=True)
    train_data_loader = data.DataLoader(train_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=opt.num_workers)

    test_dataset = datasets.CIFAR10(root='cifar10',
                                     train=False, 
                                     transform=test_transforms, 
                                     download=True)
    test_data_loader = data.DataLoader(test_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False,
                                        num_workers=opt.num_workers)

    
    # instantiate network (which has been imported from *networks.py*)
    net = MyModel()
    
    # create losses (criterion in pytorch)
    criterion_CE = nn.CrossEntropyLoss()
    
    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
    
    # create optimizers
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr)
    
    # load checkpoint if needed/ wanted
    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint) # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")
        
    
    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    
    # now we start the main loop
    n_iter = start_n_iter
    for epoch in range(start_epoch, opt.epochs):
        # set models to train mode
        net.train()
        
        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_data_loader),
                    total=len(train_data_loader))
        start_time = time.time()
        
        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            
            # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
            prepare_time = start_time-time.time()
            
            # forward and backward pass
            out = net(img)
            loss = criterion_CE(out, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # udpate tensorboardX
            writer.add_scalar('train_loss', n_iter)
            
            # compute computation time and *compute_efficiency*
            process_time = start_time-time.time()-prepare_time
            compute_efficiency = process_time/(process_time+prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, ' 
                f'loss: {loss.item():.2f},  epoch: {epoch}/{opt.epochs}')
            start_time = time.time()
            
        # maybe do a test pass every N=1 epochs
        if epoch % 1 == 0:
            # bring models to evaluation mode
            net.eval()

            correct = 0
            total = 0

            pbar = tqdm(enumerate(test_data_loader),
                    total=len(test_data_loader)) 
            with torch.no_grad():
                for i, data in pbar:
                    # data preparation
                    img, label = data
                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()
                    
                    out = net(img)
                    _, predicted = torch.max(out.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            print(f'Accuracy on test set: {100*correct/total:.2f}')

                
            # save checkpoint if needed
            cpkt = {
                'net': net.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
                'optim': optim.state_dict()
            }
            save_checkpoint(cpkt, 'model_checkpoint.ckpt')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
import torch.cuda.amp as amp
from tqdm import tqdm
import math

from models import Discriminator, Generator
from dataset import FaceDataset


config = {
    'image_size': 128,
    'batch_size': 100,
    'ngf': 512,
    'ndf': 16,
    'nzf': 100,
    'optimizer': 'sgd',
    'lr_D': 0.01,
    'lr_G': 0.01,
    'momentum': 0.9,
    'nestrov': False,
    'lr_schedule': None,
    'epochs': 100
    }
image_size = config['image_size']
batch_size = config['batch_size']

data_path = '/home/scott/Desktop/dataset/face3k/train.txt'
train_set = FaceDataset(data_path, 128, augment=True, cache_image=False)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)

netG = Generator(input_size=(batch_size, 100, 1, 1),
                 image_size=image_size,
                 ngf=config['ngf'],
                 leaky_relu=True)
netD = Discriminator(image_size=config['image_size'],
                     ndf=config['ndf'],
                     leaky_relu=True)
'''
load trained weight
'''

'''
create optimizer & lr schedule
TTL
'''
# filter 使用方式
# filter(function, iterable):
# function: 判斷函數
# iterable: 可迭代對象
optimizerD = None
optimizerG = None
if config['optimizer'] == 'sgd':
    optimizerD = SGD(params=filter(lambda p: p.requires_grad, netD.parameters()),
                     lr=config['lr_D'],
                     momentum=config['momentum'],
                     nesterov=config['nestrov'])
    optimizerG = SGD(params=filter(lambda p: p.requires_grad, netG.parameters()),
                     lr=config['lr_G'],
                     momentum=config['momentum'],
                     nesterov=config['nestrov'])


epochs = config['epochs']
lr_sch = None
if config['lr_schedule'] == 'exp-1.5pi':
    lr_lambda = lambda x: math.exp(-1.5 * x * math.pi / epochs)
    lr_sch = lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda, last_epoch=epochs)


'''
create loss
'''
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netD.to(device)
netG.to(device)
for epoch in range(epochs):
    pbar = tqdm(train_loader, leave=True, total=len(train_loader))
    for b_idx, (image, target) in enumerate(pbar):
        optimizerG.zero_grad()
        optimizerD.zero_grad()

        netD.train()
        netG.train()
        '''
        train Discriminator
        1. train D with real data & real labels
        2. train G with fake data which is created by G and noise z
         
        '''

        out = netD(image.to(device)).view(-1)

        real_targets = torch.ones(image.shape[0]).to(device)

        loss_D_real = criterion(out, real_targets)
        loss_D_real.backward()

        z = torch.randn([config['batch_size'], config['nzf'], 1, 1], device=device)
        fake_data = netG(z)
        fake_targets = torch.zeros(fake_data.shape[0]).to(device)
        out = netD(fake_data.detach()).view(-1)
        loss_D_fake = criterion(out, fake_targets)
        loss_D_fake.backward()
        optimizerD.step()


        fake_data = torch.randn([batch_size, 100, 1, 1]).to(device)


        out_G = netG(fake_data)
        fake_target = torch.ones(out_G.shape[0]).to(device)
        loss_G = -1 * criterion(netD(out_G).view(-1), fake_target)
        loss_G.backward()
        optimizerG.step()


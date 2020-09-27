import torch
from torch.utils.data import DataLoader
from models import Discriminator, Generator
from dataset import FaceDataset
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
import math
config = {
    'image_size': 128,
    'batch_size': 128,
    'ngf': 512,
    'ndf': 16,
    'optimizer': 'sgd',
    'lr': 0.01,
    'momentum': 0.9,
    'nestrov': False,
    'lr_schedule': None,
    'epochs': 100
    }

data_path = '/Users/lionl771128/Documents/DL_Project/GAN/DCGAN/train_128.txt'
train_set = FaceDataset(data_path, 128)
train_loader = DataLoader(train_set,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True)

netG = Generator(input_size=(config['batch_size'], 100, 1, 1),
                 image_size=config['image_size'],
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
'''
# filter 使用方式
optimizerD = None
optimizerG = None
if config['optimizer'] == 'sgd':
    optimizerD = SGD(params=filter(lambda p: p.requires_grad, netD.parameters()),
                     lr=config['lr'],
                     momentum=config['momentum'],
                     nesterov=config['nesterov'])
    optimizerG = SGD(params=filter(lambda p: p.requires_grad, netG.parameters()),
                     lr=config['lr'],
                     momentum=config['momentum'],
                     nesterov=config['nesterov'])


epochs = config['epochs']
lr_scheduler = None
if config['lr_schedule'] == 'exp-1.5pi':
    lr_lambda = lambda x: math.exp(-1.5 * x * math.pi / epochs)
    lr_scheduler.LambdaLR(optimizerD, lr_lambda=lr_lambda, last_epoch=epochs)



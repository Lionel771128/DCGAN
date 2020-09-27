import torch
from models import Discriminator, Generator

config = {
    'image_size': 128,
    'batch_size': 128,
    'ngf': 512,
    'ndf': 16,
    'lr': 0.01,
    'momentum': 0.9,
    'nestrov': False,
    'lr_schedule': None
    }



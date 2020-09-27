import torch.nn as nn
import math
from collections import OrderedDict


class Generator(nn.Module):
    def __init__(self, input_size, image_size, ngf, leaky_relu=False):
        self.input_size = input_size # b,c,h,w
        self.image_size = image_size
        self.num_of_block = int(math.log(image_size, 2))
        self.activation = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(1, inplace=True)
        for i in range(self.num_of_block-1):
            if i == 0:
                in_channel = self.input_size[1]
                padding = 0
                stride = 1
            else:
                in_channel = ngf / (2 ** (i - 1))
                padding = 2
                stride = 2
            out_channel = ngf / (2 ** i)
            g_block = OrderedDict([(f'g{i}_Tconv', nn.ConvTranspose2d(in_channels=in_channel,
                                                                      out_channels=out_channel,
                                                                      kernel_size=4,
                                                                      stride=stride,
                                                                      padding=padding)),
                                   (f'g{i}_bn', nn.BatchNorm2d(out_channel)),
                                   (f'g{i}_leaky_relu' if leaky_relu else f'g{i}_relu', self.activation)])
            setattr(self, f'g{i}_block', g_block)
        self.last_gblock = nn.Sequential(OrderedDict([('last_conv', nn.ConvTranspose2d(in_channels=ngf // pow(2, self.num_of_block-1),
                                                                                       out_channels=3,
                                                                                       kernel_size=4,
                                                                                       stride=2,
                                                                                       padding=1))]))
        self.tanh = nn.Tanh()

    def foward(self, x):
        for i in range(self.num_of_block - 1):
            x = getattr(self, f'g{i}_block')(x)
        x = self.last_gblock(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size=128, ndf=16, leaky_relu=False):
        self.image_size = image_size
        self.num_of_block = int(math.log(self.image_size // 4, 2))
        self.activation = nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(1, inplace=True)
        for i in range(self.num_of_block):
            if i == 0:
                in_channel = 3
                out_channel = ndf
            else:
                in_channel = ndf * pow(2, i-1)
                out_channel = ndf * pow(2, i)
            d_block = OrderedDict([(f'd{i}_conv',
                                    nn.Conv2d(in_channels=in_channel,
                                              out_channels=out_channel,
                                              kernel_size=4,
                                              stride=2,
                                              padding=1)),
                                   (f'd{i}_bn', nn.BatchNorm2d(out_channel)),
                                   (f'd{i}_leaky_relu' if leaky_relu else f'd{i}_relu', self.activation)])
            setattr(self, f'dblock{i}', nn.Sequential(d_block))

        self.last_dblock = nn.Sequential(OrderedDict([('last_conv', nn.Conv2d(in_channels=ndf*pow(2, self.num_of_block),
                                                                              out_channels=1,
                                                                              kernel_size=4,
                                                                              stride=1,
                                                                              padding=0))]))

    def forward(self, x):
        for i in range(self.num_of_block):
            x = getattr(self, f'dblock{i}')(x)
        x = self.last_dblock(x)
        return x

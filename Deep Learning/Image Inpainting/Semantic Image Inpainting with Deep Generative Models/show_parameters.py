import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
from torchvision.utils import make_grid
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
import argparse
from DataSet.RemoteSensingData import NWPUDataSet
from SNGANPart.dcgan import Generator, Discriminator

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=3, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=30, help='training batch size')

opt = parser.parse_args()
print(opt)

n_dis = opt.n_dis
nz = opt.nz

G = Generator(nz, 3)
SND = Discriminator(3)

for i, param in enumerate(G.parameters()):
    print(i, ':', param.size())
    # print(param[0, 0].detach().numpy())
    # print(param[1, 0].detach().numpy())
    if i is 20:
        vutils.save_image(param[:, :3],
                          '%s/parameters%03d.png' % ('output', 1),
                          normalize=True)
    # break

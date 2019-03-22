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

device = torch.device("cuda:0")

dataset = NWPUDataSet()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize
                                         , shuffle=True, num_workers=0)


def show_img(img):
    img = make_grid(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


n_dis = opt.n_dis
nz = opt.nz

input = torch.FloatTensor(opt.batchsize, 3, 128, 128)
noise = torch.FloatTensor(opt.batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchsize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchsize)
real_label = 1
fake_label = 0

G = Generator(nz, 3)
SND = Discriminator(3)
print(G)
print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

G = torch.load('netG.pth')
SND = torch.load('netD.pth')

criterion = nn.BCELoss()

G.to(device)
SND.to(device)
criterion.to(device)
input, label = input.to(device), label.to(device)
noise, fixed_noise = noise.to(device), fixed_noise.to(device)

optimizerG = optim.Adam(G.parameters(), lr=0.00002, betas=(0, 0.999))
optimizerSND = optim.Adam(SND.parameters(), lr=0.00002, betas=(0, 0.999))

list_G_loss = []
list_D_loss = []

for epoch in range(2):
    for i, data in enumerate(dataloader, 0):
        step = epoch * len(dataloader) + i
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        # if opt.cuda:
        #    real_cpu = real_cpu.cuda()
        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = input.to(device)
        labelv = label.to(device)
        output = SND(inputv)

        # errD_real = torch.mean(F.softplus(-output))
        errD_real = criterion(output, labelv)
        errD_real.backward()

        D_x = output.data.mean()
        # train with fake
        noise.resize_(batch_size, noise.size(1), noise.size(2), noise.size(3)).normal_(0, 1)
        noisev = noise.to(device)
        fake = G(noisev)
        labelv = label.fill_(fake_label).to(device)
        output = SND(fake.detach())
        # errD_fake = torch.mean(F.softplus(output))
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake

        optimizerSND.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if step % n_dis == 0:
            G.zero_grad()
            labelv = label.fill_(real_label).to(device)  # fake labels are real for generator cost
            output = SND(fake)
            # errG = torch.mean(F.softplus(-output))
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 200, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            list_G_loss.append(errG.item())
            list_D_loss.append(errD.item())
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              '%s/real_samples.png' % 'output',
                              normalize=True)
            fake = G(fixed_noise)
            vutils.save_image(fake.data,
                              '%s/fake_samples_epoch_%03d.png' % ('output', epoch + 26),
                              normalize=True)

torch.save(G, 'netG.pth')
torch.save(SND, 'netD.pth')

f = open("loss_G.txt", "a+")
for i in list_G_loss:
    f.write(str(i) + '\n')
f.close()

f = open("loss_D.txt", "a+")
for i in list_D_loss:
    f.write(str(i) + '\n')
f.close()

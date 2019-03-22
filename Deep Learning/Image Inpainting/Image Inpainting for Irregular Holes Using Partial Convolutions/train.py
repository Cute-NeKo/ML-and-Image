import argparse
import os
import torch
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import make_grid
import torchvision.utils as vutils

import numpy as np
import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from dataset.RS_Mask_Data import RSMaskDataSet

from util.image import unnormalize
from util.io import load_ckpt
from util.io import save_ckpt

mean = np.array(opt.MEAN)
std = np.array(opt.STD)

in_transform = transforms.Compose(
    [transforms.Normalize(list(-1 * mean / std), list(1.0 / std))])


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_finetune', type=float, default=0.00005)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batchsize', type=int, default=5)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

dataset_train = RSMaskDataSet()

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize
                                               , shuffle=True, num_workers=0)

model = PConvUNet().to(device)

model = torch.load('PConvUNet.pth')

if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00005)
criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

list_loss_dict = []
list_tocal_loss = []
for epoch in range(5):
    model.train()
    for i, data in enumerate(dataloader_train):
        image, mask, gt = data
        image = image.to(device)
        mask = mask.to(device)
        gt = gt.to(device)
        output, _ = model(image, mask)
        loss_dict = criterion(image, mask, output, gt)

        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value
            ##parameters

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print(
                '[%d] [%d/%d] total_loss: %.4f valid_loss: %.4f hole_loss: %.4f tv_loss: %.4f  prc_loss: %.4f style_loss: %.4f'
                % (epoch, i, len(dataloader_train),
                   loss.item(), loss_dict['valid'].item(), loss_dict['hole'].item(), loss_dict['tv'].item(),
                   loss_dict['prc'].item(), loss_dict['style'].item()))
            list_tocal_loss.append(loss.item())
            list_loss_dict.append(loss_dict)

        if i % 100 == 0:

            vutils.save_image(gt,
                              '%s/real_samples.png' % 'output',
                              normalize=True)

            output_trans = torch.ones_like(output)
            for i in range(args.batchsize):
                # output_trans[i] = in_transform(output[i])
                output_trans[i][0] = output[i][0] * opt.STD[0] + opt.MEAN[0]
                output_trans[i][1] = output[i][1] * opt.STD[1] + opt.MEAN[1]
                output_trans[i][2] = output[i][2] * opt.STD[2] + opt.MEAN[2]
            vutils.save_image(output_trans,
                              '%s/output%03d.png' % ('output', epoch + 193),
                              normalize=True)

            vutils.save_image(mask,
                              '%s/mask%03d.png' % ('output', epoch + 193),
                              normalize=True)

torch.save(model, 'PConvUNet.pth')

f = open("txt/total_loss.txt", "a+")
for a in list_tocal_loss:
    f.write(str(a) + '\n')
f.close()

f = open("txt/loss_dict.txt", "a+")
for a in list_loss_dict:
    f.write(str(a['valid'].item()) + ' ')
    f.write(str(a['hole'].item()) + ' ')
    f.write(str(a['tv'].item()) + ' ')
    f.write(str(a['prc'].item()) + ' ')
    f.write(str(a['style'].item()) + '\n')
f.close()

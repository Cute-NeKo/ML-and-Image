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
from data.RS_Mask_Data import RSMaskDataSet
from model.gan import InpaintSAGNet, InpaintSADirciminator
from model.loss import SNDisLoss, SNGenLoss, ReconLoss
import time

mean = np.array(opt.MEAN)
std = np.array(opt.STD)

in_transform = transforms.Compose(
    [transforms.Normalize(list(-1 * mean / std), list(1.0 / std))])

parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_finetune', type=float, default=0.00005)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batchsize', type=int, default=3)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=5000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0')

dataset_train = RSMaskDataSet()

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize
                                               , shuffle=True, num_workers=0)


def train(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, dataloader, val_datas=None):
    """
    Train Phase, for training and spectral normalization patch gan in
    Free-Form Image Inpainting with Gated Convolution (snpgan)

    """
    netG.to(device)
    netD.to(device)

    list_dloss = []
    list_gloss = []
    list_rloss = []
    for epoch in range(5):
        netG.train()
        netD.train()
        for i, (imgs, masks) in enumerate(dataloader):
            # Optimize Discriminator
            optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()
            # mask is 1 on masked region
            imgs, masks = imgs.to(device), masks.to(device)

            coarse_imgs, recon_imgs = netG(imgs, masks)
            # vutils.save_image(imgs,
            #                   '%s/real_samples.png' % 'output',
            #                   normalize=True)
            # time.sleep(10000)

            complete_imgs = recon_imgs * masks + imgs * (1 - masks)

            pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
            neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
            d_loss = DLoss(pred_pos, pred_neg)
            d_loss.backward(retain_graph=True)

            optD.step()

            # Optimize Generator
            optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
            pred_neg = netD(neg_imgs)
            # pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
            g_loss = GANLoss(pred_neg)
            r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

            whole_loss = g_loss + r_loss

            # Update the recorder for losses
            whole_loss.backward()

            optG.step()

            if i % 20 == 0:
                print(
                    '[%d/%d][%d/%d] d_loss: %.4f g_loss: %.4f r_loss: %.4f'
                    % (epoch, 3, i, len(dataloader_train),
                       d_loss.item(), g_loss.item(), r_loss.item()))

                list_dloss.append(d_loss.item())
                list_gloss.append(g_loss.item())
                list_rloss.append(r_loss.item())

            if i % 100 == 0:
                vutils.save_image(imgs,
                                  '%s/real_samples.png' % 'output',
                                  normalize=True)
                output_trans = torch.ones_like(recon_imgs)
                for i in range(args.batchsize):
                    # output_trans[i] = in_transform(output[i])
                    output_trans[i][0] = recon_imgs[i][0] * opt.STD[0] + opt.MEAN[0]
                    output_trans[i][1] = recon_imgs[i][1] * opt.STD[1] + opt.MEAN[1]
                    output_trans[i][2] = recon_imgs[i][2] * opt.STD[2] + opt.MEAN[2]

                masks = 1 - masks
                # aaa = in_transform(imgs[0])
                # result = aaa.cpu() * masks[0].cpu() + output_trans[0].cpu() * (1 - masks[0].cpu())
                # result = transforms.ToPILImage()(result.cpu()).convert('RGB')
                # result.show()
                # aaa = in_transform(imgs[1])
                # result = aaa.cpu() * masks[1].cpu() + output_trans[1].cpu() * (1 - masks[1].cpu())
                # result = transforms.ToPILImage()(result.cpu()).convert('RGB')
                # result.show()
                # aaa = in_transform(imgs[2])
                # result = aaa.cpu() * masks[2].cpu() + output_trans[2].cpu() * (1 - masks[2].cpu())
                # result = transforms.ToPILImage()(result.cpu()).convert('RGB')
                # result.show()

                vutils.save_image(output_trans,
                                  '%s/output%03d.png' % ('output', epoch + 585),
                                  normalize=True)
                vutils.save_image(1-masks,
                                  '%s/mask%03d.png' % ('output', epoch + 585),
                                  normalize=True)

    torch.save(netG, 'netG.pth')
    torch.save(netD, 'netD.pth')

    f = open("txt/dis_loss.txt", "a+")
    for a in list_dloss:
        f.write(str(a) + '\n')
    f.close()

    f = open("txt/gan_loss.txt", "a+")
    for a in list_gloss:
        f.write(str(a) + '\n')
    f.close()

    f = open("txt/recon_loss.txt", "a+")
    for a in list_rloss:
        f.write(str(a) + '\n')
    f.close()


netG = InpaintSAGNet()
netD = InpaintSADirciminator()
netG = torch.load('netG.pth')
netD = torch.load('netD.pth')

# Define loss
recon_loss = ReconLoss(1.2, 1.2, 1.2, 1.2)
gan_loss = SNGenLoss(0.005)
dis_loss = SNDisLoss()
lr, decay = 0.0001, 0.0
optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay)
optD = torch.optim.Adam(netD.parameters(), lr=4 * lr, weight_decay=decay)

train(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, dataloader_train)

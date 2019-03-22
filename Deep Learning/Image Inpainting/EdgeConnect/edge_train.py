import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from model.networks import EdgeGenerator, InpaintGenerator, Discriminator
from model.loss import AdversarialLoss, PerceptualLoss, StyleLoss

from data.RS_Edge_Data import RSEdgeDataSet
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
# training options

parser.add_argument('--batchsize', type=int, default=3)
parser.add_argument('--LR', type=float, default=0.0001)
parser.add_argument('--BETA1', type=float, default=0.0)
parser.add_argument('--BETA2', type=float, default=0.9)
parser.add_argument('--D2G_LR', type=float, default=0.1)
parser.add_argument('--GAN_LOSS', type=str, default='nsgan')
parser.add_argument('--INPAINT_ADV_LOSS_WEIGHT', type=float, default=0.1)
parser.add_argument('--L1_LOSS_WEIGHT', type=float, default=1)
parser.add_argument('--STYLE_LOSS_WEIGHT', type=float, default=250)
parser.add_argument('--CONTENT_LOSS_WEIGHT', type=float, default=0.1)
parser.add_argument('--FM_LOSS_WEIGHT', type=float, default=10)

args = parser.parse_args()

device = torch.device('cuda:0')

dataset_train = RSEdgeDataSet()
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize
                                               , shuffle=True, num_workers=0)

generator = EdgeGenerator(use_spectral_norm=True).to(device)
discriminator = Discriminator(in_channels=2, use_sigmoid=args.GAN_LOSS != 'hinge').to(device)

generator = torch.load('Edge_generator.pth')
discriminator = torch.load('Edge_discriminator.pth')

l1_loss = nn.L1Loss().to(device)
adversarial_loss = AdversarialLoss(type=args.GAN_LOSS).to(device)

gen_optimizer = optim.Adam(
    params=generator.parameters(),
    lr=float(args.LR),
    betas=(args.BETA1, args.BETA2)
)

dis_optimizer = optim.Adam(
    params=discriminator.parameters(),
    lr=float(args.LR) * float(args.D2G_LR),
    betas=(args.BETA1, args.BETA2)
)

list_dloss = []
list_gloss = []
for epoch in range(5):
    for i, (imgs, edges, masks) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        edges = edges.to(device)
        masks = masks.to(device)

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        imgs_masked = (imgs * (1 - masks).float()) + masks
        edges_masked = (edges * (1 - masks))
        inputs = torch.cat((imgs_masked, edges_masked), dim=1)
        inputs = inputs.to(device)
        outputs = generator(inputs)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((imgs, edges), dim=1)
        dis_input_fake = torch.cat((imgs, outputs.detach()), dim=1)
        dis_real, dis_real_feat = discriminator(dis_input_real)  # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = discriminator(dis_input_fake)  # in: (grayscale(1) + edge(1))
        dis_real_loss = adversarial_loss(dis_real, True, True)
        dis_fake_loss = adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((imgs, outputs), dim=1)
        gen_fake, gen_fake_feat = discriminator(gen_input_fake)  # in: (grayscale(1) + edge(1))
        gen_gan_loss = adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for ii in range(len(dis_real_feat)):
            gen_fm_loss += l1_loss(gen_fake_feat[ii], dis_real_feat[ii].detach())
        gen_fm_loss = gen_fm_loss * args.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        # backward
        if dis_loss != 0:
            dis_loss.backward()
        dis_optimizer.step()

        if gen_loss != 0:
            gen_loss.backward()
        gen_optimizer.step()

        if i % 20 == 0:
            print(
                '[%d/%d][%d/%d] d_loss: %.4f g_loss: %.4f g_adv_loss: %.4f g_fm_loss: %.4f'
                % (epoch, 3, i, len(dataloader_train),
                   dis_loss, gen_loss, gen_gan_loss, gen_fm_loss))

            list_gloss.append(gen_loss)
            list_dloss.append(dis_loss)

        if i % 100 == 0:
            vutils.save_image(imgs,
                              '%s/real_samples.png' % 'edge_output',
                              normalize=True)
            vutils.save_image(edges,
                              '%s/real_edges.png' % 'edge_output',
                              normalize=True)
            vutils.save_image(imgs_masked,
                              '%s/img_mask%03d.png' % ('edge_output', epoch + 100),
                              normalize=True)

            vutils.save_image(edges_masked,
                              '%s/edge_mask%03d.png' % ('edge_output', epoch + 100),
                              normalize=True)
            vutils.save_image(outputs,
                              '%s/output%03d.png' % ('edge_output', epoch + 100),
                              normalize=True)

torch.save(generator, 'Edge_generator.pth')
torch.save(discriminator, 'Edge_discriminator.pth')

f = open("txt/edge_dis_loss.txt", "a+")
for a in list_dloss:
    f.write(str(a) + '\n')
f.close()

f = open("txt/edge_gen_loss.txt", "a+")
for a in list_gloss:
    f.write(str(a) + '\n')
f.close()

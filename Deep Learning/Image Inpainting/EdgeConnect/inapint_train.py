import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from model.loss import AdversarialLoss
from model.networks import EdgeGenerator, InpaintGenerator, Discriminator
from model.loss import AdversarialLoss, PerceptualLoss, StyleLoss

from data.RS_Mask_Data import RSMaskDataSet
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
# training options

parser.add_argument('--batchsize', type=int, default=3)
parser.add_argument('--LR', type=float, default=0.00001)
parser.add_argument('--BETA1', type=float, default=0.0)
parser.add_argument('--BETA2', type=float, default=0.9)
parser.add_argument('--D2G_LR', type=float, default=0.1)
parser.add_argument('--GAN_LOSS', type=str, default='nsgan')
parser.add_argument('--INPAINT_ADV_LOSS_WEIGHT', type=float, default=0.1)
parser.add_argument('--L1_LOSS_WEIGHT', type=float, default=1)
parser.add_argument('--STYLE_LOSS_WEIGHT', type=float, default=250)
parser.add_argument('--CONTENT_LOSS_WEIGHT', type=float, default=0.1)

args = parser.parse_args()

device = torch.device('cuda:0')

dataset_train = RSMaskDataSet()
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize
                                               , shuffle=True, num_workers=0)

generator = InpaintGenerator().to(device)
discriminator = Discriminator(in_channels=3, use_sigmoid=args.GAN_LOSS != 'hinge').to(device)

generator = torch.load('generator.pth')
discriminator = torch.load('discriminator.pth')

l1_loss = nn.L1Loss().to(device)
perceptual_loss = PerceptualLoss().to(device)
style_loss = StyleLoss().to(device)
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
        inputs = torch.cat((imgs_masked, edges), dim=1)

        inputs = inputs.to(device)
        outputs = generator(inputs)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = imgs
        dis_input_fake = outputs.detach()
        dis_real, _ = discriminator(dis_input_real)  # in: [rgb(3)]
        dis_fake, _ = discriminator(dis_input_fake)  # in: [rgb(3)]
        dis_real_loss = adversarial_loss(dis_real, True, True)
        dis_fake_loss = adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = discriminator(gen_input_fake)  # in: [rgb(3)]
        gen_gan_loss = adversarial_loss(gen_fake, True, False) * args.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss

        # generator l1 loss
        gen_l1_loss = l1_loss(outputs, imgs) * args.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss

        # generator perceptual loss
        gen_content_loss = perceptual_loss(outputs, imgs)
        gen_content_loss = gen_content_loss * args.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss

        # generator style loss
        gen_style_loss = style_loss(outputs * masks, imgs * masks)
        gen_style_loss = gen_style_loss * args.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        # backward
        dis_loss.backward()
        dis_optimizer.step()

        gen_loss.backward()
        gen_optimizer.step()

        if i % 20 == 0:
            print(
                '[%d/%d][%d/%d] d_loss: %.4f g_loss: %.4f g_adv_loss: %.4f g_l1_loss: %.4f g_per_loss: %.4f g_sty_loss: %.4f'
                % (epoch, 3, i, len(dataloader_train),
                   dis_loss, gen_loss, gen_gan_loss, gen_l1_loss, gen_content_loss, gen_style_loss))

            list_gloss.append(gen_loss)
            list_dloss.append(dis_loss)

        if i % 100 == 0:
            vutils.save_image(imgs,
                              '%s/real_samples.png' % 'output',
                              normalize=True)
            vutils.save_image(edges,
                              '%s/edge%03d.png' % ('output', epoch + 225),
                              normalize=True)
            vutils.save_image(imgs_masked,
                              '%s/mask%03d.png' % ('output', epoch + 225),
                              normalize=True)
            vutils.save_image(outputs,
                              '%s/output%03d.png' % ('output', epoch + 225),
                              normalize=True)

torch.save(generator, 'generator.pth')
torch.save(discriminator, 'discriminator.pth')

f = open("txt/dis_loss.txt", "a+")
for a in list_dloss:
    f.write(str(a) + '\n')
f.close()

f = open("txt/gen_loss.txt", "a+")
for a in list_gloss:
    f.write(str(a) + '\n')
f.close()

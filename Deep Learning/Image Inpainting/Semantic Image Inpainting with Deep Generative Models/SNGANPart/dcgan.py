import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from SNGANPart.snconv2d import SNConv2d

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.00002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=20, help='interval between image sampling')
opt = parser.parse_args()

device = torch.device('cuda:0')


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self, input_d=100, output_d=3):
        super(Generator, self).__init__()
        self.d = 64
        self.input_d = input_d
        self.output_d = output_d
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_d, self.d * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(self.d * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.d * 16, self.d * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.d * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.d * 8, self.d * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.d * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.d * 4, self.d * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.d * 2),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.d * 2, self.d, 4, 2, 1, bias=True),
            nn.BatchNorm2d(self.d),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(self.d, int(self.d / 2), 4, 2, 1, bias=True),
            nn.BatchNorm2d(int(self.d / 2)),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(int(self.d / 2), self.output_d, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def wight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = self.main(input)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_d=3):
        super(Discriminator, self).__init__()
        self.d = 64
        self.input_d = input_d

        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            # SNConv2d()

            SNConv2d(self.input_d, int(self.d / 2), 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 64 x 64

            SNConv2d(int(self.d / 2), self.d, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 32 x 32

            SNConv2d(self.d, self.d * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16

            SNConv2d(self.d * 2, self.d * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 8 x 8

            SNConv2d(self.d * 4, self.d * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(self.d * 8, self.d * 16, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(self.d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # self.snlinear = nn.Sequential(SNLinear(ndf * 4 * 4 * 4, 1),
        #                              nn.Sigmoid())

    def forward(self, input):
        output = self.main(input)
        # output = output.view(output.size(0), -1)
        # output = self.snlinear(output)
        return output.view(-1, 1).squeeze(1)

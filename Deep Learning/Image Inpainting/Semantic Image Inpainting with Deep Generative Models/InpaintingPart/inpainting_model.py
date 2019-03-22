import numpy as np
import argparse
from scipy.signal import convolve2d
from InpaintingPart.mask_generator import gen_mask
import matplotlib.pyplot as plt


class InpaintingModel():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = (opt.img_size, opt.img_size)

    @staticmethod
    def create_weight_mask(masks, nsize=7):
        wmasks = np.zeros_like(masks)
        ker = np.ones((nsize, nsize), dtype=np.float32)
        ker = ker / np.sum(ker)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            inv_mask = 1.0 - mask
            temp = mask * convolve2d(inv_mask, ker, mode='same', boundary='symm')
            wmasks[idx] = mask * temp

        return wmasks

    @staticmethod
    def create3_channel_masks(masks):
        masks_3c = np.zeros((*masks.shape, 3), dtype=np.float32)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            masks_3c[idx, :, :, :] = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        return masks_3c


parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=3, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=30, help='training batch size')

parser.add_argument('--img_size', type=int, default=128, help='image height or width')
parser.add_argument('--mask_type', type=str, default='center', help='mask type choice in [center|random|half|pattern]')

opt = parser.parse_args()
aa = gen_mask(opt)
aa = InpaintingModel.create_weight_mask(aa, 7)
plt.imshow(aa[0], interpolation='nearest')
plt.show()
print('finash')

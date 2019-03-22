import argparse

parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--n_dis', type=int, default=3, help='discriminator critic iters')
parser.add_argument('--nz', type=int, default=100, help='dimention of lantent noise')
parser.add_argument('--batchsize', type=int, default=30, help='training batch size')

parser.add_argument('--img_size', type=int, default=128, help='image height or width')
parser.add_argument('--mask_type', type=str, default='center', help='mask type choice in [center|random|half|pattern]')

opt = parser.parse_args()
print(opt)

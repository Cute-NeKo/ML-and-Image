import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
parser.add_argument('--imageSize', type=int, default=90, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--out', type=str, default='pth', help='folder to output model checkpoints')

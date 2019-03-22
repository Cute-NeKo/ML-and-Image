import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import os
from PIL import Image
import random
import numpy as np
import imageio
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.viewer import ImageViewer


class RSEdgeDataSet(Dataset):
    def __init__(self):
        self.transform_train = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.transform_mask = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor()])

        # load all image
        # dir_name_train = self.listdir(r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45')
        # self.data_list_train = []
        # for i in dir_name_train:
        #     self.data_list_train = self.data_list_train + self.listdir(i)
        # print('Read ' + str(len(self.data_list_train)) + ' images')

        # load part image
        dir_name_train = self.listdir(r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45\freeway')
        # dir_name_train = self.listdir(r'E:\DataSet\UCMerced_LandUse\Images\freeway')

        self.data_list_train = dir_name_train
        print('Read ' + str(len(self.data_list_train)) + ' images')

        data_route_mask = r'E:\DataSet\mask\mask_test - delete\testing_mask_dataset'
        self.data_list_mask = self.listdir(data_route_mask)
        print('Read ' + str(len(self.data_list_mask)) + ' images')

    def __len__(self):
        return len(self.data_list_train)

    def __getitem__(self, item):
        img = self.data_list_train[item]
        img = Image.open(img)
        img = self.transform_mask(img.convert('L'))

        mask = self.data_list_mask[random.randint(0, len(self.data_list_mask) - 1)]
        mask = Image.open(mask)
        mask = self.transform_mask(mask)
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0

        edge = self.get_edge(self.data_list_train[item])
        edge = Image.fromarray(edge)
        edge = self.transform_mask(edge)

        return img, edge, mask

    @staticmethod
    def listdir(path):
        name_list = []
        for file in os.listdir(path):
            name_list.append(os.path.join(path, file))
        return name_list

    @staticmethod
    def get_edge(img_name):
        img = imageio.imread(img_name)

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        # create grayscale image
        img_gray = rgb2gray(img)

        result = canny(img[:, :, 0], sigma=2).astype(np.uint8)
        result[result < 0.5] = 0
        result[result > 0.5] = 255

        return result

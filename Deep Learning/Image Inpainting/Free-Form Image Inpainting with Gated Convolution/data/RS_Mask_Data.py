import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import random
import numpy as np


class RSMaskDataSet(Dataset):
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

        # load meadow image
        dir_name_train = self.listdir(r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45\mountain')
        self.data_list_train = dir_name_train
        print('Read ' + str(len(self.data_list_train)) + ' images')

        data_route_mask = r'E:\DataSet\mask\mask_test - delete\testing_mask_dataset'
        # data_route_mask = r'E:\DataSet\mask\mask_test - delete2'
        self.data_list_mask = self.listdir(data_route_mask)
        print('Read ' + str(len(self.data_list_mask)) + ' images')

    def __len__(self):
        return len(self.data_list_train)

    def __getitem__(self, item):
        img = self.data_list_train[item]
        img = Image.open(img)
        img = self.transform_train(img.convert('RGB'))

        mask = self.data_list_mask[random.randint(0, len(self.data_list_mask) - 1)]
        mask = Image.open(mask)
        mask = self.transform_mask(mask)
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        return img, mask

    @staticmethod
    def listdir(path):
        name_list = []
        for file in os.listdir(path):
            name_list.append(os.path.join(path, file))
        return name_list

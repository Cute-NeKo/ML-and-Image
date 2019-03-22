import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import numpy as np


class NWPUDataSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.RandomCrop((128, 128)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dir_name = self.listdir(r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45')
        self.data_list = []
        for i in dir_name:
            self.data_list = self.data_list + self.listdir(i)
        print('Read ' + str(len(self.data_list)) + ' images')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img = self.data_list[item]
        img = Image.open(img)
        img = self.transform(img)
        return img

    @staticmethod
    def listdir(path):
        name_list = []
        for file in os.listdir(path):
            name_list.append(os.path.join(path, file))
        return name_list

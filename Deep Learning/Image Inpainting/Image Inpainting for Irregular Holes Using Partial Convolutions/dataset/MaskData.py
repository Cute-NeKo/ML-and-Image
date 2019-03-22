import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import numpy as np


class MaskDataSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor()])

        data_route = r'E:\DataSet\mask\mask_test\testing_mask_dataset'
        self.data_list = self.listdir(data_route)
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

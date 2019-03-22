import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os
from PIL import Image
import numpy as np


class CCFDataSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()])
        data_list = []
        label_list = []
        data_route = r'E:\DataSet\CCF remote sensing\remote_sensing_image\seg_train\train\src'
        label_route = r'E:\DataSet\CCF remote sensing\remote_sensing_image\seg_train\train\label'
        for i in range(15000):
            data_list.append(os.path.join(data_route, str(i) + '.png'))
            label_list.append(os.path.join(label_route, str(i) + '.png'))
        self.data_list = data_list
        self.label_list = label_list
        print('Read ' + str(len(self.data_list)) + ' images')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label)
        img = self.transform(img)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return img, label

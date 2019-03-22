import warnings
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
import model.fcn as fcn
from config import opt
from data import voc
from data.voc import VocSegDataset, img_transforms, COLORMAP, inverse_normalization, CLASSES
from PIL import Image

voc_root = r"E:\DataSet\VOC2012"


class VOCSegDataset(Dataset):
    '''
    voc dataset
    '''

    def __init__(self, train, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = voc.read_images(voc_root, train=train)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)
        print('Read ' + str(len(self.data_list)) + ' images')

    def _filter(self, images):  # 过滤掉图片大小小于 crop 大小的图片
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = Image.open(img)
        label = Image.open(label).convert('RGB')
        img, label = self.transforms(img, label, self.crop_size)
        return img, label

    def __len__(self):
        return len(self.data_list)


#####################################################################
# 实例化数据集
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape, img_transforms)
voc_test = VOCSegDataset(False, input_shape, img_transforms)

train_data = DataLoader(voc_train, batch_size=10, shuffle=True, num_workers=0)
valid_data = DataLoader(voc_test, batch_size=1, num_workers=0)

num_classes = len(CLASSES)

net = fcn.FcnResNet(num_classes)
net.load_state_dict(torch.load('FCN_resnet34.pkl'))
device = torch.device("cuda:0")
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

for e in range(3):
    train_loss = 0

    prev_time = datetime.now()
    net = net.train()
    for batch_size, data in enumerate(train_data):
        im = data[0].to(device)
        labels = data[1].to(device)
        print(labels.type())

        # forward
        out = net(im)
        loss = criterion(out, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(batch_size, ':', loss.item())

        pred_labels = out.max(dim=1)[1].data.cpu().numpy()


    print(train_loss/len(train_data))



torch.save(net.state_dict(), 'FCN_resnet34.pkl')

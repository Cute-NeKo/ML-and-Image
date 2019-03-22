import torch
from torch.utils.data import DataLoader

import torch.nn as nn

from model import fcn
from data.CCF import CCFDataSet


data_train = CCFDataSet()
train_data = DataLoader(data_train, batch_size=10, shuffle=True, num_workers=0)

num_classes = 5

# net = fcn.FcnResNet(num_classes)
net=torch.load('CCF_FCN_resnet34.pkl')
device = torch.device("cuda:0")
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.000005)

for e in range(1):
    train_loss = 0

    net = net.train()
    for batch_size, data in enumerate(train_data):
        im = data[0].to(device)
        labels = data[1].to(device)

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

    print(train_loss / len(train_data))

torch.save(net,'CCF_FCN_resnet34.pkl')

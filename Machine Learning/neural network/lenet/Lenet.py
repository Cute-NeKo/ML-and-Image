import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = torchvision.datasets.MNIST(root="../data/mnist/", train=True, transform=trans, download=True)
test_set = torchvision.datasets.MNIST(root="../data/mnist/", train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False)


model = LeNet()
device = torch.device("cuda:0")
model.to(device)
#
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
# criterion = nn.CrossEntropyLoss()
#
# for epoch in range(10):
#     for batch_size, (x, target) in enumerate(train_loader):
#         optimizer.zero_grad()
#         x = x.to(device)
#         target = target.to(device)
#         out = model(x)
#         loss = criterion(out, target)
#         loss.backward()
#         optimizer.step()
#
#         if batch_size % 200 == 0:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, batch_size, loss.item()))

#保存
# torch.save(model.state_dict(), 'lenet.pkl')
model.load_state_dict(torch.load('lenet.pkl'))

# test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        print(images.shape)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

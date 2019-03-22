import numpy as np
import torch
import struct
import torchvision

device = torch.device("cuda:0")

if __name__ == '__main__':
    mnist_train = torchvision.datasets.MNIST(root="../data/mnist/", train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data/mnist/", train=False,
                                            transform=torchvision.transforms.ToTensor(), download=True)

    trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=100,
                                              shuffle=True, num_workers=2)

    data_train = mnist_train.train_data.view([60000, 784]).float()
    label_train = mnist_train.train_labels.view(60000).float()
    label_train[label_train != 5] = 0
    label_train[label_train == 5] = 1

    w = torch.randn(784, 1,device=device, dtype=torch.float, requires_grad=True)
    b = torch.randn(1, 1, device=device, dtype=torch.float, requires_grad=True)


    for i in range(10):

        for j, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            inputs = inputs.view([100, 784]).float()
            # inputs[inputs < 10] = 0
            # inputs[inputs >= 10] = 1
            labels = labels.view([100, 1]).float()
            labels[labels != 5] = 0
            labels[labels == 5] = 1

            y_pred = (inputs.mm(w) + b) / 10
            # h = 1.0 / (1.0 + (-y_pred).exp())
            h = torch.nn.functional.sigmoid(y_pred)
            loss = (-labels * torch.log(h) - (1 - labels) * torch.log(1 - h)).sum()
            loss.backward()
            with torch.no_grad():
                w -= (0.3 * w.grad)
                b -= (0.3 * b.grad)

                w.grad.zero_()
                b.grad.zero_()

            print(loss)

    data_test = mnist_test.test_data.view([10000, 784]).float().cuda()
    # data_test[data_test < 10] = 0
    # data_test[data_test >= 10] = 1
    label_test = mnist_test.test_labels.view([10000, 1]).float().cuda()
    label_test[label_test != 5] = 0
    label_test[label_test == 5] = 1


    result = data_test.mm(w) + b
    result[result > 0] = 1
    result[result <= 0] = 0

    all = 0
    right = 0
    for i in range(10000):
        if label_test[i] == 1:
            all += 1
            if result[i] == 1:
                right += 1
    print(right / all)

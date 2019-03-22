import torch
import matplotlib.pyplot as plot
import torch.nn as nn
import torch.nn.functional as F

dtype = torch.float
data = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).float()
label = torch.tensor([[0], [1], [1], [0]]).float()

w1 = torch.randn(2, 4, dtype=dtype, requires_grad=True)
c1 = torch.randn(1, 4, dtype=dtype, requires_grad=True)
w2 = torch.randn(4, 1, dtype=dtype, requires_grad=True)
c2 = torch.randn(1, 1, dtype=dtype, requires_grad=True)


for i in range(10000):
    y1 = data.mm(w1)+c1
    y1 = F.relu(y1)
    y2 = y1.mm(w2)+c2

    loss = (label - y2).pow(2).sum()

    loss.backward()


    with torch.no_grad():
        w1 -= (0.01 * w1.grad)
        w2 -= (0.01 * w2.grad)
        c1 -= (0.01 * c1.grad)
        c2 -= (0.01 * c2.grad)

        w1.grad.zero_()
        w2.grad.zero_()
        c1.grad.zero_()
        c2.grad.zero_()

x = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).float()
print(F.relu(x.mm(w1)+c1).mm(w2)+c2)

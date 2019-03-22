import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from DataSet.RemoteSensingData import NWPUDataSet
import torch.utils.data

# show image

# def show_img(img):
#     img = make_grid(img)
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
#     plt.show()
#
#
# dataset = NWPUDataSet()
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=20
#                                          , shuffle=True, num_workers=0)
#
# for i, data in enumerate(dataloader):
#     data = data * 0.5 + 0.5
#     show_img([data[0],data[1],data[2]])
#     break


bb = [22, 3.256, 2.654, 55.23, 45.36]
f = open("loss_G.txt", "a+")
for i in bb:
    f.write(str(i) + '\n')
f.close()

f = open("loss_D.txt", "a+")
for i in bb:
    f.write(str(i) + '\n')
f.close()

import torch
from model import fcn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

COLORMAP = [[0, 0, 0], [0, 128, 0], [128, 128, 128], [0, 0, 128], [128, 128, 0]]

num_classes = 5
model = torch.load('CCF_FCN_resnet34.pkl', map_location=lambda storage, loc: storage)

input_shape = (320, 480)
img_transform = transforms.Compose([
    transforms.ToTensor()])

# img = Image.open(r'E:\DataSet\CCF remote sensing\remote_sensing_image\seg_train\train\src\407.png')
img = Image.open(r'C:\Users\uygug\Desktop\2222.png')
label_img = Image.open(r'E:\DataSet\CCF remote sensing\remote_sensing_image\seg_train\train\label\407.png')
img.show()
# label_img.show()

img = img_transform(img)
img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
result = model(img)
print(result.shape)

# for i in range(256):
#     for j in range(256):
#         xx = result[:, :, i, j]
#         # print(xx)
#         print(xx.max(1))

pred = result.max(1)[1].squeeze().cpu().data.numpy()
cm = np.array(COLORMAP).astype('uint8')
pred = cm[pred]
pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
cv2.imshow('ff', pred)

ll = cm[label_img]
ll = cv2.cvtColor(ll, cv2.COLOR_BGR2RGB)
cv2.imshow('55', ll)
cv2.waitKey(0)

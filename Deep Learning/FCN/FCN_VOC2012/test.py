import torch
from model import fcn
from data.voc import VocSegDataset, img_transforms, COLORMAP, inverse_normalization, CLASSES
from PIL import Image
import transforms.transforms as tfs
import numpy as np
import cv2
from data import voc

num_classes = len(CLASSES)
model = fcn.FcnResNet(num_classes)
model.load_state_dict(torch.load('FCN_resnet34.pkl'))

input_shape = (320, 480)
img_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(r'E:\DataSet\VOC2012\JPEGImages\2007_000033.jpg')
# img = Image.open(r'H:/20180706204556.png').convert('RGB')
label_img = Image.open(r'E:\DataSet\VOC2012\SegmentationClass\2007_000033.png').convert('RGB')
img, label_img = voc.random_crop(img, label_img, (320, 480))
img.show()
label_img.show()

img = img_tfs(img)
img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
result = model(img)
print(result.shape)

# for i in range(320):
#     for j in range(480):
#         xx = result[:, :, i, j]
#         # print(xx)
#         print(xx.max(1))
# print(result[:, :, 2, 2])

pred = result.max(1)[1].squeeze().cpu().data.numpy()
cm = np.array(COLORMAP).astype('uint8')
pred = cm[pred]
pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
cv2.imshow('ff', pred)
cv2.waitKey(0)

###########################################################

# img = Image.open(r'E:\DataSet\VOC2012\JPEGImages\2007_001763.jpg')
# label_img = Image.open(r'E:\DataSet\VOC2012\SegmentationClass\2007_001763.png').convert('RGB')
# img, label_img = voc.random_crop(img, label_img, (320, 480))
# img.show()
# label_img.show()
#
# label = voc.image2label(label_img)
# label = torch.from_numpy(label)
# pred = label.squeeze().cpu().data.numpy()
# cm = np.array(COLORMAP).astype('uint8')
# pred = cm[pred]
# pred=cv2.cvtColor(pred,cv2.COLOR_RGB2BGR)
# cv2.imshow('ff', pred)
# cv2.waitKey(0)


#############################################
# img = Image.open(r'E:\DataSet\VOC2012\JPEGImages\2008_000661.jpg')
# label_img = Image.open(r'E:\DataSet\VOC2012\SegmentationClass\2008_000661.png').convert('RGB')
# img.show()
#
# label_array = np.asarray(label_img)
# print(label_array.shape)
# for i in label_array:
#     for j in i:
#         if j[0] != 0 or j[1] != 0 or j[2] != 0:
#             print(j)

import torch
import numpy as np
import imageio
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from skimage.viewer import ImageViewer
from PIL import Image
import cv2

# filename = r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45\mountain\mountain_186.jpg'
filename = r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45\bridge\bridge_060.jpg'
# filename = r'C:\Users\uygug\Desktop\20190211200316.jpg'

#######################################################
img = cv2.imread(filename, 0)
img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow('ggg', img)
edge = cv2.Canny(img, 50, 150)
# cv2.imshow('gg', edge)
# # cv2.waitKey()

edge = Image.fromarray(edge)
edge.show()

###########################################################
# this is good

img = imageio.imread(filename)

# gray to rgb
if len(img.shape) < 3:
    img = gray2rgb(img)

# create grayscale image
img_gray = rgb2gray(img)

result = canny(img[:, :, 0], sigma=1).astype(np.uint8)
result[result < 0.5] = 0
result[result > 0.5] = 255

# viewer = ImageViewer(result)
# viewer.show()

edge = Image.fromarray(result)
edge.show()

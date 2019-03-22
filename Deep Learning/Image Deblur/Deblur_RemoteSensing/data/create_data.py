import cv2
from os import listdir
import os
import random
import time


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def blur_img(img):
    t = random.randint(0, )


file_path = r'E:\DataSet\DOTA\part1\images'
simage_filenames = [x for x in listdir(file_path) if is_image_file(x)]

save_path_sharp = r'E:\DataSet\DOTA_blur_sharp\sharp'
save_path_blur = r'E:\DataSet\DOTA_blur_sharp\blur'

k = 1
for e in range(1):
    for i in range(len(simage_filenames)):
        path1 = os.path.join(file_path, simage_filenames[i])
        img = cv2.imread(path1)
        height = img.shape[0] - 260
        width = img.shape[1] - 260
        height = random.randint(0, height)
        width = random.randint(0, width)
        crop_img = img[height:height + 256, width:width + 256]
        crop_img_blur = cv2.blur(crop_img, (11, 11))
        print(k, ':', crop_img.shape)

        s1 = save_path_sharp + '\\' + str(k) + '.png'
        cv2.imwrite(s1, crop_img)
        s2 = save_path_blur + '\\' + str(k) + '.png'
        cv2.imwrite(s2, crop_img_blur)

        k += 1

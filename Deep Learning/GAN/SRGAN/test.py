from data.CCF import CCFDataSet
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Parser import parser
from model.models import Generator
import torch
from PIL import Image
import cv2
import numpy as np

device = torch.device("cuda:0")

opt = parser.parse_args()

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize * opt.upSampling),
                                transforms.ToTensor()])
transform2 = transforms.Compose([transforms.ToTensor()])

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                            ])
unnormalize = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

generator = Generator(16, opt.upSampling)
generator.load_state_dict(torch.load('pth/generator_final.pth'))

#
# data_train = CCFDataSet()
# dataloader = DataLoader(data_train, batch_size=opt.batchSize, shuffle=True, num_workers=0)
#
#
# for i, data in enumerate(dataloader):
#     img = data[0]
#     print(img.shape)
#
#     img_lr = scale(img)
#     img_lr = img_lr.view([1, 3, 90, 90])
#     img_lr = unnormalize(img_lr[0])
#     img_lr = transforms.ToPILImage()(img_lr).convert('RGB')
#     img_lr.show()
#
#     img = normalize(img)
#     img_real = unnormalize(img)
#     img_real = transforms.ToPILImage()(img_real).convert('RGB')
#     img_real.show()
#
#     low_res = scale(img)
#     low_res = low_res.view([1, 3, 90, 90])
#     low_res = generator(low_res)
#     print(low_res.shape)
#     low_res = unnormalize(low_res[0])
#     low_res = transforms.ToPILImage()(low_res).convert('RGB')
#     low_res.show()
#
#     img_interp = scale(img)
#     img_interp = img_interp.view([1, 3, 90, 90])
#     img_interp = unnormalize(img_interp[0])
#     img_interp = transforms.ToPILImage()(img_interp).convert('RGB')
#     img_interp = img_interp.resize([180, 180])
#     img_interp.show()
#     break


# img = Image.open(r'E:\DataSet\NWPU VHR-10 dataset\NWPU VHR-10 dataset\positive image set\418.jpg')
# dd = transform(img)
#
# img = normalize(dd)
# img_real = unnormalize(img)
# img_real = transforms.ToPILImage()(img_real).convert('RGB')
# img_real.show()
#
# low_res = scale(dd)
# low_res = low_res.view([1, 3, 90, 90])
# low_res = generator(low_res)
# print(low_res.shape)
# low_res = unnormalize(low_res[0])
# low_res[low_res > 1] = 1
# low_res[low_res < 0] = 0
# low_res = transforms.ToPILImage()(low_res).convert('RGB')
# low_res.show()
#
# img_interp = scale(dd)
# img_interp = img_interp.view([1, 3, 90, 90])
# img_interp = unnormalize(img_interp[0])
#
# img_interp = transforms.ToPILImage()(img_interp).convert('RGB')
# img_interp = img_interp.resize([180, 180])
# img_interp.show()


# generator.to(device)
# stride = 90
# image_size = 90
# image = cv2.imread(r'H:\city9.jpg')
# h, w, _ = image.shape
# padding_h = (h // stride + 1) * stride
# padding_w = (w // stride + 1) * stride
# padding_img = np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
# padding_img[0:h, 0:w, :] = image[:, :, :]
# print('src:', padding_img.shape)
# mask_whole = np.zeros((padding_h * 2, padding_w * 2, 3), dtype=np.uint8)
#
# for i in range(padding_h // stride):
#     for j in range(padding_w // stride):
#         print(i, ' ', j)
#         crop = padding_img[i * stride:i * stride + image_size, j * stride:j * stride + image_size, :]
#         ch, cw, _ = crop.shape
#         if ch != 90 or cw != 90:
#             print('invalid size!')
#             continue
#
#         crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#         crop = transform2(crop)
#         crop = normalize(crop)
#         crop = crop.view(1, crop.shape[0], crop.shape[1], crop.shape[2])
#         result = generator(crop.to(device)).cpu()
#         result = unnormalize(result[0])
#         result[result > 1] = 1
#         result[result < 0] = 0
#         result = transforms.ToPILImage()(result).convert('RGB')
#         result = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
#         mask_whole[i * stride * 2:i * stride * 2 + image_size * 2, j * stride * 2:j * stride * 2 + image_size * 2,
#         :] = result[
#              :, :]
#
# result_img = mask_whole[:h * 2, :w * 2, :]
#
# img_interp = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
# # cv2.imshow('11', image)
# # cv2.imshow('55', result_img)
# cv2.imwrite(r'C:\Users\uygug\Desktop\ffff.png', result_img)
# cv2.imwrite(r'C:\Users\uygug\Desktop\interp.png', img_interp)
# # cv2.waitKey(0)


generator.to(device)
img = Image.open(r'E:\DataSet\20180821173107.png').convert('RGB')
print(img.size)
dd = transform2(img)
dd = dd.view(1, 3, img.size[1], img.size[0])
result = generator(dd.to(device)).cpu()
result = unnormalize(result[0])
result[result > 1] = 1
result[result < 0] = 0
result = transforms.ToPILImage()(result).convert('RGB')
print(result.size)
result.show()

img.show()
img = img.resize([img.size[0] * 2, img.size[1] * 2])
img.show()

import torch
from model.gan import InpaintSAGNet, InpaintSADirciminator
from PIL import Image
from torchvision import transforms
import opt


def in_transform(img):
    result = torch.ones_like(img)
    result[0] = img[0] * opt.STD[0] + opt.MEAN[0]
    result[1] = img[1] * opt.STD[1] + opt.MEAN[1]
    result[2] = img[2] * opt.STD[2] + opt.MEAN[2]
    result[result < 0] = 0
    result[result > 1] = 1
    return result


transform_img = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform_mask = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

device = torch.device('cuda:0')

model = InpaintSAGNet().to(device)

model = torch.load('netG.pth')

# img = Image.open(r'E:\DataSet\NWPU-RESISC45-dataset\NWPU-RESISC45\NWPU-RESISC45\mountain\mountain_131.jpg')
img = Image.open(r'C:\Users\uygug\Desktop\搜狗截图20190211195106.jpg')
img.show()
img = transform_img(img.convert('RGB')).view(-1, 3, 256, 256)
img = torch.cat([img, torch.full_like(img, 0.3), torch.full_like(img, 0.3)], dim=0)

mask = Image.open(r'E:\DataSet\mask\mask_test - delete\testing_mask_dataset\00222.png')
mask = transform_mask(mask).view(-1, 1, 256, 256)
mask = torch.cat([mask, torch.full_like(mask, 0.3), torch.full_like(mask, 0.3)], dim=0)

mask[mask > 0.5] = 1
mask[mask < 0.5] = 0
# mask = 1.0 - mask


output, _ = model(img.to(device), mask.to(device))
result = in_transform(output[0])
gt = in_transform(img[0])

mask = 1.0 - mask
gt_mask = gt.cpu() * mask[0].cpu()
gt_mask = transforms.ToPILImage()(gt_mask).convert('RGB')
gt_mask.show()

result = gt.cpu() * mask[0].cpu() + result.cpu() * (1 - mask[0].cpu())
result = transforms.ToPILImage()(result.cpu()).convert('RGB')
result.show()

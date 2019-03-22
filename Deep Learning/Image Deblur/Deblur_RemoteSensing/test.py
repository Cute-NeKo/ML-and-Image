from models.models import create_model
from options.train_options import TrainOptions
from PIL import Image
import torchvision.transforms as transforms
import torch

device = torch.device("cuda:0")

opt = TrainOptions().parse()
model = create_model(opt)

img = Image.open(r'E:\DataSet\DOTA_blur_sharp\blur\252.png').convert('RGB')
# img = Image.open(r'C:\Users\uygug\Desktop\1212.png').convert('RGB')

img.show()
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img = transform(img).view(-1, 3, img.size[0], img.size[1])

result = model.netG(img.to(device))
result = result.data.cpu()[0] * 0.5 + 0.5
result = transforms.ToPILImage()(result).convert('RGB')
result.show()

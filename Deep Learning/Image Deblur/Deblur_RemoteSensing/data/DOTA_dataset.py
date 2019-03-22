from os import listdir
from os.path import join
import numpy as np
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms


class DOTADatasetFolder(data.Dataset):
    def __init__(self):
        super(DOTADatasetFolder, self).__init__()
        self.sharp_path = join(r'E:\DataSet\DOTA_blur_sharp', "sharp")
        self.blur_path = join(r'E:\DataSet\DOTA_blur_sharp', "blur")
        self.image_filenames_sharp = [x for x in listdir(self.sharp_path) if self.is_image_file(x)]
        self.image_filenames_blur = [x for x in listdir(self.blur_path) if self.is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        input = self.load_img(join(self.blur_path, self.image_filenames_blur[index]))
        input = self.transform(input)
        target = self.load_img(join(self.sharp_path, self.image_filenames_sharp[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames_sharp)

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

    def load_img(self, filepath):
        img = Image.open(filepath).convert('RGB')
        # img = img.resize((256, 256), Image.BICUBIC)
        return img

    def save_img(self, image_tensor, filename):
        image_numpy = image_tensor.float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = image_numpy.astype(np.uint8)
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(filename)
        print("Image saved as {}".format(filename))

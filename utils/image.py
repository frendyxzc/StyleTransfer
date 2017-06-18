# -*- coding: utf-8 -*-

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mimage
from PIL import Image


use_cuda = torch.cuda.is_available()
image_size = 512 if use_cuda else 512  # use small size if no gpu

class ImageUtil(object):
    """
    this class load an image from file path,
    return a torch tensor
    """
    def __init__(self):
        self.loader = transforms.Compose([
            transforms.Scale(image_size),
            transforms.ToTensor()
        ])
        self.un_loader = transforms.ToPILImage()

    def load_image(self, image_name):
        image = Image.open(image_name)
        image = Variable(self.loader(image))
        image = image.unsqueeze(0)
        return image

    def show_image(self, tensor, title=None):
        """
        this method will convert
        :return:
        """
        image = tensor.clone().cpu()
        image = image.view(3, image_size, image_size)
        image = self.un_loader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.show()

    def save_image(self, tensor, name='default.jpg'):
        image = tensor.clone().cpu()
        image = image.view(3, image_size, image_size)
        image = self.un_loader(image)
        print(image)
        try:
            im = Image.fromarray(image)
            im.save(name)
            print('image saved.')
        except Exception as e:
            print(e)
            mimage.imsave(name, image)
            print('image saved.')

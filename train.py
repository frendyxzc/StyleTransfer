# -*- coding: utf-8 -*-

import torch
from models import get_input_param_optimizer, get_model_and_losses
from utils.image import ImageUtil
import sys

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def train():
    image_util = ImageUtil()

    image_1 = "images/monet.jpg"
    image_2 = "images/sun.jpg"
    if len(sys.argv) > 2:
        image_1 = sys.argv[1]
        image_2 = sys.argv[2]
    print('image style: {}, image content: {}'.format(image_1, image_2))
    style_img = image_util.load_image(image_1).type(dtype)
    content_img = image_util.load_image(image_2).type(dtype)

    assert style_img.size() == content_img.size(), \
        "The style and content images should be the same size on height and width."

    style_weight = 1000
    content_weight = 3
    num_steps = 300

    input_img = content_img.clone()
    model, style_losses, content_losses = get_model_and_losses(style_img, content_img,
                                                                     style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()

            return style_score + style_score

        optimizer.step(closure)
    input_param.data.clamp_(0, 1)
    output = input_param.data
    image_util.save_image(output, name='output.jpg')


def main():
    train()


if __name__ == '__main__':
    main()
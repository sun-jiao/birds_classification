import torch
import torch.nn as nn


class Augmentations(nn.Module):

    def __init__(self):
        super(Augmentations, self).__init__()
        self.gaussian_blur = nn.Conv2d(3, 3, (3, 3), padding=1, groups=3, bias=False)

    def forward(self, image):
        self.gaussian_blur.weight.data.normal_(0.111, 0.01)
        image = self.gaussian_blur(image)

        gaussian_noise = torch.randn((3, 200, 200)).cuda()
        image += gaussian_noise * 10
        return image

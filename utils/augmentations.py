import torch
import torch.nn as nn


class Augmentations(nn.Module):

    def __init__(self):
        super(Augmentations, self).__init__()
        self.gaussian_blur = nn.Conv2d(3, 3, (3, 3), padding=1, groups=3, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, image):
        self.gaussian_blur.weight.data.normal_(0.111, 0.01)
        image = self.gaussian_blur(image)

        gaussian_noise = torch.randn((3, 200, 200), device='cuda')
        image += gaussian_noise * 10

        # Contrast 0.6 ~ 1.4
        gamma = 0.8 * torch.rand(1, device='cuda') + 0.6
        image *= gamma

        # Brightness -30 ~ 30
        brightness = 60 * torch.rand(1, device='cuda') - 30
        image += brightness

        # Dropout
        image = self.dropout(image)
        return image

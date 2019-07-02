import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Augmentations(nn.Module):

    def __init__(self):
        super(Augmentations, self).__init__()
        self.gaussian_blur = nn.Conv2d(3, 3, (3, 3), padding=1, groups=3, bias=False)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, image):
        self.gaussian_blur.weight.data.normal_(0.111, 0.01)
        image = self.gaussian_blur(image)

        gaussian_noise = torch.randn((3, 200, 200), device='cuda')
        noise_scale = torch.rand(1, device='cuda') * 10
        image += gaussian_noise * noise_scale

        # Contrast 0.6 ~ 1.4
        gamma = 0.8 * torch.rand(1, device='cuda') + 0.6
        image *= gamma

        # Brightness -30 ~ 30
        brightness = 60 * torch.rand(1, device='cuda') - 30
        image += brightness

        # Dropout
        image = self.dropout(image)

        # Flip horizontal or vertical for NCHW
        dim = torch.rand(1, device='cuda').round().long().data.item() + 2
        image = torch.flip(image, (dim,))

        # Rotate -90 ~ 90
        angle = torch.rand(1, device='cuda') * 180 - 90
        elastic_scale = torch.rand((4), device='cuda')*0.4 + 0.8  # Also scale the image
        theta = torch.tensor([
                    [math.cos(angle) * elastic_scale[0], math.sin(-angle) * elastic_scale[1], 0],
                    [math.sin(angle) * elastic_scale[2], math.cos(angle) * elastic_scale[3], 0]
                ], device='cuda', dtype=torch.float)

        # 'theta' need to be Nx2x3 shape
        theta = theta.unsqueeze(0).repeat(image.size(0), 1, 1)
        grid = F.affine_grid(theta, image.size())
        image = F.grid_sample(image, grid)

        # Crop and resize
        orig_size = (image.size(2), image.size(3))
        crop_offset = torch.rand(1, device='cuda') * image.size(2) * 0.1
        crop_offset = crop_offset.long()
        crop_length = image.size(2) - crop_offset
        crop_length = crop_length.long()
        image = image[:, :, crop_offset:crop_length, crop_offset:crop_length]
        image = F.interpolate(image, orig_size)
        return image

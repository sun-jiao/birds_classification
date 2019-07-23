import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Augmentations(nn.Module):

    def __init__(self):
        super(Augmentations, self).__init__()
        self.gaussian_blur = nn.Conv2d(3, 3, (3, 3), padding=1, groups=3, bias=False).cuda()
        self.heavy_blur = nn.Conv2d(3, 3, (5, 5), padding=2, groups=3, bias=False).cuda()
        self.dropout = nn.Dropout(p=0.1).cuda()

    def forward(self, image):
        self.gaussian_blur.weight.data.normal_(0.111, 0.02)
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

        # Dropout or Heavy Gaussian Blur
        rand = torch.rand(1, device='cuda')
        if rand > 0.5:
            image = self.dropout(image)
        else:
            self.heavy_blur.weight.data.normal_(0.04, 0.02)
            image = self.heavy_blur(image)

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

        # Randomly to gray
        rand = torch.rand(1, device='cuda')
        if rand > 0.5:
            gray = image.sum(dim=1, dtype=torch.int16) / image.shape[1]
            image = torch.stack([gray, gray, gray], dim=1)
        return image


if __name__ == '__main__':
    import cv2
    files = ['/home/haodong/Downloads/WechatIMG17.jpeg',
             '/home/haodong/Downloads/17test.png',
             '/home/haodong/Downloads/example2.png']
    images = []
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, (200, 200))
        images.append(image)

    images = torch.Tensor(images).cuda().permute(0, 3, 1, 2).float()

    aug = Augmentations().cuda()
    imgs = aug(images).permute(0, 2, 3, 1).byte()

    for index, img in enumerate(imgs):
        cv2.imwrite('/home/haodong/Downloads/draw{}.jpg'.format(index), img.cpu().detach().numpy())

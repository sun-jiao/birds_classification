import cv2
import time
import argparse

import torch

from nets import resnet
from train import cfg

IMAGE_SHAPE = (200, 200)

def predict(args):
    net = resnet.resnext50_32x4d(num_classes=cfg['num_classes'])
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()

    img = cv2.imread(args.image_file)
    img = cv2.resize(img, IMAGE_SHAPE)
    t0 = time.time()
    tensor_img = torch.from_numpy(img)
    result = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
    t1 = time.time()
    print(result, 'time:', t1 - t0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Imag file to be predicted')
    parser.add_argument('--trained_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    predict(args)

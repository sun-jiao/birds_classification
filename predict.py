import cv2
import time
import argparse

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from train import cfg

IMAGE_SHAPE = (100, 100)


def predict(args):
    net = EfficientNet.from_name('efficientnet-b0', override_params={'num_classes': cfg['num_classes']})
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()

    softmax = nn.Softmax()

    img = cv2.imread(args.image_file)
    img = cv2.resize(img, IMAGE_SHAPE)
    t0 = time.time()
    tensor_img = torch.from_numpy(img)
    result = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
    result = softmax(result)
    values, indices = torch.max(result, 1)
    t1 = time.time()
    print(values, indices, 'time:', t1 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Imag file to be predicted')
    parser.add_argument('--trained_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    predict(args)

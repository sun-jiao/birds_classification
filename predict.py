import cv2
import csv
import time
import argparse

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

IMAGE_SHAPE = (300, 300)


def load_label_map(filename):
    label_map = {}

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            id = int(row[0])
            name = row[1]
            label_map[id] = name

    return label_map


def predict(args):
    net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})
    net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
    net.eval()

    softmax = nn.Softmax()

    img = cv2.imread(args.image_file)
    img = cv2.resize(img, IMAGE_SHAPE)
    t0 = time.time()
    tensor_img = torch.from_numpy(img)
    result = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
    result = softmax(result)
    values, indices = torch.topk(result, 10)
    t1 = time.time()

    print(indices)
    labelmap = load_label_map("labelmap.csv")
    for id, accu in zip(indices[0].tolist(), values[0].tolist()):
        print("{:1.4f}, {}".format(accu, labelmap.get(id, "Unknown")))
    print('time:', t1 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Imag file to be predicted')
    parser.add_argument('--trained_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    predict(args)

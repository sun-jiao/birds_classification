import os
import cv2
import csv
import sys
import time
import argparse

import torch

from efficientnet_pytorch import EfficientNet

IMAGE_SHAPE = (300, 300)


def check_top5(label_map, result, real_name):
    for id in result:
        pred_name = label_map.get(id, None)
        if real_name == pred_name:
            return True
    return False


def load_cn_label(filename):
    label_map = {}

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[1].isdigit():
                id = int(row[1])
                name = row[0]
                label_map[id] = name

    return label_map


def predict(net, image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, IMAGE_SHAPE)

    tensor_img = torch.from_numpy(img)
    result = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
    values, indices = torch.topk(result, 500)
    return indices[0].tolist()


def main(args):
    net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})
    net.load_state_dict(torch.load(args.classify_model, map_location='cpu'))
    net.eval()

    label_map = load_cn_label("V7.ioc101.id.mapping.txt")

    total_top1 = 0
    total_top5 = 0
    total_count = 0
    directory = next(os.walk(args.eval_data))
    for dir_name in directory[1]:
        top1 = 0
        top5 = 0
        count = 0
        real_name = dir_name
        for image_file in os.listdir(os.path.join(args.eval_data, dir_name)):
            full_path = os.path.join(args.eval_data, dir_name, image_file)
            result = predict(net, full_path)
            # filter out non-chinses-bird and get top5
            result = list(filter(lambda id: label_map.get(id, None) is not None, result))[:5]
            if len(result) != 5:
                print(f"Totally wrong prediction! {len(result)}", file=sys.stderr)
                sys.exit(1)
            # check top1
            pred_name = label_map.get(result[0], None)
            if pred_name == real_name:
                top1 += 1
                top5 += 1
            # check top5
            if check_top5(label_map, result[1:], real_name):
                top5 += 1
            count += 1
        if count == 0:
            print(f"{real_name}, {count:.4f}, {count:.4f}")
        else:
            print(f"{real_name}, {count}, {top1}, {top5}, {top1/count:.4f}, {top5/count:.4f}")

        total_top1 += top1
        total_top5 += top5
        total_count += count
    print(f"top1 accuracy: {total_top1/total_count:.4f}")
    print(f"top5 accuracy: {total_top5/total_count:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', default=None, type=str)
    parser.add_argument('--classify_model', default=None, type=str)
    args = parser.parse_args()
    main(args)

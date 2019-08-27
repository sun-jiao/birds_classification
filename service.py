from flask import Flask
from flask import request
from flask_cors import CORS
from efficientnet_pytorch import EfficientNet

from ssd import build_ssd
from data import BaseTransform

import torch
import torch.nn as nn

import cv2
import time
import json
import codecs
import numpy as np

IMAGE_SHAPE = (200, 200)

app = Flask(__name__)
CORS(app)

# For detection
dnet = build_ssd('test', 300, 21)
ckpt = torch.load('./ssd300_voc_90000.pth', map_location='cpu')
dnet.load_state_dict(ckpt)
dnet.eval()

transform = BaseTransform(dnet.size, (104, 117, 123))

# For classification
net = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 11000})
net.load_state_dict(torch.load('/media/data/sanbai/0.8005_big_categories.pth', map_location='cpu'))
net.eval()

softmax = nn.Softmax(dim=1)

with codecs.open('labelmap.csv', encoding='gbk') as fp:
    content = fp.readlines()

bird_map = {}
for line in content:
    arr = line.strip().split(',')
    id = int(arr[0])
    name = arr[1]
    bird_map[id] = name


def get_rectangles(detections, height, width):
    bird_index = 3  # For VOC
    scale = torch.Tensor([width, height, width, height])
    j = 0
    score = detections[0, bird_index, j, 0]
    locations = []
    while score >= 0.5:
        pt = (detections[0, bird_index, j, 1:] * scale).cpu().numpy()
        print('pt', pt, score)
        locations.append(pt)
        j += 1
        score = detections[0, bird_index, j, 0]
    return locations


@app.route('/birds', methods=['GET', 'POST'])
def birds():
    if request.method == 'POST':
        if 'image' not in request.files:
            print('No file')
        binary = request.files['image'].read()

        t0 = time.time()
        img = cv2.imdecode(np.fromstring(binary, dtype=np.uint8), 1)
        height, width = img.shape[:2]

        # For detection
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        detections = dnet(x.unsqueeze(0)).data
        locations = get_rectangles(detections, height, width)

        result = []
        for index, loc in enumerate(locations):
            print(index, loc)
            subimg = img[int(loc[1]):int(loc[3]), int(loc[0]):int(loc[2])]
            print('shape:', subimg.shape)
            cv2.imwrite('{}_try.jpg'.format(index), subimg)

            # For classification
            subimg = cv2.resize(subimg, IMAGE_SHAPE)

            tensor_img = torch.from_numpy(subimg)
            classified = net(tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float())
            classified = softmax(classified)
            values, indices = torch.topk(classified, 3)
            values = values.tolist()
            top3 = []
            index = 0
            for id in indices[0]:
                id = int(id)
                if id in bird_map:
                    top3.append([values[0][index] * 100, bird_map[id]])
                index += 1
            result.append({
                'loc': [int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])],
                'top3': top3
            })

        t1 = time.time()
        print('time:', t1 - t0)
        print(result)
        return json.dumps(result)
    if request.method == 'GET':
        return 'Get done'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

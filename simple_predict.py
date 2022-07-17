import sys
import cv2
import time
import json
import queue
import torch
import argparse
import threading

import torch
import numpy as np
import torch.nn as nn

import pycls.core.builders as model_builder
from pycls.core.config import cfg

def pressure_predict(net, tensor_img):
    t0 = time.time()
    for _ in range(10):
        result = net(tensor_img)
        result = softmax(result)
        values, indices = torch.topk(result, 10)
    t1 = time.time()
    print("time:", t1 - t0)
    print(values)

if __name__ == "__main__":
    cfg.MODEL.TYPE = "regnet"
    # RegNetY-8.0GF
    cfg.REGNET.DEPTH = 17
    cfg.REGNET.SE_ON = False
    cfg.REGNET.W0 = 192
    cfg.REGNET.WA = 76.82
    cfg.REGNET.WM = 2.19
    cfg.REGNET.GROUP_W = 56
    cfg.BN.NUM_GROUPS = 4
    cfg.MODEL.NUM_CLASSES = 11120
    net = model_builder.build_model()
    net.load_state_dict(torch.load("bird_cls_2754696.pth", map_location="cpu"))
    net.eval()
    model = net
    softmax = nn.Softmax(dim=1).eval()

    # read image
    img = cv2.imread("blujay.jpg")
    img = cv2.resize(img, (300, 300))
    tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    pressure_predict(net, tensor_img)

    dummy_input = torch.randn(1, 3, 300, 300)
    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(net, dummy_input)

    net = torch.jit.optimize_for_inference(traced_script_module)
    pressure_predict(net, tensor_img)

    import intel_extension_for_pytorch as ipex
    net = net.to(memory_format=torch.channels_last)
    net = ipex.optimize(net)
    tensor_img = tensor_img.to(memory_format=torch.channels_last)

    with torch.no_grad():
        pressure_predict(net, tensor_img)

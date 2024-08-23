import sys
from collections import OrderedDict

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
import torchvision
from pycls.core.config import cfg
from torch import softmax


def pressure_predict(net, tensor_img):
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            result = net(tensor_img)
            result = softmax(result)
            values, indices = torch.topk(result, 10)
    t1 = time.time()
    print("time:", t1 - t0)
    print(values)


if __name__ == "__main__":
    net = torchvision.models.resnet34(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 10320)

    # Load the state dict
    state_dict = torch.load('/home/sunjiao/Downloads/LBird-31_checkpoint.pth.tar', map_location=torch.device('cpu'))[
        'state_dict']

    # Create a new state dict with keys modified
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v

    # Load the new state dict
    net.load_state_dict(new_state_dict)

    # read image
    img = cv2.imread("blujay.jpg")
    img = cv2.resize(img, (300, 300))
    tensor_img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    pressure_predict(net, tensor_img)

    # quantization
    model_int8 = torch.quantization.quantize_dynamic(
        net,
        {torch.nn.Linear, torch.nn.Conv2d, torch.nn.GroupNorm},
        dtype=torch.qint8)
    torch.save(model_int8, "int8.pth")
    pressure_predict(model_int8, tensor_img)

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

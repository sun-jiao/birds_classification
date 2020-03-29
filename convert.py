import cv2
import time
import torch
import argparse

from torch.autograd import Variable
from efficientnet_pytorch import EfficientNet

IMAGE_SHAPE = (300, 300)


def convert(args):
    net = EfficientNet.from_name('efficientnet-b2', override_params={'num_classes': 11000})
    net.load_state_dict(torch.load(args.classify_model, map_location='cpu'))
    net.eval()
    dummy_input = Variable(torch.randn(1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    torch.onnx.export(
        net,
        dummy_input,
        args.output_onnx_model,
        export_params=True,
        input_names=['main_input'],
        output_names=['main_output'],
        verbose=False
    )

    img = cv2.imread(args.image_file)

    t0 = time.time()
    img = cv2.resize(img, IMAGE_SHAPE)

    tensor_img = torch.from_numpy(img)
    tensor_img = tensor_img.unsqueeze(0).permute(0, 3, 1, 2).float()
    print("Pytorch input shape:", tensor_img)
    result = net(tensor_img)
    print("Pytorch output:", result, result.shape)
    t1 = time.time()
    print('time:', t1 - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', default=None, type=str, help='Image file to be predicted')
    parser.add_argument('--classify_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    parser.add_argument('--output_onnx_model', default='my.onnx',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()

    convert(args)

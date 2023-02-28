import torch
import argparse
import pycls.core.builders as model_builder

from pycls.core.config import cfg
from torch.autograd import Variable

IMAGE_SHAPE = (300, 300)


def export(args):
    state_net = torch.load(args.classify_model, map_location="cpu")
    cfg.merge_from_other_cfg(state_net["_config"])
    del state_net["_config"]
    net = model_builder.build_model()
    net.load_state_dict(state_net)
    net.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    with torch.jit.optimized_execution(True):
        traced_script_module = torch.jit.trace(net, dummy_input)
        traced_script_module.save("bird_classification.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify_model', default='ckpt/bird_cls_0.pth',
                        type=str, help='Trained ckpt file path to open')
    args = parser.parse_args()
    export(args)

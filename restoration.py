import pycls.core.builders as model_builder
import torch
import torchvision.models
from pycls.core.config import cfg
from torch import jit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# RegNetY-8.0GF
cfg.MODEL.TYPE = "regnet"
cfg.REGNET.DEPTH = 20
cfg.REGNET.SE_ON = False
cfg.REGNET.W0 = 128
cfg.MODEL.NUM_CLASSES = 11000
net = model_builder.build_model()

net = net.to(device).eval()
print("net", net)

print(torchvision.models.regnet_y_8gf())

jnet = torch.load('model20200824.pth')
net.load_state_dict(jnet)

print()

import torch.nn as nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet(nn.Module):

    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.base = models.resnet34(pretrained=True)
        self.base.avgpool = Identity()
        self.base.fc = nn.Linear(7 * 7 * 512, num_classes)

    def forward(self, x):
        x = self.base(x)
        return x
